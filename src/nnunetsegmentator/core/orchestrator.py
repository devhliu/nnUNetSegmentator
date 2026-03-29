"""
Segmentation Orchestrator

This module provides the main orchestrator for managing segmentation tasks.
"""

from typing import Union, List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import os
from .. import image as sitk
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
import logging

from .registry import TaskRegistry, TaskDefinition
from .config import Config
from .exceptions import PipelineConfigurationError
from .gpu_manager import get_gpu_monitor
from ..io.readers import ImageReader, MultiModalReader
from ..io.writers import ImageWriter
from ..pipeline.base import Pipeline, PipelineContext

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """
    Container for segmentation results.
    
    This class holds the segmentation output along with metadata and metrics.
    """
    segmentation: sitk.Image
    labels: Dict[str, sitk.Image]
    metadata: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_array(self) -> np.ndarray:
        """Get segmentation as numpy array"""
        return sitk.GetArrayFromImage(self.segmentation)
    
    def get_label_array(self, label_name: str) -> np.ndarray:
        """Get specific label as numpy array"""
        if label_name not in self.labels:
            raise KeyError(f"Label '{label_name}' not found")
        return sitk.GetArrayFromImage(self.labels[label_name])


class SegmentationOrchestrator:
    """
    Main orchestrator for nnUNet-based segmentation tasks.
    
    This class provides a unified interface for:
    - Single subject and batch processing
    - Multiple input formats (NIfTI, DICOM)
    - Custom pipeline configuration
    - Task management
    """
    
    def __init__(
        self,
        task_name: str = None,
        config: Config = None,
        model_path: Union[str, Path] = None,
        pipeline: Pipeline = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            task_name: Name of registered task (e.g., 'gtrc', 'lion', 'deep_psma')
            config: Configuration object
            model_path: Override model path
            pipeline: Custom pipeline (overrides task default)
        """
        self.config = config or Config()
        self.model_path = Path(model_path) if model_path else None
        self.pipeline_mode = self.config.custom.get("pipeline_mode", "default")
        
        # Setup environment
        self.config.setup_nnunet_environment()
        
        if task_name:
            self.task = TaskRegistry.get_task(task_name)
            self.model_path = self.model_path or self._resolve_primary_model_path(self.task)
            self.pipeline = pipeline or self._build_pipeline(self.task)
        else:
            self.task = None
            self.pipeline = pipeline

        
        logger.info(f"Orchestrator initialized for task: {task_name or 'custom'}")
    
    def _build_pipeline(self, task: TaskDefinition) -> Pipeline:
        """
        Build pipeline from task definition.
        
        Args:
            task: Task definition
            
        Returns:
            Configured pipeline
        """
        from ..pipeline.builders import PipelineBuilder
        builder = PipelineBuilder(
            task.pipeline_config,
            variables={
                "model_path": str(self.model_path) if self.model_path else None,
                "threshold": self.config.custom.get("threshold"),
                "_task_models": task.models,
                "_output_labels": task.output_config.get("labels", {}),
            },
            mode=self.pipeline_mode,
        )
        return builder.build()

    def _resolve_primary_model_path(self, task: TaskDefinition) -> Path:
        """
        Resolve default model path for the task's primary model.
        """
        if not task.models:
            raise PipelineConfigurationError(f"Task '{task.name}' has no models configured")

        primary_name = next(iter(task.models))
        model_info = task.models[primary_name]

        configured_model_path = self.config.get_model_path(task.name, model_info.task_id)
        if configured_model_path.exists():
            TaskRegistry.register_model_path(primary_name, configured_model_path)
            return configured_model_path

        return TaskRegistry.get_model_path(primary_name)

    def _validate_runtime(self) -> None:
        """
        Validate runtime requirements before executing segmentation.
        """
        if self.pipeline is None:
            raise PipelineConfigurationError("No pipeline configured. Provide task_name or pipeline.")

        if not self.pipeline.validate():
            raise PipelineConfigurationError("Pipeline validation failed")

        required_env_vars = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
        missing_env_vars = [var for var in required_env_vars if not os.environ.get(var)]
        if missing_env_vars:
            raise PipelineConfigurationError(
                f"Missing required nnUNet environment variables: {', '.join(missing_env_vars)}"
            )

        for step in self.pipeline.steps:
            unresolved = self._find_unresolved_placeholders(step.config)
            if unresolved:
                raise PipelineConfigurationError(
                    f"Unresolved placeholders in step '{step.name}': {', '.join(unresolved)}"
                )

            model_path = step.config.get("model_path")
            if isinstance(model_path, str):
                if not Path(model_path).exists():
                    raise FileNotFoundError(
                        f"Configured model path does not exist for step '{step.name}': {model_path}"
                    )

            model_name = step.config.get("model_name")
            if isinstance(model_name, str):
                resolved = TaskRegistry.get_model_path(model_name)
                if not resolved.exists():
                    raise FileNotFoundError(
                        f"Configured model '{model_name}' resolved to missing path: {resolved}"
                    )

    def _find_unresolved_placeholders(self, value: Any) -> List[str]:
        """
        Recursively find unresolved ${...} placeholders in a config structure.
        """
        unresolved: List[str] = []

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                for item in node.values():
                    _walk(item)
                return
            if isinstance(node, list):
                for item in node:
                    _walk(item)
                return
            if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
                unresolved.append(node)

        _walk(value)
        return unresolved
    
    def segment(
        self,
        input_data: Union[str, Path, Dict[str, str], sitk.Image, np.ndarray],
        output_path: Union[str, Path] = None,
        return_labels: bool = True,
        compute_metrics: bool = False,
        accelerator: str = None,
        **kwargs
    ) -> SegmentationResult:
        """
        Perform segmentation on single subject.
        
        Args:
            input_data: Input data (file path, dict of paths, or image)
            output_path: Optional output path
            return_labels: Return individual label images
            compute_metrics: Compute segmentation metrics
            accelerator: Optional accelerator device (e.g., 'cuda:0', 'mps')
            **kwargs: Additional options
            
        Returns:
            SegmentationResult object
        """
        logger.info("Starting segmentation")
        self._validate_runtime()
        
        # Read input
        images, metadata = self._read_input(input_data)
        if self.task and "labels" in self.task.output_config:
            metadata["labels"] = self.task.output_config["labels"]

        if output_path:
            output_path = Path(output_path)
            metadata["output_dir"] = str(output_path.parent)
            metadata["case_id"] = output_path.stem.replace(".nii", "")
        else:
            metadata["output_dir"] = str(self.config.output_dir)
        
        # Create pipeline context
        context = PipelineContext(
            input_image=images,
            input_array=sitk.GetArrayFromImage(images),
            metadata=metadata
        )
        
        # Execute pipeline
        # Add accelerator to context metadata for Herd Mode
        if accelerator:
            context.metadata['accelerator'] = accelerator
        
        context = self.pipeline.execute(context)
        
        # Extract results
        segmentation = self._extract_segmentation(context)
        
        # Split into labels if requested
        labels = {}
        if return_labels and self.task:
            labels = self._split_labels(segmentation, self.task)
        
        # Compute metrics if requested
        metrics = None
        if compute_metrics:
            metrics = self._compute_metrics(segmentation, images)
        
        # Write output if path provided
        if output_path:
            self._write_output(segmentation, output_path, labels)
        
        logger.info("Segmentation complete")
        
        return SegmentationResult(
            segmentation=segmentation,
            labels=labels,
            metadata=metadata,
            metrics=metrics or {}
        )
    
    def segment_batch(
        self,
        input_list: List[Union[str, Path, Dict[str, str]]],
        output_dir: Union[str, Path],
        num_workers: int = 1,
        use_multiprocessing: bool = False,
        progress_callback: Callable = None,
        continue_on_error: bool = False,
        **kwargs
    ) -> List[Optional[SegmentationResult]]:
        """
        Perform segmentation on multiple subjects with optional distributed inference (Herd Mode).
        
        Args:
            input_list: List of input data
            output_dir: Output directory
            num_workers: Number of parallel workers
            use_multiprocessing: Use multiprocessing instead of threading
            progress_callback: Optional callback for progress updates
            **kwargs: Additional options passed to segment()
            
        Returns:
            List of SegmentationResult objects
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting batch segmentation with {len(input_list)} subjects")

        gpu_monitor = get_gpu_monitor()
        accelerator_assignments = gpu_monitor.assign_devices(len(input_list))

        if gpu_monitor.has_gpu:
            if gpu_monitor.device_count > 1:
                logger.info(f"Herd Mode: Distributing across {gpu_monitor.device_count} GPUs")
                if not use_multiprocessing:
                    logger.warning("Multiple GPUs detected. Forcing multiprocessing for Herd Mode.")
                    use_multiprocessing = True
            else:
                logger.info(f"Herd Mode: Using device {gpu_monitor.devices[0]}")
        
        results: List[Optional[SegmentationResult]] = []
        
        executor_class = ProcessPoolExecutor if use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=num_workers) as executor:
            futures = []
            for i, input_data in enumerate(input_list):
                # Determine output path
                if isinstance(input_data, dict):
                    name = Path(list(input_data.values())[0]).stem
                else:
                    name = Path(input_data).stem
                output_path = output_dir / f"{name}_seg.nii.gz"
                
                # Get assigned accelerator
                accelerator = accelerator_assignments[i]
                
                future = executor.submit(
                    self.segment,
                    input_data,
                    output_path,
                    accelerator=accelerator,
                    **kwargs
                )
                futures.append((i, future))
            
            # Collect results
            for i, future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    if progress_callback:
                        progress_callback(i + 1, len(input_list), result)
                except (RuntimeError, ValueError, OSError) as exc:
                    logger.exception("Failed to process subject %s", i)
                    if continue_on_error:
                        results.append(None)
                        continue
                    raise RuntimeError(f"Batch segmentation failed for subject index {i}") from exc
        
        logger.info(f"Batch segmentation complete: {len([r for r in results if r is not None])}/{len(input_list)} successful")
        
        return results
    
    def _read_input(
        self,
        input_data: Union[str, Path, Dict[str, str], sitk.Image, np.ndarray]
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        """
        Read and prepare input data.
        
        Args:
            input_data: Input in various formats
            
        Returns:
            Tuple of (image, metadata)
        """
        if isinstance(input_data, sitk.Image):
            metadata = {'source': 'sitk_image'}
            return input_data, metadata
        
        if isinstance(input_data, np.ndarray):
            image = sitk.GetImageFromArray(input_data)
            metadata = {'source': 'numpy_array', 'shape': input_data.shape}
            return image, metadata
        
        if isinstance(input_data, dict):
            # Multi-modal input
            results = MultiModalReader.read(input_data)
            # Return primary modality
            primary_modality = list(input_data.keys())[0]
            return results[primary_modality]
        
        # Single file/directory
        return ImageReader.read(input_data)
    
    def _extract_segmentation(self, context: PipelineContext) -> sitk.Image:
        """
        Extract final segmentation from context.
        
        Args:
            context: Pipeline context after execution
            
        Returns:
            Segmentation as SimpleITK image
        """
        seg_array = context.intermediate_results.get('final_prediction')
        if seg_array is None:
            seg_array = context.intermediate_results.get('refined_prediction')
        if seg_array is None:
            seg_array = context.intermediate_results.get('raw_prediction')
        if seg_array is None:
            raise RuntimeError("Pipeline produced no segmentation output")
        
        seg_image = sitk.GetImageFromArray(seg_array)
        seg_image.CopyInformation(context.input_image)
        
        return seg_image
    
    def _split_labels(
        self,
        segmentation: sitk.Image,
        task: TaskDefinition
    ) -> Dict[str, sitk.Image]:
        """
        Split multi-label segmentation into individual labels.
        
        Args:
            segmentation: Multi-label segmentation
            task: Task definition with label information
            
        Returns:
            Dictionary mapping label names to binary images
        """
        labels = {}
        seg_array = sitk.GetArrayFromImage(segmentation)
        
        model_info = list(task.models.values())[0]
        for label_name, label_value in model_info.labels.items():
            label_array = (seg_array == label_value).astype(np.uint8)
            label_image = sitk.GetImageFromArray(label_array)
            label_image.CopyInformation(segmentation)
            labels[label_name] = label_image
        
        return labels
    
    def _compute_metrics(
        self,
        segmentation: sitk.Image,
        reference: sitk.Image
    ) -> Dict[str, float]:
        """
        Compute segmentation metrics.
        
        Args:
            segmentation: Segmentation image
            reference: Reference image
            
        Returns:
            Dictionary of metrics
        """
        from ..utils.metrics import compute_volume
        
        seg_array = sitk.GetArrayFromImage(segmentation)
        spacing = reference.GetSpacing()
        
        metrics = {
            'volume_mm3': compute_volume(seg_array, spacing),
            'num_voxels': int(np.sum(seg_array > 0))
        }
        
        return metrics
    
    def _write_output(
        self,
        segmentation: sitk.Image,
        output_path: Path,
        labels: Dict[str, sitk.Image]
    ):
        """
        Write segmentation outputs.
        
        Args:
            segmentation: Main segmentation
            output_path: Output path
            labels: Individual label images
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write main segmentation
        ImageWriter.write(segmentation, output_path)
        
        # Write individual labels
        if labels:
            label_dir = output_path.parent / f"{output_path.stem}_labels"
            label_dir.mkdir(exist_ok=True)
            for label_name, label_image in labels.items():
                ImageWriter.write(label_image, label_dir / f"{label_name}.nii.gz")
        
        logger.info(f"Output written to: {output_path}")
