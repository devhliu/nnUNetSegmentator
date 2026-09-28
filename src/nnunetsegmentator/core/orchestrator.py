"""
Segmentation Orchestrator

This module provides the main orchestrator for managing segmentation tasks.
"""

from typing import Union, List, Dict, Any, Optional, Callable, Tuple
from pathlib import Path
import json
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
        pipeline: Pipeline = None,
        models: List[str] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            task_name: Name of registered task (e.g., 'gtrc', 'lion', 'deep_psma', 'moose')
            config: Configuration object
            model_path: Override model path
            pipeline: Custom pipeline (overrides task default)
            models: Optional subset of task models to run (e.g. selected
                automatically by ModelSelector or specified manually)
        """
        self.config = config or Config()
        self.model_path = Path(model_path) if model_path else None
        self.pipeline_mode = self.config.custom.get("pipeline_mode", "default")

        # Setup environment
        self.config.setup_nnunet_environment()

        if task_name:
            self.task = TaskRegistry.get_task(task_name)
            self.selected_models = self._resolve_selected_models(models)
            primary_name = self.selected_models[0]
            self.model_path = self.model_path or self._resolve_model_path(self.task, primary_name)
            self.pipeline = pipeline or self._build_pipeline(self.task, primary_name)
        else:
            self.task = None
            self.selected_models = []
            self.pipeline = pipeline


        logger.info(f"Orchestrator initialized for task: {task_name or 'custom'}")

    def _resolve_selected_models(self, models: List[str] = None) -> List[str]:
        """
        Resolve and validate the model subset to run (defaults to the task's
        primary model order).
        """
        if not self.task.models:
            raise PipelineConfigurationError(f"Task '{self.task.name}' has no models configured")

        if not models:
            return [next(iter(self.task.models))]

        unknown = [name for name in models if name not in self.task.models]
        if unknown:
            raise PipelineConfigurationError(
                f"Unknown model(s) for task '{self.task.name}': {', '.join(unknown)}. "
                f"Available: {', '.join(self.task.models)}"
            )
        return list(models)

    def _build_pipeline(self, task: TaskDefinition, model_name: str = None) -> Pipeline:
        """
        Build pipeline from task definition.

        Args:
            task: Task definition
            model_name: Model whose pipeline is built (defaults to primary model).
                Tasks may provide per-model step lists via
                ``pipeline_config['model_steps']``.

        Returns:
            Configured pipeline
        """
        from ..pipeline.builders import PipelineBuilder
        from ..pipeline.steps import MergeLabelsStep
        pipeline_config = task.pipeline_config
        if model_name and 'model_steps' in pipeline_config:
            steps = pipeline_config['model_steps'].get(model_name, pipeline_config.get('steps', []))
            pipeline_config = {'name': pipeline_config.get('name', 'pipeline'), 'steps': steps}
        builder = PipelineBuilder(
            pipeline_config,
            variables={
                "model_path": str(self.model_path) if self.model_path else None,
                "threshold": self.config.custom.get("threshold"),
                "_task_models": task.models,
                "_output_labels": task.output_config.get("labels", {}),
            },
            mode=self.pipeline_mode,
        )
        pipeline = builder.build()
        # Every task finishes by collapsing the fine-grained labels that the
        # central SNOMED mapping assigns to the same coarser concept (e.g. the
        # left lung lobes become 'Left Lung'), so label and segmentation
        # outputs stay consistent across tasks.
        pipeline.add_step(MergeLabelsStep('merge_labels', {}))
        return pipeline

    def _resolve_model_path(self, task: TaskDefinition, model_name: str = None) -> Path:
        """
        Resolve model path for a task model (defaults to the primary model).
        """
        if not task.models:
            raise PipelineConfigurationError(f"Task '{task.name}' has no models configured")

        model_name = model_name or next(iter(task.models))
        model_info = task.models[model_name]

        configured_model_path = self.config.get_model_path(task.name, model_info.task_id)
        if configured_model_path.exists():
            TaskRegistry.register_model_path(model_name, configured_model_path)
            return configured_model_path

        return TaskRegistry.get_model_path(model_name)

    def _enrich_runtime_labels(self, model_name: str) -> Dict[str, int]:
        """
        Enrich model labels from the downloaded payload when the task opts in
        (output_config['runtime_labels']); returns the model's label dict.
        """
        model_info = self.task.models[model_name]
        if not self.task.output_config.get("runtime_labels"):
            return model_info.labels
        try:
            from ..tasks.moose import ensure_runtime_labels
        except ImportError:
            return model_info.labels
        if self.task.name != "moose":
            return model_info.labels
        return ensure_runtime_labels(model_name)

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

        # Per-model tasks (e.g. MOOSE) run each selected model independently
        # and emit one segmentation + organ-mapping sidecar per model.
        if self._is_per_model() and len(self.selected_models) > 1:
            return self._segment_per_model(
                input_data,
                output_path=output_path,
                return_labels=return_labels,
                compute_metrics=compute_metrics,
                accelerator=accelerator,
                **kwargs
            )

        primary_model = self.selected_models[0] if self.selected_models else None
        return self._segment_single(
            input_data,
            output_path=output_path,
            return_labels=return_labels,
            compute_metrics=compute_metrics,
            accelerator=accelerator,
            model_name=primary_model,
            **kwargs
        )

    def _is_per_model(self) -> bool:
        """Whether the task requests per-model output handling."""
        return bool(self.task and self.task.output_config.get("strategy") == "per_model")

    def _segment_single(
        self,
        input_data: Union[str, Path, Dict[str, str], sitk.Image, np.ndarray],
        output_path: Union[str, Path] = None,
        return_labels: bool = True,
        compute_metrics: bool = False,
        accelerator: str = None,
        model_name: str = None,
        **kwargs
    ) -> SegmentationResult:
        """
        Run the pipeline once for a single (selected) model.
        """
        # Read input
        images, metadata = self._read_input(input_data)
        if self.task and "labels" in self.task.output_config:
            if self._is_per_model() and model_name:
                # Use the selected model's (runtime-enriched) label set
                model_labels = self._enrich_runtime_labels(model_name)
                metadata["labels"] = {label_id: name for name, label_id in model_labels.items()}
            else:
                metadata["labels"] = self.task.output_config["labels"]

        if output_path:
            output_path = Path(output_path)
            metadata["output_dir"] = str(output_path.parent)
            metadata["case_id"] = output_path.stem.replace(".nii", "")
        else:
            metadata["output_dir"] = str(self.config.output_dir)

        # Unified DICOM handling: persist the input series as NIfTI before
        # segmentation and keep the reference series for DICOM-SEG export.
        dicom_reference = None
        if metadata.get("format") == "dicom" and output_path is not None:
            dicom_reference = self._prepare_dicom_input(
                images, metadata, save_converted_nifti=kwargs.get("save_converted_nifti", True))

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
            labels = self._split_labels(
                segmentation,
                self.task,
                model_name,
                merge_map=context.intermediate_results.get('label_merge_map'),
            )

        # Build SNOMED organ mapping when the task opts in
        organ_mapping = None
        if self._is_per_model() and self.task.output_config.get("emit_label_mapping") and model_name:
            final_labels = self._normalize_labels_id_to_name(metadata.get("labels") or {}) or None
            organ_mapping = self._build_organ_mapping(model_name, final_labels)
            metadata["organ_mapping"] = organ_mapping

        # Compute metrics if requested
        metrics = None
        if compute_metrics:
            metrics = self._compute_metrics(segmentation, images)

        # Write output if path provided
        if output_path:
            self._write_output(segmentation, output_path, labels, organ_mapping=organ_mapping)

            # Unified DICOM handling: export the segmentation as DICOM-SEG
            # alongside the NIfTI output.
            if dicom_reference and kwargs.get("export_dicom_seg", True):
                self._export_dicom_seg(segmentation, metadata, output_path, dicom_reference)

        logger.info("Segmentation complete")

        return SegmentationResult(
            segmentation=segmentation,
            labels=labels,
            metadata=metadata,
            metrics=metrics or {}
        )

    def _segment_per_model(
        self,
        input_data: Union[str, Path, Dict[str, str], sitk.Image, np.ndarray],
        output_path: Union[str, Path] = None,
        return_labels: bool = True,
        compute_metrics: bool = False,
        accelerator: str = None,
        **kwargs
    ) -> SegmentationResult:
        """
        Run every selected model sequentially, writing one segmentation and
        organ-mapping sidecar per model (per_model output strategy).

        The combined result uses '<model>/<label>' label keys and reports
        per-model outputs under metadata['per_model'].
        """
        if output_path:
            output_path = Path(output_path)
            base_dir = output_path.parent
            base_stem = self._strip_nifti_suffix(output_path.name)
            if base_stem.endswith("_seg"):
                base_stem = base_stem[: -len("_seg")]
        else:
            base_dir = None
            base_stem = "segmentation"

        original_model_path = self.model_path
        original_pipeline = self.pipeline

        combined_labels: Dict[str, sitk.Image] = {}
        organ_mappings: Dict[str, Any] = {}
        per_model_metadata: Dict[str, Any] = {}
        primary_result: Optional[SegmentationResult] = None

        try:
            for model_name in self.selected_models:
                logger.info("Running per-model segmentation for '%s'", model_name)
                self.model_path = self._resolve_model_path(self.task, model_name)
                self.pipeline = self._build_pipeline(self.task, model_name)

                model_output = (
                    base_dir / f"{base_stem}__{model_name}_seg.nii.gz"
                    if base_dir else None
                )
                result = self._segment_single(
                    input_data,
                    output_path=model_output,
                    return_labels=return_labels,
                    compute_metrics=compute_metrics,
                    accelerator=accelerator,
                    model_name=model_name,
                    **kwargs
                )
                if primary_result is None:
                    primary_result = result
                for label_name, label_image in result.labels.items():
                    combined_labels[f"{model_name}/{label_name}"] = label_image
                if result.metadata.get("organ_mapping"):
                    organ_mappings[model_name] = result.metadata["organ_mapping"]
                per_model_metadata[model_name] = {
                    "output": str(model_output) if model_output else None,
                    "labels": result.metadata.get("labels", {}),
                }
        finally:
            self.model_path = original_model_path
            self.pipeline = original_pipeline

        metadata = dict(primary_result.metadata) if primary_result else {}
        metadata.pop("labels", None)
        metadata["per_model"] = per_model_metadata
        metadata["organ_mapping"] = organ_mappings

        return SegmentationResult(
            segmentation=primary_result.segmentation if primary_result else sitk.Image(),
            labels=combined_labels,
            metadata=metadata,
            metrics=primary_result.metrics if primary_result else {}
        )

    @staticmethod
    def _strip_nifti_suffix(name: str) -> str:
        """Strip the full NIfTI suffix chain ('.nii.gz') from a filename."""
        suffixes = "".join(Path(name).suffixes)
        return name[: -len(suffixes)] if suffixes else name
    
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
        task: TaskDefinition,
        model_name: str = None,
        merge_map: Dict[str, str] = None
    ) -> Dict[str, sitk.Image]:
        """
        Split multi-label segmentation into individual labels.

        Args:
            segmentation: Multi-label segmentation
            task: Task definition with label information
            model_name: Model whose label set drives the split (defaults to
                the task's first model)
            merge_map: Optional {label name: merged label name} map produced by
                the merge_labels pipeline step; every name sharing a merged
                label is combined into that single output label

        Returns:
            Dictionary mapping label names to binary images
        """
        labels = {}
        seg_array = sitk.GetArrayFromImage(segmentation)

        if model_name and model_name in task.models:
            model_info = task.models[model_name]
        else:
            model_info = list(task.models.values())[0]

        # Label tables are declared in either orientation ({name: id} or
        # {id: name}); normalize so both split correctly.
        id_to_name = self._normalize_labels_id_to_name(model_info.labels)
        for label_id, label_name in id_to_name.items():
            label_array = (seg_array == label_id).astype(np.uint8)
            label_image = sitk.GetImageFromArray(label_array)
            label_image.CopyInformation(segmentation)

            target_name = (merge_map or {}).get(label_name, label_name)
            if target_name in labels:
                label_image = self._combine_label_images(labels[target_name], label_image)

            labels[target_name] = label_image

        return labels

    @staticmethod
    def _combine_label_images(first: sitk.Image, second: sitk.Image) -> sitk.Image:
        """Union two binary label images (used when merging fine labels)."""
        combined = np.maximum(
            sitk.GetArrayFromImage(first), sitk.GetArrayFromImage(second)
        ).astype(np.uint8)
        combined_image = sitk.GetImageFromArray(combined)
        combined_image.CopyInformation(first)
        return combined_image
    
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
    
    def _prepare_dicom_input(
        self,
        image: sitk.Image,
        metadata: Dict[str, Any],
        save_converted_nifti: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Unified DICOM input handling: convert the series to NIfTI before
        segmentation and collect the reference series for DICOM-SEG export.

        The converted NIfTI is named after the input series (not the output
        case), so per-model runs reuse the same file (idempotent write).
        """
        reference = metadata.get("dicom_files") or metadata.get("dicom_dir")
        if not reference:
            logger.warning("DICOM input lacks series reference; skipping NIfTI conversion")
            return None

        if save_converted_nifti:
            input_stem = Path(
                metadata.get("dicom_dir") or metadata.get("filepath") or "dicom_series"
            ).name
            nifti_path = Path(metadata["output_dir"]) / f"{input_stem}_image"
            if not nifti_path.with_suffix(".nii.gz").exists():
                written = ImageWriter.write(image, nifti_path)
                logger.info("DICOM series converted to NIfTI: %s", written)
            metadata["nifti_path"] = str(nifti_path) + ".nii.gz"

        return {"files": reference}

    def _export_dicom_seg(
        self,
        segmentation: sitk.Image,
        metadata: Dict[str, Any],
        output_path: Path,
        dicom_reference: Dict[str, Any],
    ) -> Optional[Path]:
        """
        Export the segmentation as a DICOM-SEG object next to the NIfTI
        output, using the source DICOM series as reference.
        """
        try:
            from ..io.dicom_export import DICOMSEGExporter
            exporter = DICOMSEGExporter()
        except ImportError as exc:
            logger.warning("DICOM-SEG export skipped: %s", exc)
            return None

        labels = self._normalize_labels_id_to_name(metadata.get("labels") or {})
        if not labels:
            logger.warning("DICOM-SEG export skipped: no label mapping available")
            return None

        case_id = metadata.get("case_id") or self._strip_nifti_suffix(output_path.name)
        seg_path = output_path.parent / f"{case_id}.dcm"

        try:
            return exporter.export(
                segmentation=segmentation,
                reference_dicom_dir=dicom_reference["files"],
                output_path=seg_path,
                labels=labels,
                series_description=(
                    f"nnunetsegmentator {self.task.name}" if self.task else "nnunetsegmentator"
                ),
            )
        except (ValueError, RuntimeError) as exc:
            logger.warning("DICOM-SEG export failed for '%s': %s", seg_path, exc)
            return None

    @staticmethod
    def _normalize_labels_id_to_name(labels: Dict[Any, Any]) -> Dict[int, str]:
        """
        Normalize a label mapping to {label_id: name}.

        Task metadata may hold either orientation ({1: 'spleen'} or
        {'spleen': 1}); the DICOM-SEG exporter requires {id: name}.
        """
        normalized: Dict[int, str] = {}
        for key, value in labels.items():
            if isinstance(key, int):
                normalized[key] = str(value)
            else:
                try:
                    normalized[int(value)] = str(key)
                except (TypeError, ValueError):
                    continue
        return normalized

    def _write_output(
        self,
        segmentation: sitk.Image,
        output_path: Path,
        labels: Dict[str, sitk.Image],
        organ_mapping: Dict[str, Any] = None
    ):
        """
        Write segmentation outputs.

        Args:
            segmentation: Main segmentation
            output_path: Output path
            labels: Individual label images
            organ_mapping: Optional SNOMED organ-mapping payload written as a
                JSON sidecar next to the segmentation
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

        # Write organ-mapping sidecar (model-native names <-> SNOMED)
        if organ_mapping:
            self._write_organ_mapping_sidecar(output_path, organ_mapping)

        logger.info(f"Output written to: {output_path}")

    def _build_organ_mapping(
        self, model_name: str, label_table: Dict[int, str] = None
    ) -> Dict[str, Any]:
        """
        Build the SNOMED organ-mapping payload for a model: label names
        cross-referenced with canonical names, SNOMED codes, laterality and
        display RGB.

        ``label_table`` is the final ``{label_id: name}`` table (i.e. after the
        merge_labels step) and defaults to the model's own labels when omitted.
        """
        from ..mapping import LabelMapper

        model_info = self.task.models[model_name]
        if label_table is None:
            label_table = self._normalize_labels_id_to_name(
                self._enrich_runtime_labels(model_name)
            )

        mapper = LabelMapper.get_instance()
        organ_indices: Dict[str, Any] = {}
        for label_id, label_name in label_table.items():
            entry = mapper.resolve(label_name)
            organ_indices[str(label_id)] = {
                "name": label_name,
                "canonical": entry.canonical_name,
                "SNOMED": entry.type.to_dict() if entry.type else None,
                "laterality": entry.laterality,
                "rgb": entry.rgb,
            }

        return {
            "model": model_name,
            "task": self.task.name,
            "task_id": model_info.task_id,
            "organ_indices": organ_indices,
        }

    def _write_organ_mapping_sidecar(
        self,
        output_path: Path,
        organ_mapping: Dict[str, Any]
    ) -> Path:
        """Write the organ-mapping payload as '<stem>_organ_mapping.json'."""
        stem = self._strip_nifti_suffix(output_path.name)
        if stem.endswith("_seg"):
            stem = stem[: -len("_seg")]
        sidecar_path = output_path.parent / f"{stem}_organ_mapping.json"
        with open(sidecar_path, "w", encoding="utf-8") as f:
            json.dump(organ_mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"Organ mapping sidecar written to: {sidecar_path}")
        return sidecar_path
