"""
Pipeline Builders

This module provides utilities for constructing pipelines from configurations.
"""

from typing import Dict, Any, List, Optional
import logging

from .base import Pipeline, PipelineStep
from .steps import (
    ResampleStep,
    ClipIntensityStep,
    NormalizeStep,
    SUVThresholdStep,
    SUVConversionStep,
    MultiChannelStackStep,
    ResampleNibabelStep,
    ProjectionStep,
    nnUNetInferenceStep,
    CascadeInferenceStep,
    ExpandContractStep,
    LargestComponentStep,
    MorphologicalOpsStep,
    FillHolesStep,
    FilterLabelStep,
    IntensityMaskStep,
    RemoveSmallObjectsStep,
    ThresholdProbabilityStep,
    MultiLabelPostprocessingStep,
    AnatomicalConstraintsStep,
    DistanceRefinementStep,
    ShapeConstraintStep,
    PetThresholdStep,
    VertebraeLocalizationStep,
    BodyCompositionMetricsStep,
    PreserveOriginalStep,
    TumorMetricsStep,
    EnsembleInferenceStep,
    MultiModelConcatStep,
    TotalSegmentatorEnsembleStep,
    ROIProcessingStep,
)
from .steps.visualization import MIPGenerationStep

logger = logging.getLogger(__name__)


class PipelineBuilder:
    """
    Builder for constructing pipelines from configuration dictionaries.
    
    This class provides a declarative way to build pipelines without
    manually instantiating each step.
    """
    
    # Registry of available steps
    STEP_REGISTRY = {
        # Preprocessing
        'resample': ResampleStep,
        'resample_nibabel': ResampleNibabelStep,
        'clip_intensity': ClipIntensityStep,
        'normalize': NormalizeStep,
        'suv_threshold': SUVThresholdStep,
        'suv_conversion': SUVConversionStep,
        'multi_channel_stack': MultiChannelStackStep,
        'projection': ProjectionStep,
        'preserve_original': PreserveOriginalStep,
        # Inference
        'nnunet_inference': nnUNetInferenceStep,
        'cascade_inference': CascadeInferenceStep,
        'ensemble_inference': EnsembleInferenceStep,
        'multi_model_concat': MultiModelConcatStep,
        'totalsegmentator_ensemble': TotalSegmentatorEnsembleStep,
        'roi': ROIProcessingStep,
        # Postprocessing
        'expand_contract': ExpandContractStep,
        'largest_component': LargestComponentStep,
        'morphological_ops': MorphologicalOpsStep,
        'fill_holes': FillHolesStep,
        'filter_label': FilterLabelStep,
        'intensity_mask': IntensityMaskStep,
        'remove_small_objects': RemoveSmallObjectsStep,
        'threshold_probability': ThresholdProbabilityStep,
        'multi_label_postprocessing': MultiLabelPostprocessingStep,
        'anatomical_constraints': AnatomicalConstraintsStep,
        'distance_refinement': DistanceRefinementStep,
        'shape_constraint': ShapeConstraintStep,
        'pet_threshold': PetThresholdStep,
        # Visualization
        'mip_generation': MIPGenerationStep,
        # Quantification
        'vertebrae_localization': VertebraeLocalizationStep,
        'body_composition_metrics': BodyCompositionMetricsStep,
        'tumor_metrics': TumorMetricsStep,
    }
    
    def __init__(
        self,
        config: Dict[str, Any],
        variables: Optional[Dict[str, Any]] = None,
        mode: str = "default",
    ):
        """
        Initialize builder.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.variables = variables or {}
        self.mode = mode
    
    def build(self) -> Pipeline:
        """
        Build pipeline from configuration.
        
        Returns:
            Constructed pipeline
        """
        normalized = self._normalize_pipeline_config(self.config)
        pipeline_name = normalized.get('name', 'custom_pipeline')
        steps_config = normalized.get('steps', [])
        
        pipeline = Pipeline(name=pipeline_name)
        
        for step_config in steps_config:
            step = self._build_step(step_config)
            pipeline.add_step(step)
        
        logger.info(f"Built pipeline '{pipeline_name}' with {len(pipeline)} steps")
        
        return pipeline

    def _normalize_pipeline_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize task pipeline config to canonical {'name', 'steps'} format.
        Supports both:
        - Flat config with explicit 'steps'
        - Mode-based config with 'default'/'fast' and pre/inference/post sections
        """
        if 'steps' in config:
            return self._resolve_placeholders(config)

        if {'preprocessing', 'inference', 'postprocessing'} & set(config.keys()):
            mode_config = config
        else:
            if self.mode in config and isinstance(config[self.mode], dict):
                mode_config = config[self.mode]
            elif 'default' in config and isinstance(config['default'], dict):
                mode_config = config['default']
            else:
                raise ValueError(
                    "Invalid pipeline config: expected 'steps' or a mode-based "
                    "configuration with 'default' and pre/inference/post sections"
                )

        steps: List[Dict[str, Any]] = []
        for step in mode_config.get('preprocessing', []):
            steps.append(self._normalize_step(step))

        inference_config = mode_config.get('inference')
        if inference_config:
            steps.extend(self._normalize_inference(inference_config))

        for step in mode_config.get('postprocessing', []):
            steps.append(self._normalize_step(step))

        pipeline_name = config.get('name', f'pipeline_{self.mode}')
        return self._resolve_placeholders({'name': pipeline_name, 'steps': steps})

    def _normalize_inference(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize inference section from mode-based task configuration.
        """
        if config.get('type') == 'roi':
            params = {k: v for k, v in config.items() if k != 'type'}
            return [{
                'type': 'roi',
                'name': config.get('name', 'roi_processing'),
                'params': params,
            }]

        models = config.get('models', [])
        if not models:
            raise ValueError("Inference configuration must include a non-empty 'models' list")

        common_params = {
            k: v for k, v in config.items()
            if k not in {'models', 'ensemble_mode', 'type', 'name'}
        }

        if len(models) == 1:
            params = {'model_name': models[0], **common_params}
            return [{
                'type': 'nnunet_inference',
                'name': config.get('name', 'inference'),
                'params': params,
            }]

        ensemble_mode = config.get('ensemble_mode', 'concatenate')
        if ensemble_mode == 'concatenate':
            params = {
                'model_names': models,
                'label_mappings': self._build_label_mappings(models),
                **common_params,
            }
            step_type = 'multi_model_concat'
            step_name = config.get('name', 'multi_model_concat')
        else:
            params = {'model_names': models, 'method': ensemble_mode, **common_params}
            step_type = 'ensemble_inference'
            step_name = config.get('name', 'ensemble_inference')

        return [{'type': step_type, 'name': step_name, 'params': params}]

    def _build_label_mappings(self, model_names: List[str]) -> List[Dict[int, int]]:
        """
        Build per-model local->global label mappings from task metadata when available.
        """
        task_models = self.variables.get("_task_models", {})
        output_labels = self.variables.get("_output_labels", {})

        if not task_models or not output_labels:
            return []

        global_name_to_id = self._normalize_labels_to_name_to_id(output_labels)
        mappings: List[Dict[int, int]] = []

        for model_name in model_names:
            model_info = task_models.get(model_name)
            if model_info is None:
                mappings.append({})
                continue

            local_name_to_id = self._normalize_labels_to_name_to_id(model_info.labels)
            mapping: Dict[int, int] = {}
            for label_name, local_id in local_name_to_id.items():
                global_id = global_name_to_id.get(label_name)
                if global_id is not None:
                    mapping[local_id] = global_id
            mappings.append(mapping)

        return mappings

    def _normalize_labels_to_name_to_id(self, labels: Dict[Any, Any]) -> Dict[str, int]:
        """
        Normalize labels to {'name': id} regardless of input direction.
        """
        if not labels:
            return {}

        first_key = next(iter(labels.keys()))
        if isinstance(first_key, int):
            return {str(name): int(label_id) for label_id, name in labels.items()}
        return {str(name): int(label_id) for name, label_id in labels.items()}

    def _normalize_step(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize step config to canonical {'type', 'name', 'params'} format.
        """
        if 'type' not in config:
            raise ValueError(f"Step config missing required field 'type': {config}")

        if 'params' in config:
            return config

        step_type = config['type']
        params = {
            k: v for k, v in config.items()
            if k not in {'type', 'name'}
        }

        if step_type == 'clip_intensity':
            if 'min' in params and 'lower' not in params:
                params['lower'] = params.pop('min')
            if 'max' in params and 'upper' not in params:
                params['upper'] = params.pop('max')

        return {
            'type': step_type,
            'name': config.get('name', step_type),
            'params': params,
        }

    def _resolve_placeholders(self, value: Any) -> Any:
        """
        Recursively resolve ${var} placeholders in config from builder variables.
        """
        if isinstance(value, dict):
            return {k: self._resolve_placeholders(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(v) for v in value]
        if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
            key = value[2:-1]
            if key in self.variables:
                return self.variables[key]
        return value
    
    def _build_step(self, config: Dict[str, Any]) -> PipelineStep:
        """
        Build a single step from configuration.
        
        Args:
            config: Step configuration
            
        Returns:
            Pipeline step instance
        """
        step_type = config.get('type')
        step_name = config.get('name', step_type)
        step_params = config.get('params', {})
        
        if step_type not in self.STEP_REGISTRY:
            raise ValueError(f"Unknown step type: {step_type}. "
                           f"Available: {list(self.STEP_REGISTRY.keys())}")
        
        step_class = self.STEP_REGISTRY[step_type]
        step = step_class(step_name, step_params)
        
        return step
    
    @classmethod
    def register_step(cls, name: str, step_class: type) -> None:
        """
        Register a custom step type.
        
        Args:
            name: Step type name
            step_class: Step class
        """
        cls.STEP_REGISTRY[name] = step_class
        logger.info(f"Registered custom step type: {name}")


def build_default_preprocessing_pipeline(
    modality: str = 'CT',
    target_spacing: tuple = (1.5, 1.5, 1.5)
) -> Pipeline:
    """
    Build a default preprocessing pipeline for a given modality.
    
    Args:
        modality: Imaging modality ('CT', 'PET', 'MR')
        target_spacing: Target spacing for resampling
        
    Returns:
        Preprocessing pipeline
    """
    pipeline = Pipeline(name=f"{modality}_preprocessing")
    
    # Resample
    pipeline.add_step(ResampleStep(
        'resample',
        {'spacing': target_spacing}
    ))
    
    # Modality-specific processing
    if modality == 'CT':
        pipeline.add_step(ClipIntensityStep(
            'clip_intensity',
            {'lower': -1000, 'upper': 1000}
        ))
        pipeline.add_step(NormalizeStep(
            'normalize',
            {'method': 'ct'}
        ))
    elif modality == 'PET':
        pipeline.add_step(SUVThresholdStep(
            'suv_threshold',
            {'threshold': 2.5}
        ))
        pipeline.add_step(NormalizeStep(
            'normalize',
            {'method': 'minmax'}
        ))
    else:
        pipeline.add_step(NormalizeStep(
            'normalize',
            {'method': 'zscore'}
        ))
    
    return pipeline


def build_default_postprocessing_pipeline() -> Pipeline:
    """
    Build a default postprocessing pipeline.
    
    Returns:
        Postprocessing pipeline
    """
    pipeline = Pipeline(name="default_postprocessing")
    
    pipeline.add_step(LargestComponentStep(
        'largest_component',
        {'keep_top_n': 1}
    ))
    
    pipeline.add_step(MorphologicalOpsStep(
        'morphological_close',
        {'operation': 'close', 'radius': 1}
    ))
    
    return pipeline


def build_inference_pipeline(
    model_path: str,
    use_tta: bool = True,
    folds: List[int] = None
) -> Pipeline:
    """
    Build a simple inference pipeline.
    
    Args:
        model_path: Path to model
        use_tta: Use test-time augmentation
        folds: Folds to use
        
    Returns:
        Inference pipeline
    """
    if folds is None:
        folds = [0]
    
    pipeline = Pipeline(name="inference")
    
    pipeline.add_step(nnUNetInferenceStep(
        'inference',
        {
            'model_path': model_path,
            'folds': folds,
            'use_mirroring': use_tta
        }
    ))
    
    return pipeline
