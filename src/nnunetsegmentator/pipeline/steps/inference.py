"""
Inference Pipeline Steps

This module provides inference steps for running nnUNet models.
"""

import copy
import numpy as np
from pathlib import Path
from typing import Dict, List
import logging

from ..base import PipelineStep, PipelineContext
from ...core.registry import TaskRegistry

logger = logging.getLogger(__name__)


class nnUNetInferenceStep(PipelineStep):
    """
    Execute nnUNet inference.
    
    This step runs nnUNet inference on the prepared input data.
    It uses the nnUNetv2 Python API.
    """
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self._predictor = None
        self._initialized = False
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute nnUNet inference.
        
        Config options:
            model_path: Path to the trained model
            folds: List of folds to use (default: [0])
            tile_step_size: Tile step size for sliding window (default: 0.5)
            use_gaussian: Use Gaussian importance weighting (default: True)
            use_mirroring: Use test-time augmentation (default: True)
            device: Device to use (default: 'cuda')
        """
        # Check for accelerator in context (Herd Mode)
        device = context.metadata.get('accelerator')
        if not device:
            device = self.config.get('device', 'cuda')
            
        self._init_predictor(device=device)
        
        input_array = context.input_array
        spacing = context.metadata.get('current_spacing', (1.5, 1.5, 1.5))
        
        logger.debug(f"Running inference with spacing {spacing} on {device}")
        prediction = self._run_api_inference(input_array, spacing)

        save_probabilities = self.config.get('save_or_return_probabilities', False)
        if save_probabilities:
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                raw_prediction = prediction[0]
                probabilities = prediction[1]
            elif isinstance(prediction, np.ndarray) and prediction.ndim >= 4:
                probabilities = prediction
                raw_prediction = np.argmax(probabilities, axis=0).astype(np.uint8)
            else:
                probabilities = None
                raw_prediction = prediction

            context.intermediate_results['probabilities'] = probabilities
            context.intermediate_results['raw_prediction'] = raw_prediction
            logger.debug(
                "Inference complete. Prediction shape: %s; probabilities: %s",
                getattr(raw_prediction, "shape", None),
                None if probabilities is None else probabilities.shape,
            )
        else:
            context.intermediate_results['raw_prediction'] = prediction
            logger.debug(f"Inference complete. Prediction shape: {prediction.shape}")
        
        return context

    def _init_predictor(self, device=None):
        """
        Initialize the nnUNet predictor.
        """
        target_device = device or self.config.get('device', 'cuda')
        
        if self._initialized:
            # Check if we need to re-initialize for a different device
            if self._predictor and hasattr(self._predictor, 'device'):
                current_device = str(self._predictor.device)
                if current_device == target_device or (target_device == 'cuda' and 'cuda' in current_device):
                    return
            
            logger.info(f"Re-initializing predictor for device: {target_device}")
        
        model_path = self._resolve_model_path()
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        try:
            from nnunetv2.inference.predict import nnUNetPredictor
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "nnUNetv2 is required for inference. Install the 'nnunet' optional dependency."
            ) from exc

        folds = self.config.get('folds', [0])
        tile_step_size = self.config.get('tile_step_size', 0.5)
        use_gaussian = self.config.get('use_gaussian', True)
        use_mirroring = self.config.get('use_mirroring', True)
        perform_everything_on_gpu = self.config.get('perform_everything_on_gpu', True)
        verbose = self.config.get('verbose', False)

        logger.info(f"Initializing nnUNetv2 predictor from {model_path} on {target_device}")

        self._predictor = nnUNetPredictor(
            tile_step_size=tile_step_size,
            use_gaussian=use_gaussian,
            use_mirroring=use_mirroring,
            perform_everything_on_gpu=perform_everything_on_gpu,
            device=torch.device(target_device),
            verbose=verbose,
            allow_tqdm=self.config.get('allow_tqdm', True),
        )

        self._predictor.initialize_from_trained_model_folder(str(model_path), folds)
        logger.info("nnUNetv2 predictor initialized successfully")
        
        self._initialized = True

    def _resolve_model_path(self) -> Path:
        """
        Resolve model path from direct path or registered model name.
        """
        model_path = self.config.get('model_path')
        if model_path:
            return Path(model_path)

        model_name = self.config.get('model_name')
        if model_name:
            return TaskRegistry.get_model_path(model_name)

        raise ValueError("Either 'model_path' or 'model_name' must be specified in config")
    
    def _run_api_inference(self, input_array: np.ndarray, spacing: tuple) -> np.ndarray:
        """
        Run inference using nnUNetv2 Python API.
        
        Args:
            input_array: Input image array
            spacing: Image spacing
            
        Returns:
            Prediction array
        """
        # Ensure proper shape for nnUNet (c, d, h, w)
        if input_array.ndim == 3:
            # Single channel, add channel dimension
            input_array = input_array[np.newaxis, ...]
        elif input_array.ndim != 4:
            raise ValueError(f"Invalid input shape: {input_array.shape}")
        
        # Run prediction
        # Note: nnUNetv2 expects properties dict with spacing
        properties = {
            'spacing': spacing
        }
        
        prediction = self._predictor.predict_single_npy_array(
            input_array,
            properties,
            None,  # segmentation_previous_stage
            None,  # output_file_truncated
            self.config.get('save_or_return_probabilities', False),
        )
        
        return prediction


class CascadeInferenceStep(PipelineStep):
    """
    Cascade multiple models.
    
    This step runs multiple models in sequence and combines their predictions
    using various strategies (intersection, crop, concatenate).
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute cascade inference.
        
        Config options:
            models: List of model configurations
            mode: Combination mode ('intersect', 'crop', 'concatenate')
        """
        models = self.config['models']
        mode = self.config.get('mode', 'intersect')
        
        logger.debug(f"Running cascade inference with {len(models)} models, mode: {mode}")
        
        predictions = []
        for i, model_config in enumerate(models):
            step = nnUNetInferenceStep(f"cascade_{model_config.get('name', i)}", model_config)
            context = step.execute(context)
            predictions.append(context.intermediate_results['raw_prediction'].copy())
        
        # Combine predictions
        if mode == 'intersect':
            # Intersection of all predictions
            combined = np.all(predictions, axis=0).astype(np.uint8)
        elif mode == 'union':
            # Union of all predictions
            combined = np.any(predictions, axis=0).astype(np.uint8)
        elif mode == 'crop':
            # Use first prediction to crop region for second
            # This is useful for two-stage detection + segmentation
            combined = predictions[-1]
        elif mode == 'concatenate':
            # Concatenate along channel dimension
            combined = np.concatenate(predictions, axis=0)
        elif mode == 'majority_vote':
            # Majority voting
            combined = (np.sum(predictions, axis=0) > len(predictions) / 2).astype(np.uint8)
        else:
            raise ValueError(f"Unknown cascade mode: {mode}")
        
        context.intermediate_results['raw_prediction'] = combined
        logger.debug(f"Cascade inference complete. Combined shape: {combined.shape}")
        
        return context


class EnsembleInferenceStep(PipelineStep):
    """
    Ensemble multiple model predictions.
    
    This step runs multiple models and combines their probability outputs
    for more robust predictions.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute ensemble inference.
        
        Config options:
            models: List of model configurations
            weights: Optional weights for each model
            method: Ensemble method ('average', 'max', 'weighted')
        """
        models = self._get_models()
        weights = self.config.get('weights', None)
        method = self.config.get('method', 'average')
        
        logger.debug(f"Running ensemble inference with {len(models)} models")
        
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        
        # Collect probability predictions
        # Note: This requires models to output probabilities
        probabilities = []
        for i, model_config in enumerate(models):
            model_config_with_prob = copy.deepcopy(model_config)
            model_config_with_prob['save_or_return_probabilities'] = True
            step = nnUNetInferenceStep(
                f"ensemble_{model_config_with_prob.get('name', i)}",
                model_config_with_prob,
            )
            context = step.execute(context)
            # Assuming probabilities are stored
            probabilities.append(context.intermediate_results.get('probabilities'))
        
        # Combine probabilities
        if method == 'average':
            combined_prob = np.mean(probabilities, axis=0)
        elif method == 'max':
            combined_prob = np.max(probabilities, axis=0)
        elif method == 'weighted':
            combined_prob = np.sum(
                [p * w for p, w in zip(probabilities, weights)],
                axis=0
            )
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        # Get final prediction from probabilities
        prediction = np.argmax(combined_prob, axis=0).astype(np.uint8)
        
        context.intermediate_results['raw_prediction'] = prediction
        context.intermediate_results['probabilities'] = combined_prob
        
        return context

    def _get_models(self) -> List[Dict]:
        """
        Accept either explicit model configs or model_names + shared settings.
        """
        if 'models' in self.config:
            return self.config['models']

        model_names = self.config.get('model_names')
        if not model_names:
            raise ValueError("EnsembleInferenceStep requires 'models' or 'model_names'")

        shared = {
            k: v for k, v in self.config.items()
            if k not in {'model_names', 'models', 'weights', 'method'}
        }
        return [{'model_name': name, **shared} for name in model_names]


class MultiModelConcatStep(PipelineStep):
    """
    Multi-model concatenation for TotalSegmentator 5-part strategy.
    
    This step runs multiple models that each predict a subset of labels,
    then concatenates the predictions into a single multilabel segmentation.
    This is the strategy used by TotalSegmentator for efficient whole-body segmentation.
    """
    
    def __init__(self, name: str, config: dict = None):
        super().__init__(name, config)
        self._label_mappings = None
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute multi-model concatenation.
        
        Config options:
            models: List of model configurations
            label_mappings: List of label mappings for each model
                          Each mapping is a dict {local_label: global_label}
            model_names: Optional list of model names for logging
        
        Example:
            config = {
                'models': [
                    {'model_path': '/path/to/organs_model', ...},
                    {'model_path': '/path/to/vertebrae_model', ...},
                ],
                'label_mappings': [
                    {1: 1, 2: 2, ...},  # organs: local -> global
                    {1: 25, 2: 26, ...},  # vertebrae: local -> global
                ]
            }
        """
        models = self._get_models()
        label_mappings = self.config.get('label_mappings', [])
        model_names = self.config.get('model_names', [f"model_{i}" for i in range(len(models))])
        
        logger.info(f"Running multi-model concatenation with {len(models)} models")
        
        # Determine output shape
        input_shape = context.input_array.shape
        if input_shape.ndim == 4:
            input_shape = input_shape[1:]  # Remove channel dimension
        
        # Find maximum label value to determine output size
        max_label = 0
        for mapping in label_mappings:
            if mapping:
                max_label = max(max_label, max(mapping.values()))
        
        # Initialize combined segmentation
        combined = np.zeros(input_shape, dtype=np.uint8)
        
        # Run each model and map labels
        for i, (model_config, model_name) in enumerate(zip(models, model_names)):
            logger.info(f"Running model {i+1}/{len(models)}: {model_name}")
            
            # Run inference
            step = nnUNetInferenceStep(f"multimodel_{model_name}", model_config)
            context = step.execute(context)
            
            # Get prediction
            prediction = context.intermediate_results['raw_prediction'].copy()
            
            # Map labels
            if i < len(label_mappings) and label_mappings[i]:
                mapped_prediction = self._map_labels(prediction, label_mappings[i])
            else:
                # No mapping, use as-is
                mapped_prediction = prediction
            
            # Combine (later models overwrite earlier for overlapping labels)
            # Use maximum to handle overlaps
            combined = np.maximum(combined, mapped_prediction)
            
            logger.debug(f"Model {model_name} complete. "
                        f"Unique labels: {np.unique(mapped_prediction)}")
        
        context.intermediate_results['raw_prediction'] = combined
        logger.info(f"Multi-model concatenation complete. "
                   f"Total unique labels: {len(np.unique(combined)) - 1}")
        
        return context

    def _get_models(self) -> List[Dict]:
        """
        Accept either explicit model configs or model_names + shared settings.
        """
        if 'models' in self.config:
            return self.config['models']

        model_names = self.config.get('model_names')
        if not model_names:
            raise ValueError("MultiModelConcatStep requires 'models' or 'model_names'")

        shared = {
            k: v for k, v in self.config.items()
            if k not in {'model_names', 'models', 'label_mappings'}
        }
        return [{'model_name': name, **shared} for name in model_names]
    
    def _map_labels(self, prediction: np.ndarray, 
                   mapping: Dict[int, int]) -> np.ndarray:
        """
        Map local labels to global labels.
        
        Args:
            prediction: Prediction with local labels
            mapping: Dictionary mapping local to global labels
        
        Returns:
            Mapped prediction
        """
        mapped = np.zeros_like(prediction)
        
        for local_label, global_label in mapping.items():
            mapped[prediction == local_label] = global_label
        
        return mapped


class TotalSegmentatorEnsembleStep(PipelineStep):
    """
    Specialized ensemble step for TotalSegmentator 5-part strategy.
    
    This implements the exact strategy used by TotalSegmentator:
    - Run 5 separate models (organs, vertebrae, cardiac, muscles, ribs)
    - Each model predicts a subset of the 117 classes
    - Concatenate results into single multilabel segmentation
    """
    
    # Predefined label mappings for TotalSegmentator 5-part strategy
    PART_MAPPINGS = {
        'organs': {
            # 24 classes -> global labels 1-24
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10,
            11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
            19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24
        },
        'vertebrae': {
            # 26 classes -> global labels 25-50
            1: 25, 2: 26, 3: 27, 4: 28, 5: 29, 6: 30, 7: 31, 8: 32, 9: 33,
            10: 34, 11: 35, 12: 36, 13: 37, 14: 38, 15: 39, 16: 40, 17: 41,
            18: 42, 19: 43, 20: 44, 21: 45, 22: 46, 23: 47, 24: 48, 25: 49,
            26: 50
        },
        'cardiac': {
            # 18 classes -> global labels 51-68
            1: 51, 2: 52, 3: 53, 4: 54, 5: 55, 6: 56, 7: 57, 8: 58, 9: 59,
            10: 60, 11: 61, 12: 62, 13: 63, 14: 64, 15: 65, 16: 66, 17: 67,
            18: 68
        },
        'muscles': {
            # 23 classes -> global labels 69-91
            1: 69, 2: 70, 3: 71, 4: 72, 5: 73, 6: 74, 7: 75, 8: 76, 9: 77,
            10: 78, 11: 79, 12: 80, 13: 81, 14: 82, 15: 83, 16: 84, 17: 85,
            18: 86, 19: 87, 20: 88, 21: 89, 22: 90, 23: 91
        },
        'ribs': {
            # 26 classes -> global labels 92-117
            1: 92, 2: 93, 3: 94, 4: 95, 5: 96, 6: 97, 7: 98, 8: 99, 9: 100,
            10: 101, 11: 102, 12: 103, 13: 104, 14: 105, 15: 106, 16: 107,
            17: 108, 18: 109, 19: 110, 20: 111, 21: 112, 22: 113, 23: 114,
            24: 115, 25: 116, 26: 117
        }
    }
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute TotalSegmentator 5-part ensemble.
        
        Config options:
            model_paths: Dict mapping part names to model paths
                        {'organs': '/path/to/organs_model', ...}
            device: Device to use (default: 'cuda')
            folds: Folds to use (default: [0])
        """
        model_paths = self.config['model_paths']
        device = self.config.get('device', 'cuda')
        folds = self.config.get('folds', [0])
        
        # Determine which parts to run
        parts_to_run = ['organs', 'vertebrae', 'cardiac', 'muscles', 'ribs']
        available_parts = [p for p in parts_to_run if p in model_paths]
        
        logger.info(f"Running TotalSegmentator ensemble with {len(available_parts)} parts: "
                   f"{available_parts}")
        
        # Determine output shape
        input_shape = context.input_array.shape
        if context.input_array.ndim == 4:
            input_shape = input_shape[1:]
        
        # Initialize combined segmentation
        combined = np.zeros(input_shape, dtype=np.uint8)
        
        # Run each part
        for part_name in available_parts:
            model_path = model_paths[part_name]
            
            logger.info(f"Running {part_name} model...")
            
            # Create model config
            model_config = {
                'model_path': model_path,
                'device': device,
                'folds': folds,
            }
            
            # Run inference
            step = nnUNetInferenceStep(f"totalseg_{part_name}", model_config)
            context = step.execute(context)
            
            # Get prediction
            prediction = context.intermediate_results['raw_prediction'].copy()
            
            # Map labels
            mapping = self.PART_MAPPINGS[part_name]
            mapped_prediction = self._map_labels(prediction, mapping)
            
            # Combine
            combined = np.maximum(combined, mapped_prediction)
            
            # Log statistics
            num_labels = len(np.unique(mapped_prediction)) - 1
            logger.info(f"{part_name} complete: {num_labels} structures segmented")
        
        context.intermediate_results['raw_prediction'] = combined
        
        # Log final statistics
        total_structures = len(np.unique(combined)) - 1
        logger.info(f"TotalSegmentator ensemble complete: "
                   f"{total_structures} structures segmented")
        
        return context
    
    def _map_labels(self, prediction: np.ndarray, 
                   mapping: Dict[int, int]) -> np.ndarray:
        """Map local labels to global labels."""
        mapped = np.zeros_like(prediction)
        
        for local_label, global_label in mapping.items():
            mapped[prediction == local_label] = global_label
        
        return mapped
