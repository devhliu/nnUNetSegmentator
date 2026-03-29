"""
ROI Processing Pipeline Steps

This module provides steps for Region of Interest (ROI) based processing,
enabling efficient multi-stage segmentation (low-res -> crop -> high-res).
"""

import numpy as np
from ... import image as sitk
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from copy import deepcopy

from ..base import PipelineStep, PipelineContext, Pipeline
from .inference import nnUNetInferenceStep
from .preprocessing import ResampleStep

logger = logging.getLogger(__name__)


class ROIProcessingStep(PipelineStep):
    """
    Perform ROI-based multi-stage segmentation.
    
    This step orchestrates a coarse-to-fine segmentation strategy:
    1. Run low-resolution segmentation on the whole image
    2. Identify ROIs for specific anatomical groups
    3. Crop the original image to these ROIs
    4. Run high-resolution models on the crops
    5. Merge results back into the full image space
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute ROI processing.
        
        Config options:
            low_res_model: Name of the low-resolution model (e.g., 'total_fast')
            roi_groups: Dict mapping group names to config:
                {
                    "organs": {
                        "model": "total_organs",
                        "labels": [1, 2, 3...] (ids in low-res segmentation)
                    },
                    ...
                }
            crop_margin: Margin in mm to add around ROI (default: [20, 20, 20])
            low_res_spacing: Target spacing for low-res step (default: [3.0, 3.0, 3.0])
        """
        low_res_model = self.config.get('low_res_model')
        roi_groups = self.config.get('roi_groups', {})
        crop_margin = self.config.get('crop_margin', [20.0, 20.0, 20.0])
        low_res_spacing = self.config.get('low_res_spacing', [3.0, 3.0, 3.0])
        
        if not low_res_model:
            raise ValueError("low_res_model must be specified for ROIProcessingStep")
            
        logger.info(f"Starting ROI processing with low-res model: {low_res_model}")
        
        # 1. Low-resolution segmentation
        # We perform this in a sub-context to avoid polluting the main context
        # but we need the input image
        
        # Create low-res image
        resampler = ResampleStep(name="roi_low_res_resample", config={'spacing': low_res_spacing})
        
        # Create a temporary context for low-res
        low_res_context = PipelineContext(
            input_image=context.input_image,
            input_array=context.input_array,
            metadata=deepcopy(context.metadata)
        )
        
        # Resample
        low_res_context = resampler.execute(low_res_context)
        low_res_image = low_res_context.input_image
        
        # Run low-res inference
        inference_step = nnUNetInferenceStep(name="roi_low_res_inference", config={'model_name': low_res_model})
        low_res_context = inference_step.execute(low_res_context)
        
        low_res_seg = low_res_context.intermediate_results.get('raw_prediction')
        if low_res_seg is None:
            raise RuntimeError("Low-resolution inference failed to produce 'raw_prediction'")
            
        logger.info("Low-resolution segmentation completed")
        
        # 2. Process each ROI group
        final_combined_seg = np.zeros_like(context.input_array, dtype=np.uint8)
        
        # We need mapping from high-res model output labels to final labels
        # This assumes the high-res models output class 1..N which map to specific final classes
        
        original_spacing = context.input_image.GetSpacing()
        original_size = context.input_image.GetSize()
        
        # Get low-res metadata for coordinate mapping
        low_res_spacing_actual = low_res_image.GetSpacing()
        
        results_map = {}  # Store results per structure/label
        
        for group_name, group_config in roi_groups.items():
            high_res_model = group_config.get('model')
            target_labels = group_config.get('labels', [])
            
            if not high_res_model:
                logger.warning(f"No model specified for group {group_name}, skipping")
                continue
                
            logger.info(f"Processing ROI group: {group_name} with model {high_res_model}")
            
            # 3. Generate crop mask from low-res segmentation
            # Create a binary mask including all target labels for this group
            group_mask_low_res = np.zeros_like(low_res_seg, dtype=np.uint8)
            for label_idx in target_labels:
                group_mask_low_res[low_res_seg == label_idx] = 1
                
            # If mask is empty, skip this group (nothing to segment)
            if np.sum(group_mask_low_res) == 0:
                logger.warning(f"No ROI found for group {group_name}, skipping")
                continue
                
            # Calculate bounding box in low-res space
            coords = np.where(group_mask_low_res > 0)
            x_min, x_max = coords[0].min(), coords[0].max()
            y_min, y_max = coords[1].min(), coords[1].max()
            z_min, z_max = coords[2].min(), coords[2].max()
            
            # Convert crop margin from mm to voxels in low-res space
            margin_voxels = [
                int(crop_margin[0] / low_res_spacing_actual[0]),
                int(crop_margin[1] / low_res_spacing_actual[1]),
                int(crop_margin[2] / low_res_spacing_actual[2]),
            ]
            
            # Apply margin
            x_min = max(0, x_min - margin_voxels[0])
            x_max = min(low_res_seg.shape[0], x_max + margin_voxels[0])
            y_min = max(0, y_min - margin_voxels[1])
            y_max = min(low_res_seg.shape[1], y_max + margin_voxels[1])
            z_min = max(0, z_min - margin_voxels[2])
            z_max = min(low_res_seg.shape[2], z_max + margin_voxels[2])
            
            # Convert bounding box to physical coordinates
            # Low res image origin
            origin = low_res_image.GetOrigin()
            direction = low_res_image.GetDirection()
            
            # We need to map this back to the original image space to crop the HIGH res image
            # The most robust way is to use physical coordinates
            
            # Simplified approach: Map indices based on spacing ratio
            # This assumes images are aligned and share origin/direction (which ResampleStep should preserve)
            
            hr_x_min = int(x_min * (low_res_spacing_actual[0] / original_spacing[0]))
            hr_x_max = int(x_max * (low_res_spacing_actual[0] / original_spacing[0]))
            hr_y_min = int(y_min * (low_res_spacing_actual[1] / original_spacing[1]))
            hr_y_max = int(y_max * (low_res_spacing_actual[1] / original_spacing[1]))
            hr_z_min = int(z_min * (low_res_spacing_actual[2] / original_spacing[2]))
            hr_z_max = int(z_max * (low_res_spacing_actual[2] / original_spacing[2]))
            
            # Clip to image bounds
            hr_shape = context.input_array.shape
            hr_x_min = max(0, hr_x_min)
            hr_x_max = min(hr_shape[0], hr_x_max)
            hr_y_min = max(0, hr_y_min)
            hr_y_max = min(hr_shape[1], hr_y_max)
            hr_z_min = max(0, hr_z_min)
            hr_z_max = min(hr_shape[2], hr_z_max)
            
            # 4. Crop original image
            crop_array = context.input_array[hr_x_min:hr_x_max, hr_y_min:hr_y_max, hr_z_min:hr_z_max]
            
            if crop_array.size == 0:
                logger.warning(f"Empty crop for group {group_name}, skipping")
                continue
                
            # Create crop image (preserving metadata)
            crop_image = sitk.GetImageFromArray(crop_array)
            
            # Calculate new origin for crop
            # original origin + index * spacing * direction_matrix (simplified for now)
            # For aligned axes:
            new_origin = [
                origin[0] + hr_x_min * original_spacing[0],
                origin[1] + hr_y_min * original_spacing[1],
                origin[2] + hr_z_min * original_spacing[2]
            ]
            
            crop_image.SetOrigin(new_origin)
            crop_image.SetSpacing(original_spacing)
            crop_image.SetDirection(direction)
            
            # 5. Run high-res inference on crop
            crop_context = PipelineContext(
                input_image=crop_image,
                input_array=crop_array,
                metadata=deepcopy(context.metadata)
            )
            
            # NOTE: We assume the high-res model handles its own resampling if needed
            # via its own internal pipeline or the InferenceStep
            
            hr_inference = nnUNetInferenceStep(name=f"roi_{group_name}_inference", 
                                              config={'model_name': high_res_model})
            
            crop_context = hr_inference.execute(crop_context)
                
            crop_seg = crop_context.intermediate_results.get('raw_prediction')
            if crop_seg is None:
                continue
                
            # 6. Merge back
            # crop_seg has labels 1..M specific to that model
            # We need to know what they map to in the final segmentation
            # This mapping should be provided in the group config or derived from model info
            
            # For simplicity, we paste the raw values for now, assuming post-processing
            # or the user handles the label mapping/conflict resolution
            # Or we can store them in a way that preserves the source
            
            # Update final combined segmentation
            # Note: This simple overwrite might clobber overlaps.
            # A more sophisticated approach would handle overlaps (e.g. max confidence)
            
            # We only overwrite non-zero values
            mask_indices = np.where(crop_seg > 0)
            
            # Calculate target indices in full array
            target_x = mask_indices[0] + hr_x_min
            target_y = mask_indices[1] + hr_y_min
            target_z = mask_indices[2] + hr_z_min
            
            # Clip to be safe (though should be within bounds)
            valid = (target_x < hr_shape[0]) & (target_y < hr_shape[1]) & (target_z < hr_shape[2])
            
            final_combined_seg[target_x[valid], target_y[valid], target_z[valid]] = crop_seg[
                mask_indices[0][valid], mask_indices[1][valid], mask_indices[2][valid]
            ]
            
            # Store in results map if needed for multi-label handling
            # results_map[group_name] = ...
            
        context.intermediate_results['raw_prediction'] = final_combined_seg
        
        return context
