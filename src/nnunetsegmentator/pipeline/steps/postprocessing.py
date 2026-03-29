"""
Postprocessing Pipeline Steps

This module provides postprocessing steps for refining segmentation results.
"""

import numpy as np
from scipy import ndimage
from typing import Dict, Any, List, Tuple
import logging

from ..base import PipelineStep, PipelineContext

logger = logging.getLogger(__name__)


class ExpandContractStep(PipelineStep):
    """
    Expand-contract boundary refinement.
    
    This step refines segmentation boundaries by expanding and then
    contracting the segmentation, which can help smooth boundaries
    and remove small protrusions.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute expand-contract refinement.
        
        Config options:
            distance: Distance for expansion/contraction (default: 2.0)
        """
        distance = self.config.get('distance', 2.0)
        
        seg = context.intermediate_results['raw_prediction']
        
        logger.debug(f"Applying expand-contract with distance {distance}")
        
        # Compute signed distance map
        dist_map = ndimage.distance_transform_edt(seg) - \
                   ndimage.distance_transform_edt(1 - seg)
        
        # Expand and contract
        expanded = (dist_map > -distance).astype(np.uint8)
        contracted = (dist_map > distance).astype(np.uint8)
        
        # Final refinement
        refined = expanded & contracted
        
        context.intermediate_results['refined_prediction'] = refined
        
        return context


class LargestComponentStep(PipelineStep):
    """
    Keep only largest connected component.
    
    This step removes all but the largest connected component from
    the segmentation, which is useful for removing small spurious
    regions.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute largest component selection.
        
        Config options:
            min_size: Minimum component size to keep (default: None, keep only largest)
            keep_top_n: Keep top N largest components (default: 1)
        """
        min_size = self.config.get('min_size', None)
        keep_top_n = self.config.get('keep_top_n', 1)
        
        seg = context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction'])
        
        logger.debug(f"Selecting largest {keep_top_n} components")
        
        labeled, num_features = ndimage.label(seg)
        
        if num_features == 0:
            # No components found
            context.intermediate_results['final_prediction'] = seg
            return context
        
        # Get component sizes
        component_sizes = ndimage.sum(seg, labeled, range(1, num_features + 1))
        
        # Sort by size
        sorted_indices = np.argsort(component_sizes)[::-1]
        
        # Select components to keep
        if min_size is not None:
            # Keep all components above minimum size
            keep_indices = [i for i in sorted_indices 
                          if component_sizes[i] >= min_size]
        else:
            # Keep top N
            keep_indices = sorted_indices[:keep_top_n]
        
        # Create output
        output = np.zeros_like(seg)
        for idx in keep_indices:
            output[labeled == idx + 1] = 1
        
        context.intermediate_results['final_prediction'] = output
        
        return context


class MorphologicalOpsStep(PipelineStep):
    """
    Apply morphological operations.
    
    This step applies various morphological operations to refine
    the segmentation.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute morphological operation.
        
        Config options:
            operation: Operation type ('close', 'open', 'dilate', 'erode')
            radius: Structuring element radius (default: 2)
            iterations: Number of iterations (default: 1)
        """
        operation = self.config.get('operation', 'close')
        radius = self.config.get('radius', 2)
        iterations = self.config.get('iterations', 1)
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        logger.debug(f"Applying morphological {operation} with radius {radius}")
        
        # Create structuring element
        struct = ndimage.generate_binary_structure(3, 1)
        
        # Apply operation
        if operation == 'close':
            seg = ndimage.binary_closing(seg, structure=struct, 
                                        iterations=iterations * radius)
        elif operation == 'open':
            seg = ndimage.binary_opening(seg, structure=struct, 
                                        iterations=iterations * radius)
        elif operation == 'dilate':
            seg = ndimage.binary_dilation(seg, structure=struct, 
                                         iterations=iterations * radius)
        elif operation == 'erode':
            seg = ndimage.binary_erosion(seg, structure=struct, 
                                        iterations=iterations * radius)
        else:
            raise ValueError(f"Unknown morphological operation: {operation}")
        
        context.intermediate_results['final_prediction'] = seg.astype(np.uint8)
        
        return context


class FillHolesStep(PipelineStep):
    """
    Fill holes in segmentation.
    
    This step fills holes inside the segmentation, which can occur
    due to imperfect predictions.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute hole filling.
        
        Config options:
            connectivity: Connectivity for hole detection (default: 1)
        """
        connectivity = self.config.get('connectivity', 1)
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        logger.debug("Filling holes in segmentation")
        
        # Fill holes
        filled = ndimage.binary_fill_holes(seg, structure=ndimage.generate_binary_structure(3, connectivity))
        
        context.intermediate_results['final_prediction'] = filled.astype(np.uint8)
        
        return context


class FilterLabelStep(PipelineStep):
    """
    Filter segmentation to keep only specific labels.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute label filtering.
        
        Config options:
            label: Label(s) to keep (int or list of ints)
            binary: Convert result to binary mask (default: False)
        """
        target_label = self.config.get('label')
        binary = self.config.get('binary', False)
        
        if target_label is None:
            return context
            
        seg_keys = ['final_prediction', 'refined_prediction', 'raw_prediction']
        seg_key = next((k for k in seg_keys if k in context.intermediate_results), None)
        
        if not seg_key:
            return context
            
        seg = context.intermediate_results[seg_key]
        
        if isinstance(target_label, (list, tuple)):
            mask = np.isin(seg, target_label)
        else:
            mask = seg == target_label
            
        output = np.zeros_like(seg)
        output[mask] = 1 if binary else seg[mask]
        
        context.intermediate_results['final_prediction'] = output
        
        return context


class IntensityMaskStep(PipelineStep):
    """
    Mask segmentation based on image intensity (e.g., PET SUV).
    
    This step removes segmented regions where the underlying image intensity
    is below a certain threshold.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute intensity masking.
        
        Config options:
            threshold: Intensity threshold (keep > threshold)
            image_key: Key for intensity image (default: 'input_array' or 'original_input_array')
        """
        threshold = self.config.get('threshold')
        
        if threshold is None:
            return context
            
        seg_keys = ['final_prediction', 'refined_prediction', 'raw_prediction']
        seg_key = next((k for k in seg_keys if k in context.intermediate_results), None)
        
        if not seg_key:
            return context
            
        seg = context.intermediate_results[seg_key]
        
        # Get intensity image
        img = context.metadata.get('original_input_array', context.input_array)
        
        # Ensure dimensions match
        if img.shape != seg.shape:
            # Handle channel dim
            if img.ndim == seg.ndim + 1:
                img = img[0]
            elif seg.ndim == img.ndim + 1:
                seg = seg[0]
        
        mask = img > threshold
        output = seg * mask
        
        context.intermediate_results['final_prediction'] = output
        
        return context


class RemoveSmallObjectsStep(PipelineStep):
    """
    Remove small objects from segmentation.
    
    This step removes objects smaller than a specified size threshold.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute small object removal.
        
        Config options:
            min_size: Minimum object size in voxels (default: 100)
        """
        min_size = self.config.get('min_size', 100)
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        logger.debug(f"Removing objects smaller than {min_size} voxels")
        
        # Label connected components
        labeled, num_features = ndimage.label(seg)
        
        # Get component sizes
        component_sizes = ndimage.sum(seg, labeled, range(1, num_features + 1))
        
        # Create mask of components to keep
        keep_mask = np.zeros(num_features + 1, dtype=bool)
        keep_mask[0] = False  # Background
        for i, size in enumerate(component_sizes):
            if size >= min_size:
                keep_mask[i + 1] = True
        
        # Apply mask
        output = keep_mask[labeled]
        
        context.intermediate_results['final_prediction'] = output.astype(np.uint8)
        
        return context


class ThresholdProbabilityStep(PipelineStep):
    """
    Apply probability threshold.
    
    This step applies a threshold to probability maps to generate
    binary segmentations.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute probability thresholding.
        
        Config options:
            threshold: Probability threshold (default: 0.5)
            probability_key: Key for probability map in intermediate_results
        """
        threshold = self.config.get('threshold', 0.5)
        probability_key = self.config.get('probability_key', 'probabilities')
        
        probabilities = context.intermediate_results.get(probability_key)
        
        if probabilities is None:
            logger.warning("No probability map found, skipping thresholding")
            return context
        
        logger.debug(f"Applying probability threshold: {threshold}")
        
        # Apply threshold
        if probabilities.ndim == 4:
            # Multi-class probabilities: (C, D, H, W)
            prediction = np.argmax(probabilities, axis=0).astype(np.uint8)
        else:
            # Binary probabilities: (D, H, W)
            prediction = (probabilities > threshold).astype(np.uint8)
        
        context.intermediate_results['raw_prediction'] = prediction
        
        return context


class MultiLabelPostprocessingStep(PipelineStep):
    """
    Postprocessing for multi-label segmentations.
    
    This step applies label-specific postprocessing operations.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute multi-label postprocessing.
        
        Config options:
            label_configs: Dict mapping label values to their postprocessing configs
        """
        label_configs = self.config.get('label_configs', {})
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        logger.debug("Applying multi-label postprocessing")
        
        output = np.zeros_like(seg)
        
        for label_value, config in label_configs.items():
            # Extract label
            label_mask = (seg == label_value).astype(np.uint8)
            
            # Apply label-specific operations
            if config.get('keep_largest', False):
                labeled, num_features = ndimage.label(label_mask)
                if num_features > 0:
                    component_sizes = ndimage.sum(label_mask, labeled, range(1, num_features + 1))
                    largest = np.argmax(component_sizes) + 1
                    label_mask = (labeled == largest).astype(np.uint8)
            
            if config.get('fill_holes', False):
                label_mask = ndimage.binary_fill_holes(label_mask).astype(np.uint8)
            
            min_size = config.get('min_size', 0)
            if min_size > 0:
                labeled, num_features = ndimage.label(label_mask)
                if num_features > 0:
                    component_sizes = ndimage.sum(label_mask, labeled, range(1, num_features + 1))
                    for i, size in enumerate(component_sizes):
                        if size < min_size:
                            label_mask[labeled == i + 1] = 0
            
            # Add to output
            output[label_mask > 0] = label_value
        
        context.intermediate_results['final_prediction'] = output
        
        return context


class AnatomicalConstraintsStep(PipelineStep):
    """
    Apply anatomical constraints to segmentation.
    
    This step applies constraints such as bone avoidance, airway avoidance,
    and boundary refinement to improve segmentation quality.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute anatomical constraints.
        
        Config options:
            constraints: Dict mapping structure names to constraint types or masks
        """
        constraints = self.config.get('constraints', {})
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        if not constraints:
            return context
            
        logger.debug("Applying anatomical constraints")
        
        # Check if we have label mapping available
        # This step assumes 'labels' in metadata or config
        labels = context.metadata.get('labels', {})
        label_values = {name: i for i, name in labels.items()} if labels else {}
        
        processed_seg = seg.copy()
        
        for structure_name, constraint in constraints.items():
            # Find label value for structure
            # Handle both string keys (if labels loaded from json) and int keys
            label_val = None
            for k, v in labels.items():
                if v == structure_name:
                    label_val = k
                    break
            
            if label_val is None:
                continue
                
            # Create mask for this structure
            structure_mask = (seg == label_val).astype(np.uint8)
            
            if isinstance(constraint, str):
                # Apply named constraint type
                refined_mask = self._apply_structure_constraint(structure_mask, constraint)
            elif isinstance(constraint, np.ndarray):
                # Apply mask constraint directly
                refined_mask = structure_mask * constraint
            else:
                continue
                
            # Update segmentation
            processed_seg[structure_mask > 0] = 0  # Clear old
            processed_seg[refined_mask > 0] = label_val  # Set new
            
        context.intermediate_results['final_prediction'] = processed_seg
        return context

    def _apply_structure_constraint(self, mask: np.ndarray, constraint_type: str) -> np.ndarray:
        """Apply structure-specific constraints."""
        if constraint_type == "bone_avoidance":
            # Remove segmentations from bone regions
            return self._apply_bone_avoidance(mask)
        elif constraint_type == "airway_avoidance":
            # Remove segmentations from airway regions
            return self._apply_airway_avoidance(mask)
        elif constraint_type == "organ_boundary_refinement":
            # Refine boundaries at organ interfaces
            return self._apply_organ_boundary_refinement(mask)
        else:
            return mask

    def _apply_bone_avoidance(self, mask: np.ndarray) -> np.ndarray:
        """Remove segmentations from high-intensity regions (likely bone in CT)."""
        # This would typically require the original image for intensity analysis
        # For now, return the original mask
        return mask

    def _apply_airway_avoidance(self, mask: np.ndarray) -> np.ndarray:
        """Remove segmentations from airway regions."""
        # This would typically use airway segmentation as constraint
        # For now, return the original mask
        return mask

    def _apply_organ_boundary_refinement(self, mask: np.ndarray) -> np.ndarray:
        """Refine segmentations at organ boundaries."""
        # Apply gentle boundary smoothing
        processed_mask = mask.copy()

        # Remove isolated single voxels
        # using scipy.ndimage.label
        labeled_mask, num_features = ndimage.label(processed_mask)

        if num_features > 0:
            # Component sizes
            component_sizes = ndimage.sum(processed_mask, labeled_mask, range(1, num_features + 1))
            
            # Find small components (size 1)
            # component_sizes is a list/array where index i corresponds to label i+1
            for i, size in enumerate(component_sizes):
                if size <= 1:
                    processed_mask[labeled_mask == i + 1] = 0

        return processed_mask


class DistanceRefinementStep(PipelineStep):
    """
    Apply distance-based refinement.
    
    This step refines segmentation based on distance from reference structures.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute distance refinement.
        
        Config options:
            reference_masks: Dict of reference masks
            min_distance: Minimum distance in voxels (default: 5)
        """
        reference_masks = self.config.get('reference_masks', {})
        min_distance = self.config.get('min_distance', 5)
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        if not reference_masks:
            return context
            
        logger.debug(f"Applying distance refinement (min_dist={min_distance})")
        
        processed_seg = seg.copy()
        
        for ref_name, ref_mask in reference_masks.items():
            if np.sum(ref_mask) == 0:
                continue

            # Calculate distance transform
            # distance_transform_edt computes distance to nearest non-zero pixel
            # We want distance from the reference mask (where ref_mask > 0)
            # So we compute distance on the inverted mask (~ref_mask)
            # The result is distance to the nearest ref_mask voxel
            distance_transform = ndimage.distance_transform_edt(1 - (ref_mask > 0).astype(int))

            # Apply distance constraint
            too_close = distance_transform < min_distance

            # Remove voxels too close to reference structure
            processed_seg[too_close] = 0
            
        context.intermediate_results['final_prediction'] = processed_seg
        return context


class ShapeConstraintStep(PipelineStep):
    """
    Apply shape constraints (volume, sphericity).
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute shape constraints.
        
        Config options:
            constraints: Dict mapping structure names to constraint dicts
                {
                    "min_volume": int,
                    "max_volume": int,
                    "min_sphericity": float
                }
        """
        constraints = self.config.get('constraints', {})
        
        seg = context.intermediate_results.get('final_prediction',
                                               context.intermediate_results.get('refined_prediction',
                                               context.intermediate_results['raw_prediction']))
        
        if not constraints:
            return context
            
        logger.debug("Applying shape constraints")
        
        processed_seg = seg.copy()
        labels = context.metadata.get('labels', {})
        
        for structure_name, struct_constraints in constraints.items():
            # Find label value
            label_val = None
            for k, v in labels.items():
                if v == structure_name:
                    label_val = k
                    break
            
            if label_val is None:
                continue
                
            mask = (seg == label_val).astype(np.uint8)
            if np.sum(mask) == 0:
                continue

            # Apply volume constraint
            if "min_volume" in struct_constraints:
                if np.sum(mask) < struct_constraints["min_volume"]:
                    processed_seg[processed_seg == label_val] = 0
                    continue

            # Apply maximum volume constraint
            if "max_volume" in struct_constraints:
                if np.sum(mask) > struct_constraints["max_volume"]:
                    # Keep largest component(s)
                    labeled_mask, num_features = ndimage.label(mask)
                    if num_features > 0:
                        component_sizes = ndimage.sum(mask, labeled_mask, range(1, num_features + 1))
                        
                        # Sort components by size (descending)
                        sorted_indices = np.argsort(component_sizes)[::-1] + 1
                        
                        new_mask = np.zeros_like(mask)
                        current_volume = 0
                        
                        for idx in sorted_indices:
                            if current_volume >= struct_constraints["max_volume"]:
                                break
                            
                            comp_mask = (labeled_mask == idx)
                            comp_vol = np.sum(comp_mask)
                            
                            if current_volume + comp_vol <= struct_constraints["max_volume"]:
                                new_mask[comp_mask] = 1
                                current_volume += comp_vol
                        
                        # Update processed seg
                        processed_seg[processed_seg == label_val] = 0
                        processed_seg[new_mask > 0] = label_val

            # Apply sphericity constraint
            if "min_sphericity" in struct_constraints:
                sphericity = self._calculate_sphericity(mask)
                if sphericity < struct_constraints["min_sphericity"]:
                    # Filter components
                    filtered_mask = self._filter_by_sphericity(
                        mask, struct_constraints["min_sphericity"]
                    )
                    # Update processed seg
                    processed_seg[processed_seg == label_val] = 0
                    processed_seg[filtered_mask > 0] = label_val

        context.intermediate_results['final_prediction'] = processed_seg
        return context

    def _calculate_sphericity(self, mask: np.ndarray) -> float:
        """Calculate sphericity of a 3D object."""
        if np.sum(mask) == 0:
            return 0.0

        # Calculate volume
        volume = np.sum(mask)

        # Calculate surface area
        surface_area = self._estimate_surface_area(mask)

        # Calculate sphericity: (π^(1/3) * (6V)^(2/3)) / A
        if surface_area > 0:
            sphericity = (np.pi ** (1/3) * (6 * volume) ** (2/3)) / surface_area
        else:
            sphericity = 0.0

        return sphericity

    def _estimate_surface_area(self, mask: np.ndarray) -> float:
        """Estimate surface area of a 3D binary object."""
        # Simple estimation using morphological operations
        struct = ndimage.generate_binary_structure(3, 1)
        eroded = ndimage.binary_erosion(mask, structure=struct)
        surface = mask.astype(bool) & ~eroded.astype(bool)
        return np.sum(surface)

    def _filter_by_sphericity(self, mask: np.ndarray, min_sphericity: float) -> np.ndarray:
        """Filter components by minimum sphericity."""
        labeled_mask, num_features = ndimage.label(mask)

        if num_features == 0:
            return np.zeros_like(mask)

        filtered_mask = np.zeros_like(mask, dtype=np.uint8)

        for component_id in range(1, num_features + 1):
            component = (labeled_mask == component_id).astype(np.uint8)
            sphericity = self._calculate_sphericity(component)

            if sphericity >= min_sphericity:
                filtered_mask[component > 0] = 1

        return filtered_mask


class PetThresholdStep(PipelineStep):
    """
    Thresholds the segmentation to only contain voxels with intensity higher than a specified value in the PET image.
    Uses 'original_input_array' from metadata if available, otherwise current input_array.
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute thresholding.
        
        Config options:
        - threshold (float): Intensity threshold. Voxels <= threshold will be removed from segmentation.
        """
        threshold = self.config.get('threshold')
        
        if threshold is None:
            logger.info(f"{self.name}: No threshold specified, skipping.")
            return context
            
        if 'original_input_array' in context.metadata:
            pet_array = context.metadata['original_input_array']
            logger.debug(f"{self.name}: Using preserved original array for thresholding")
        else:
            pet_array = context.input_array
            logger.warning(f"{self.name}: 'original_input_array' not found, using current input_array which might be preprocessed/normalized!")
            
        seg_array = context.intermediate_results.get('refined_prediction',
                                                    context.intermediate_results.get('raw_prediction'))
                                                    
        if seg_array is None:
            logger.warning(f"{self.name}: No segmentation found in context.")
            return context
            
        if seg_array.shape != pet_array.shape:
             logger.warning(f"{self.name}: Shape mismatch between PET ({pet_array.shape}) and segmentation ({seg_array.shape}).")
             if context.input_array.shape == seg_array.shape:
                 logger.warning(f"{self.name}: Falling back to current input_array (normalized/resampled). Threshold might be invalid for normalized data.")
                 pet_array = context.input_array
             else:
                 logger.error(f"{self.name}: Cannot find matching PET array. Skipping threshold.")
                 return context

        logger.debug(f"{self.name}: Applying threshold {threshold}")
        mask = (pet_array > threshold)
        new_seg = seg_array.copy()
        new_seg[~mask] = 0
        
        context.intermediate_results['refined_prediction'] = new_seg
        
        return context
