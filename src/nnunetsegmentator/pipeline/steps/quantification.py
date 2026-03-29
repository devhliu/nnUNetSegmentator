import numpy as np
from ... import image as sitk
from ..base import PipelineStep, PipelineContext
import csv
import os
import logging

logger = logging.getLogger(__name__)

class TumorMetricsStep(PipelineStep):
    """
    Compute tumor metrics (volume, average intensity) and optionally save to CSV.
    """
    
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.output_csv = self.config.get('output_csv', 'tumor_metrics.csv')
        self.save_csv = self.config.get('save_csv', True)
        self.intensity_scaling_factor = self.config.get('intensity_scaling_factor', 1.0) # E.g. for SUV conversion if not already done

    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Compute metrics and store in context.metadata['metrics'].
        """
        # Get segmentation
        # Try to get refined segmentation first, then raw
        seg_array = context.intermediate_results.get('final_prediction',
                                                    context.intermediate_results.get('refined_prediction',
                                                    context.intermediate_results.get('raw_prediction')))
                                                    
        if seg_array is None:
            logger.warning(f"{self.name}: No segmentation found.")
            return context
            
        # Get PET image (intensity)
        # Use preserved original array if available (before normalization)
        if 'original_input_array' in context.metadata:
            pet_array = context.metadata['original_input_array']
            logger.debug(f"{self.name}: Using preserved original array for intensity metrics")
        else:
            pet_array = context.input_array
            logger.warning(f"{self.name}: 'original_input_array' not found, using current input_array for intensity metrics.")

        if seg_array.shape != pet_array.shape:
            logger.warning(f"{self.name}: Shape mismatch between PET ({pet_array.shape}) and segmentation ({seg_array.shape}). Cannot compute intensity metrics correctly.")
            # If shape mismatch, we can still compute volume if spacing is known, but intensity requires alignment.
            # Assuming spacing is in metadata
            
        spacing = context.metadata.get('original_spacing', (1.0, 1.0, 1.0))
        # Ensure spacing is tuple of floats
        if isinstance(spacing, (list, tuple)):
             voxel_volume = np.prod(spacing)
        else:
             voxel_volume = 1.0 # Default if unknown

        # Calculate metrics per label or total?
        # LION calculated total tumor volume and average intensity for label 1.
        # Let's support multi-label if present, or just treat all non-zero as tumor.
        
        mask = (seg_array > 0)
        tumor_voxel_count = np.sum(mask)
        
        if tumor_voxel_count == 0:
            logger.info(f"{self.name}: No tumor found.")
            metrics = {
                'tumor_volume_cm3': 0.0,
                'average_intensity': 0.0
            }
        else:
            # Volume in cm^3 (assuming spacing in mm)
            # 1000 mm^3 = 1 cm^3
            tumor_volume_cm3 = (tumor_voxel_count * voxel_volume) / 1000.0
            
            # Average intensity
            if seg_array.shape == pet_array.shape:
                intensities = pet_array[mask]
                avg_intensity = np.mean(intensities) * self.intensity_scaling_factor
            else:
                avg_intensity = 0.0 # Cannot compute
                
            metrics = {
                'tumor_volume_cm3': tumor_volume_cm3,
                'average_intensity': avg_intensity
            }
            
        logger.info(f"Tumor Metrics: Volume={metrics['tumor_volume_cm3']:.2f} cm3, Avg Intensity={metrics['average_intensity']:.2f}")
        
        context.metadata['tumor_metrics'] = metrics
        
        if self.save_csv:
            self._save_to_csv(metrics, context)
            
        return context

    def _save_to_csv(self, metrics: dict, context: PipelineContext):
        # Determine output path
        # context.metadata might have 'output_dir'
        output_dir = context.metadata.get('output_dir', '.')
        csv_path = os.path.join(output_dir, self.output_csv)
        
        # Check if file exists to write header
        write_header = not os.path.exists(csv_path)
        
        try:
            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['Case ID', 'Tumor Volume (cm^3)', 'Average Intensity']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if write_header:
                    writer.writeheader()
                    
                row = {
                    'Case ID': context.metadata.get('case_id', 'unknown'),
                    'Tumor Volume (cm^3)': metrics['tumor_volume_cm3'],
                    'Average Intensity': metrics['average_intensity']
                }
                writer.writerow(row)
                logger.info(f"Saved metrics to {csv_path}")
        except OSError as exc:
            logger.error(f"Failed to save metrics to CSV: {exc}")
            raise

class BodyCompositionMetricsStep(PipelineStep):
    """
    Calculate body composition metrics (Muscle/Fat area, volume, density).
    """
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute metrics calculation.
        
        Config options:
            label_mapping: Map segmentation labels to metric types
                (1=SFAT, 2=VFAT, 3=Muscle, 4=IMAT) -> (Muscle=1, SFAT=2, VFAT=3, IMAT=4)
                Default assumes TotalSegmentator output structure if not provided.
            csv_output: Whether to save CSV (default: True)
        """
        # Default mapping from TotalSegmentator (v2) tissue_types to Codebase A logic
        # TotalSeg Task 485: 1=SFAT, 2=VFAT, 3=Muscle, 4=IMAT
        # Codebase A Logic: 1=Muscle, 2=SFAT, 3=VFAT, 4=IMAT
        # We need to map Input Label -> Logic Label
        # So: 1->2, 2->3, 3->1, 4->4
        default_mapping = {
            1: 2, # SFAT -> SFAT
            2: 3, # VFAT -> VFAT
            3: 1, # Muscle -> Muscle
            4: 4  # IMAT -> IMAT
        }
        
        # If the keys in config are strings, convert to int
        raw_mapping = self.config.get('label_mapping', default_mapping)
        label_mapping = {int(k): int(v) for k, v in raw_mapping.items()}
        
        csv_output = self.config.get('csv_output', True)
        
        # Get segmentation and image
        # 'final_prediction' usually holds the segmentation
        seg_arr = context.intermediate_results.get('final_prediction')
        if seg_arr is None:
            logger.warning("No segmentation found for metrics.")
            return context
            
        # Get original image for density (HU)
        # Use 'original_input_array' if preserved, else current 'input_array' (might be normalized!)
        # Codebase A uses original Hounsfield Units.
        # If 'input_array' is normalized, we can't use it for HU density.
        # We should check if 'original_input_array' exists.
        img_arr = context.metadata.get('original_input_array')
        if img_arr is None:
            # Fallback, but warn if density is needed
            logger.warning("Original input array not found. Density metrics might be incorrect if image is normalized.")
            img_arr = context.input_array
            
        # Get vertebrae masks
        vertebrae_masks = context.metadata.get('vertebrae_masks', {})
        if not vertebrae_masks:
            logger.warning("No vertebrae masks found. Skipping L3/T12-L4 metrics.")
            return context
            
        # Metadata
        spacing = context.metadata.get('original_spacing', (1.0, 1.0, 1.0))
        # Arrays follow nibabel/native order (X, Y, Z)
        # We need Z spacing for slice thickness, X/Y for area
        pixel_area = spacing[0] * spacing[1]
        voxel_volume = spacing[0] * spacing[1] * spacing[2]
        
        metrics = {}
        
        # --- 2D Metrics at L3 ---
        if 'vertebrae_L3' in vertebrae_masks:
            l3_mask = vertebrae_masks['vertebrae_L3']
            # Find center slice of L3
            # l3_mask is (X, Y, Z)
            z_indices = np.where(np.sum(l3_mask, axis=(0, 1)) > 0)[0]
            if len(z_indices) > 0:
                # Codebase A uses argmax of sum, which is similar (slice with most L3)
                l3_slice_idx = int(np.argmax(np.sum(l3_mask, axis=(0, 1))))
                
                logger.info(f"Calculating 2D metrics at L3 (slice {l3_slice_idx})")
                
                # Extract 2D slice
                seg_slice = seg_arr[:, :, l3_slice_idx]
                img_slice = img_arr[:, :, l3_slice_idx]
                
                # Calculate metrics for each tissue
                # Mapped labels: 1=Muscle, 2=SFAT, 3=VFAT, 4=IMAT
                for source_label, logic_label in label_mapping.items():
                    mask = (seg_slice == source_label)
                    area = np.sum(mask) * pixel_area
                    density = np.mean(img_slice[mask]) if np.sum(mask) > 0 else 0
                    
                    name_map = {1: 'muscle', 2: 'sfat', 3: 'vfat', 4: 'mfat'}
                    name = name_map.get(logic_label, f'tissue_{logic_label}')
                    
                    metrics[f'{name}_area_mm2'] = float(area)
                    metrics[f'{name}_density_hu'] = float(density)
                
                # Total fat (2+3+4)
                fat_labels = [k for k, v in label_mapping.items() if v in [2, 3, 4]]
                fat_mask = np.isin(seg_slice, fat_labels)
                metrics['total_fat_area_mm2'] = float(np.sum(fat_mask) * pixel_area)
                
                # Body area (> -500 HU)
                # Assuming img_slice is HU
                body_mask = (img_slice > -500)
                metrics['body_area_mm2'] = float(np.sum(body_mask) * pixel_area)
                
                metrics['l3_slice_index'] = l3_slice_idx
            else:
                logger.warning("L3 mask is empty")
                
        # --- 3D Metrics T12-L4 ---
        if 'vertebrae_T12' in vertebrae_masks and 'vertebrae_L4' in vertebrae_masks:
            t12_mask = vertebrae_masks['vertebrae_T12']
            l4_mask = vertebrae_masks['vertebrae_L4']
            
            z_t12 = np.where(np.sum(t12_mask, axis=(0, 1)) > 0)[0]
            z_l4 = np.where(np.sum(l4_mask, axis=(0, 1)) > 0)[0]
            
            if len(z_t12) > 0 and len(z_l4) > 0:
                # Codebase A range: min(t12, l4) to max(t12, l4) centers?
                # Codebase A: slice_range = range(min(t12_slice, l4_slice), max(t12_slice, l4_slice) + 1)
                # It uses the center slice of each vertebra as bounds.
                t12_slice = int(np.argmax(np.sum(t12_mask, axis=(0, 1))))
                l4_slice = int(np.argmax(np.sum(l4_mask, axis=(0, 1))))
                
                start_slice = min(t12_slice, l4_slice)
                end_slice = max(t12_slice, l4_slice)
                
                logger.info(f"Calculating 3D metrics T12-L4 (slices {start_slice}-{end_slice})")
                
                # Extract volume
                # Note: seg_arr is (X, Y, Z)
                seg_vol = seg_arr[:, :, start_slice:end_slice+1]
                img_vol = img_arr[:, :, start_slice:end_slice+1]
                
                # Calculate metrics
                for source_label, logic_label in label_mapping.items():
                    mask = (seg_vol == source_label)
                    vol = np.sum(mask) * voxel_volume
                    density = np.mean(img_vol[mask]) if np.sum(mask) > 0 else 0
                    
                    name_map = {1: 'muscle', 2: 'sfat', 3: 'vfat', 4: 'mfat'}
                    name = name_map.get(logic_label, f'tissue_{logic_label}')
                    
                    metrics[f'{name}_volume_mm3'] = float(vol)
                    metrics[f'{name}_density_3d_hu'] = float(density)
                
                # Total fat
                fat_labels = [k for k, v in label_mapping.items() if v in [2, 3, 4]]
                fat_mask = np.isin(seg_vol, fat_labels)
                metrics['total_fat_volume_mm3'] = float(np.sum(fat_mask) * voxel_volume)
                
                # Body volume
                body_mask = (img_vol > -500)
                metrics['body_volume_mm3'] = float(np.sum(body_mask) * voxel_volume)
                
                metrics['scan_length_mm'] = float((end_slice - start_slice + 1) * spacing[2])
            else:
                logger.warning("T12 or L4 mask is empty")

        # Save to CSV
        if csv_output and metrics:
            output_dir = context.metadata.get('output_dir', '.')
            case_id = context.metadata.get('case_id', 'unknown_case')
            csv_path = os.path.join(output_dir, f"{case_id}_metrics.csv")
            
            # Convert to DF
            # If pandas is not available, use csv module (already imported in this file)
            try:
                import pandas as pd
                df = pd.DataFrame([metrics])
                df.insert(0, 'case_id', case_id)
                df.to_csv(csv_path, index=False)
            except ImportError:
                 # Fallback to csv module
                 fieldnames = ['case_id'] + list(metrics.keys())
                 with open(csv_path, 'w', newline='') as f:
                     writer = csv.DictWriter(f, fieldnames=fieldnames)
                     writer.writeheader()
                     row = {'case_id': case_id}
                     row.update(metrics)
                     writer.writerow(row)
                     
            logger.info(f"Saved metrics to {csv_path}")
            
            context.metadata['metrics_path'] = csv_path
            existing_metrics = context.metadata.get('metrics')
            if not isinstance(existing_metrics, dict):
                existing_metrics = {}
            existing_metrics.update(metrics)
            context.metadata['metrics'] = existing_metrics
            
        return context
