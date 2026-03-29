"""
Vertebrae Localization Step.

This module provides a step to locate specific vertebrae using TotalSegmentator.
"""

import logging
import tempfile
import shutil
import subprocess
import os
import nibabel as nib
import numpy as np
from pathlib import Path
from ..base import PipelineStep, PipelineContext
from ... import image as sitk

logger = logging.getLogger(__name__)

class VertebraeLocalizationStep(PipelineStep):
    """
    Locate vertebrae using TotalSegmentator.
    
    This step runs TotalSegmentator on the input image to segment specific vertebrae,
    which are then used as landmarks for body composition analysis.
    """
    
    def validate_config(self) -> bool:
        # Check if TotalSegmentator is installed
        try:
            import totalsegmentator
            return True
        except ImportError:
            logger.error("TotalSegmentator not found. Please install it with 'pip install TotalSegmentator'")
            return False
            
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute vertebrae localization.
        
        Config options:
            roi_subset: List of ROIs to segment (default: ['vertebrae_T12', 'vertebrae_L3', 'vertebrae_L4'])
            fast: Use fast mode (default: True)
        """
        rois = self.config.get('roi_subset', ['vertebrae_T12', 'vertebrae_L3', 'vertebrae_L4'])
        fast = self.config.get('fast', True)
        
        logger.info(f"Localizing vertebrae: {rois}")
        
        # We need to save the current input image to a temporary file for TotalSegmentator
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_input = os.path.join(temp_dir, "input.nii.gz")
            temp_output = os.path.join(temp_dir, "output") # Output folder for multi-label or individual files
            
            # Save current image
            # Note: TotalSegmentator expects NIfTI
            sitk.WriteImage(context.input_image, temp_input)
            
            # Construct command
            # TotalSegmentator -i input -o output --roi_subset ... --fast
            cmd = [
                "TotalSegmentator",
                "-i", temp_input,
                "-o", temp_output,
                "--roi_subset"
            ] + rois
            
            if fast:
                cmd.append("--fast")
                
            logger.info(f"Running TotalSegmentator: {' '.join(cmd)}")
            
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                logger.error(f"TotalSegmentator failed: {e}")
                raise RuntimeError("TotalSegmentator vertebrae localization failed") from e
                
            # Read back results
            # TotalSegmentator with --roi_subset usually outputs individual files in the output folder
            # e.g., output/vertebrae_L3.nii.gz
            
            vertebrae_masks = {}
            for roi in rois:
                roi_path = os.path.join(temp_output, f"{roi}.nii.gz")
                if os.path.exists(roi_path):
                    # Load mask
                    mask_img = sitk.ReadImage(roi_path)
                    mask_arr = sitk.GetArrayFromImage(mask_img)
                    vertebrae_masks[roi] = mask_arr
                    logger.debug(f"Loaded mask for {roi}")
                else:
                    logger.warning(f"Mask for {roi} not found in output")
            
            context.metadata['vertebrae_masks'] = vertebrae_masks
            
        return context
