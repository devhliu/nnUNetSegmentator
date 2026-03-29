"""
Image Writers

This module provides writers for various medical image formats.
"""

from .. import image as sitk
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any
import logging

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logger = logging.getLogger(__name__)


class ImageWriter:
    """
    Universal image writer supporting NIfTI and DICOM.
    
    This class provides a unified interface for writing medical images
    to various formats.
    """
    
    @staticmethod
    def write(
        image: Union[sitk.Image, np.ndarray],
        path: Union[str, Path],
        reference: sitk.Image = None,
        format: str = 'nifti',
        compress: bool = True,
        dtype = None
    ) -> Path:
        """
        Write image to file.
        
        Args:
            image: Image to write (SimpleITK Image or numpy array)
            path: Output path
            reference: Reference image for geometry (if image is numpy array)
            format: Output format ('nifti' or 'dicom')
            compress: Compress NIfTI output
            dtype: Data type for output (default: auto-detect)
            
        Returns:
            Path to written file
        """
        path = Path(path)
        
        logger.debug(f"Writing image to: {path}")
        
        # Convert numpy to SimpleITK if needed
        if isinstance(image, np.ndarray):
            if dtype is not None:
                image = image.astype(dtype)
            
            sitk_image = sitk.GetImageFromArray(image)
            
            if reference is not None:
                sitk_image.CopyInformation(reference)
        else:
            sitk_image = image
        
        # Write
        if format == 'nifti':
            path = path.with_suffix('.nii.gz' if compress else '.nii')
            sitk.WriteImage(sitk_image, str(path))
        elif format == 'dicom':
            # DICOM writing requires additional setup
            ImageWriter._write_dicom_series(sitk_image, path)
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Image written to: {path}")
        
        return path
    
    @staticmethod
    def _write_dicom_series(image: sitk.Image, directory: Path):
        """
        Write as DICOM series.
        
        This is a simplified implementation. For full DICOM writing,
        additional metadata and proper DICOM tags are needed.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Use SimpleITK's DICOM writer
        # Note: This creates a basic DICOM series without full metadata
        writer = sitk.ImageFileWriter()
        writer.SetImageIO("GDCM")
        
        # Write each slice
        depth = image.GetSize()[2]
        for z in range(depth):
            slice_image = image[:, :, z]
            slice_path = directory / f"slice_{z:04d}.dcm"
            writer.SetFileName(str(slice_path))
            writer.Execute(slice_image)
        
        logger.info(f"DICOM series written to: {directory}")
    
    @staticmethod
    def write_segmentation(
        segmentation: Union[sitk.Image, np.ndarray],
        path: Union[str, Path],
        reference: sitk.Image = None,
        labels: Dict[str, int] = None,
        compress: bool = True
    ) -> Path:
        """
        Write segmentation with optional label metadata.
        
        Args:
            segmentation: Segmentation image
            path: Output path
            reference: Reference image for geometry
            labels: Dictionary mapping label names to values
            compress: Compress output
            
        Returns:
            Path to written file
        """
        path = Path(path)
        
        # Write segmentation
        seg_path = ImageWriter.write(
            segmentation,
            path,
            reference=reference,
            format='nifti',
            compress=compress,
            dtype=np.uint8
        )
        
        # Write label metadata if provided
        if labels and HAS_NIBABEL:
            import json
            
            # Create label metadata file
            meta_path = path.with_suffix('.json')
            
            # Invert labels dict for saving
            label_names = {str(v): k for k, v in labels.items()}
            
            with open(meta_path, 'w') as f:
                json.dump(label_names, f, indent=2)
            
            logger.debug(f"Label metadata written to: {meta_path}")
        
        return seg_path
    
    @staticmethod
    def write_multi_label(
        segmentation: Union[sitk.Image, np.ndarray],
        output_dir: Union[str, Path],
        labels: Dict[str, int],
        reference: sitk.Image = None,
        prefix: str = ""
    ) -> Dict[str, Path]:
        """
        Write each label as a separate binary mask.
        
        Args:
            segmentation: Multi-label segmentation
            output_dir: Output directory
            labels: Dictionary mapping label names to values
            reference: Reference image for geometry
            prefix: Prefix for output filenames
            
        Returns:
            Dictionary mapping label names to output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(segmentation, sitk.Image):
            seg_array = sitk.GetArrayFromImage(segmentation)
        else:
            seg_array = segmentation
        
        output_paths = {}
        
        for label_name, label_value in labels.items():
            # Create binary mask
            mask = (seg_array == label_value).astype(np.uint8)
            
            # Write
            filename = f"{prefix}{label_name}.nii.gz" if prefix else f"{label_name}.nii.gz"
            output_path = output_dir / filename
            
            ImageWriter.write(
                mask,
                output_path,
                reference=reference,
                dtype=np.uint8
            )
            
            output_paths[label_name] = output_path
        
        logger.info(f"Written {len(output_paths)} label masks to: {output_dir}")
        
        return output_paths


class NumpyWriter:
    """
    Writer for numpy arrays.
    
    This class handles writing numpy arrays with optional metadata.
    """
    
    @staticmethod
    def write(
        array: np.ndarray,
        path: Union[str, Path],
        metadata: Dict[str, Any] = None,
        compress: bool = True
    ) -> Path:
        """
        Write numpy array to file.
        
        Args:
            array: Numpy array to write
            path: Output path
            metadata: Optional metadata dictionary
            compress: Use npz compression
            
        Returns:
            Path to written file
        """
        path = Path(path)
        
        logger.debug(f"Writing numpy array to: {path}")
        
        if compress:
            path = path.with_suffix('.npz')
            if metadata:
                np.savez_compressed(path, array=array, **metadata)
            else:
                np.savez_compressed(path, array=array)
        else:
            path = path.with_suffix('.npy')
            np.save(path, array)
        
        logger.info(f"Array written to: {path}")
        
        return path
