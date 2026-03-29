"""
Image Readers

This module provides readers for various medical image formats.
"""

from .. import image as sitk
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Dict, Any, List
import logging

try:
    import pydicom
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

logger = logging.getLogger(__name__)


class ImageReader:
    """
    Universal image reader supporting NIfTI and DICOM.
    
    This class provides a unified interface for reading medical images
    from various formats.
    """
    
    @staticmethod
    def read(
        path: Union[str, Path, list],
        modality: str = None
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        """
        Read image from file(s).
        
        Args:
            path: File path (NIfTI) or directory/list of files (DICOM)
            modality: Optional modality hint ('CT', 'PET', 'MR')
            
        Returns:
            Tuple of (SimpleITK Image, metadata dict)
        """
        if isinstance(path, list):
            return ImageReader._read_dicom(path, modality)
        
        path = Path(path)
        
        if path.is_dir():
            return ImageReader._read_dicom(path, modality)
        else:
            return ImageReader._read_nifti(path, modality)
    
    @staticmethod
    def _read_nifti(
        path: Path,
        modality: str = None
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        """Read NIfTI file"""
        logger.debug(f"Reading NIfTI file: {path}")
        
        image = sitk.ReadImage(str(path))
        
        # Extract metadata
        metadata = {
            'format': 'nifti',
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'size': image.GetSize(),
            'filepath': str(path),
        }
        
        if HAS_NIBABEL:
            nii = nib.load(str(path))
            metadata['affine'] = nii.affine
            metadata['header'] = dict(nii.header)
        
        if modality:
            metadata['modality'] = modality
        
        return image, metadata
    
    @staticmethod
    def _read_dicom(
        path: Union[Path, List[str]],
        modality: str = None
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        """Read DICOM series"""
        logger.debug(f"Reading DICOM from: {path}")
        
        if isinstance(path, list):
            series_files = path
        else:
            reader = sitk.ImageSeriesReader()
            series_files = reader.GetGDCMSeriesFileNames(str(path))
            
            if not series_files:
                raise ValueError(f"No DICOM series found in {path}")
        
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(series_files)
        image = reader.Execute()
        
        # Extract DICOM metadata
        metadata = {
            'format': 'dicom',
            'spacing': image.GetSpacing(),
            'origin': image.GetOrigin(),
            'direction': image.GetDirection(),
            'size': image.GetSize(),
            'num_slices': len(series_files),
        }
        
        if HAS_PYDICOM and series_files:
            ds = pydicom.dcmread(series_files[0])
            metadata.update({
                'modality': ds.get('Modality', 'Unknown'),
                'patient_id': ds.get('PatientID', 'Unknown'),
                'patient_name': str(ds.get('PatientName', 'Unknown')),
                'study_date': ds.get('StudyDate', 'Unknown'),
                'study_time': ds.get('StudyTime', 'Unknown'),
                'series_description': ds.get('SeriesDescription', 'Unknown'),
                'series_number': ds.get('SeriesNumber', 'Unknown'),
                'slice_thickness': float(ds.get('SliceThickness', 0)),
                'pixel_spacing': ds.get('PixelSpacing', [0, 0]),
            })
        
        if modality:
            metadata['modality'] = modality
        
        return image, metadata


class MultiModalReader:
    """
    Reader for multi-modal inputs (e.g., PET/CT).
    
    This class handles reading and aligning multiple imaging modalities.
    """
    
    @staticmethod
    def read(
        paths: Dict[str, Union[str, Path]],
        align_to: str = 'ct'
    ) -> Dict[str, Tuple[sitk.Image, Dict[str, Any]]]:
        """
        Read multiple modalities and align them.
        
        Args:
            paths: Dictionary mapping modality to path
                   e.g., {'pet': '/path/to/pet', 'ct': '/path/to/ct'}
            align_to: Reference modality for alignment
            
        Returns:
            Dictionary mapping modality to (image, metadata) tuples
        """
        logger.debug(f"Reading multi-modal data: {list(paths.keys())}")
        
        results = {}
        
        # Read all modalities
        for modality, path in paths.items():
            image, metadata = ImageReader.read(path, modality=modality)
            metadata['modality'] = modality
            results[modality] = (image, metadata)
        
        # Align all to reference
        if align_to in results:
            ref_image = results[align_to][0]
            ref_spacing = ref_image.GetSpacing()
            ref_size = ref_image.GetSize()
            ref_origin = ref_image.GetOrigin()
            ref_direction = ref_image.GetDirection()
            
            for modality in results:
                if modality != align_to:
                    image, metadata = results[modality]
                    
                    logger.debug(f"Resampling {modality} to match {align_to}")
                    
                    # Resample to reference
                    resampled = sitk.Resample(
                        image,
                        ref_image,
                        sitk.sitkLinear,
                        sitk.sitkFloat32,
                        ref_origin,
                        ref_spacing,
                        ref_direction
                    )
                    
                    results[modality] = (resampled, metadata)
        
        return results
    
    @staticmethod
    def read_petct(
        pet_path: Union[str, Path],
        ct_path: Union[str, Path],
        align_to: str = 'ct'
    ) -> Tuple[Dict[str, sitk.Image], Dict[str, Dict[str, Any]]]:
        """
        Convenience method for reading PET/CT pairs.
        
        Args:
            pet_path: Path to PET data
            ct_path: Path to CT data
            align_to: Reference modality for alignment
            
        Returns:
            Tuple of (images dict, metadata dict)
        """
        paths = {'pet': pet_path, 'ct': ct_path}
        results = MultiModalReader.read(paths, align_to=align_to)
        
        images = {mod: img for mod, (img, _) in results.items()}
        metadata = {mod: meta for mod, (_, meta) in results.items()}
        
        return images, metadata


class NumpyReader:
    """
    Reader for numpy arrays.
    
    This class handles reading numpy arrays and converting them to
    SimpleITK images with proper metadata.
    """
    
    @staticmethod
    def read(
        array: np.ndarray,
        spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        direction: Tuple[float, ...] = None
    ) -> Tuple[sitk.Image, Dict[str, Any]]:
        """
        Read numpy array as SimpleITK image.
        
        Args:
            array: Numpy array (D, H, W) or (C, D, H, W)
            spacing: Image spacing
            origin: Image origin
            direction: Image direction matrix
            
        Returns:
            Tuple of (SimpleITK Image, metadata dict)
        """
        logger.debug(f"Reading numpy array with shape {array.shape}")
        
        # Handle multi-channel
        if array.ndim == 4:
            # Multi-channel: take first channel
            array = array[0]
        
        # Create SimpleITK image
        image = sitk.GetImageFromArray(array)
        image.SetSpacing(spacing)
        image.SetOrigin(origin)
        
        if direction is not None:
            image.SetDirection(direction)
        
        metadata = {
            'format': 'numpy',
            'spacing': spacing,
            'origin': origin,
            'direction': image.GetDirection(),
            'size': image.GetSize(),
            'array_shape': array.shape,
        }
        
        return image, metadata
