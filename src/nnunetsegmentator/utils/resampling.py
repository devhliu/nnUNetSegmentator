"""
Resampling Utilities

This module provides functions for resampling NIfTI images using nibabel exclusively.
All resampling operations use nibabel.processing for accurate and efficient results.
"""

import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output
import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def resample_image(
    image: Union[nib.Nifti1Image, str, Path, np.ndarray],
    target_spacing: Tuple[float, float, float],
    interpolator: str = 'linear',
    original_spacing: Optional[Tuple[float, float, float]] = None,
    affine: Optional[np.ndarray] = None
) -> Union[nib.Nifti1Image, np.ndarray]:
    """
    Resample image to target spacing using nibabel.
    
    Args:
        image: Input image (NIfTI image, file path, or numpy array)
        target_spacing: Target spacing in mm (tuple of 3 floats)
        interpolator: Interpolation method
            - 'nearest': Nearest neighbor (for segmentations)
            - 'linear': Linear interpolation (default, for images)
            - 'cubic': Cubic interpolation
        original_spacing: Original spacing (required if image is numpy array)
        affine: Affine matrix (required if image is numpy array)
        
    Returns:
        Resampled image (NIfTI image or numpy array, same type as input)
        
    Example:
        >>> import nibabel as nib
        >>> img = nib.load('image.nii.gz')
        >>> resampled = resample_image(img, (1.0, 1.0, 1.0), interpolator='linear')
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # Convert numpy array to NIfTI image if needed.
    # Arrays are expected in nibabel/native order (X, Y, Z).
    if is_numpy:
        if original_spacing is None:
            raise ValueError("original_spacing required for numpy arrays")
        if affine is None:
            # Create default affine from spacing
            affine = np.diag(list(original_spacing) + [1.0])
        nifti_img = nib.Nifti1Image(image, affine)
    else:
        # Load image if path provided
        if isinstance(image, (str, Path)):
            nifti_img = nib.load(str(image))
        else:
            nifti_img = image
    
    # Map interpolator string to order
    interp_order_map = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3
    }
    order = interp_order_map.get(interpolator, 1)
    
    # Use nibabel's resample_to_output
    resampled = resample_to_output(
        nifti_img,
        voxel_sizes=target_spacing,
        order=order,
        mode='constant',
        cval=0.0
    )
    
    if is_numpy:
        return resampled.get_fdata()
    
    return resampled


def resample_segmentation(
    segmentation: Union[nib.Nifti1Image, str, Path, np.ndarray],
    target_spacing: Tuple[float, float, float],
    original_spacing: Optional[Tuple[float, float, float]] = None,
    affine: Optional[np.ndarray] = None
) -> Union[nib.Nifti1Image, np.ndarray]:
    """
    Resample segmentation to target spacing using nearest neighbor interpolation.
    
    This function preserves discrete label values during resampling.
    
    Args:
        segmentation: Input segmentation (NIfTI image, file path, or numpy array)
        target_spacing: Target spacing in mm
        original_spacing: Original spacing (required if numpy array)
        affine: Affine matrix (required if numpy array)
        
    Returns:
        Resampled segmentation (same type as input)
        
    Example:
        >>> seg = nib.load('segmentation.nii.gz')
        >>> resampled_seg = resample_segmentation(seg, (1.0, 1.0, 1.0))
    """
    return resample_image(
        segmentation,
        target_spacing,
        interpolator='nearest',  # Always use nearest neighbor for segmentations
        original_spacing=original_spacing,
        affine=affine
    )


def resample_to_reference(
    image: Union[nib.Nifti1Image, str, Path, np.ndarray],
    reference: Union[nib.Nifti1Image, str, Path],
    interpolator: str = 'linear'
) -> Union[nib.Nifti1Image, np.ndarray]:
    """
    Resample image to match reference image geometry using nibabel.
    
    Args:
        image: Input image to resample
        reference: Reference image defining output geometry
        interpolator: Interpolation method ('nearest', 'linear', 'cubic')
        
    Returns:
        Resampled image matching reference geometry
        
    Example:
        >>> img = nib.load('image.nii.gz')
        >>> ref = nib.load('reference.nii.gz')
        >>> resampled = resample_to_reference(img, ref, interpolator='cubic')
    """
    is_numpy = isinstance(image, np.ndarray)
    
    # Load images if paths provided
    if isinstance(image, (str, Path)):
        image = nib.load(str(image))
    elif is_numpy:
        image = nib.Nifti1Image(image, np.eye(4))
    if isinstance(reference, (str, Path)):
        reference = nib.load(str(reference))
    
    # Map interpolator string to order
    interp_order_map = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3
    }
    order = interp_order_map.get(interpolator, 1)
    
    # Use nibabel's resample_from_to
    resampled = resample_from_to(
        image,
        (reference.shape, reference.affine),
        order=order,
        mode='constant',
        cval=0.0
    )
    
    if is_numpy:
        return resampled.get_fdata()
    
    return resampled


def resample_segmentation_to_reference(
    segmentation: Union[nib.Nifti1Image, str, Path],
    reference: Union[nib.Nifti1Image, str, Path]
) -> nib.Nifti1Image:
    """
    Resample segmentation to match reference image using nearest neighbor.
    
    Args:
        segmentation: Segmentation image
        reference: Reference image defining output geometry
        
    Returns:
        Resampled segmentation matching reference geometry
        
    Example:
        >>> seg = nib.load('segmentation.nii.gz')
        >>> ref = nib.load('image.nii.gz')
        >>> resampled_seg = resample_segmentation_to_reference(seg, ref)
    """
    return resample_to_reference(segmentation, reference, interpolator='nearest')


def compute_new_shape(
    original_shape: Tuple[int, int, int],
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """
    Compute target shape after resampling to new spacing.
    
    Args:
        original_shape: Original image shape
        original_spacing: Original voxel spacing
        target_spacing: Target voxel spacing
        
    Returns:
        Target image shape
        
    Example:
        >>> new_shape = compute_new_shape(
        ...     (512, 512, 100),
        ...     (0.5, 0.5, 3.0),
        ...     (1.0, 1.0, 1.0)
        ... )
    """
    new_shape = tuple(
        int(round(original_shape[i] * original_spacing[i] / target_spacing[i]))
        for i in range(3)
    )
    return new_shape


def is_isotropic(
    spacing: Tuple[float, float, float],
    tolerance: float = 0.01
) -> bool:
    """
    Check if spacing is isotropic.
    
    Args:
        spacing: Voxel spacing
        tolerance: Relative tolerance for comparison
        
    Returns:
        True if spacing is isotropic within tolerance
    """
    mean_spacing = np.mean(spacing)
    max_diff = max(abs(s - mean_spacing) for s in spacing)
    return max_diff < tolerance * mean_spacing


def get_spacing_from_nifti(img: nib.Nifti1Image) -> Tuple[float, float, float]:
    """
    Get voxel spacing from NIfTI image.
    
    Args:
        img: NIfTI image
        
    Returns:
        Voxel spacing tuple
    """
    return img.header.get_zooms()[:3]
