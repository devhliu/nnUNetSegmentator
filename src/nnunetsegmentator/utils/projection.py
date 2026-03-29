"""
2D Projection Utilities

This module provides utilities for creating 2D projections from 3D medical images,
enabling fast 2D-based segmentation approaches like TotalSegmentator 2D.
"""

from .. import image as sitk
import numpy as np
from typing import Tuple, Optional, Union, List
import logging

logger = logging.getLogger(__name__)


def create_coronal_projection(
    image: Union[sitk.Image, np.ndarray],
    projection_type: str = 'maximum',
    axis: int = 1
) -> Union[sitk.Image, np.ndarray]:
    """
    Create 2D projection from 3D image along specified axis.
    
    This function creates coronal, sagittal, or axial projections by applying
    intensity projection along the specified axis.
    
    Args:
        image: Input 3D image (SimpleITK or numpy array)
        projection_type: Type of intensity projection
            - 'maximum': Maximum intensity projection (MIP)
            - 'average': Average intensity projection (AIP)
            - 'minimum': Minimum intensity projection
        axis: Axis along which to project
            - 0: Sagittal projection (along x-axis)
            - 1: Coronal projection (along y-axis) - default
            - 2: Axial projection (along z-axis)
            
    Returns:
        2D projection image (same type as input)
        
    Example:
        >>> from .. import image as sitk
        >>> img = sitk.ReadImage('ct_scan.nii.gz')
        >>> mip = create_coronal_projection(img, projection_type='maximum')
        >>> aip = create_coronal_projection(img, projection_type='average')
    """
    is_sitk = isinstance(image, sitk.Image)
    
    if is_sitk:
        array = sitk.GetArrayFromImage(image)
    else:
        array = image
    
    if array.ndim != 3:
        raise ValueError(f"Expected 3D image, got {array.ndim}D")
    
    logger.debug(f"Creating {projection_type} projection along axis {axis}")
    logger.debug(f"Input shape: {array.shape}")
    
    # Apply projection
    if projection_type == 'maximum':
        projection = np.max(array, axis=axis)
    elif projection_type == 'average':
        projection = np.mean(array, axis=axis)
    elif projection_type == 'minimum':
        projection = np.min(array, axis=axis)
    else:
        raise ValueError(f"Unknown projection type: {projection_type}")
    
    logger.debug(f"Projection shape: {projection.shape}")
    
    if is_sitk:
        # Create SimpleITK image from projection
        projection_sitk = sitk.GetImageFromArray(projection)
        
        # Copy metadata and adjust spacing
        original_spacing = image.GetSpacing()
        original_origin = image.GetOrigin()
        original_direction = image.GetDirection()
        
        # Remove the projected axis from spacing and origin
        new_spacing = list(original_spacing)
        new_spacing.pop(axis)
        projection_sitk.SetSpacing(new_spacing)
        
        new_origin = list(original_origin)
        new_origin.pop(axis)
        projection_sitk.SetOrigin(new_origin)
        
        # Adjust direction matrix (remove projected axis)
        direction_matrix = np.array(original_direction).reshape(3, 3)
        # Remove the axis row and column
        mask = [i for i in range(3) if i != axis]
        new_direction = direction_matrix[np.ix_(mask, mask)]
        projection_sitk.SetDirection(new_direction.flatten())
        
        return projection_sitk
    
    return projection


def create_dual_projection(
    image: Union[sitk.Image, np.ndarray],
    axis: int = 1
) -> Tuple[Union[sitk.Image, np.ndarray], Union[sitk.Image, np.ndarray]]:
    """
    Create dual-channel projection (MIP + AIP) for 2D segmentation.
    
    This creates a two-channel input suitable for 2D U-Net models,
    combining maximum and average intensity projections.
    
    Args:
        image: Input 3D image
        axis: Projection axis (default: 1 for coronal)
        
    Returns:
        Tuple of (MIP, AIP) projections
        
    Example:
        >>> mip, aip = create_dual_projection(ct_image)
        >>> # Stack for 2-channel input
        >>> dual_channel = np.stack([mip, aip], axis=0)
    """
    mip = create_coronal_projection(image, projection_type='maximum', axis=axis)
    aip = create_coronal_projection(image, projection_type='average', axis=axis)
    
    return mip, aip


def create_multichannel_projection(
    image: Union[sitk.Image, np.ndarray],
    projections: List[str] = ['maximum', 'average'],
    axis: int = 1
) -> np.ndarray:
    """
    Create multi-channel projection for 2D segmentation.
    
    Args:
        image: Input 3D image
        projections: List of projection types to include
        axis: Projection axis
        
    Returns:
        Multi-channel numpy array (channels, height, width)
        
    Example:
        >>> multi_proj = create_multichannel_projection(
        ...     ct_image,
        ...     projections=['maximum', 'average', 'minimum']
        ... )
    """
    is_sitk = isinstance(image, sitk.Image)
    
    channels = []
    for proj_type in projections:
        proj = create_coronal_projection(image, projection_type=proj_type, axis=axis)
        if is_sitk:
            proj = sitk.GetArrayFromImage(proj)
        channels.append(proj)
    
    return np.stack(channels, axis=0)


def expand_2d_to_3d(
    segmentation_2d: Union[sitk.Image, np.ndarray],
    reference_3d: sitk.Image,
    axis: int = 1
) -> sitk.Image:
    """
    Expand 2D segmentation back to 3D by replicating along projection axis.
    
    This is useful for visualizing 2D segmentation results in 3D space
    or for combining with 3D segmentations.
    
    Args:
        segmentation_2d: 2D segmentation result
        reference_3d: Original 3D image for geometry reference
        axis: Axis along which the projection was made
        
    Returns:
        3D segmentation with the 2D result replicated along the axis
        
    Example:
        >>> seg_2d = model.predict(projection)
        >>> seg_3d = expand_2d_to_3d(seg_2d, original_ct)
    """
    if isinstance(segmentation_2d, sitk.Image):
        seg_array = sitk.GetArrayFromImage(segmentation_2d)
    else:
        seg_array = segmentation_2d
    
    # Get 3D shape
    size_3d = list(reference_3d.GetSize())
    
    # Insert the projection axis dimension
    target_shape = list(seg_array.shape)
    target_shape.insert(axis, size_3d[axis])
    
    # Expand by replicating
    seg_3d_array = np.expand_dims(seg_array, axis=axis)
    seg_3d_array = np.repeat(seg_3d_array, size_3d[axis], axis=axis)
    
    # Create SimpleITK image
    seg_3d = sitk.GetImageFromArray(seg_3d_array)
    seg_3d.CopyInformation(reference_3d)
    
    return seg_3d


def get_projection_statistics(
    image: Union[sitk.Image, np.ndarray],
    axis: int = 1
) -> dict:
    """
    Compute statistics for different projection types.
    
    This can help in choosing the optimal projection type for a given image.
    
    Args:
        image: Input 3D image
        axis: Projection axis
        
    Returns:
        Dictionary with statistics for each projection type
    """
    is_sitk = isinstance(image, sitk.Image)
    
    if is_sitk:
        array = sitk.GetArrayFromImage(image)
    else:
        array = image
    
    stats = {}
    
    for proj_type in ['maximum', 'average', 'minimum']:
        proj = create_coronal_projection(array, projection_type=proj_type, axis=axis)
        
        stats[proj_type] = {
            'mean': float(np.mean(proj)),
            'std': float(np.std(proj)),
            'min': float(np.min(proj)),
            'max': float(np.max(proj)),
            'median': float(np.median(proj))
        }
    
    return stats


def determine_optimal_projection_axis(
    image: sitk.Image
) -> int:
    """
    Determine the optimal projection axis based on image geometry.
    
    For CT scans, coronal projection (axis=1) is typically used,
    but this function can help determine the best axis based on
    image dimensions and spacing.
    
    Args:
        image: Input 3D image
        
    Returns:
        Optimal axis for projection (0, 1, or 2)
    """
    size = image.GetSize()
    spacing = image.GetSpacing()
    
    # Typically, we want to project along the axis with the most slices
    # to get the most information in the projection
    
    # For standard CT scans:
    # - Axial slices are usually the most numerous (axis=2)
    # - Coronal projection projects along anterior-posterior (axis=1)
    
    # Default to coronal (axis=1) as used in TotalSegmentator 2D
    # But can be adjusted based on image geometry
    
    # If one dimension is much larger, project along that
    max_dim = max(size)
    if size[0] == max_dim and size[0] > size[1] * 1.5:
        return 0  # Sagittal
    elif size[2] == max_dim and size[2] > size[1] * 1.5:
        return 2  # Axial
    else:
        return 1  # Coronal (default)
