"""
Metrics Utilities

This module provides functions for computing segmentation metrics.
"""

import numpy as np
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_volume(
    segmentation: np.ndarray,
    spacing: Tuple[float, float, float]
) -> float:
    """
    Compute volume of segmentation in mm³.
    
    Args:
        segmentation: Binary segmentation array
        spacing: Voxel spacing in mm
        
    Returns:
        Volume in mm³
    """
    voxel_volume = spacing[0] * spacing[1] * spacing[2]
    num_voxels = np.sum(segmentation > 0)
    return float(num_voxels * voxel_volume)


def compute_surface_area(
    segmentation: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute surface area of segmentation.
    
    Args:
        segmentation: Binary segmentation array
        spacing: Voxel spacing
        
    Returns:
        Surface area
    """
    from scipy import ndimage
    
    # Compute gradient
    grad = np.gradient(segmentation.astype(float), spacing)
    
    # Surface area is integral of gradient magnitude over boundary
    grad_mag = np.sqrt(sum(g**2 for g in grad))
    
    # Approximate surface area
    surface_area = np.sum(grad_mag) * np.prod(spacing)
    
    return float(surface_area)


def compute_dice(
    segmentation1: np.ndarray,
    segmentation2: np.ndarray
) -> float:
    """
    Compute Dice similarity coefficient.
    
    Args:
        segmentation1: First binary segmentation
        segmentation2: Second binary segmentation
        
    Returns:
        Dice coefficient (0-1)
    """
    intersection = np.sum(segmentation1 & segmentation2)
    union = np.sum(segmentation1) + np.sum(segmentation2)
    
    if union == 0:
        return 1.0  # Both empty
    
    return float(2.0 * intersection / union)


def compute_jaccard(
    segmentation1: np.ndarray,
    segmentation2: np.ndarray
) -> float:
    """
    Compute Jaccard index (IoU).
    
    Args:
        segmentation1: First binary segmentation
        segmentation2: Second binary segmentation
        
    Returns:
        Jaccard index (0-1)
    """
    intersection = np.sum(segmentation1 & segmentation2)
    union = np.sum(segmentation1 | segmentation2)
    
    if union == 0:
        return 1.0  # Both empty
    
    return float(intersection / union)


def compute_hausdorff_distance(
    segmentation1: np.ndarray,
    segmentation2: np.ndarray,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> float:
    """
    Compute Hausdorff distance between segmentations.
    
    Args:
        segmentation1: First binary segmentation
        segmentation2: Second binary segmentation
        spacing: Voxel spacing
        
    Returns:
        Hausdorff distance in mm
    """
    from scipy import ndimage
    
    # Get surface points
    surf1 = _get_surface_points(segmentation1)
    surf2 = _get_surface_points(segmentation2)
    
    if len(surf1) == 0 or len(surf2) == 0:
        return float('inf')
    
    # Compute distances
    # Distance from surf1 to surf2
    dist1 = ndimage.distance_transform_edt(
        ~segmentation2,
        sampling=spacing
    )
    max_dist1 = np.max(dist1[tuple(surf1.T)])
    
    # Distance from surf2 to surf1
    dist2 = ndimage.distance_transform_edt(
        ~segmentation1,
        sampling=spacing
    )
    max_dist2 = np.max(dist2[tuple(surf2.T)])
    
    return float(max(max_dist1, max_dist2))


def _get_surface_points(segmentation: np.ndarray) -> np.ndarray:
    """
    Get surface points of a binary segmentation.
    
    Args:
        segmentation: Binary segmentation
        
    Returns:
        Array of surface point coordinates
    """
    from scipy import ndimage
    
    # Erode
    eroded = ndimage.binary_erosion(segmentation)
    
    # Surface is boundary
    surface = segmentation & ~eroded
    
    # Get coordinates
    coords = np.where(surface)
    
    return np.array(coords).T


def compute_mean_intensity(
    segmentation: np.ndarray,
    image: np.ndarray
) -> float:
    """
    Compute mean intensity of image within segmentation mask.
    
    Args:
        segmentation: Binary segmentation array
        image: Intensity image array
        
    Returns:
        Mean intensity
    """
    if np.sum(segmentation) == 0:
        return 0.0
    return float(np.mean(image[segmentation > 0]))


def compute_suv_max(
    segmentation: np.ndarray,
    image: np.ndarray
) -> float:
    """
    Compute max SUV (intensity) within segmentation mask.
    
    Args:
        segmentation: Binary segmentation array
        image: Intensity image array (assumed to be SUV if called for SUV max)
        
    Returns:
        Max intensity
    """
    if np.sum(segmentation) == 0:
        return 0.0
    return float(np.max(image[segmentation > 0]))


def compute_all_metrics(
    segmentation: np.ndarray,
    reference: np.ndarray = None,
    image: np.ndarray = None,
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Compute all available metrics.
    
    Args:
        segmentation: Segmentation array
        reference: Optional reference segmentation for comparison
        image: Optional intensity image for intensity metrics (SUV, etc.)
        spacing: Voxel spacing
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'volume_mm3': compute_volume(segmentation, spacing),
        'num_voxels': int(np.sum(segmentation > 0)),
        'surface_area': compute_surface_area(segmentation, spacing),
    }
    
    if image is not None:
        metrics.update({
            'mean_intensity': compute_mean_intensity(segmentation, image),
            'max_intensity': compute_suv_max(segmentation, image),
            # Alias for PET context
            'suv_mean': compute_mean_intensity(segmentation, image),
            'suv_max': compute_suv_max(segmentation, image),
            'tlg': compute_mean_intensity(segmentation, image) * compute_volume(segmentation, spacing) / 1000.0, # Total Lesion Glycolysis (SUV * ml)
        })
    
    if reference is not None:
        metrics.update({
            'dice': compute_dice(segmentation, reference),
            'jaccard': compute_jaccard(segmentation, reference),
            'hausdorff_mm': compute_hausdorff_distance(segmentation, reference, spacing),
        })
    
    return metrics
