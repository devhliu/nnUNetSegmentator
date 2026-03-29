"""
Visualization Utilities

This module provides functions for visualizing segmentations.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def visualize_segmentation(
    image: np.ndarray,
    segmentation: np.ndarray,
    slice_idx: int = None,
    axis: int = 0,
    alpha: float = 0.5,
    cmap: str = 'gray',
    seg_cmap: str = 'jet',
    title: str = None,
    save_path: str = None
):
    """
    Visualize segmentation overlay on image.
    
    Args:
        image: Image array
        segmentation: Segmentation array
        slice_idx: Slice index (default: middle)
        axis: Axis for slicing (0, 1, or 2)
        alpha: Overlay transparency
        cmap: Image colormap
        seg_cmap: Segmentation colormap
        title: Plot title
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    # Get slice
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        seg_slice = segmentation[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        seg_slice = segmentation[:, slice_idx, :]
    else:
        img_slice = image[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Image
    axes[0].imshow(img_slice, cmap=cmap)
    axes[0].set_title('Image')
    axes[0].axis('off')
    
    # Segmentation
    axes[1].imshow(seg_slice, cmap=seg_cmap)
    axes[1].set_title('Segmentation')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(img_slice, cmap=cmap)
    mask = seg_slice > 0
    axes[2].imshow(np.ma.masked_where(~mask, seg_slice), cmap=seg_cmap, alpha=alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    if title:
        fig.suptitle(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def create_comparison_plot(
    image: np.ndarray,
    segmentation1: np.ndarray,
    segmentation2: np.ndarray,
    label1: str = "Segmentation 1",
    label2: str = "Segmentation 2",
    slice_idx: int = None,
    axis: int = 0,
    save_path: str = None
):
    """
    Create comparison plot of two segmentations.
    
    Args:
        image: Image array
        segmentation1: First segmentation
        segmentation2: Second segmentation
        label1: Label for first segmentation
        label2: Label for second segmentation
        slice_idx: Slice index
        axis: Slicing axis
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    # Get slices
    if slice_idx is None:
        slice_idx = image.shape[axis] // 2
    
    if axis == 0:
        img_slice = image[slice_idx, :, :]
        seg1_slice = segmentation1[slice_idx, :, :]
        seg2_slice = segmentation2[slice_idx, :, :]
    elif axis == 1:
        img_slice = image[:, slice_idx, :]
        seg1_slice = segmentation1[:, slice_idx, :]
        seg2_slice = segmentation2[:, slice_idx, :]
    else:
        img_slice = image[:, :, slice_idx]
        seg1_slice = segmentation1[:, :, slice_idx]
        seg2_slice = segmentation2[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Image
    axes[0, 0].imshow(img_slice, cmap='gray')
    axes[0, 0].set_title('Image')
    axes[0, 0].axis('off')
    
    # Segmentation 1
    axes[0, 1].imshow(img_slice, cmap='gray')
    axes[0, 1].imshow(np.ma.masked_where(seg1_slice == 0, seg1_slice), 
                     cmap='jet', alpha=0.5)
    axes[0, 1].set_title(label1)
    axes[0, 1].axis('off')
    
    # Segmentation 2
    axes[1, 0].imshow(img_slice, cmap='gray')
    axes[1, 0].imshow(np.ma.masked_where(seg2_slice == 0, seg2_slice), 
                     cmap='jet', alpha=0.5)
    axes[1, 0].set_title(label2)
    axes[1, 0].axis('off')
    
    # Difference
    diff = seg1_slice != seg2_slice
    axes[1, 1].imshow(img_slice, cmap='gray')
    axes[1, 1].imshow(np.ma.masked_where(~diff, diff), cmap='Reds', alpha=0.7)
    axes[1, 1].set_title('Difference')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()


def create_multi_slice_view(
    image: np.ndarray,
    segmentation: np.ndarray = None,
    num_slices: int = 6,
    axis: int = 0,
    save_path: str = None
):
    """
    Create multi-slice view of image and optional segmentation.
    
    Args:
        image: Image array
        segmentation: Optional segmentation array
        num_slices: Number of slices to show
        axis: Slicing axis
        save_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available, skipping visualization")
        return
    
    # Determine slice indices
    total_slices = image.shape[axis]
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    # Create figure
    cols = 3
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten()
    
    for i, slice_idx in enumerate(slice_indices):
        if axis == 0:
            img_slice = image[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :] if segmentation is not None else None
        elif axis == 1:
            img_slice = image[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :] if segmentation is not None else None
        else:
            img_slice = image[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx] if segmentation is not None else None
        
        axes[i].imshow(img_slice, cmap='gray')
        
        if seg_slice is not None:
            mask = seg_slice > 0
            axes[i].imshow(np.ma.masked_where(~mask, seg_slice), 
                          cmap='jet', alpha=0.5)
        
        axes[i].set_title(f'Slice {slice_idx}')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(len(slice_indices), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Figure saved to: {save_path}")
    
    plt.show()
