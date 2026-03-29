"""Utility modules for the nnUNet framework"""

from .metrics import compute_volume, compute_surface_area, compute_dice
from .resampling import (
    resample_image,
    resample_segmentation,
    resample_to_reference,
    resample_segmentation_to_reference,
    compute_new_shape,
    is_isotropic,
    get_spacing_from_nifti,
)
from .projection import (
    create_coronal_projection,
    create_dual_projection,
    create_multichannel_projection,
    expand_2d_to_3d,
    get_projection_statistics,
    determine_optimal_projection_axis,
)
from .visualization import visualize_segmentation, create_comparison_plot
from .resource_management import (
    gpu_memory_management,
    temp_directory_cleanup,
    model_loader_context,
    ResourcePool,
)

__all__ = [
    "compute_volume",
    "compute_surface_area",
    "compute_dice",
    "resample_image",
    "resample_segmentation",
    "resample_to_reference",
    "resample_segmentation_to_reference",
    "compute_new_shape",
    "is_isotropic",
    "get_spacing_from_nifti",
    "create_coronal_projection",
    "create_dual_projection",
    "create_multichannel_projection",
    "expand_2d_to_3d",
    "get_projection_statistics",
    "determine_optimal_projection_axis",
    "visualize_segmentation",
    "create_comparison_plot",
    "gpu_memory_management",
    "temp_directory_cleanup",
    "model_loader_context",
    "ResourcePool",
]
