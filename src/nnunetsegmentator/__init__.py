"""
nnUNetSegmentator - A Unified Framework for Medical Image Segmentation

nnUNetSegmentator provides a unified interface to multiple state-of-the-art
nnUNet-based medical image segmentation models. It simplifies access to 22+
pretrained models across various modalities (CT, MR, PET/CT) without requiring
deep expertise in each model's implementation.

Key Features:
    - 22+ segmentation models (GTRC-Net, LION, TotalSegmentator, etc.)
    - Multi-modality support (CT, MR, PET, PET/CT, SPECT/CT)
    - Flexible processing pipelines
    - Multi-format I/O (NIfTI, DICOM, MHA, NRRD)
    - Batch processing capabilities
    - Production-ready with comprehensive error handling

Quick Start:
    >>> from nnunetsegmentator import SegmentationOrchestrator
    >>> orchestrator = SegmentationOrchestrator(task_name='total')
    >>> result = orchestrator.segment('ct_scan.nii.gz', 'output.nii.gz')

For more information, visit: https://github.com/devhliu/nnunetsegmentator

Example:
    Basic usage::

        from nnunetsegmentator import SegmentationOrchestrator

        # Initialize with a task
        orchestrator = SegmentationOrchestrator(task_name='total')

        # Segment an image
        result = orchestrator.segment('input.nii.gz', 'output.nii.gz')

        # Access results
        print(f"Segmentation shape: {result.segmentation.shape}")
        print(f"Number of labels: {len(result.labels)}")

    Multi-modal segmentation::

        from nnunetsegmentator import SegmentationOrchestrator

        orchestrator = SegmentationOrchestrator(task_name='gtrc')

        # Segment with PET and CT
        result = orchestrator.segment(
            input_data={'pet': 'pet.nii.gz', 'ct': 'ct.nii.gz'},
            output_path='tumor.nii.gz'
        )

Note:
    All segmentation models are developed by their respective original authors.
    This framework provides a unified interface to access these models.
    Please cite the original publications when using specific models.

See Also:
    - Documentation: https://github.com/devhliu/nnunetsegmentator#readme
    - Examples: https://github.com/devhliu/nnunetsegmentator/tree/main/examples
    - Issues: https://github.com/devhliu/nnunetsegmentator/issues
"""

__version__ = "0.2.1"
__author__ = "devhliu"
__license__ = "MIT"

from .core.orchestrator import SegmentationOrchestrator
from .core.registry import TaskRegistry, TaskDefinition, ModelInfo
from .core.config import Config
from .core.exceptions import (
    nnunetsegmentatorError,
    ConfigurationError,
    TaskNotFoundError,
    ModelNotFoundError,
    ModelDownloadError,
    PipelineError,
    PipelineValidationError,
    PipelineConfigurationError,
    StepNotFoundError,
    InputError,
    OutputError,
    FormatNotSupportedError,
    DeviceError,
    GPUNotAvailableError,
    InferenceError,
    PostprocessingError,
    RegistrationError,
)
from . import image

__all__ = [
    "SegmentationOrchestrator",
    "TaskRegistry",
    "TaskDefinition",
    "ModelInfo",
    "Config",
    "nnunetsegmentatorError",
    "ConfigurationError",
    "TaskNotFoundError",
    "ModelNotFoundError",
    "ModelDownloadError",
    "PipelineError",
    "PipelineValidationError",
    "PipelineConfigurationError",
    "StepNotFoundError",
    "InputError",
    "OutputError",
    "FormatNotSupportedError",
    "DeviceError",
    "GPUNotAvailableError",
    "InferenceError",
    "PostprocessingError",
    "RegistrationError",
    "image",
]
