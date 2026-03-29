"""
nnunetsegmentator - Unified nnUNet Segmentation Framework

A comprehensive framework for managing multiple nnUNet-based segmentation tasks
with support for complex processing pipelines, multiple I/O formats, and both
Python and CLI interfaces.
"""

__version__ = "0.2.0"

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
