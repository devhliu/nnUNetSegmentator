"""
Custom Exceptions for nnunetsegmentator

This module defines the exception hierarchy for the nnunetsegmentator framework.
All custom exceptions inherit from nnunetsegmentatorError.
"""

from typing import Optional


class nnunetsegmentatorError(Exception):
    """
    Base exception for all nnunetsegmentator errors.

    All custom exceptions in this framework inherit from this class,
    making it easy to catch any framework-specific error with a single
    except clause.
    """

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message} (caused by: {self.cause})"
        return self.message


class ConfigurationError(nnunetsegmentatorError):
    """
    Raised when configuration is invalid or cannot be loaded.
    """
    pass


class TaskNotFoundError(nnunetsegmentatorError):
    """
    Raised when a requested task is not registered.
    """

    def __init__(self, task_name: str, available_tasks: Optional[list] = None):
        self.task_name = task_name
        self.available_tasks = available_tasks or []
        message = f"Task '{task_name}' not found"
        if available_tasks:
            message += f". Available tasks: {', '.join(available_tasks)}"
        super().__init__(message)


class ModelNotFoundError(nnunetsegmentatorError):
    """
    Raised when a model cannot be found or downloaded.
    """

    def __init__(self, model_name: str, task_name: Optional[str] = None):
        self.model_name = model_name
        self.task_name = task_name
        message = f"Model '{model_name}' not found"
        if task_name:
            message += f" in task '{task_name}'"
        super().__init__(message)


class ModelDownloadError(nnunetsegmentatorError):
    """
    Raised when model download fails.
    """

    def __init__(self, model_name: str, url: str, cause: Optional[Exception] = None):
        self.model_name = model_name
        self.url = url
        message = f"Failed to download model '{model_name}' from {url}"
        super().__init__(message, cause)


class PipelineError(nnunetsegmentatorError):
    """
    Raised when a pipeline execution fails.
    """

    def __init__(self, pipeline_name: str, step_name: str, cause: Optional[Exception] = None):
        self.pipeline_name = pipeline_name
        self.step_name = step_name
        message = f"Pipeline '{pipeline_name}' failed at step '{step_name}'"
        super().__init__(message, cause)


class PipelineValidationError(nnunetsegmentatorError):
    """
    Raised when pipeline configuration is invalid.
    """
    pass


class PipelineConfigurationError(nnunetsegmentatorError):
    """
    Raised when no pipeline is configured.
    """
    pass


class StepNotFoundError(nnunetsegmentatorError):
    """
    Raised when a requested pipeline step type is not registered.
    """

    def __init__(self, step_type: str, available_types: Optional[list] = None):
        self.step_type = step_type
        self.available_types = available_types or []
        message = f"Step type '{step_type}' not found"
        if available_types:
            message += f". Available types: {', '.join(available_types)}"
        super().__init__(message)


class InputError(nnunetsegmentatorError):
    """
    Raised when input data is invalid or cannot be read.
    """
    pass


class OutputError(nnunetsegmentatorError):
    """
    Raised when output cannot be written.
    """
    pass


class FormatNotSupportedError(nnunetsegmentatorError):
    """
    Raised when a file format is not supported.
    """

    def __init__(self, format_name: str, supported_formats: Optional[list] = None):
        self.format_name = format_name
        self.supported_formats = supported_formats or []
        message = f"Format '{format_name}' not supported"
        if supported_formats:
            message += f". Supported formats: {', '.join(supported_formats)}"
        super().__init__(message)


class DeviceError(nnunetsegmentatorError):
    """
    Raised when there is an issue with compute device (GPU/MPS/CPU).
    """
    pass


class GPUNotAvailableError(DeviceError):
    """
    Raised when GPU is requested but not available.
    """
    pass


class InferenceError(nnunetsegmentatorError):
    """
    Raised when inference fails.
    """
    pass


class PostprocessingError(nnunetsegmentatorError):
    """
    Raised when postprocessing fails.
    """
    pass


class RegistrationError(nnunetsegmentatorError):
    """
    Raised when task or model registration fails.
    """
    pass
