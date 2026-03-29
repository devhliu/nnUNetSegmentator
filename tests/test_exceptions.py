"""Tests for nnunetsegmentator exceptions"""

import pytest

from nnunetsegmentator.core.exceptions import (
    nnunetsegmentatorError,
    ConfigurationError,
    TaskNotFoundError,
    ModelNotFoundError,
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
)


class TestExceptionHierarchy:
    """Test that all exceptions follow the proper hierarchy"""

    def test_base_exception_inheritance(self):
        """All exceptions should inherit from nnunetsegmentatorError"""
        exc = ConfigurationError("test message")
        assert isinstance(exc, nnunetsegmentatorError)
        assert isinstance(exc, Exception)

    def test_all_exceptions_inherit_from_base(self):
        """Verify all exception classes inherit from base"""
        exceptions_with_args = [
            (ConfigurationError, ("test message",)),
            (TaskNotFoundError, ("task_name",)),
            (ModelNotFoundError, ("model_name",)),
            (PipelineError, ("pipe", "step")),
            (PipelineValidationError, ("pipe",)),
            (PipelineConfigurationError, ("pipe",)),
            (StepNotFoundError, ("step_name",)),
            (InputError, ("test message",)),
            (OutputError, ("test message",)),
            (FormatNotSupportedError, ("format",)),
            (DeviceError, ("test message",)),
            (GPUNotAvailableError, ("test message",)),
            (InferenceError, ("test message",)),
        ]
        for exc_class, args in exceptions_with_args:
            exc = exc_class(*args)
            assert isinstance(exc, nnunetsegmentatorError)


class TestTaskNotFoundError:
    """Tests for TaskNotFoundError"""

    def test_message_without_available_tasks(self):
        """Test message when no available tasks provided"""
        exc = TaskNotFoundError("my_task")
        assert "my_task" in str(exc)
        assert "not found" in str(exc)
        assert exc.task_name == "my_task"
        assert exc.available_tasks == []

    def test_message_with_available_tasks(self):
        """Test message includes available tasks when provided"""
        available = ["task1", "task2", "task3"]
        exc = TaskNotFoundError("unknown_task", available)
        assert "unknown_task" in str(exc)
        assert "task1" in str(exc)
        assert exc.available_tasks == available


class TestModelNotFoundError:
    """Tests for ModelNotFoundError"""

    def test_message_simple(self):
        """Test basic error message"""
        exc = ModelNotFoundError("model_v1")
        assert "model_v1" in str(exc)

    def test_message_with_task(self):
        """Test message includes task name when provided"""
        exc = ModelNotFoundError("model_v1", task_name="my_task")
        assert "model_v1" in str(exc)
        assert "my_task" in str(exc)


class TestPipelineError:
    """Tests for PipelineError"""

    def test_message_contains_pipeline_and_step(self):
        """Test that error message contains pipeline and step names"""
        exc = PipelineError("my_pipeline", "preprocessing_step")
        assert "my_pipeline" in str(exc)
        assert "preprocessing_step" in str(exc)
        assert exc.pipeline_name == "my_pipeline"
        assert exc.step_name == "preprocessing_step"

    def test_message_with_cause(self):
        """Test error chaining with cause"""
        cause = ValueError("original error")
        exc = PipelineError("pipeline", "step", cause=cause)
        assert exc.cause is cause
        assert "original error" in str(exc)


class TestStepNotFoundError:
    """Tests for StepNotFoundError"""

    def test_message_without_available_types(self):
        """Test message when no available types provided"""
        exc = StepNotFoundError("unknown_step")
        assert "unknown_step" in str(exc)

    def test_message_with_available_types(self):
        """Test message includes available types when provided"""
        available = ["resample", "normalize", "inference"]
        exc = StepNotFoundError("unknown", available)
        assert "unknown" in str(exc)
        assert "resample" in str(exc)


class TestFormatNotSupportedError:
    """Tests for FormatNotSupportedError"""

    def test_message_simple(self):
        """Test basic error message"""
        exc = FormatNotSupportedError("jpeg")
        assert "jpeg" in str(exc)
        assert "not supported" in str(exc)

    def test_message_with_supported_formats(self):
        """Test message includes supported formats"""
        supported = ["nifti", "dicom", "nrrd"]
        exc = FormatNotSupportedError("jpeg", supported)
        assert "jpeg" in str(exc)
        assert "nifti" in str(exc)


class TestErrorChaining:
    """Test that exceptions properly chain causes"""

    def test_cause_is_stored(self):
        """Test that cause exception is stored"""
        original = RuntimeError("original")
        exc = PipelineError("pipe", "step", cause=original)
        assert exc.cause is original

    def test_str_includes_cause(self):
        """Test that string representation includes cause"""
        original = ValueError("bad value")
        exc = ConfigurationError("config error", cause=original)
        error_str = str(exc)
        assert "config error" in error_str
        assert "bad value" in error_str
