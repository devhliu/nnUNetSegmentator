"""
Resource Management Utilities

This module provides context managers for resource management including
GPU memory and temporary files.
"""

from typing import Optional, Iterator
from contextlib import contextmanager
import logging
import tempfile
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


@contextmanager
def gpu_memory_management(device: str = "cuda:0") -> Iterator[None]:
    """
    Context manager for GPU memory management.

    This context manager ensures proper GPU memory cleanup after operations.
    It clears the CUDA cache on entry and exit to ensure clean memory state.

    Args:
        device: Device string (e.g., 'cuda:0', 'cuda:1', 'mps')

    Example:
        with gpu_memory_management("cuda:0"):
            # Run inference
            result = predictor.predict(input_data)
        # GPU memory is cleaned up
    """
    try:
        import torch
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        yield
    except ImportError:
        yield
    finally:
        try:
            import torch
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except (ImportError, AssertionError):
            pass


@contextmanager
def temp_directory_cleanup(dir_path: Optional[Path] = None) -> Iterator[Path]:
    """
    Context manager for temporary directory management.

    Creates a temporary directory on entry and cleans it up on exit.
    If dir_path is provided, uses that directory instead.

    Args:
        dir_path: Optional directory path to use instead of creating temp

    Yields:
        Path to the temporary directory

    Example:
        with temp_directory_cleanup() as temp_dir:
            # Process files in temp directory
            output_path = temp_dir / "output.nii.gz"
        # Temp directory is cleaned up
    """
    if dir_path is not None:
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        yield dir_path
        return

    temp_dir = tempfile.mkdtemp(prefix="nnunet_")
    temp_path = Path(temp_dir)

    try:
        yield temp_path
    finally:
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except OSError as exc:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {exc}")


@contextmanager
def model_loader_context(model_path: str, device: str = "cuda:0") -> Iterator:
    """
    Context manager for loading and unloading models.

    This context manager handles model loading, GPU transfer, and cleanup.

    Args:
        model_path: Path to the model
        device: Device to load the model on

    Yields:
        Loaded model instance

    Example:
        with model_loader_context("/path/to/model", "cuda:0") as model:
            prediction = model.predict(input_data)
        # Model is unloaded and memory is freed
    """
    model = None
    try:
        from nnunetv2.inference.predict import nnUNetPredictor
        import torch

        model = nnUNetPredictor(device=torch.device(device))
        model.initialize_from_trained_model_folder(model_path, folds=[0])
        logger.info(f"Loaded model from {model_path} on {device}")

        yield model
    finally:
        if model is not None:
            del model
            try:
                import torch
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
            except (ImportError, AssertionError):
                pass
            logger.debug(f"Unloaded model from {device}")


class ResourcePool:
    """
    Pool for managing shared resources like GPU memory.

    This class provides a simple resource pooling mechanism for
    managing expensive objects like model predictors across
    multiple operations.
    """

    def __init__(self, max_size: int = 2):
        self._max_size = max_size
        self._resources: dict = {}
        self._lock = None

    def acquire(self, key: str, factory: callable) -> object:
        """
        Acquire a resource from the pool.

        Args:
            key: Resource identifier
            factory: Factory function to create resource if not in pool

        Returns:
            Resource object
        """
        if key not in self._resources:
            self._resources[key] = factory()
        return self._resources[key]

    def release(self, key: str) -> None:
        """
        Release a resource back to the pool.

        Args:
            key: Resource identifier
        """
        if key in self._resources:
            del self._resources[key]

    def clear(self) -> None:
        """Clear all resources from the pool"""
        self._resources.clear()
        try:
            import torch
            torch.cuda.empty_cache()
        except (ImportError, AssertionError):
            pass
