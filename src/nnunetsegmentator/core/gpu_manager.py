"""
GPU Manager for Herd Mode

This module provides GPU detection and assignment for distributed inference.
"""

from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Information about a compute device"""
    name: str
    type: str
    index: Optional[int] = None

    def __str__(self) -> str:
        if self.index is not None:
            return f"{self.type}:{self.index} ({self.name})"
        return f"{self.type} ({self.name})"


class GPUMonitor:
    """
    Monitor for compute devices (GPU/MPS/CPU).

    This class provides device detection and round-robin assignment
    for distributed inference across multiple devices.
    """

    def __init__(self):
        self._devices: List[DeviceInfo] = []
        self._initialize_devices()

    def _initialize_devices(self) -> None:
        """Detect available compute devices"""
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    self._devices.append(DeviceInfo(
                        name=device_name,
                        type="cuda",
                        index=i
                    ))
                logger.info(f"Detected {len(self._devices)} CUDA GPU(s)")
                return

            if torch.backends.mps.is_available():
                self._devices.append(DeviceInfo(
                    name="Apple MPS",
                    type="mps",
                    index=None
                ))
                logger.info("Detected Apple MPS")
                return

        except ImportError:
            logger.warning("PyTorch not available. GPU monitoring disabled.")

        self._devices.append(DeviceInfo(
            name="CPU",
            type="cpu",
            index=None
        ))
        logger.info("Using CPU for computation")

    @property
    def devices(self) -> List[DeviceInfo]:
        """Get list of available devices"""
        return self._devices

    @property
    def device_count(self) -> int:
        """Get number of available devices"""
        return len(self._devices)

    @property
    def has_gpu(self) -> bool:
        """Check if GPU is available"""
        return any(d.type in ("cuda", "mps") for d in self._devices)

    def get_device(self, index: int) -> str:
        """
        Get device string for a given index.

        Args:
            index: Device index

        Returns:
            Device string (e.g., 'cuda:0', 'mps', 'cpu')
        """
        if not self._devices:
            return "cpu"

        device = self._devices[index % len(self._devices)]
        if device.index is not None:
            return f"{device.type}:{device.index}"
        return device.type

    def assign_devices(self, num_subjects: int) -> List[Optional[str]]:
        """
        Assign devices to subjects using round-robin distribution.

        Args:
            num_subjects: Number of subjects to process

        Returns:
            List of device strings for each subject
        """
        if not self.has_gpu or self.device_count == 0:
            return [None] * num_subjects

        return [self.get_device(i) for i in range(num_subjects)]

    def requires_multiprocessing(self, num_workers: int) -> bool:
        """
        Check if multiprocessing should be used for thread safety.

        Args:
            num_workers: Number of workers

        Returns:
            True if multiprocessing is recommended
        """
        cuda_count = sum(1 for d in self._devices if d.type == "cuda")
        return cuda_count > 1 and num_workers > 1


def get_gpu_monitor() -> GPUMonitor:
    """
    Get a singleton GPU monitor instance.

    Returns:
        GPUMonitor instance
    """
    if not hasattr(get_gpu_monitor, "_instance"):
        get_gpu_monitor._instance = GPUMonitor()
    return get_gpu_monitor._instance
