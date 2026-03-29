"""Tests for GPU Manager"""

import pytest
from unittest.mock import patch, MagicMock

from nnunetsegmentator.core.gpu_manager import (
    GPUMonitor,
    DeviceInfo,
    get_gpu_monitor,
)


class TestDeviceInfo:
    """Tests for DeviceInfo dataclass"""

    def test_creation_with_index(self):
        """Test DeviceInfo creation with device index"""
        device = DeviceInfo(name="NVIDIA RTX 3090", type="cuda", index=0)
        assert device.name == "NVIDIA RTX 3090"
        assert device.type == "cuda"
        assert device.index == 0

    def test_creation_without_index(self):
        """Test DeviceInfo creation without device index (CPU/MPS)"""
        device = DeviceInfo(name="Apple MPS", type="mps")
        assert device.name == "Apple MPS"
        assert device.type == "mps"
        assert device.index is None

    def test_str_with_index(self):
        """Test string representation with index"""
        device = DeviceInfo(name="NVIDIA RTX 3090", type="cuda", index=0)
        assert str(device) == "cuda:0 (NVIDIA RTX 3090)"

    def test_str_without_index(self):
        """Test string representation without index"""
        device = DeviceInfo(name="CPU", type="cpu")
        assert str(device) == "cpu (CPU)"


class TestGPUMonitor:
    """Tests for GPUMonitor class"""

    @patch('nnunetsegmentator.core.gpu_manager.logger')
    def test_init_without_torch(self, mock_logger):
        """Test initialization when torch is not available"""
        with patch.dict('sys.modules', {'torch': None}):
            monitor = GPUMonitor()
            assert len(monitor.devices) == 1
            assert monitor.devices[0].type == "cpu"

    @patch('nnunetsegmentator.core.gpu_manager.logger')
    def test_init_with_cuda(self, mock_logger):
        """Test initialization with CUDA available"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        mock_torch.cuda.get_device_name.side_effect = ["GPU 0", "GPU 1"]

        with patch.dict('sys.modules', {'torch': mock_torch}):
            monitor = GPUMonitor()
            assert monitor.device_count == 2
            assert monitor.has_gpu is True

    @patch('nnunetsegmentator.core.gpu_manager.logger')
    def test_init_with_mps(self, mock_logger):
        """Test initialization with MPS available"""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict('sys.modules', {'torch': mock_torch}):
            monitor = GPUMonitor()
            assert monitor.device_count == 1
            assert monitor.devices[0].type == "mps"
            assert monitor.has_gpu is True

    def test_get_device_single_device(self):
        """Test get_device with single device"""
        monitor = GPUMonitor()
        monitor._devices = [DeviceInfo(name="CPU", type="cpu")]

        assert monitor.get_device(0) == "cpu"
        assert monitor.get_device(5) == "cpu"

    def test_get_device_multiple_devices(self):
        """Test get_device with multiple devices (round-robin)"""
        monitor = GPUMonitor()
        monitor._devices = [
            DeviceInfo(name="GPU 0", type="cuda", index=0),
            DeviceInfo(name="GPU 1", type="cuda", index=1),
        ]

        assert monitor.get_device(0) == "cuda:0"
        assert monitor.get_device(1) == "cuda:1"
        assert monitor.get_device(2) == "cuda:0"
        assert monitor.get_device(3) == "cuda:1"

    def test_assign_devices_empty(self):
        """Test assign_devices with no GPU"""
        monitor = GPUMonitor()
        monitor._devices = [DeviceInfo(name="CPU", type="cpu")]

        assignments = monitor.assign_devices(3)
        assert assignments == [None, None, None]

    def test_assign_devices_with_gpu(self):
        """Test assign_devices with GPU"""
        monitor = GPUMonitor()
        monitor._devices = [
            DeviceInfo(name="GPU 0", type="cuda", index=0),
            DeviceInfo(name="GPU 1", type="cuda", index=1),
        ]

        assignments = monitor.assign_devices(3)
        assert assignments == ["cuda:0", "cuda:1", "cuda:0"]

    def test_requires_multiprocessing_single_gpu(self):
        """Test multiprocessing not required with single GPU"""
        monitor = GPUMonitor()
        monitor._devices = [DeviceInfo(name="GPU 0", type="cuda", index=0)]

        assert monitor.requires_multiprocessing(4) is False

    def test_requires_multiprocessing_multi_gpu(self):
        """Test multiprocessing required with multiple GPUs"""
        monitor = GPUMonitor()
        monitor._devices = [
            DeviceInfo(name="GPU 0", type="cuda", index=0),
            DeviceInfo(name="GPU 1", type="cuda", index=1),
        ]

        assert monitor.requires_multiprocessing(2) is True
        assert monitor.requires_multiprocessing(1) is False


class TestGetGPUMonitorSingleton:
    """Tests for get_gpu_monitor singleton"""

    def test_returns_singleton(self):
        """Test that get_gpu_monitor returns the same instance"""
        if hasattr(get_gpu_monitor, "_instance"):
            del get_gpu_monitor._instance

        monitor1 = get_gpu_monitor()
        monitor2 = get_gpu_monitor()
        assert monitor1 is monitor2
