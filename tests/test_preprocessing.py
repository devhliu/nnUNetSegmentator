"""Tests for Preprocessing Pipeline Steps"""

import pytest
import numpy as np
from nnunetsegmentator import image as sitk

from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.steps.preprocessing import (
    NormalizeStep,
    ClipIntensityStep,
    SUVThresholdStep,
    WindowLevelStep,
)


class TestNormalizeStep:
    """Tests for NormalizeStep"""

    def _create_context(self, array):
        """Helper to create pipeline context"""
        image = sitk.GetImageFromArray(array.astype(np.float32))
        image.SetSpacing((1.5, 1.5, 1.5))
        image.SetOrigin((0, 0, 0))
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return PipelineContext(
            input_image=image,
            input_array=array.astype(np.float32),
            metadata={}
        )

    def test_normalize_minmax(self):
        """Test minmax normalization"""
        array = np.array([[[100.0]]])
        context = self._create_context(array)

        step = NormalizeStep("normalize", {"method": "minmax"})
        result = step.execute(context)

        normalized = result.input_array
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_zscore(self):
        """Test z-score normalization"""
        array = np.array([[[50.0, 100.0], [150.0, 200.0]]])
        array = array[:, :, np.newaxis]
        context = self._create_context(array.astype(np.float32))

        step = NormalizeStep("normalize", {"method": "zscore"})
        result = step.execute(context)

        normalized = result.input_array
        np.testing.assert_array_almost_equal(normalized.mean(), 0.0, decimal=5)
        np.testing.assert_array_almost_equal(normalized.std(), 1.0, decimal=5)

    def test_normalize_ct(self):
        """Test CT-specific normalization"""
        array = np.array([[[[-1500.0, 0.0, 1000.0, 2000.0]]]])
        context = self._create_context(array)

        step = NormalizeStep("normalize", {"method": "ct"})
        result = step.execute(context)

        normalized = result.input_array
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0

    def test_normalize_preserves_metadata(self):
        """Test that normalization preserves image metadata"""
        array = np.array([[[100.0]]])
        context = self._create_context(array)

        original_spacing = context.input_image.GetSpacing()
        original_origin = context.input_image.GetOrigin()

        step = NormalizeStep("normalize", {"method": "minmax"})
        result = step.execute(context)

        assert result.input_image.GetSpacing() == original_spacing
        assert result.input_image.GetOrigin() == original_origin

    def test_normalize_unknown_method_raises(self):
        """Test that unknown method raises ValueError"""
        array = np.array([[[100.0]]])
        context = self._create_context(array)

        step = NormalizeStep("normalize", {"method": "unknown"})
        with pytest.raises(ValueError, match="Unknown normalization method"):
            step.execute(context)


class TestClipIntensityStep:
    """Tests for ClipIntensityStep"""

    def _create_context(self, array):
        """Helper to create pipeline context"""
        image = sitk.GetImageFromArray(array)
        return PipelineContext(
            input_image=image,
            input_array=array,
            metadata={}
        )

    def test_clip_default_bounds(self):
        """Test clipping with default bounds"""
        array = np.array([[[-1500.0, 0.0, 500.0, 1500.0]]])
        context = self._create_context(array)

        step = ClipIntensityStep("clip")
        result = step.execute(context)

        clipped = sitk.GetArrayFromImage(result.input_image)
        assert clipped.min() >= -1000
        assert clipped.max() <= 1000

    def test_clip_custom_bounds(self):
        """Test clipping with custom bounds"""
        array = np.array([[[-100.0, 0.0, 50.0, 100.0]]])
        context = self._create_context(array)

        step = ClipIntensityStep("clip", {"lower": -50, "upper": 75})
        result = step.execute(context)

        clipped = sitk.GetArrayFromImage(result.input_image)
        assert clipped.min() >= -50
        assert clipped.max() <= 75


class TestSUVThresholdStep:
    """Tests for SUVThresholdStep"""

    def _create_context(self, array):
        """Helper to create pipeline context"""
        image = sitk.GetImageFromArray(array.astype(np.float32))
        image.SetSpacing((1.5, 1.5, 1.5))
        image.SetOrigin((0, 0, 0))
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return PipelineContext(
            input_image=image,
            input_array=array.astype(np.float32),
            metadata={}
        )

    def test_suv_threshold_default(self):
        """Test SUV threshold with default value"""
        array = np.array([[[0.0, 1.0, 2.5, 5.0, 10.0]]])
        context = self._create_context(array)

        step = SUVThresholdStep("suv_threshold")
        result = step.execute(context)

        thresholded = result.input_array
        assert thresholded[0, 0, 0] == 0.0
        assert thresholded[0, 0, 1] == 0.0
        assert thresholded[0, 0, 2] == 0.0
        assert thresholded[0, 0, 3] == 5.0
        assert thresholded[0, 0, 4] == 10.0

    def test_suv_threshold_custom(self):
        """Test SUV threshold with custom value"""
        array = np.array([[[0.0, 1.0, 2.5, 5.0]]])
        context = self._create_context(array)

        step = SUVThresholdStep("suv_threshold", {"threshold": 4.0})
        result = step.execute(context)

        thresholded = result.input_array
        assert thresholded[0, 0, 2] == 0.0
        assert thresholded[0, 0, 3] == 5.0

    def test_suv_threshold_preserves_metadata(self):
        """Test that SUV thresholding preserves image metadata"""
        array = np.array([[[0.0, 5.0]]])
        context = self._create_context(array)

        original_spacing = context.input_image.GetSpacing()
        original_origin = context.input_image.GetOrigin()

        step = SUVThresholdStep("suv_threshold")
        result = step.execute(context)

        assert result.input_image.GetSpacing() == original_spacing
        assert result.input_image.GetOrigin() == original_origin


class TestWindowLevelStep:
    """Tests for WindowLevelStep"""

    def _create_context(self, array):
        """Helper to create pipeline context"""
        image = sitk.GetImageFromArray(array.astype(np.float32))
        image.SetSpacing((1.5, 1.5, 1.5))
        image.SetOrigin((0, 0, 0))
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return PipelineContext(
            input_image=image,
            input_array=array.astype(np.float32),
            metadata={}
        )

    def test_window_level_default(self):
        """Test window/level with defaults (400/40)"""
        array = np.array([[[0.0, 40.0, 100.0, 200.0, 300.0]]])
        context = self._create_context(array)

        step = WindowLevelStep("window_level")
        result = step.execute(context)

        windowed = result.input_array
        assert windowed.min() >= 0.0
        assert windowed.max() <= 1.0

    def test_window_level_custom(self):
        """Test window/level with custom values"""
        array = np.array([[[-50.0, 0.0, 50.0, 100.0, 150.0]]])
        context = self._create_context(array)

        step = WindowLevelStep("window_level", {"window": 100, "level": 50})
        result = step.execute(context)

        windowed = result.input_array
        assert windowed.min() >= 0.0
        assert windowed.max() <= 1.0

    def test_window_level_preserves_metadata(self):
        """Test that window/level preserves image metadata"""
        array = np.array([[[0.0, 100.0]]])
        context = self._create_context(array)

        original_spacing = context.input_image.GetSpacing()
        original_origin = context.input_image.GetOrigin()

        step = WindowLevelStep("window_level")
        result = step.execute(context)

        assert result.input_image.GetSpacing() == original_spacing
        assert result.input_image.GetOrigin() == original_origin
