"""Tests for CropFOVStep / FOVRestrictStep (MOOSE cascade workflow)"""

import numpy as np
import pytest

from nnunetsegmentator import image as sitk
from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.builders import PipelineBuilder
from nnunetsegmentator.pipeline.steps.inference import CropFOVStep, FOVRestrictStep


def _context(array):
    image = sitk.GetImageFromArray(array)
    return PipelineContext(
        input_image=image,
        input_array=array,
        metadata={"spacing": (1.0, 1.0, 1.0)},
    )


@pytest.fixture
def patched_inference(monkeypatch):
    """
    Patch nnUNetInferenceStep.execute to emit a caller-provided prediction
    (no model download / GPU required).
    """
    def _install(prediction):
        def fake_execute(self, context):
            context.intermediate_results["raw_prediction"] = prediction.copy()
            return context

        monkeypatch.setattr(
            "nnunetsegmentator.pipeline.steps.inference.nnUNetInferenceStep.execute",
            fake_execute,
        )

    return _install


class TestCropFOVStep:
    def test_z_crop_and_bookkeeping(self, patched_inference):
        # Arrays are (x, y, z): the crop model marks a band along z (axis 2).
        input_array = np.zeros((10, 6, 10), dtype=np.float32)
        crop_prediction = np.zeros((10, 6, 10), dtype=np.uint8)
        crop_prediction[:, :, 3:7] = 22  # vertebrae band in slices 3..6

        patched_inference(crop_prediction)

        step = CropFOVStep("crop_fov", {
            "crop_model_name": "clin_ct_fast_vertebrae",
            "fov_intensities": [20, 24],
        })
        context = step.execute(_context(input_array))

        assert context.input_array.shape == (10, 6, 4)
        fov_crop = context.intermediate_results["fov_crop"]
        assert fov_crop["z_start"] == 3
        assert fov_crop["z_end"] == 7
        assert np.array_equal(fov_crop["crop_prediction"], crop_prediction)

    def test_empty_mask_keeps_full_fov(self, patched_inference):
        input_array = np.zeros((10, 6, 6), dtype=np.float32)
        patched_inference(np.zeros((10, 6, 6), dtype=np.uint8))

        step = CropFOVStep("crop_fov", {
            "crop_model_name": "clin_ct_fast_vertebrae",
            "fov_intensities": [20, 24],
        })
        context = step.execute(_context(input_array))

        assert context.input_array.shape == (10, 6, 6)
        assert "fov_crop" not in context.intermediate_results

    def test_missing_crop_model_raises(self):
        step = CropFOVStep("crop_fov", {})
        with pytest.raises(ValueError):
            step.execute(_context(np.zeros((4, 4, 4))))


class TestFOVRestrictStep:
    def test_paste_and_band_zeroing(self, patched_inference):
        # Arrays are (x, y, z): the crop mask band spans z slices 3..6.
        full = np.zeros((10, 6, 10), dtype=np.float32)
        crop_prediction = np.zeros((10, 6, 10), dtype=np.uint8)
        crop_prediction[:, :, 3:7] = 22

        step = FOVRestrictStep("restrict_fov", {
            "crop_label": 22,
            "largest_component_only": True,
        })
        context = _context(full)
        context.intermediate_results["fov_crop"] = {
            "z_start": 3,
            "z_end": 7,
            "crop_prediction": crop_prediction,
        }
        # Target model prediction on the cropped volume
        cropped_prediction = np.full((10, 6, 4), 5, dtype=np.uint8)
        context.intermediate_results["raw_prediction"] = cropped_prediction

        context = step.execute(context)
        result = context.intermediate_results["raw_prediction"]

        assert result.shape == (10, 6, 10)
        assert np.all(result[:, :, 3:7] == 5)
        assert np.all(result[:, :, :3] == 0)
        assert np.all(result[:, :, 7:] == 0)

    def test_without_crop_info_passthrough(self):
        step = FOVRestrictStep("restrict_fov", {"crop_label": 22})
        prediction = np.full((4, 4, 4), 2, dtype=np.uint8)
        context = _context(np.zeros((4, 4, 4)))
        context.intermediate_results["raw_prediction"] = prediction

        context = step.execute(context)
        assert np.array_equal(context.intermediate_results["raw_prediction"], prediction)

    def test_prefers_refined_prediction(self):
        step = FOVRestrictStep("restrict_fov", {})
        refined = np.full((2, 4, 4), 9, dtype=np.uint8)
        context = _context(np.zeros((4, 4, 4)))
        context.intermediate_results["raw_prediction"] = np.zeros((2, 4, 4), dtype=np.uint8)
        context.intermediate_results["refined_prediction"] = refined

        context = step.execute(context)
        assert np.array_equal(context.intermediate_results["raw_prediction"], refined)


class TestStepRegistry:
    def test_cascade_steps_registered(self):
        assert PipelineBuilder.STEP_REGISTRY["crop_fov"] is CropFOVStep
        assert PipelineBuilder.STEP_REGISTRY["fov_restrict"] is FOVRestrictStep
