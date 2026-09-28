"""Tests for RegionCropStep / RegionRestoreStep (TotalSegmentator crop workflow)
and the per-label postprocessing options (body task postprocessing)."""

import numpy as np
import pytest

from nnunetsegmentator import image as sitk
from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.builders import PipelineBuilder
from nnunetsegmentator.pipeline.steps.postprocessing import (
    LargestComponentStep,
    RemoveSmallObjectsStep,
)
from nnunetsegmentator.pipeline.steps.roi import RegionCropStep, RegionRestoreStep

# Importing the tasks package registers all built-in tasks, which
# RegionCropStep relies on (TaskRegistry.find_model) for label resolution.
import nnunetsegmentator.tasks  # noqa: F401


LUNG_LABEL = "lung_upper_lobe_left"  # id 10 in the official total labels


def _context(array, spacing=(1.0, 1.0, 1.0)):
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(tuple(spacing))
    return PipelineContext(
        input_image=image,
        input_array=array,
        metadata={"spacing": spacing},
    )


@pytest.fixture
def patched_inference(monkeypatch):
    """
    Patch nnUNetInferenceStep.execute to emit a caller-provided prediction
    (no model download / GPU required). Optionally records the configs the
    step was constructed with.
    """
    seen_configs = []

    def _install(prediction):
        def fake_execute(self, context):
            seen_configs.append(dict(self.config))
            context.intermediate_results["raw_prediction"] = prediction.copy()
            return context

        monkeypatch.setattr(
            "nnunetsegmentator.pipeline.steps.inference.nnUNetInferenceStep.execute",
            fake_execute,
        )

    _install.seen_configs = seen_configs
    return _install


class TestRegionCropStep:
    def test_crop_and_bookkeeping(self, patched_inference):
        # Arrays are (x, y, z): the crop model marks a lung-lobe band along z.
        input_array = np.zeros((10, 6, 10), dtype=np.float32)
        crop_prediction = np.zeros((10, 6, 10), dtype=np.uint8)
        crop_prediction[:, :, 3:7] = 10  # lung_upper_lobe_left

        patched_inference(crop_prediction)

        step = RegionCropStep("region_crop", {
            "crop_model_name": "total_fast",
            "crop_labels": [LUNG_LABEL],
            "crop_addon": [0, 0, 0],
        })
        context = step.execute(_context(input_array))

        assert context.input_array.shape == (10, 6, 4)
        region_crop = context.intermediate_results["region_crop"]
        assert region_crop["bbox"] == [[0, 10], [0, 6], [3, 7]]
        assert region_crop["full_shape"] == (10, 6, 10)
        assert np.array_equal(region_crop["crop_prediction"], crop_prediction)
        assert region_crop["crop_model_name"] == "total_fast"

    def test_addon_converts_mm_to_voxels(self, patched_inference):
        input_array = np.zeros((10, 6, 12), dtype=np.float32)
        crop_prediction = np.zeros((10, 6, 12), dtype=np.uint8)
        crop_prediction[:, :, 4:6] = 10  # lung band z=4..5

        patched_inference(crop_prediction)

        # spacing (1.0, 1.0, 2.0): addon 3 mm along z -> int(3/2) = 1 voxel
        step = RegionCropStep("region_crop", {
            "crop_model_name": "total_fast",
            "crop_labels": [LUNG_LABEL],
            "crop_addon": [0, 0, 3],
        })
        context = step.execute(_context(input_array, spacing=(1.0, 1.0, 2.0)))

        region_crop = context.intermediate_results["region_crop"]
        assert region_crop["bbox"] == [[0, 10], [0, 6], [3, 7]]
        assert context.input_array.shape == (10, 6, 4)

    def test_addon_is_truncated_not_rounded(self, patched_inference):
        input_array = np.zeros((10, 6, 12), dtype=np.float32)
        crop_prediction = np.zeros((10, 6, 12), dtype=np.uint8)
        crop_prediction[:, :, 4:6] = 10

        patched_inference(crop_prediction)

        # spacing 1.0: addon 2.9 mm -> int(2.9) = 2 voxels (truncation)
        step = RegionCropStep("region_crop", {
            "crop_model_name": "total_fast",
            "crop_labels": [LUNG_LABEL],
            "crop_addon": [2.9, 2.9, 2.9],
        })
        context = step.execute(_context(input_array))

        region_crop = context.intermediate_results["region_crop"]
        assert region_crop["bbox"] == [[0, 10], [0, 6], [2, 8]]

    def test_empty_mask_skips_crop(self, patched_inference):
        input_array = np.zeros((10, 6, 6), dtype=np.float32)
        patched_inference(np.zeros((10, 6, 6), dtype=np.uint8))

        step = RegionCropStep("region_crop", {
            "crop_model_name": "total_fast",
            "crop_labels": [LUNG_LABEL],
        })
        context = step.execute(_context(input_array))

        assert context.input_array.shape == (10, 6, 6)
        assert context.intermediate_results["region_crop"] is None

    def test_crop_inference_config(self, patched_inference):
        """The crop model runs with use_mirroring=False and model_name set."""
        input_array = np.zeros((10, 6, 6), dtype=np.float32)
        patched_inference(np.zeros((10, 6, 6), dtype=np.uint8))

        step = RegionCropStep("region_crop", {
            "crop_model_name": "total_fast",
            "crop_labels": [LUNG_LABEL],
        })
        step.execute(_context(input_array))

        assert len(patched_inference.seen_configs) == 1
        config = patched_inference.seen_configs[0]
        assert config["model_name"] == "total_fast"
        assert config["use_mirroring"] is False

    def test_missing_crop_model_raises(self):
        step = RegionCropStep("region_crop", {})
        with pytest.raises(ValueError):
            step.execute(_context(np.zeros((4, 4, 4))))

    def test_unknown_crop_label_raises(self, patched_inference):
        input_array = np.zeros((10, 6, 6), dtype=np.float32)
        patched_inference(np.zeros((10, 6, 6), dtype=np.uint8))

        step = RegionCropStep("region_crop", {
            "crop_model_name": "total_fast",
            "crop_labels": ["not_a_total_label"],
        })
        with pytest.raises(ValueError):
            step.execute(_context(input_array))


class TestRegionRestoreStep:
    def test_paste_into_full_grid(self):
        full = np.zeros((10, 6, 10), dtype=np.float32)
        crop_prediction = np.zeros((10, 6, 10), dtype=np.uint8)
        crop_prediction[:, :, 3:7] = 10

        step = RegionRestoreStep("region_restore", {})
        context = _context(full)
        context.intermediate_results["region_crop"] = {
            "bbox": [[0, 10], [0, 6], [3, 7]],
            "full_shape": (10, 6, 10),
            "crop_prediction": crop_prediction,
            "crop_model_name": "total_fast",
        }
        context.intermediate_results["raw_prediction"] = np.full((10, 6, 4), 5, dtype=np.uint8)

        context = step.execute(context)
        result = context.intermediate_results["raw_prediction"]

        assert result.shape == (10, 6, 10)
        assert np.all(result[:, :, 3:7] == 5)
        assert np.all(result[:, :, :3] == 0)
        assert np.all(result[:, :, 7:] == 0)

    def test_prefers_refined_prediction(self):
        step = RegionRestoreStep("region_restore", {})
        refined = np.full((2, 4, 4), 9, dtype=np.uint8)
        context = _context(np.zeros((4, 4, 4)))
        context.intermediate_results["raw_prediction"] = np.zeros((2, 4, 4), dtype=np.uint8)
        context.intermediate_results["refined_prediction"] = refined

        context = step.execute(context)
        assert np.array_equal(context.intermediate_results["raw_prediction"], refined)

    def test_without_crop_info_passthrough(self):
        step = RegionRestoreStep("region_restore", {})
        prediction = np.full((4, 4, 4), 2, dtype=np.uint8)
        context = _context(np.zeros((4, 4, 4)))
        context.intermediate_results["raw_prediction"] = prediction

        context = step.execute(context)
        assert np.array_equal(context.intermediate_results["raw_prediction"], prediction)

    def test_remove_outside_zeroes_outside_mask(self):
        """remove_outside zeros the prediction outside the crop-model mask."""
        crop_prediction = np.zeros((10, 6, 10), dtype=np.uint8)
        crop_prediction[:, :, 3:7] = 10  # mask label region

        # Prediction is already full-frame (e.g. no cropping was possible)
        prediction = np.zeros((10, 6, 10), dtype=np.uint8)
        prediction[:, :, 4:6] = 5

        step = RegionRestoreStep("region_restore", {
            "remove_outside": {"labels": [LUNG_LABEL], "dilation_mm": 0},
        })
        context = _context(np.zeros((10, 6, 10)))
        context.intermediate_results["region_crop"] = {
            "bbox": [[0, 10], [0, 6], [0, 10]],
            "full_shape": (10, 6, 10),
            "crop_prediction": crop_prediction,
            "crop_model_name": "total_fast",
        }
        context.intermediate_results["raw_prediction"] = prediction

        context = step.execute(context)
        result = context.intermediate_results["raw_prediction"]

        assert np.all(result[:, :, 4:6] == 5)
        assert np.all(result[:, :, :3] == 0)
        assert np.all(result[:, :, 7:] == 0)

    def test_remove_outside_dilates_mask(self):
        crop_prediction = np.zeros((20, 6, 20), dtype=np.uint8)
        crop_prediction[8:12, :, 8:12] = 10  # small core

        prediction = np.full((20, 6, 20), 5, dtype=np.uint8)

        step = RegionRestoreStep("region_restore", {
            "remove_outside": {"labels": [LUNG_LABEL], "dilation_mm": 2},
        })
        context = _context(np.zeros((20, 6, 20)))
        context.intermediate_results["region_crop"] = {
            "bbox": [[0, 20], [0, 6], [0, 20]],
            "full_shape": (20, 6, 20),
            "crop_prediction": crop_prediction,
            "crop_model_name": "total_fast",
        }
        context.intermediate_results["raw_prediction"] = prediction

        context = step.execute(context)
        result = context.intermediate_results["raw_prediction"]

        # Dilation by int(2/1)=2 grows the mask; a point 2 voxels from the
        # core is kept, one 5 voxels away is zeroed.
        assert result[8, 3, 8] == 5  # adjacent to the core (kept)
        assert result[3, 3, 3] == 0  # far away (zeroed)


class TestPerLabelPostprocessing:
    METADATA_LABELS = {1: "body_trunc", 2: "body_extremities"}

    def _context(self, seg):
        context = _context(np.zeros(seg.shape, dtype=np.float32))
        context.metadata = {
            "spacing": (1.0, 1.0, 1.0),
            "labels": dict(self.METADATA_LABELS),
        }
        context.intermediate_results["raw_prediction"] = seg
        return context

    def test_largest_component_per_label(self):
        seg = np.zeros((12, 4, 4), dtype=np.uint8)
        seg[0:2, :, :] = 1   # body_trunc, small blob (2 slices)
        seg[6:12, :, :] = 1  # body_trunc, large blob (6 slices)
        seg[3, 0, 0] = 2     # body_extremities, isolated single voxel

        step = LargestComponentStep("largest_component", {"labels": ["body_trunc"]})
        context = step.execute(self._context(seg))
        result = context.intermediate_results["final_prediction"]

        # Only the largest body_trunc blob survives; the small one is gone.
        assert np.all(result[6:12, :, :] == 1)
        assert np.all(result[0:2, :, :] == 0)
        # body_extremities is untouched (per-label operation).
        assert result[3, 0, 0] == 2

    def test_remove_small_objects_mm3(self):
        seg = np.zeros((12, 4, 4), dtype=np.uint8)
        seg[0:2, :, :] = 1   # body_trunc, 32 voxels (kept: not in labels list)
        seg[6, 0:2, 0:2] = 2    # body_extremities, 4 voxels (removed)
        seg[8:12, :, :] = 2  # body_extremities, 64 voxels (kept)

        step = RemoveSmallObjectsStep("remove_small_objects", {
            "labels": ["body_extremities"],
            "min_size_mm3": 50000,
        })
        context = step.execute(self._context(seg))
        result = context.intermediate_results["final_prediction"]

        # With min_size_mm3=50000 and 1mm voxels, both extremity blobs are
        # below the threshold and removed.
        assert np.all(result == 0) or np.all(result[0:2, :, :] == 1)
        assert np.all(result[6, 0:2, 0:2] == 0)
        assert np.all(result[8:12, :, :] == 0)
        # body_trunc is not in the labels list, so it is preserved as-is.
        assert np.all(result[0:2, :, :] == 1)

    def test_remove_small_objects_threshold_in_mm3(self):
        seg = np.zeros((12, 4, 4), dtype=np.uint8)
        seg[0:2, :, :] = 1   # 32 mm3 -> removed
        seg[6:12, :, :] = 1  # 96 mm3 -> kept

        step = RemoveSmallObjectsStep("remove_small_objects", {
            "labels": ["body_trunc"],
            "min_size_mm3": 50,
        })
        context = step.execute(self._context(seg))
        result = context.intermediate_results["final_prediction"]

        assert np.all(result[0:2, :, :] == 0)
        assert np.all(result[6:12, :, :] == 1)

    def test_unknown_label_name_raises(self):
        seg = np.zeros((4, 4, 4), dtype=np.uint8)
        step = LargestComponentStep("largest_component", {"labels": ["not_a_label"]})
        with pytest.raises(ValueError):
            step.execute(self._context(seg))


class TestStepRegistry:
    def test_region_steps_registered(self):
        assert PipelineBuilder.STEP_REGISTRY["region_crop"] is RegionCropStep
        assert PipelineBuilder.STEP_REGISTRY["region_restore"] is RegionRestoreStep
