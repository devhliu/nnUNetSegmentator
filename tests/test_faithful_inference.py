"""Tests for faithful nnUNet inference (spacing resampling + grid restore)."""

from types import SimpleNamespace

import numpy as np

from nnunetsegmentator import image as sitk
from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.steps.inference import nnUNetInferenceStep


def _make_step(config=None):
    return nnUNetInferenceStep("inference", config or {})


def _context(array, spacing_xyz=(1.0, 1.0, 1.0)):
    img = sitk.GetImageFromArray(array)
    img.SetSpacing(tuple(float(v) for v in spacing_xyz))
    return PipelineContext(
        input_image=img,
        input_array=array,
        metadata={
            "spacing": tuple(float(v) for v in spacing_xyz),
            "current_spacing": tuple(float(v) for v in spacing_xyz),
        },
    )


def _fake_predictor(spacing=None, transpose_forward=(0, 1, 2)):
    """Minimal predictor stub; omit spacing to emulate mocked predictors."""
    predictor = SimpleNamespace()
    if spacing is not None:
        predictor.plans_manager = SimpleNamespace(
            transpose_forward=list(transpose_forward))
        predictor.configuration_manager = SimpleNamespace(spacing=list(spacing))
    return predictor


class TestModelSpacing:
    def test_inverse_transpose_permutation(self):
        # plans spacing is stored in transposed axis order; the raw
        # SimpleITK (z, y, x) order is recovered via transpose_forward.
        step = _make_step()
        step._predictor = _fake_predictor([2.0, 1.0, 0.5], [2, 0, 1])
        assert step._get_model_spacing_zyx((1.0, 1.0, 3.0)) == (1.0, 0.5, 2.0)

    def test_identity_transpose(self):
        step = _make_step()
        step._predictor = _fake_predictor([0.5, 0.5, 1.0], [0, 1, 2])
        assert step._get_model_spacing_zyx((1.0, 1.0, 3.0)) == (0.5, 0.5, 1.0)

    def test_2d_configuration_keeps_slice_spacing(self):
        # 2D configs only define in-plane spacing; the slice axis keeps the
        # source spacing (nnUNetv2 target_spacing convention).
        step = _make_step()
        step._predictor = _fake_predictor([1.5, 1.5], [2, 0, 1])
        assert step._get_model_spacing_zyx((1.0, 1.0, 3.0)) == (1.5, 1.5, 3.0)

    def test_mocked_predictor_returns_none(self):
        step = _make_step()
        step._predictor = _fake_predictor()
        assert step._get_model_spacing_zyx((1.0, 1.0, 1.0)) is None


class TestResampleArray:
    def test_identity_shape_preserves_values(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        step = _make_step()
        out = step._resample_array(data, (2, 3, 4), order=3)
        assert out.shape == (2, 3, 4)
        assert np.allclose(out, data, atol=1e-4)

    def test_nearest_downsample_picks_anchor_voxels(self):
        # sitk identity-transform convention: output voxel i samples input
        # index i * input_size / output_size.
        data = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
        step = _make_step()
        out = step._resample_array(data, (2, 2, 2), order=0)
        assert np.array_equal(out, data[::2, ::2, ::2])

    def test_4d_channels_resampled_independently(self):
        planes = [np.arange(64, dtype=np.float32).reshape(4, 4, 4),
                  np.arange(64, dtype=np.float32).reshape(4, 4, 4) * 2.0]
        data = np.stack(planes)
        step = _make_step()
        out = step._resample_array(data, (2, 2, 2), order=0)
        assert out.shape == (2, 2, 2, 2)
        assert np.array_equal(out[0], planes[0][::2, ::2, ::2])
        assert np.array_equal(out[1], planes[1][::2, ::2, ::2])


class TestFaithfulInference:
    def test_transposes_resamples_and_restores_grid(self):
        # Source: (x, y, z) = (4, 6, 8) at spacing (1.0, 1.0, 2.0).
        # Model spacing zyx (0.5, 1.0, 1.0) -> z extent 8 * 2 / 0.5 = 32.
        spacing_xyz = (1.0, 1.0, 2.0)
        input_array = np.zeros((4, 6, 8), dtype=np.float32)
        input_array[:, :, 2:6] = 10.0

        step = _make_step()
        step._predictor = _fake_predictor([0.5, 1.0, 1.0], [0, 1, 2])

        calls = {}

        def fake_api(array, spacing):
            calls["array"] = array
            calls["spacing"] = spacing
            return np.full((32, 6, 4), 7, dtype=np.uint8)

        step._run_api_inference = fake_api

        result = step._run_faithful_inference(_context(input_array, spacing_xyz),
                                              input_array)

        assert calls["array"].shape == (32, 6, 4)
        assert calls["spacing"] == (0.5, 1.0, 1.0)
        assert result.shape == (4, 6, 8)
        assert np.all(result == 7)

    def test_no_resample_when_spacing_matches(self):
        input_array = np.zeros((4, 6, 8), dtype=np.float32)
        step = _make_step()
        step._predictor = _fake_predictor([1.0, 1.0, 1.0], [0, 1, 2])

        calls = {}

        def fake_api(array, spacing):
            calls["array"] = array
            calls["spacing"] = spacing
            return np.full((8, 6, 4), 3, dtype=np.uint8)

        step._run_api_inference = fake_api

        result = step._run_faithful_inference(_context(input_array), input_array)

        assert calls["array"].shape == (8, 6, 4)  # plain (x,y,z) -> (z,y,x)
        assert calls["spacing"] == (1.0, 1.0, 1.0)
        assert result.shape == (4, 6, 8)
        assert np.all(result == 3)

    def test_degrades_without_configuration_manager(self):
        # Mocked predictors (no plans/configuration) fall back to the legacy
        # path: raw (x, y, z) array with the metadata spacing.
        input_array = np.zeros((4, 6, 8), dtype=np.float32)
        step = _make_step()
        step._predictor = _fake_predictor()

        calls = {}

        def fake_api(array, spacing):
            calls["array"] = array
            calls["spacing"] = spacing
            return np.full((4, 6, 8), 5, dtype=np.uint8)

        step._run_api_inference = fake_api

        result = step._run_faithful_inference(
            _context(input_array, (1.0, 1.0, 2.0)), input_array)

        assert calls["array"] is input_array
        assert calls["spacing"] == (1.0, 1.0, 2.0)
        assert np.all(result == 5)

    def test_faithful_resampling_disabled_uses_legacy_path(self):
        input_array = np.zeros((4, 6, 8), dtype=np.float32)
        step = _make_step({"faithful_resampling": False})
        step._predictor = _fake_predictor([0.5, 1.0, 1.0], [0, 1, 2])

        calls = {}

        def fake_api(array, spacing):
            calls["array"] = array
            calls["spacing"] = spacing
            return np.full((4, 6, 8), 4, dtype=np.uint8)

        step._run_api_inference = fake_api

        result = step._run_faithful_inference(
            _context(input_array, (1.0, 1.0, 2.0)), input_array)

        assert calls["array"] is input_array
        assert calls["spacing"] == (1.0, 1.0, 2.0)
        assert np.all(result == 4)

    def test_grid_sync_fallback_uses_array_shape(self):
        # Z-cropped array with stale image geometry: the grid is derived
        # from the array shape plus the metadata spacing.
        full_array = np.zeros((4, 6, 8), dtype=np.float32)
        cropped_array = full_array[:, :, 3:]  # (4, 6, 5)
        context = _context(full_array, (1.0, 1.0, 2.0))
        context.input_array = cropped_array

        step = _make_step()
        step._predictor = _fake_predictor([2.0, 1.0, 1.0], [0, 1, 2])

        calls = {}

        def fake_api(array, spacing):
            calls["array"] = array
            calls["spacing"] = spacing
            return np.full((5, 6, 4), 9, dtype=np.uint8)

        step._run_api_inference = fake_api

        result = step._run_faithful_inference(context, cropped_array)

        assert calls["array"].shape == (5, 6, 4)
        assert calls["spacing"] == (2.0, 1.0, 1.0)
        assert result.shape == (4, 6, 5)
        assert np.all(result == 9)

    def test_probability_tuple_restored(self):
        input_array = np.zeros((4, 6, 8), dtype=np.float32)
        step = _make_step()
        step._predictor = _fake_predictor([1.0, 1.0, 1.0], [0, 1, 2])

        def fake_api(array, spacing):
            segmentation = np.full((8, 6, 4), 2, dtype=np.uint8)
            probabilities = np.zeros((2, 8, 6, 4), dtype=np.float32)
            probabilities[1] = 1.0
            return segmentation, probabilities

        step._run_api_inference = fake_api

        result = step._run_faithful_inference(_context(input_array), input_array)

        segmentation, probabilities = result
        assert segmentation.shape == (4, 6, 8)
        assert np.all(segmentation == 2)
        assert probabilities.shape == (2, 4, 6, 8)
        assert np.all(probabilities[1] == 1.0)
