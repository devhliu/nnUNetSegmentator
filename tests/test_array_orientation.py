import numpy as np

from nnunetsegmentator import image as sitk
from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.steps.quantification import (
    BodyCompositionMetricsStep,
)
from nnunetsegmentator.utils.statistics import SegmentationStatistics


def _coord_array(shape: tuple[int, int, int]) -> np.ndarray:
    x = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    z = np.arange(shape[2])[None, None, :]
    return (100 * x + 10 * y + z).astype(np.float32)


def test_nifti_round_trip_preserves_xyz_array_order() -> None:
    array_xyz = _coord_array((3, 4, 5))
    image = sitk.GetImageFromArray(array_xyz)
    image.SetSpacing((1.1, 2.2, 3.3))
    image.SetOrigin((4.0, 5.0, 6.0))
    image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))

    round_trip = sitk.Image.from_nifti(image.to_nifti())

    round_trip_array = sitk.GetArrayFromImage(round_trip)
    np.testing.assert_array_equal(round_trip_array, array_xyz)
    assert round_trip.GetSpacing() == image.GetSpacing()
    assert round_trip.GetOrigin() == image.GetOrigin()
    assert round_trip.GetDirection() == image.GetDirection()


def test_body_composition_uses_axis2_as_z(tmp_path) -> None:
    spacing = (2.0, 3.0, 5.0)
    image_array = np.full((4, 4, 5), 100.0, dtype=np.float32)
    seg_array = np.zeros_like(image_array, dtype=np.uint8)

    # Source label 3 maps to logical "muscle".
    seg_array[1, 1, 3] = 3
    seg_array[2, 2, 3] = 3
    seg_array[1, 1, 1] = 3
    seg_array[1, 1, 2] = 3
    seg_array[1, 1, 4] = 3

    l3_mask = np.zeros_like(seg_array, dtype=np.uint8)
    l3_mask[0, 0, 3] = 1
    l3_mask[1, 1, 3] = 1

    t12_mask = np.zeros_like(seg_array, dtype=np.uint8)
    t12_mask[0, 0, 1] = 1
    l4_mask = np.zeros_like(seg_array, dtype=np.uint8)
    l4_mask[0, 0, 4] = 1

    image = sitk.GetImageFromArray(image_array)
    image.SetSpacing(spacing)
    context = PipelineContext(
        input_image=image,
        input_array=image_array,
        metadata={
            "original_input_array": image_array,
            "original_spacing": spacing,
            "vertebrae_masks": {
                "vertebrae_L3": l3_mask,
                "vertebrae_T12": t12_mask,
                "vertebrae_L4": l4_mask,
            },
            "output_dir": str(tmp_path),
            "case_id": "case",
        },
    )
    context.intermediate_results["final_prediction"] = seg_array

    result = BodyCompositionMetricsStep("body_comp", {}).execute(context)
    metrics = result.metadata["metrics"]

    assert metrics["l3_slice_index"] == 3
    assert metrics["muscle_area_mm2"] == 2 * spacing[0] * spacing[1]
    assert metrics["muscle_volume_mm3"] == (
        5 * spacing[0] * spacing[1] * spacing[2]
    )
    assert metrics["scan_length_mm"] == 4 * spacing[2]


def test_shape_statistics_respect_xyz_spacing() -> None:
    seg_array = np.zeros((5, 6, 7), dtype=np.uint8)
    seg_array[1:3, 2:5, 3:6] = 1

    seg_image = sitk.GetImageFromArray(seg_array)
    seg_image.SetSpacing((2.0, 3.0, 4.0))

    stats_calc = SegmentationStatistics(segmentation=seg_image)
    stats = stats_calc.compute_shape_statistics(1)

    assert stats["centroid_x"] == 1.5 * 2.0
    assert stats["centroid_y"] == 3.0 * 3.0
    assert stats["centroid_z"] == 4.0 * 4.0
    assert stats["bbox_x"] == (2 - 1) * 2.0
    assert stats["bbox_y"] == (4 - 2) * 3.0
    assert stats["bbox_z"] == (5 - 3) * 4.0
    assert stats["volume_mm3"] == 18 * 2.0 * 3.0 * 4.0
