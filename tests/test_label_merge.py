"""Tests for cross-task label merging driven by the SNOMED ``merge_group`` column."""

from types import SimpleNamespace

import numpy as np

from nnunetsegmentator import image as sitk
from nnunetsegmentator.core.orchestrator import SegmentationOrchestrator
from nnunetsegmentator.mapping import LabelMapper
from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.builders import PipelineBuilder
from nnunetsegmentator.pipeline.steps import MergeLabelsStep

LUNG_LABELS = {
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
}

EXPECTED_MERGE_MAP = {
    "lung_upper_lobe_left": "Left Lung",
    "lung_lower_lobe_left": "Left Lung",
    "lung_upper_lobe_right": "Right Lung",
    "lung_middle_lobe_right": "Right Lung",
    "lung_lower_lobe_right": "Right Lung",
}


def _make_context(array, labels):
    context = PipelineContext(input_image=None, input_array=None, metadata={"labels": labels})
    context.intermediate_results["final_prediction"] = array
    return context


class TestMergeGroupMapping:
    def test_lung_lobes_group_by_side(self):
        mapper = LabelMapper.get_instance()
        assert mapper.get_merge_group("lung_upper_lobe_left") == "Left Lung"
        assert mapper.get_merge_group("lung_lower_lobe_left") == "Left Lung"
        assert mapper.get_merge_group("lung_middle_lobe_right") == "Right Lung"
        assert mapper.get_merge_group("lung_lower_lobe_right") == "Right Lung"

    def test_labels_without_group_are_unmerged(self):
        mapper = LabelMapper.get_instance()
        assert mapper.get_merge_group("liver") is None
        assert mapper.get_merge_group("kidney_left") is None
        assert mapper.get_merge_group("not_a_registered_label") is None

    def test_merged_concepts_keep_snomed_coding(self):
        mapper = LabelMapper.get_instance()
        left = mapper.resolve("Left Lung")
        right = mapper.resolve("Right Lung")

        assert (left.type.code, left.type.meaning) == ("39607008", "Lung")
        assert (left.type_modifier.code, left.type_modifier.meaning) == ("7771000", "Left")
        assert (right.type_modifier.code, right.type_modifier.meaning) == ("24028007", "Right")
        assert left.laterality == "left"
        assert right.laterality == "right"
        assert left.rgb and right.rgb


class TestMergeLabelsStep:
    def test_merges_lobes_into_side_labels(self):
        array = np.zeros((4, 4, 4), dtype=np.uint8)
        array[0] = 10
        array[1] = 11
        array[2] = 13
        array[3] = 5
        context = _make_context(array, {**LUNG_LABELS, 5: "liver"})

        MergeLabelsStep("merge_labels", {}).execute(context)
        merged = context.intermediate_results["final_prediction"]

        # Left lobes keep the lowest member id (10), right lobes id 13 -> 12.
        assert np.all(merged[0] == 10)
        assert np.all(merged[1] == 10)
        assert np.all(merged[2] == 12)
        assert not np.any(merged == 11)
        assert not np.any(merged == 13)
        assert np.all(merged[3] == 5)

        labels = context.metadata["labels"]
        assert labels[10] == "Left Lung"
        assert labels[12] == "Right Lung"
        assert 11 not in labels and 13 not in labels and 14 not in labels
        assert labels[5] == "liver"

        assert context.intermediate_results["label_merge_map"] == EXPECTED_MERGE_MAP

    def test_single_present_member_is_left_alone(self):
        array = np.zeros((2, 2, 2), dtype=np.uint8)
        array[0] = 11
        context = _make_context(array, {11: "lung_lower_lobe_left"})

        MergeLabelsStep("merge_labels", {}).execute(context)

        assert "label_merge_map" not in context.intermediate_results
        assert context.metadata["labels"] == {11: "lung_lower_lobe_left"}
        assert np.all(context.intermediate_results["final_prediction"][0] == 11)

    def test_missing_label_map_is_noop(self):
        context = PipelineContext(input_image=None, input_array=None, metadata={})
        context.intermediate_results["final_prediction"] = np.ones((2, 2, 2), dtype=np.uint8)

        MergeLabelsStep("merge_labels", {}).execute(context)

        assert "label_merge_map" not in context.intermediate_results
        assert context.metadata == {}

    def test_step_is_registered(self):
        assert PipelineBuilder.STEP_REGISTRY["merge_labels"] is MergeLabelsStep


class TestOrchestratorSplitMerging:
    @staticmethod
    def _orchestrator():
        # _split_labels only touches helpers, so skip full initialisation.
        return SegmentationOrchestrator.__new__(SegmentationOrchestrator)

    def test_split_applies_merge_map(self):
        array = np.zeros((2, 2, 2), dtype=np.uint8)
        array[0] = 10
        array[1] = 5
        task = SimpleNamespace(models={
            "m": SimpleNamespace(labels={
                "lung_upper_lobe_left": 10,
                "lung_lower_lobe_left": 11,
                "liver": 5,
            })
        })

        labels = self._orchestrator()._split_labels(
            sitk.GetImageFromArray(array),
            task,
            "m",
            merge_map=EXPECTED_MERGE_MAP,
        )

        assert set(labels) == {"Left Lung", "liver"}
        assert np.all(sitk.GetArrayFromImage(labels["Left Lung"])[0] == 1)
        assert np.all(sitk.GetArrayFromImage(labels["liver"])[1] == 1)

    def test_split_accepts_id_to_name_tables(self):
        array = np.zeros((2, 2, 2), dtype=np.uint8)
        array[0] = 10
        array[1] = 5
        task = SimpleNamespace(models={
            "m": SimpleNamespace(labels={10: "lung_upper_lobe_left", 5: "liver"})
        })

        labels = self._orchestrator()._split_labels(sitk.GetImageFromArray(array), task, "m")

        assert set(labels) == {"lung_upper_lobe_left", "liver"}
        assert np.all(sitk.GetArrayFromImage(labels["lung_upper_lobe_left"])[0] == 1)


class TestOrganMappingMerging:
    def test_merged_label_reported_with_lung_coding(self):
        orchestrator = SegmentationOrchestrator.__new__(SegmentationOrchestrator)
        orchestrator.task = SimpleNamespace(
            name="total",
            models={"m": SimpleNamespace(task_id="Dataset1_Test")},
        )

        payload = orchestrator._build_organ_mapping("m", {5: "liver", 10: "Left Lung"})

        left_lung = payload["organ_indices"]["10"]
        assert left_lung["name"] == "Left Lung"
        assert left_lung["canonical"] == "Left Lung"
        assert left_lung["SNOMED"]["CodeValue"] == "39607008"
        assert left_lung["laterality"] == "left"

        assert payload["organ_indices"]["5"]["canonical"] == "liver"