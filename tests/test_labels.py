"""
Tests for the nnunetsegmentator.labels sub-module.

Covers label table integrity: every registered task's output labels match its
labels-package table; per-table counts and uniqueness.
"""

import pytest

import nnunetsegmentator.tasks  # noqa: F401  (registers all tasks on import)
from nnunetsegmentator.core.registry import TaskRegistry
from nnunetsegmentator.labels import (
    ABDOMINAL_MUSCLES_LABELS,
    BODY_LABELS,
    BRAIN_ANEURYSM_LABELS,
    BREASTS_LABELS,
    CEREBRAL_BLEED_LABELS,
    CRANIOFACIAL_STRUCTURES_LABELS,
    HEART_CHAMBERS_LABELS,
    KIDNEY_CYSTS_LABELS,
    LIVER_LESIONS_LABELS,
    LIVER_SEGMENTS_LABELS,
    LIVER_VESSELS_LABELS,
    LUNG_LOBES,
    LUNG_NODULES_LABELS,
    LUNG_VESSELS_LABELS,
    MR_LABELS,
    MR_MUSCLES_LABELS,
    MR_ORGANS_LABELS,
    PART_CARDIAC_LABELS,
    PART_MUSCLES_LABELS,
    PART_ORGANS_LABELS,
    PART_RIBS_LABELS,
    PART_VERTEBRAE_LABELS,
    PLEURAL_PERICARD_EFFUSION_LABELS,
    TASK_LABEL_TABLES,
    TEETH_LABELS,
    TISSUE_TYPES_LABELS_3,
    TISSUE_TYPES_MR_LABELS,
    TOTAL_LABELS,
    TRUNK_CAVITIES_LABELS,
    VERTEBRAE_BODY_LABELS,
    VERTEBRAE_MR_LABELS,
    VERTEBRAE_PP_LABELS,
    VENTRICLE_PARTS_LABELS,
    get_labels,
)


class TestLabelTables:
    """Label tables are complete, unique and wired into every task."""

    def test_table_counts(self):
        assert len(TOTAL_LABELS) == 117
        assert len(PART_ORGANS_LABELS) == 24
        assert len(PART_VERTEBRAE_LABELS) == 26
        assert len(PART_CARDIAC_LABELS) == 18
        assert len(PART_MUSCLES_LABELS) == 23
        assert len(PART_RIBS_LABELS) == 26
        assert len(MR_ORGANS_LABELS) == 29
        assert len(MR_MUSCLES_LABELS) == 21
        assert len(MR_LABELS) == 50
        assert len(TEETH_LABELS) == 77
        assert len(VERTEBRAE_MR_LABELS) == 25
        assert len(VERTEBRAE_PP_LABELS) == 24
        assert len(ABDOMINAL_MUSCLES_LABELS) == 22
        assert len(VENTRICLE_PARTS_LABELS) == 12
        assert len(LUNG_VESSELS_LABELS) == 4
        assert len(BODY_LABELS) == 2
        assert len(TISSUE_TYPES_MR_LABELS) == 3
        assert len(HEART_CHAMBERS_LABELS) == 7
        assert len(CEREBRAL_BLEED_LABELS) == 1
        assert len(LIVER_VESSELS_LABELS) == 2
        assert len(LUNG_NODULES_LABELS) == 2
        assert len(VERTEBRAE_BODY_LABELS) == 2
        assert len(LIVER_LESIONS_LABELS) == 1
        assert len(KIDNEY_CYSTS_LABELS) == 2
        assert len(PLEURAL_PERICARD_EFFUSION_LABELS) == 3
        assert len(LIVER_SEGMENTS_LABELS) == 8
        assert len(TRUNK_CAVITIES_LABELS) == 4
        assert len(BRAIN_ANEURYSM_LABELS) == 1
        assert len(BREASTS_LABELS) == 1
        assert len(CRANIOFACIAL_STRUCTURES_LABELS) == 7

    def test_table_names_unique(self):
        for name, table in TASK_LABEL_TABLES.items():
            values = list(table.values())
            assert len(values) == len(set(values)), name

    def test_mr_labels_construction(self):
        """MR_LABELS = organs 1-29 as-is + muscles remapped to 30-50."""
        assert MR_LABELS == {**MR_ORGANS_LABELS,
                             **{i + 29: name for i, name in MR_MUSCLES_LABELS.items()}}
        assert MR_LABELS[29] == "iliac_vena_right"
        assert MR_LABELS[30] == "humerus_left"
        assert MR_LABELS[50] == "brain"

    def test_tissue_types_mr_shares_3_class_table(self):
        assert TISSUE_TYPES_MR_LABELS == TISSUE_TYPES_LABELS_3

    def test_get_labels_returns_task_output_labels(self):
        """Every registered table equals the task definition's output labels."""
        for task_name, table in TASK_LABEL_TABLES.items():
            definition = TaskRegistry.get_task(task_name)
            assert definition.output_config["labels"] == table, task_name

    def test_get_labels_covers_all_totalsegmentator_tasks(self):
        expected = {
            "total", "total_mr", "lung_vessels", "body", "body_mr",
            "tissue_types", "tissue_types_mr", "heartchambers_highres",
            "cerebral_bleed", "liver_vessels", "lung_nodules", "vertebrae_body",
            "liver_lesions", "kidney_cysts", "pleural_pericard_effusion",
            "vertebrae_mr", "breasts", "ventricle_parts", "liver_segments",
            "liver_segments_mr", "trunk_cavities", "brain_aneurysm",
            "vertebrae_pp", "abdominal_muscles", "craniofacial_structures",
            "teeth",
            # BOA body_composition_analysis
            "boa_body_parts", "boa_body_regions",
        }
        assert expected == set(TASK_LABEL_TABLES)

    def test_get_labels_unknown_task_raises_with_available_list(self):
        with pytest.raises(KeyError, match="Available:"):
            get_labels("no_such_task")

    def test_get_labels_returns_expected_tables(self):
        assert get_labels("total") is TOTAL_LABELS
        assert get_labels("total_mr") is MR_LABELS
        assert get_labels("body") is BODY_LABELS
        assert get_labels("body_mr") is BODY_LABELS
        assert get_labels("liver_segments_mr") is LIVER_SEGMENTS_LABELS

    def test_crop_label_sets(self):
        assert LUNG_LOBES == [
            "lung_upper_lobe_left", "lung_lower_lobe_left",
            "lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right",
        ]
        # Crop labels must exist in the total label vocabulary.
        assert set(LUNG_LOBES) <= set(TOTAL_LABELS.values())
