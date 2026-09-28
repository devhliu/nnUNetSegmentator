"""Tests for the MOOSE task definition and per-model orchestrator behaviour"""

import json
import re

import numpy as np
import pytest

from nnunetsegmentator.core.registry import TaskRegistry
from nnunetsegmentator.core.orchestrator import SegmentationOrchestrator
from nnunetsegmentator.core.config import Config


@pytest.fixture(scope="module", autouse=True)
def registered_tasks():
    """Register all built-in tasks once for the module."""
    import nnunetsegmentator.tasks as _tasks  # noqa: F401
    yield


@pytest.fixture
def fake_model_dir(tmp_path):
    """A minimal model payload directory (avoids triggering downloads)."""
    model_dir = tmp_path / "Dataset123_Organs"
    model_dir.mkdir()
    (model_dir / "dataset.json").write_text(json.dumps({
        "labels": {"background": 0, "liver": 1, "spleen": 2, "kidney_left": 3},
    }))
    return model_dir


@pytest.fixture
def moose_orchestrator(fake_model_dir):
    return SegmentationOrchestrator(
        task_name="moose",
        config=Config(),
        model_path=fake_model_dir,
    )


class TestTaskRegistration:
    def test_25_models_registered(self):
        task = TaskRegistry.get_task("moose")
        assert len(task.models) == 25

    def test_task_ids_canonical(self):
        task = TaskRegistry.get_task("moose")
        pattern = re.compile(r"^Dataset\d+_[A-Za-z0-9_-]+$")
        for model_info in task.models.values():
            assert pattern.match(model_info.task_id), model_info.task_id

    def test_static_labels_seeded_from_csv(self):
        task = TaskRegistry.get_task("moose")
        organs = task.models["clin_ct_organs"].labels
        assert "liver" in organs
        assert organs["liver"] >= 1

    def test_models_without_csv_labels_start_empty(self):
        task = TaskRegistry.get_task("moose")
        # Models absent from the CSV rely on runtime enrichment
        assert isinstance(task.models["clin_ct_lungs"].labels, dict)

    def test_output_config(self):
        task = TaskRegistry.get_task("moose")
        output_config = task.output_config
        assert output_config["strategy"] == "per_model"
        assert output_config["emit_label_mapping"] is True
        assert output_config["runtime_labels"] is True

    def test_cascade_workflow_declared(self):
        task = TaskRegistry.get_task("moose")
        model_steps = task.pipeline_config["model_steps"]
        assert set(model_steps.keys()) == {"clin_ct_body_composition"}
        step_types = [s["type"] for s in model_steps["clin_ct_body_composition"]]
        assert step_types == ["preserve_original", "crop_fov", "nnunet_inference", "fov_restrict"]

    def test_modalities(self):
        task = TaskRegistry.get_task("moose")
        assert task.models["clin_ct_organs"].modality == "CT"
        assert task.models["clin_pt_fdg_brain_v1"].modality == "PT"
        assert task.models["clin_mr_FVM"].modality == "MR"


class TestOrchestratorModelSelection:
    def test_default_primary_model(self, moose_orchestrator):
        assert moose_orchestrator.selected_models == ["clin_ct_body"]

    def test_explicit_model_subset(self, fake_model_dir):
        orchestrator = SegmentationOrchestrator(
            task_name="moose",
            config=Config(),
            model_path=fake_model_dir,
            models=["clin_ct_body_composition", "clin_ct_organs"],
        )
        assert orchestrator.selected_models == ["clin_ct_body_composition", "clin_ct_organs"]

    def test_cascade_pipeline_for_body_composition(self, fake_model_dir):
        orchestrator = SegmentationOrchestrator(
            task_name="moose",
            config=Config(),
            model_path=fake_model_dir,
            models=["clin_ct_body_composition"],
        )
        step_types = [type(s).__name__ for s in orchestrator.pipeline.steps]
        assert step_types == [
            "PreserveOriginalStep",
            "CropFOVStep",
            "nnUNetInferenceStep",
            "FOVRestrictStep",
            "MergeLabelsStep",
        ]

    def test_unknown_model_rejected(self, fake_model_dir):
        from nnunetsegmentator.core.exceptions import PipelineConfigurationError
        with pytest.raises(PipelineConfigurationError):
            SegmentationOrchestrator(
                task_name="moose",
                config=Config(),
                model_path=fake_model_dir,
                models=["nope"],
            )


class TestRuntimeLabels:
    def test_ensure_runtime_labels_from_dataset_json(self, fake_model_dir):
        TaskRegistry.register_model_path("clin_ct_organs", fake_model_dir)
        from nnunetsegmentator.tasks.moose import ensure_runtime_labels

        labels = ensure_runtime_labels("clin_ct_organs")
        assert labels == {"liver": 1, "spleen": 2, "kidney_left": 3}
        # Enrichment is written back into the registered ModelInfo
        assert TaskRegistry.get_task("moose").models["clin_ct_organs"].labels == labels

    def test_missing_payload_returns_static_labels(self, monkeypatch):
        from nnunetsegmentator.core.exceptions import ModelNotFoundError
        from nnunetsegmentator.tasks.moose import ensure_runtime_labels

        def _raise(cls, model_name):
            raise ModelNotFoundError(model_name, task_name="moose")

        monkeypatch.setattr(
            "nnunetsegmentator.core.registry.TaskRegistry.get_model_path",
            classmethod(_raise),
        )
        labels = ensure_runtime_labels("clin_ct_lungs")
        assert isinstance(labels, dict)


class TestPerModelHelpers:
    def test_split_labels_uses_selected_model(self, moose_orchestrator, fake_model_dir):
        from nnunetsegmentator import image as sitk

        task = moose_orchestrator.task
        TaskRegistry.register_model_path("clin_ct_organs", fake_model_dir)
        from nnunetsegmentator.tasks.moose import ensure_runtime_labels
        ensure_runtime_labels("clin_ct_organs")

        seg_array = np.zeros((4, 4, 4), dtype=np.uint8)
        seg_array[1, 1, 1] = 1  # liver
        seg_array[2, 2, 2] = 3  # kidney_left
        segmentation = sitk.GetImageFromArray(seg_array)

        labels = moose_orchestrator._split_labels(segmentation, task, "clin_ct_organs")
        assert set(labels.keys()) == {"liver", "spleen", "kidney_left"}
        assert sitk.GetArrayFromImage(labels["liver"])[1, 1, 1] == 1
        assert sitk.GetArrayFromImage(labels["kidney_left"])[2, 2, 2] == 1

    def test_build_organ_mapping_payload(self, moose_orchestrator, fake_model_dir):
        TaskRegistry.register_model_path("clin_ct_organs", fake_model_dir)
        from nnunetsegmentator.tasks.moose import ensure_runtime_labels
        ensure_runtime_labels("clin_ct_organs")

        payload = moose_orchestrator._build_organ_mapping("clin_ct_organs")
        assert payload["model"] == "clin_ct_organs"
        assert payload["task"] == "moose"
        assert payload["task_id"] == "Dataset123_Organs"
        liver = payload["organ_indices"]["1"]
        assert liver["name"] == "liver"
        assert liver["canonical"] == "liver"
        assert liver["SNOMED"]["CodeValue"] == "10200004"
        assert liver["SNOMED"]["CodingSchemeDesignator"] == "SCT"

    def test_is_per_model(self, moose_orchestrator):
        assert moose_orchestrator._is_per_model() is True

    def test_strip_nifti_suffix(self):
        assert SegmentationOrchestrator._strip_nifti_suffix("case.nii.gz") == "case"
        assert SegmentationOrchestrator._strip_nifti_suffix("case__m_seg.nii.gz") == "case__m_seg"
        assert SegmentationOrchestrator._strip_nifti_suffix("case") == "case"

    def test_sidecar_naming(self, moose_orchestrator, tmp_path):
        import pathlib

        organ_mapping = {"model": "m", "organ_indices": {}}
        sidecar = moose_orchestrator._write_organ_mapping_sidecar(
            tmp_path / "case__clin_ct_organs_seg.nii.gz", organ_mapping
        )
        assert sidecar.name == "case__clin_ct_organs_organ_mapping.json"
        assert json.loads(pathlib.Path(sidecar).read_text()) == organ_mapping
