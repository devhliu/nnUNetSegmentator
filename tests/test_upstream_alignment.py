"""
Tests for the alignment with the upstream weight sources and preprocessing.

Covers the shared registry surface (registration integrity, weight-URL hygiene)
and the newly wired BOA body-composition tasks (task ids, labels, resampling,
fold ensemble, ported postprocessing).
"""

from pathlib import Path

import numpy as np
import pytest

import nnunetsegmentator.tasks  # noqa: F401  (registers all tasks on import)
from nnunetsegmentator import image as sitk
from nnunetsegmentator.core.registry import TaskRegistry
from nnunetsegmentator.labels import BODY_PARTS_LABELS, BODY_REGION_LABELS
from nnunetsegmentator.pipeline.base import PipelineContext
from nnunetsegmentator.pipeline.steps.postprocessing import (
    BoaBodyPartsPostprocessStep,
    BoaBodyRegionsPostprocessStep,
)
from nnunetsegmentator.tasks import boa as boa_tasks
from nnunetsegmentator.utils.model_layout import (
    is_canonical_task_id,
    is_downloadable_url,
)

PACKAGE_ROOT = Path(nnunetsegmentator.tasks.__file__).resolve().parents[1]

# Weight hosts the repository may legitimately reference.
ALLOWED_URL_PREFIXES = (
    "https://github.com/",
    "https://zenodo.org/records/",
    "https://model.s.mdforge.com/",
)

# Detector module legitimately mentions unusable URL fragments in code.
URL_FRAGMENT_DETECTOR = PACKAGE_ROOT / "utils" / "model_layout.py"


def _iter_all_models():
    for task_name in TaskRegistry.list_tasks():
        definition = TaskRegistry.get_task(task_name)
        for model in definition.models.values():
            yield task_name, model


def _step_types(pipeline):
    return [type(step).__name__ for step in pipeline.steps]


def _get_step(pipeline, cls_name):
    for step in pipeline.steps:
        if type(step).__name__ == cls_name:
            return step
    raise AssertionError(f"step {cls_name} not found in {_step_types(pipeline)}")


class TestRegistrationIntegrity:
    """Every built-in task registers with a canonical, unique task id."""

    def test_all_tasks_register(self):
        from nnunetsegmentator.tasks import register_all_tasks

        task_classes = register_all_tasks()
        assert len(task_classes) >= 37

    def test_task_ids_are_canonical_and_unique(self):
        seen = set()
        for name, model in _iter_all_models():
            assert is_canonical_task_id(model.task_id), (
                f"{name}/{model.name} has non-canonical task_id {model.task_id}"
            )
            key = (name, model.name)
            assert key not in seen
            seen.add(key)

    def test_boa_tasks_registered(self):
        registered = set(TaskRegistry.list_tasks())
        assert {"boa_body_parts", "boa_body_regions"} <= registered


class TestWeightUrlHygiene:
    """No placeholder Zenodo file URLs survive; every URL is known or manual."""

    def test_no_zenodo_api_file_placeholders(self):
        offenders = []
        for path in PACKAGE_ROOT.rglob("*.py"):
            if path == URL_FRAGMENT_DETECTOR:
                continue
            if "zenodo.org/api/files/" in path.read_text(encoding="utf-8"):
                offenders.append(str(path))
        assert not offenders, f"placeholder Zenodo URLs remain in: {offenders}"

    def test_every_model_url_is_known_or_manual(self):
        unknown = []
        for task_name, model in _iter_all_models():
            url = model.url
            if url and not url.startswith(ALLOWED_URL_PREFIXES):
                # .git repositories and the manual:// sentinel are non-downloadable
                # by design and therefore acceptable.
                if is_downloadable_url(url):
                    unknown.append(f"{task_name}/{model.name}: {url}")
        assert not unknown, f"unrecognised weight URLs: {unknown}"


class TestBoaTasks:
    """BOA body-composition tasks: ids, labels, resampling and fold ensemble."""

    @pytest.mark.parametrize("task_cls,task_id,labels", [
        (boa_tasks.BoaBodyPartsTask, "Dataset543_BCA_body_parts", BODY_PARTS_LABELS),
        (boa_tasks.BoaBodyRegionsTask, "Dataset542_BCA_inference", BODY_REGION_LABELS),
    ])
    def test_registry_definition(self, task_cls, task_id, labels):
        definition = TaskRegistry.get_task(task_cls.name)
        model = definition.models[task_cls.name]
        assert model.task_id == task_id
        assert model.labels == labels
        assert definition.output_config["labels"] == labels

    @pytest.mark.parametrize("task_cls", [
        boa_tasks.BoaBodyPartsTask,
        boa_tasks.BoaBodyRegionsTask,
    ])
    def test_pipeline_resamples_thickness_only(self, task_cls):
        pipeline = task_cls.get_default_pipeline("default")
        resample = _get_step(pipeline, "ResampleStep").config
        assert resample["spacing"] == [5.0, 5.0, 5.0]
        assert resample["only_thickness"] is True

    @pytest.mark.parametrize("task_cls", [
        boa_tasks.BoaBodyPartsTask,
        boa_tasks.BoaBodyRegionsTask,
    ])
    def test_pipeline_runs_full_fold_ensemble_without_mirroring(self, task_cls):
        inference = _get_step(task_cls.get_default_pipeline("default"), "nnUNetInferenceStep")
        assert inference.config["folds"] == ("all",)
        assert inference.config["use_mirroring"] is False

    def test_body_parts_postprocessing_step(self):
        pipeline = boa_tasks.BoaBodyPartsTask.get_default_pipeline("default")
        assert _step_types(pipeline) == [
            "ResampleStep", "nnUNetInferenceStep", "BoaBodyPartsPostprocessStep",
        ]
        step = _get_step(pipeline, "BoaBodyPartsPostprocessStep")
        assert step.config["threshold"] == 3000

    def test_body_regions_postprocessing_step(self):
        pipeline = boa_tasks.BoaBodyRegionsTask.get_default_pipeline("default")
        assert _step_types(pipeline) == [
            "ResampleStep", "nnUNetInferenceStep", "BoaBodyRegionsPostprocessStep",
        ]


class TestLocalToGlobalLabelMappings:
    """Grouped multi-model tasks map local label names onto the global table."""

    CONCAT_TASKS = ("ts2d", "tsxr", "dukeseg")
    LABEL_TASKS = ("ts2d", "tsxr", "dukeseg", "body_composition")

    @pytest.mark.parametrize("task_name", CONCAT_TASKS)
    def test_concat_label_mappings_are_complete(self, task_name):
        task_cls = _task_class(task_name)
        step = _get_step(task_cls.get_default_pipeline("default"), "MultiModelConcatStep")
        mappings = step.config["label_mappings"]
        assert len(mappings) == len(step.config["model_names"])
        for mapping in mappings:
            assert mapping, f"{task_name} produced an empty label mapping"

    @pytest.mark.parametrize("task_name", LABEL_TASKS)
    def test_local_label_names_exist_globally(self, task_name):
        definition = TaskRegistry.get_task(task_name)
        global_names = set(definition.output_config["labels"].values())
        for model in definition.models.values():
            missing = set(model.labels.values()) - global_names
            assert not missing, f"{task_name}/{model.name} misses global labels {missing}"


def _task_class(task_name):
    from nnunetsegmentator.tasks import discover_task_classes

    for task_class in discover_task_classes():
        if task_class.name == task_name:
            return task_class
    raise AssertionError(f"task class for {task_name} not found")


class TestOnlyThicknessResampling:
    """The BOA ``only_thickness`` flag keeps the in-plane spacing untouched."""

    def _image(self):
        array = np.zeros((4, 6, 8), dtype=np.float32)
        image = sitk.GetImageFromArray(array)
        image.SetSpacing((0.8, 0.9, 2.0))
        image.SetOrigin((0.0, 0.0, 0.0))
        image.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
        return image, array

    def test_only_thickness_resamples_z_axis(self):
        from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

        image, array = self._image()
        context = PipelineContext(input_image=image, input_array=array, metadata={})

        step = ResampleStep("resample", {"spacing": [5.0, 5.0, 5.0], "only_thickness": True})
        result = step.execute(context)

        spacing = result.input_image.GetSpacing()
        assert spacing[0] == pytest.approx(0.8)
        assert spacing[1] == pytest.approx(0.9)
        assert spacing[2] == pytest.approx(5.0)

    def test_full_resampling_overrides_all_axes(self):
        from nnunetsegmentator.pipeline.steps.preprocessing import ResampleStep

        image, array = self._image()
        context = PipelineContext(input_image=image, input_array=array, metadata={})

        step = ResampleStep("resample", {"spacing": [2.0, 2.0, 2.0]})
        result = step.execute(context)

        assert result.input_image.GetSpacing() == pytest.approx((2.0, 2.0, 2.0))


class TestBoaBodyPartsPostprocessing:
    """Small components and holes are dropped; large ones survive."""

    def _context(self, segmentation):
        context = PipelineContext(input_image=None, input_array=None, metadata={})
        context.intermediate_results["raw_prediction"] = segmentation
        return context

    def test_removes_small_components_and_holes(self):
        seg = np.zeros((3, 40, 40), dtype=np.uint8)
        seg[:, 5:20, 5:20] = 1          # large block, kept
        seg[:, 30:31, 30:31] = 1        # tiny column, removed

        context = BoaBodyPartsPostprocessStep(
            "boa_body_parts", {"threshold": 50}
        ).execute(self._context(seg))

        out = context.intermediate_results["final_prediction"]
        assert (out == 1).sum() == 3 * 15 * 15
        assert out[0, 30, 30] == 0

    def test_fills_enclosed_hole(self):
        seg = np.zeros((1, 20, 20), dtype=np.uint8)
        seg[0, 2:18, 2:18] = 1
        seg[0, 9:11, 9:11] = 0          # enclosed hole

        context = BoaBodyPartsPostprocessStep(
            "boa_body_parts", {"threshold": 50}
        ).execute(self._context(seg))

        out = context.intermediate_results["final_prediction"]
        assert out[0, 9, 9] == 1
        assert (out == 1).sum() == 16 * 16

    def test_keeps_labels_separate(self):
        seg = np.zeros((1, 20, 20), dtype=np.uint8)
        seg[0, 2:18, 2:10] = 1
        seg[0, 2:18, 12:18] = 2

        context = BoaBodyPartsPostprocessStep(
            "boa_body_parts", {"threshold": 10}
        ).execute(self._context(seg))

        out = context.intermediate_results["final_prediction"]
        assert set(np.unique(out)) == {0, 1, 2}


class TestBoaBodyRegionsPostprocessing:
    """Non-largest components of the unique regions are marked with 255."""

    def _context(self, segmentation, labels):
        context = PipelineContext(input_image=None, input_array=None, metadata={"labels": labels})
        context.intermediate_results["raw_prediction"] = segmentation
        return context

    def test_extra_foreground_component_flagged(self):
        seg = np.zeros((1, 30, 30), dtype=np.uint8)
        seg[0, 2:10, 2:10] = 1          # large component
        seg[0, 20:23, 20:23] = 2        # small satellite

        context = BoaBodyRegionsPostprocessStep("boa_body_regions", {}).execute(
            self._context(seg, {1: "subcutaneous_tissue", 2: "muscle"})
        )

        out = context.intermediate_results["final_prediction"]
        assert (out == 255).sum() == 9
        assert out[0, 2, 2] == 1

    def test_pericardium_keeps_single_component(self):
        labels = {1: "pericardium"}
        seg = np.zeros((1, 30, 30), dtype=np.uint8)
        seg[0, 2:10, 2:10] = 1          # largest pericardium blob
        seg[0, 20:22, 20:22] = 1        # spurious second blob

        context = BoaBodyRegionsPostprocessStep("boa_body_regions", {}).execute(
            self._context(seg, labels)
        )

        out = context.intermediate_results["final_prediction"]
        assert (out == 255).sum() == 4
        assert out[0, 2, 2] == 1