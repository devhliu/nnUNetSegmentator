"""Tests for the organ-driven model selector"""

import pytest

from nnunetsegmentator.core.registry import TaskRegistry, TaskDefinition, ModelInfo
from nnunetsegmentator.mapping.label_mapper import LabelMapper
from nnunetsegmentator.mapping.model_selector import ModelSelector


def _model(name, task_id, labels, modality="CT"):
    return ModelInfo(
        name=name,
        task_id=task_id,
        url="",
        checksum=None,
        labels=labels,
        modality=modality,
    )


@pytest.fixture(autouse=True)
def registry_isolation():
    """Snapshot, clear and restore the global registry around each test."""
    tasks_backup = dict(TaskRegistry._tasks)
    paths_backup = dict(TaskRegistry._model_paths)
    TaskRegistry.clear()
    ModelSelector.reset_instance()
    LabelMapper.reset_instance()
    yield
    TaskRegistry._tasks = tasks_backup
    TaskRegistry._model_paths = paths_backup
    ModelSelector.reset_instance()
    LabelMapper.reset_instance()


def register_task(name, models):
    task = TaskDefinition(
        name=name,
        models=models,
        pipeline_config={"steps": []},
        input_requirements={},
        output_config={},
    )
    TaskRegistry.register(task)
    return task


class TestBuildIndex:
    def test_index_covers_registered_models(self):
        register_task("t_a", {"m_a": _model("m_a", "Dataset1_a", {"liver": 1, "spleen": 2})})
        index = ModelSelector().build_index()
        assert "liver" in index
        providers = index["liver"]
        assert providers[0].task_name == "t_a"
        assert providers[0].model_name == "m_a"
        assert providers[0].label_id == 1

    def test_models_without_labels_skipped(self):
        register_task("t_empty", {"m_empty": _model("m_empty", "Dataset1_e", {})})
        register_task("t_full", {"m_full": _model("m_full", "Dataset1_f", {"liver": 1})})
        index = ModelSelector().build_index()
        providers = index["liver"]
        assert all(p.model_name != "m_empty" for p in providers)

    def test_id_keyed_labels_supported(self):
        """Some tasks store {label_id: name} instead of {name: label_id}."""
        register_task("t_rev", {"m_rev": _model("m_rev", "Dataset1_r", {1: "liver"})})
        index = ModelSelector().build_index()
        assert index["liver"][0].label_id == 1
        assert index["liver"][0].label_name == "liver"

    def test_index_cached(self):
        register_task("t_a", {"m_a": _model("m_a", "Dataset1_a", {"liver": 1})})
        selector = ModelSelector()
        index = selector.build_index()
        assert selector.build_index() is index
        assert selector.build_index(refresh=True) is not index


class TestSelect:
    def test_single_model_cover(self):
        register_task("moose", {
            "clin_ct_organs": _model("clin_ct_organs", "Dataset1_o", {"liver": 1, "kidney_left": 2, "spleen": 3}),
            "clin_ct_ribs": _model("clin_ct_ribs", "Dataset1_ri", {"rib_left_1": 1}),
        })
        plan = ModelSelector().select(["liver", "left kidney"], modality="CT")
        assert plan.task_name == "moose"
        assert plan.models == ["clin_ct_organs"]
        assert plan.unmatched_organs == []
        assert set(plan.organ_to_provider.keys()) == {"liver", "kidney_left"}

    def test_greedy_cover_multiple_models(self):
        register_task("t", {
            "m1": _model("m1", "Dataset1_m1", {"liver": 1}),
            "m2": _model("m2", "Dataset1_m2", {"liver": 1, "spleen": 2, "prostate": 3}),
        })
        plan = ModelSelector().select(["liver", "spleen", "prostate"])
        assert plan.models == ["m2"]

    def test_snomed_code_input(self):
        register_task("t", {"m": _model("m", "Dataset1_m", {"liver": 1})})
        plan = ModelSelector().select(["SCT:10200004"])
        assert plan.models == ["m"]
        plan2 = ModelSelector().select(["10200004"])
        assert plan2.models == ["m"]

    def test_modality_filter(self):
        register_task("t", {
            "m_ct": _model("m_ct", "Dataset1_c", {"liver": 1}, modality="CT"),
            "m_mr": _model("m_mr", "Dataset1_r", {"liver": 1}, modality="MR"),
        })
        plan = ModelSelector().select(["liver"], modality="MR")
        assert plan.models == ["m_mr"]

    def test_task_filter(self):
        register_task("t1", {"m1": _model("m1", "Dataset1_1", {"liver": 1})})
        register_task("t2", {"m2": _model("m2", "Dataset1_2", {"liver": 1})})
        plan = ModelSelector().select(["liver"], task_filter="t2")
        assert plan.task_name == "t2"
        assert plan.models == ["m2"]

    def test_unmatched_organs(self):
        register_task("t", {"m": _model("m", "Dataset1_m", {"liver": 1})})
        plan = ModelSelector().select(["liver", "unicorn_horn"])
        assert plan.unmatched_organs == ["unicorn_horn"]
        assert plan.models == ["m"]

    def test_alias_input(self):
        register_task("t", {"m": _model("m", "Dataset1_m", {"kidney_left": 1})})
        plan = ModelSelector().select(["left kidney"])
        assert plan.models == ["m"]
        assert "kidney_left" in plan.organ_to_provider

    def test_empty_organs_raises(self):
        with pytest.raises(ValueError):
            ModelSelector().select([])

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            ModelSelector().select(["liver"], strategy="bogus")
