import importlib.util
from pathlib import Path


def _load_migration_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "migrate_workspace_models.py"
    spec = importlib.util.spec_from_file_location("migrate_workspace_models", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_migrate_workspace_models_moves_single_binding(tmp_path, monkeypatch):
    module = _load_migration_module()

    workspace_results = tmp_path / "workspace" / "results"
    source = workspace_results / "Dataset901_model_a"
    source.mkdir(parents=True)
    (source / "plans.json").write_text("{}", encoding="utf-8")

    model_root = tmp_path / "models"
    monkeypatch.setattr(
        module,
        "build_prefix_index",
        lambda: {"Dataset901": [("task_a", "model_a", "Dataset901_model_a")]},
    )

    migrated = module.migrate(workspace_results=workspace_results, model_root=model_root)

    target = model_root / "task_a" / "Dataset901_model_a"
    assert migrated == 1
    assert target.exists()
    assert (target / "plans.json").exists()
    assert not source.exists()


def test_migrate_workspace_models_copies_multi_binding(tmp_path, monkeypatch):
    module = _load_migration_module()

    workspace_results = tmp_path / "workspace" / "results"
    source = workspace_results / "Dataset901"
    source.mkdir(parents=True)
    (source / "dataset.json").write_text("{}", encoding="utf-8")
    (source / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir()

    model_root = tmp_path / "models"
    monkeypatch.setattr(
        module,
        "build_prefix_index",
        lambda: {
            "Dataset901": [
                ("task_a", "model_a", "Dataset901_model_a"),
                ("task_b", "model_b", "Dataset901_model_b"),
            ]
        },
    )

    migrated = module.migrate(workspace_results=workspace_results, model_root=model_root)

    target_a = model_root / "task_a" / "Dataset901_model_a"
    target_b = model_root / "task_b" / "Dataset901_model_b"
    assert migrated == 2
    assert source.exists()
    assert (target_a / "dataset.json").exists()
    assert (target_b / "dataset.json").exists()
    assert (target_a / "nnUNetTrainer__nnUNetPlans__3d_fullres").exists()
    assert (target_b / "nnUNetTrainer__nnUNetPlans__3d_fullres").exists()
