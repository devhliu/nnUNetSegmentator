import tarfile
import zipfile
from pathlib import Path

import pytest

from nnunetsegmentator.core.registry import ModelInfo, TaskDefinition, TaskRegistry


def _make_task(task_name: str = "local_install_task") -> TaskDefinition:
    return TaskDefinition(
        name=task_name,
        models={
            "model_a": ModelInfo(
                name="model_a",
                task_id="Dataset901_model_a",
                url="https://example.com/model_a.zip",
                checksum="",
                labels={"obj": 1},
                modality="CT",
            ),
            "model_b": ModelInfo(
                name="model_b",
                task_id="Dataset902_model_b",
                url="https://example.com/model_b.zip",
                checksum="",
                labels={"obj": 1},
                modality="CT",
            ),
        },
        pipeline_config={},
        input_requirements={},
        output_config={},
    )


@pytest.fixture
def isolated_registry():
    TaskRegistry.clear()
    try:
        yield
    finally:
        TaskRegistry.clear()


def test_install_local_model_from_directory_by_canonical_key(monkeypatch, tmp_path, isolated_registry):
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
    task = _make_task()
    TaskRegistry.register(task)

    source_dir = tmp_path / "src_model_a"
    source_dir.mkdir()
    (source_dir / "dataset.json").write_text("{}", encoding="utf-8")

    installed = TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset901_model_a": source_dir},
    )

    target = tmp_path / "models" / task.name / "Dataset901_model_a"
    assert installed["model_a"] == target
    assert target.exists()
    assert (target / "dataset.json").exists()
    assert TaskRegistry.get_model_path("model_a") == target


def test_install_local_model_from_zip_by_canonical_key(monkeypatch, tmp_path, isolated_registry):
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
    task = _make_task()
    TaskRegistry.register(task)

    source_dir = tmp_path / "zip_src"
    source_dir.mkdir()
    (source_dir / "plans.json").write_text("{}", encoding="utf-8")

    archive = tmp_path / "model_b.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(source_dir / "plans.json", arcname="plans.json")

    installed = TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset902_model_b": archive},
    )

    target = tmp_path / "models" / task.name / "Dataset902_model_b"
    assert installed["model_b"] == target
    assert target.exists()
    assert (target / "plans.json").exists()
    assert TaskRegistry.get_model_path("model_b") == target


def test_install_local_model_from_tar_by_canonical_key(monkeypatch, tmp_path, isolated_registry):
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
    task = _make_task()
    TaskRegistry.register(task)

    source_dir = tmp_path / "tar_src"
    source_dir.mkdir()
    (source_dir / "plans.json").write_text("{}", encoding="utf-8")
    (source_dir / "checkpoint.pth").write_bytes(b"dummy")

    archive = tmp_path / "model_a.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(source_dir / "plans.json", arcname="plans.json")
        tf.add(source_dir / "checkpoint.pth", arcname="checkpoint.pth")

    installed = TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset901_model_a": archive},
    )

    target = tmp_path / "models" / task.name / "Dataset901_model_a"
    assert installed["model_a"] == target
    assert (target / "plans.json").exists()
    assert (target / "checkpoint.pth").exists()


def test_install_local_model_rejects_existing_without_force(
    monkeypatch, tmp_path, isolated_registry
):
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
    task = _make_task()
    TaskRegistry.register(task)

    source_dir = tmp_path / "src_model_a"
    source_dir.mkdir()
    (source_dir / "dataset.json").write_text("{}", encoding="utf-8")

    TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset901_model_a": source_dir},
    )

    with pytest.raises(FileExistsError):
        TaskRegistry.install_task_models_from_local(
            task_name=task.name,
            model_sources={"Dataset901_model_a": source_dir},
            force=False,
        )


def test_install_local_model_supports_force_overwrite(monkeypatch, tmp_path, isolated_registry):
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
    task = _make_task()
    TaskRegistry.register(task)

    source_v1 = tmp_path / "src_v1"
    source_v1.mkdir()
    (source_v1 / "plans.json").write_text("{}", encoding="utf-8")
    (source_v1 / "version.txt").write_text("v1", encoding="utf-8")
    source_v2 = tmp_path / "src_v2"
    source_v2.mkdir()
    (source_v2 / "plans.json").write_text("{}", encoding="utf-8")
    (source_v2 / "version.txt").write_text("v2", encoding="utf-8")

    TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset901_model_a": source_v1},
    )
    TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset901_model_a": source_v2},
        force=True,
    )

    target_file = tmp_path / "models" / task.name / "Dataset901_model_a" / "version.txt"
    assert (tmp_path / "models" / task.name / "Dataset901_model_a" / "plans.json").exists()
    assert target_file.read_text(encoding="utf-8") == "v2"


def test_install_local_model_flattens_nested_task_id_dir(monkeypatch, tmp_path, isolated_registry):
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
    task = _make_task()
    TaskRegistry.register(task)

    source_dir = tmp_path / "nested_src"
    nested = source_dir / "Dataset901_model_a"
    nested.mkdir(parents=True)
    (nested / "plans.json").write_text("{}", encoding="utf-8")
    (nested / "nnUNetTrainer__nnUNetPlans__3d_fullres").mkdir()

    installed = TaskRegistry.install_task_models_from_local(
        task_name=task.name,
        model_sources={"Dataset901_model_a": source_dir},
    )

    target = tmp_path / "models" / task.name / "Dataset901_model_a"
    assert installed["model_a"] == target
    assert (target / "plans.json").exists()
    assert (target / "nnUNetTrainer__nnUNetPlans__3d_fullres").exists()
    assert not (target / "Dataset901_model_a").exists()
