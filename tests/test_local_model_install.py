import tarfile
import zipfile
from pathlib import Path

import pytest

from nnunetsegmentator.core.exceptions import ConfigurationError, ModelNotFoundError
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
        # Restore the built-in tasks so other test modules relying on the
        # global registry (moose, TotalSegmentator, ...) keep working.
        from nnunetsegmentator.tasks import register_all_tasks
        register_all_tasks()


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


class TestExplicitModelResolution:
    """Resolution only checks the installed location; download is explicit."""

    @staticmethod
    def _install_payload(target: Path, name: str = "dataset.json", content: str = "{}") -> None:
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(content, encoding="utf-8")

    def test_get_model_path_raises_without_downloading(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        TaskRegistry.register(_make_task())

        def _fail(*args, **kwargs):
            raise AssertionError("get_model_path must not download")

        monkeypatch.setattr(
            "nnunetsegmentator.utils.model_download.ModelDownloader.download_model", _fail
        )

        with pytest.raises(ModelNotFoundError):
            TaskRegistry.get_model_path("model_a")

    def test_get_model_path_returns_canonical_location_when_installed(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        target = tmp_path / "models" / task.name / "Dataset901_model_a"
        self._install_payload(target)

        assert TaskRegistry.get_model_path("model_a") == target

    def test_get_model_path_rejects_installed_dir_without_payload(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        (tmp_path / "models" / task.name / "Dataset901_model_a").mkdir(parents=True)

        with pytest.raises(ModelNotFoundError):
            TaskRegistry.get_model_path("model_a")

    def test_download_model_installs_into_canonical_dir(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        def _fake_download(
            self, model_name, url, checksum=None, force=False, target_dir_name=None
        ):
            path = self.model_dir / (target_dir_name or model_name)
            path.mkdir(parents=True, exist_ok=True)
            (path / "dataset.json").write_text("{}", encoding="utf-8")
            return path

        monkeypatch.setattr(
            "nnunetsegmentator.utils.model_download.ModelDownloader.download_model",
            _fake_download,
        )

        result = TaskRegistry.download_model("model_a")
        expected = tmp_path / "models" / task.name / "Dataset901_model_a"
        assert result == expected
        assert TaskRegistry.get_model_path("model_a") == expected

    def test_download_model_force_replaces_existing_installation(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        target = tmp_path / "models" / task.name / "Dataset901_model_a"
        self._install_payload(target, name="stale.txt", content="stale")

        def _fake_download(
            self, model_name, url, checksum=None, force=False, target_dir_name=None
        ):
            path = self.model_dir / (target_dir_name or model_name)
            path.mkdir(parents=True, exist_ok=True)
            (path / "dataset.json").write_text("{}", encoding="utf-8")
            return path

        monkeypatch.setattr(
            "nnunetsegmentator.utils.model_download.ModelDownloader.download_model",
            _fake_download,
        )

        TaskRegistry.download_model("model_a", force=True)

        assert not (target / "stale.txt").exists()
        assert (target / "dataset.json").exists()

    def test_download_task_models_downloads_all_by_default(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        def _fake_download(
            self, model_name, url, checksum=None, force=False, target_dir_name=None
        ):
            path = self.model_dir / (target_dir_name or model_name)
            path.mkdir(parents=True, exist_ok=True)
            (path / "dataset.json").write_text("{}", encoding="utf-8")
            return path

        monkeypatch.setattr(
            "nnunetsegmentator.utils.model_download.ModelDownloader.download_model",
            _fake_download,
        )

        downloaded = TaskRegistry.download_task_models(task.name)
        assert set(downloaded) == {"model_a", "model_b"}

    def test_download_task_models_rejects_unknown_model(self, tmp_path, isolated_registry):
        TaskRegistry.register(_make_task())

        with pytest.raises(ConfigurationError):
            TaskRegistry.download_task_models("local_install_task", ["does_not_exist"])


class TestDownloadModelsCli:
    """`download-models` parses flags and delegates to the registry."""

    def test_cli_download_models_invokes_registry(self, monkeypatch, capsys):
        import importlib
        import sys

        cli_main = importlib.import_module("nnunetsegmentator.cli.main")
        recorded = {}

        def _fake(task_name, model_names=None, force=False):
            recorded["task_name"] = task_name
            recorded["model_names"] = model_names
            recorded["force"] = force
            return {"model_a": Path("/tmp/model_a")}

        monkeypatch.setattr(
            TaskRegistry, "download_task_models", staticmethod(_fake)
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["nnunetsegmentator", "download-models", "-t", "fake_task",
             "--models", "model_a, model_b", "--force"],
        )

        cli_main.main()

        assert recorded == {
            "task_name": "fake_task",
            "model_names": ["model_a", "model_b"],
            "force": True,
        }

    def test_cli_download_models_defaults_to_all_models(self, monkeypatch):
        import importlib
        import sys

        cli_main = importlib.import_module("nnunetsegmentator.cli.main")
        recorded = {}

        def _fake(task_name, model_names=None, force=False):
            recorded["model_names"] = model_names
            recorded["force"] = force
            return {}

        monkeypatch.setattr(
            TaskRegistry, "download_task_models", staticmethod(_fake)
        )
        monkeypatch.setattr(
            sys, "argv", ["nnunetsegmentator", "download-models", "-t", "fake_task"]
        )

        cli_main.main()

        assert recorded == {"model_names": None, "force": False}


class TestInstallModelsFromRoot:
    """`install_models_from_root` discovers candidates and installs them."""

    @staticmethod
    def _write_payload(directory: Path, name: str = "dataset.json") -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / name).write_text("{}", encoding="utf-8")
        return directory

    def test_discovers_task_id_directory(self, monkeypatch, tmp_path, isolated_registry):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        root = tmp_path / "weights"
        self._write_payload(root / "Dataset901_model_a")

        result = TaskRegistry.install_models_from_root(root)

        target = tmp_path / "models" / task.name / "Dataset901_model_a"
        assert result["installed"] == {"model_a": target}
        assert (target / "dataset.json").exists()
        assert TaskRegistry.get_model_path("model_a") == target

    def test_discovers_task_id_archive(self, monkeypatch, tmp_path, isolated_registry):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        root = tmp_path / "weights"
        root.mkdir()
        staging = tmp_path / "zip_src"
        self._write_payload(staging, name="plans.json")
        archive = root / "Dataset902_model_b.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(staging / "plans.json", arcname="plans.json")

        result = TaskRegistry.install_models_from_root(root)

        target = tmp_path / "models" / task.name / "Dataset902_model_b"
        assert result["installed"] == {"model_b": target}
        assert (target / "plans.json").exists()

    def test_discovers_nested_payload_via_ancestor_name(
        self, monkeypatch, tmp_path, isolated_registry
    ):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        root = tmp_path / "weights"
        trainer = (
            root
            / "nnUNet_results"
            / "Dataset901_model_a"
            / "nnUNetTrainer__nnUNetPlans__3d_fullres"
        )
        self._write_payload(trainer, name="plans.json")

        result = TaskRegistry.install_models_from_root(root)

        target = tmp_path / "models" / task.name / "Dataset901_model_a"
        assert result["installed"] == {"model_a": target}
        assert (target / "nnUNetTrainer__nnUNetPlans__3d_fullres" / "plans.json").exists()

    def test_ambiguous_dataset_prefix_is_skipped(self, monkeypatch, tmp_path, isolated_registry):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        TaskRegistry.register(
            TaskDefinition(
                name="ambiguous_task",
                models={
                    "alpha": ModelInfo(
                        name="alpha", task_id="Dataset881_alpha", url="", checksum="",
                        labels={"o": 1}, modality="CT",
                    ),
                    "beta": ModelInfo(
                        name="beta", task_id="Dataset881_beta", url="", checksum="",
                        labels={"o": 1}, modality="CT",
                    ),
                },
                pipeline_config={},
                input_requirements={},
                output_config={},
            )
        )

        root = tmp_path / "weights"
        self._write_payload(root / "Dataset881_shared")

        result = TaskRegistry.install_models_from_root(root)

        assert result["installed"] == {}
        assert result["planned"] == {}
        assert len(result["ambiguous"]) == 1

    def test_unmatched_candidate_is_reported(self, monkeypatch, tmp_path, isolated_registry):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        TaskRegistry.register(_make_task())

        root = tmp_path / "weights"
        self._write_payload(root / "random_weights")

        result = TaskRegistry.install_models_from_root(root)

        assert result["installed"] == {}
        assert len(result["unmatched"]) == 1

    def test_dry_run_does_not_write(self, monkeypatch, tmp_path, isolated_registry):
        monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(tmp_path / "models"))
        task = _make_task()
        TaskRegistry.register(task)

        root = tmp_path / "weights"
        self._write_payload(root / "Dataset901_model_a")

        result = TaskRegistry.install_models_from_root(root, dry_run=True)

        target = tmp_path / "models" / task.name / "Dataset901_model_a"
        assert "model_a" in result["planned"]
        assert result["installed"] == {}
        assert not target.exists()

    def test_cli_from_root_delegates_to_registry(self, monkeypatch):
        import importlib
        import sys

        cli_main = importlib.import_module("nnunetsegmentator.cli.main")
        recorded = {}

        def _fake(root, task_name=None, force=False, dry_run=False):
            recorded.update(root=root, task_name=task_name, force=force, dry_run=dry_run)
            return {
                "installed": {},
                "planned": {},
                "unmatched": [],
                "ambiguous": [],
                "skipped": [],
            }

        monkeypatch.setattr(
            TaskRegistry, "install_models_from_root", staticmethod(_fake)
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["nnunetsegmentator", "install-models", "--from", "/tmp/weights", "--force"],
        )

        cli_main.main()

        assert recorded == {
            "root": "/tmp/weights",
            "task_name": None,
            "force": True,
            "dry_run": False,
        }
