from pathlib import Path

import pytest

from nnunetsegmentator.core.config import Config


def test_setup_nnunet_environment_uses_workspace_not_model_dir(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    workspace_root = tmp_path / "nnunet_workspace"
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(model_root))
    monkeypatch.setenv("NNUNETSEGMENTATOR_NNUNET_WORKSPACE", str(workspace_root))

    cfg = Config()
    cfg.setup_nnunet_environment()

    assert cfg.nnunet_raw == workspace_root / "raw"
    assert cfg.nnunet_preprocessed == workspace_root / "preprocessed"
    assert cfg.nnunet_results == workspace_root / "results"
    assert cfg.nnunet_raw.exists()
    assert cfg.nnunet_preprocessed.exists()
    assert cfg.nnunet_results.exists()
    assert not (model_root / "raw").exists()
    assert not (model_root / "preprocessed").exists()
    assert not (model_root / "results").exists()


def test_setup_nnunet_environment_rejects_paths_inside_model_dir(monkeypatch, tmp_path):
    model_root = tmp_path / "models"
    monkeypatch.setenv("NNUNETSEGMENTATOR_MODEL_ROOTPATH", str(model_root))

    cfg = Config(
        model_dir=model_root,
        nnunet_raw=model_root / "raw",
        nnunet_preprocessed=model_root / "preprocessed",
        nnunet_results=model_root / "results",
    )

    with pytest.raises(ValueError, match="must not be inside model_dir"):
        cfg.setup_nnunet_environment()
