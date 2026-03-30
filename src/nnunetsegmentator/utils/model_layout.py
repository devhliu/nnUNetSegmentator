"""Model layout helpers for canonical nnUNet task/model directories."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

CANONICAL_TASK_ID_PATTERN = re.compile(r"^Dataset\d+_[A-Za-z0-9_]+$")
CANONICAL_MODEL_KEY_PATTERN = re.compile(r"^Dataset\d+_[A-Za-z0-9_]+$")
_SKIP_DIR_NAMES = {".git", "__pycache__"}


def is_canonical_task_id(task_id: str) -> bool:
    """Return True when task_id matches canonical form Dataset<数字>_<model_name>."""
    return bool(CANONICAL_TASK_ID_PATTERN.match(task_id))


def is_canonical_model_key(model_key: str) -> bool:
    """Return True when model key matches canonical form Dataset<数字>_<model_name>."""
    return bool(CANONICAL_MODEL_KEY_PATTERN.match(model_key))


def dataset_prefix(task_id: str) -> str:
    """Extract Dataset<数字>_<model_name> prefix from canonical task_id."""
    if not is_canonical_task_id(task_id):
        raise ValueError(
            f"Invalid task_id '{task_id}'. Expected canonical format Dataset<数字>_<model_name>."
        )
    return task_id.split("_", 1)[0]


def has_model_payload(path: Path) -> bool:
    """Check whether a directory looks like an nnUNet model payload root."""
    if not path.exists() or not path.is_dir():
        return False
    if (path / "dataset.json").exists() or (path / "plans.json").exists():
        return True
    return any(
        child.is_dir() and child.name.startswith("nnUNetTrainer")
        for child in path.iterdir()
    )


def _iter_dirs_limited(base_path: Path, max_depth: int) -> Iterable[Path]:
    """Breadth-first directory traversal with a depth limit."""
    queue = [(base_path, 0)]
    while queue:
        current, depth = queue.pop(0)
        if not current.exists() or not current.is_dir():
            continue
        yield current
        if depth >= max_depth:
            continue
        for child in current.iterdir():
            if child.is_dir() and child.name not in _SKIP_DIR_NAMES:
                queue.append((child, depth + 1))


def find_model_payload_root(base_path: Path, task_id: str, max_depth: int = 6) -> Optional[Path]:
    """
    Find model payload root under base_path for task_id.

    Preferred matches are exact task directory names, then task-prefix directory names
    (e.g., Dataset711_PSMA), then any payload directory under common nnUNet roots.
    """
    base_path = Path(base_path).expanduser().resolve()
    if not base_path.exists():
        return None
    ds_prefix = dataset_prefix(task_id)

    direct_candidates = [
        base_path,
        base_path / task_id,
        base_path / ds_prefix,
        base_path / "nnUNet_results" / task_id,
        base_path / "nnUNet_results" / ds_prefix,
        base_path / "results" / task_id,
        base_path / "results" / ds_prefix,
    ]
    for candidate in direct_candidates:
        if has_model_payload(candidate):
            return candidate

    exact_matches = []
    prefixed_matches = []
    generic_matches = []

    for directory in _iter_dirs_limited(base_path, max_depth=max_depth):
        if not has_model_payload(directory):
            continue
        if directory.name == task_id:
            exact_matches.append(directory)
            continue
        if directory.name == ds_prefix:
            prefixed_matches.append(directory)
            continue
        if directory.name.startswith(f"{ds_prefix}_"):
            prefixed_matches.append(directory)
            continue
        if "nnUNet_results" in directory.parts or "results" in directory.parts:
            generic_matches.append(directory)

    if exact_matches:
        return exact_matches[0]
    if prefixed_matches:
        return prefixed_matches[0]
    if generic_matches:
        return generic_matches[0]
    return None


def _move_children_to_target(source_dir: Path, target_dir: Path) -> None:
    for child in list(source_dir.iterdir()):
        destination = target_dir / child.name
        if destination.exists():
            raise FileExistsError(
                f"Cannot normalize model directory due to conflict: {destination}"
            )
        shutil.move(str(child), str(destination))


def normalize_model_directory(target_path: Path, task_id: str) -> Path:
    """
    Normalize a model directory so payload exists directly under target_path.

    Supports flattening duplicated task-id folders, single wrappers, and nested
    nnUNet_results/results wrappers.
    """
    if not target_path.exists() or not target_path.is_dir():
        return target_path

    while True:
        changed = False

        nested_same = target_path / task_id
        if nested_same.exists() and nested_same.is_dir():
            _move_children_to_target(nested_same, target_path)
            nested_same.rmdir()
            changed = True

        if not has_model_payload(target_path):
            payload_root = find_model_payload_root(target_path, task_id)
            if payload_root and payload_root != target_path:
                _move_children_to_target(payload_root, target_path)
                current = payload_root
                while current != target_path and current.exists():
                    parent = current.parent
                    if any(current.iterdir()):
                        break
                    current.rmdir()
                    current = parent
                changed = True

        if not has_model_payload(target_path):
            subdirs = [child for child in target_path.iterdir() if child.is_dir()]
            if len(subdirs) == 1 and has_model_payload(subdirs[0]):
                _move_children_to_target(subdirs[0], target_path)
                subdirs[0].rmdir()
                changed = True

        if not changed:
            break

    return target_path
