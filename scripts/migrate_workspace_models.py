#!/usr/bin/env python3
"""
Migrate nnUNet workspace results models into canonical nnunetsegmentator model store.

Source (default):
  ~/.nnunetsegmentator/nnunet_workspace/results

Target (default):
  ~/.nnunetsegmentator/models/{task_name}/{task_id}

Where canonical task_id is:
  Dataset<digits>_<model_name>
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunetsegmentator import tasks as _builtin_tasks  # noqa: F401,E402
from nnunetsegmentator.core.registry import TaskRegistry  # noqa: E402
from nnunetsegmentator.utils.model_layout import (  # noqa: E402
    dataset_prefix,
    has_model_payload,
    normalize_model_directory,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def default_workspace_results() -> Path:
    return Path(
        os.environ.get(
            "NNUNETSEGMENTATOR_NNUNET_WORKSPACE",
            str(Path.home() / ".nnunetsegmentator" / "nnunet_workspace"),
        )
    ) / "results"


def default_model_root() -> Path:
    return Path(
        os.environ.get(
            "NNUNETSEGMENTATOR_MODEL_ROOTPATH",
            str(Path.home() / ".nnunetsegmentator" / "models"),
        )
    )


def build_prefix_index() -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Build index: Dataset prefix -> list of (task_name, model_name, canonical_task_id).
    """
    index: Dict[str, List[Tuple[str, str, str]]] = {}
    for task_name in TaskRegistry.list_tasks().keys():
        task_def = TaskRegistry.get_task(task_name)
        for model_name, model_info in task_def.models.items():
            prefix = dataset_prefix(model_info.task_id)
            index.setdefault(prefix, []).append((task_name, model_name, model_info.task_id))
    return index


def _recursive_payload_search(root: Path, name_match: str) -> Optional[Path]:
    for path in root.rglob("*"):
        if path.is_dir() and path.name == name_match and has_model_payload(path):
            return path
    return None


def find_workspace_payload(workspace_results: Path, canonical_task_id: str) -> Optional[Path]:
    """
    Find a payload in workspace results with strict preference:
    1) canonical task_id dirs
    2) Dataset prefix dirs
    3) recursive canonical name
    4) recursive Dataset prefix name
    """
    ds_prefix = dataset_prefix(canonical_task_id)

    canonical_candidates = [
        workspace_results / canonical_task_id,
        workspace_results / "nnUNet_results" / canonical_task_id,
        workspace_results / "results" / canonical_task_id,
    ]
    for cand in canonical_candidates:
        if has_model_payload(cand):
            return cand

    prefix_candidates = [
        workspace_results / ds_prefix,
        workspace_results / "nnUNet_results" / ds_prefix,
        workspace_results / "results" / ds_prefix,
    ]
    for cand in prefix_candidates:
        if has_model_payload(cand):
            return cand

    found = _recursive_payload_search(workspace_results, canonical_task_id)
    if found:
        return found
    return _recursive_payload_search(workspace_results, ds_prefix)


def migrate(workspace_results: Path, model_root: Path, force: bool = False, dry_run: bool = False) -> int:
    prefix_index = build_prefix_index()
    migrated = 0

    for prefix, bindings in sorted(prefix_index.items()):
        payload_root = None
        for _, _, canonical_task_id in bindings:
            payload_root = find_workspace_payload(workspace_results, canonical_task_id)
            if payload_root is not None:
                break
        if payload_root is None:
            continue

        # If one source maps to multiple bindings, copy for all; otherwise move/copy once.
        for i, (task_name, model_name, canonical_task_id) in enumerate(bindings):
            binding_source = find_workspace_payload(workspace_results, canonical_task_id) or payload_root
            target_dir = model_root / task_name / canonical_task_id
            logger.info("Mapping %s -> %s (%s/%s)", binding_source, target_dir, task_name, model_name)

            if dry_run:
                continue

            target_dir.parent.mkdir(parents=True, exist_ok=True)
            if target_dir.exists():
                if not force:
                    logger.info("Skip existing target (use --force): %s", target_dir)
                    continue
                shutil.rmtree(target_dir)

            # Move only for single binding and first write; otherwise copy.
            do_move = len(bindings) == 1 and i == 0
            if do_move:
                shutil.move(str(binding_source), str(target_dir))
            else:
                shutil.copytree(binding_source, target_dir)

            normalize_model_directory(target_dir, canonical_task_id)
            if not has_model_payload(target_dir):
                raise RuntimeError(f"Migrated target has no nnUNet payload: {target_dir}")
            migrated += 1

    return migrated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate models from nnunet workspace results to canonical model store"
    )
    parser.add_argument(
        "--workspace-results",
        type=str,
        default=str(default_workspace_results()),
        help="Path to nnUNet workspace results directory",
    )
    parser.add_argument(
        "--model-root",
        type=str,
        default=str(default_model_root()),
        help="Canonical model root directory",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing canonical targets")
    parser.add_argument("--dry-run", action="store_true", help="Show planned migrations only")
    args = parser.parse_args()

    source = Path(args.workspace_results).expanduser().resolve()
    target = Path(args.model_root).expanduser().resolve()

    if not source.exists():
        logger.error("Workspace results directory does not exist: %s", source)
        sys.exit(1)

    count = migrate(source, target, force=args.force, dry_run=args.dry_run)
    logger.info("Migration complete. Migrated %d model directory(s).", count)


if __name__ == "__main__":
    main()
