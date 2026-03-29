#!/usr/bin/env python3
"""
Example: Install TotalSegmentator MR model(s) into nnunetsegmentator model root.

This script searches for MR model folders under:
  - ~/.totalsegmetnator/nnunet/results   (requested path spelling)
  - ~/.totalsegmentator/nnunet/results   (common path spelling)

Then installs discovered models for task "total_mr" into:
  ~/.nnunetsegmentator/models/total_mr/<task_id>

Supported task IDs:
  - 850 (total_mr_organs)
  - 852 (total_mr_fast)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict

# Add parent directory to path for editable/local runs
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunetsegmentator import TaskRegistry  # noqa: E402
from nnunetsegmentator import tasks as _builtin_tasks  # noqa: E402,F401


def find_totalsegmentator_results_root() -> Path:
    candidates = [
        Path.home() / ".totalsegmetnator" / "nnunet" / "results",
        Path.home() / ".totalsegmentator" / "nnunet" / "results",
    ]
    for root in candidates:
        if root.exists():
            return root
    raise FileNotFoundError(
        "Could not find TotalSegmentator results directory. "
        "Checked: ~/.totalsegmetnator/nnunet/results and ~/.totalsegmentator/nnunet/results"
    )


def discover_total_mr_sources(results_root: Path) -> Dict[str, Path]:
    """
    Discover model source directories keyed by task ID ("850", "852").
    """
    patterns = {
        "850": re.compile(r"dataset850", re.IGNORECASE),
        "852": re.compile(r"dataset852", re.IGNORECASE),
    }

    discovered: Dict[str, Path] = {}
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        for task_id, rx in patterns.items():
            if rx.search(name):
                discovered[task_id] = child.resolve()

    return discovered


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Install total_mr models from ~/.totalsegmentator into nnunetsegmentator."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing installed model directories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered sources and expected install targets.",
    )
    args = parser.parse_args()

    results_root = find_totalsegmentator_results_root()
    discovered = discover_total_mr_sources(results_root)

    if not discovered:
        raise RuntimeError(
            f"No total_mr model sources found under: {results_root}\n"
            "Expected directories containing Dataset850* or Dataset852*."
        )

    print("Discovered source model directories:")
    for task_id, src in discovered.items():
        print(f"  - {task_id}: {src}")

    if args.dry_run:
        target_root = Path.home() / ".nnunetsegmentator" / "models" / "total_mr"
        print("\nDry run (no files copied):")
        for task_id in sorted(discovered):
            print(f"  - would install task_id={task_id} -> {target_root / task_id}")
        return

    installed = TaskRegistry.install_task_models_from_local(
        task_name="total_mr",
        model_sources=discovered,  # keyed by task ID as requested
        force=args.force,
    )

    print("\nInstalled model directories:")
    for model_name, dst in installed.items():
        print(f"  - {model_name}: {dst}")


if __name__ == "__main__":
    main()
