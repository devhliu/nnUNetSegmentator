#!/usr/bin/env python3
"""
Run segmentation on Case_1 DICOM PET/CT and export outputs as DICOM series.

Default case path:
  ../data/Case_1

Expected Case_1 layout:
  Case_1/
    CT-ACCT_201/
      *.dcm
    PET-TB 10min_302/
      *.dcm

Notes:
- This example runs two segmentations by default:
  - PET series with `lion` (PSMA model)
  - CT series with `total` (pipeline mode: `fast`)
- Output is written as DICOM slice series (not NIfTI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

# Allow running directly from repository root without installation.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from nnunetsegmentator import Config, SegmentationOrchestrator  # noqa: E402
from nnunetsegmentator.core.exceptions import ModelNotFoundError  # noqa: E402
from nnunetsegmentator.io.writers import ImageWriter  # noqa: E402
from nnunetsegmentator import tasks as _builtin_tasks  # noqa: E402,F401


def _list_dicom_files(series_dir: Path) -> list[Path]:
    files = sorted(series_dir.rglob("*.dcm"))
    if files:
        return files
    return sorted(p for p in series_dir.rglob("*") if p.is_file())


def _read_modality(series_dir: Path) -> Optional[str]:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError("pydicom is required to inspect DICOM series.") from exc

    files = _list_dicom_files(series_dir)
    if not files:
        return None
    ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    modality = ds.get("Modality")
    return str(modality) if modality is not None else None


def _discover_case_series(case_dir: Path) -> Tuple[Path, Optional[Path]]:
    pet_candidates: list[Path] = []
    ct_candidates: list[Path] = []

    for child in sorted(p for p in case_dir.iterdir() if p.is_dir()):
        modality = _read_modality(child)
        if modality == "PT":
            pet_candidates.append(child)
        elif modality == "CT":
            ct_candidates.append(child)

    if not pet_candidates:
        raise FileNotFoundError(f"No PET (PT) DICOM series found in: {case_dir}")

    pet_dir = pet_candidates[0]
    ct_dir = ct_candidates[0] if ct_candidates else None
    return pet_dir, ct_dir


def _print_series_info(case_dir: Path, pet_dir: Path, ct_dir: Optional[Path]) -> None:
    print("=" * 80)
    print("Case_1 DICOM Segmentation Example")
    print("=" * 80)
    print(f"Case directory: {case_dir}")
    print(f"PET series:     {pet_dir}")
    if ct_dir is not None:
        print(f"CT series:      {ct_dir}")
    else:
        print("CT series:      <not found>")
    print("-" * 80)


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_case_dir = repo_root / "data" / "Case_1"

    parser = argparse.ArgumentParser(
        description=(
            "Run segmentation on Case_1 PET/CT DICOM and export segmentation outputs as DICOM series."
        )
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=default_case_dir,
        help=f"Case directory containing PET/CT DICOM series (default: {default_case_dir})",
    )
    parser.add_argument(
        "--pet-dir",
        type=Path,
        default=None,
        help="Optional explicit PET DICOM series directory (overrides auto-discovery).",
    )
    parser.add_argument(
        "--ct-dir",
        type=Path,
        default=None,
        help="Optional explicit CT DICOM series directory (for logging/reference).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("example_output_case01"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--pet-task",
        type=str,
        default="lion",
        help="PET task to run (default: lion).",
    )
    parser.add_argument(
        "--pet-model-path",
        type=Path,
        default=None,
        help="Optional local PET model path to bypass registry lookup/download.",
    )
    parser.add_argument(
        "--ct-task",
        type=str,
        default="total",
        help="CT task to run (default: total).",
    )
    parser.add_argument(
        "--ct-pipeline-mode",
        type=str,
        default="fast",
        help="CT pipeline mode via Config.custom['pipeline_mode'] (default: fast).",
    )
    parser.add_argument(
        "--ct-model-path",
        type=Path,
        default=None,
        help=(
            "Optional local CT model path. "
            "Useful for tasks whose pipeline uses ${model_path} placeholders "
            "(not required for default total/fast, which resolves by model name)."
        ),
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Skip exporting per-label DICOM series.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device ID used when use_gpu=True.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU mode.",
    )
    return parser.parse_args()


def _resolve_default_pet_model_path(pet_task: str, pet_model_path: Optional[Path]) -> Optional[Path]:
    if pet_model_path is not None:
        return pet_model_path

    if pet_task == "lion":
        default_psma_model = Path.home() / ".nnunetsegmentator" / "models" / "lion" / "Dataset711_psma"
        if default_psma_model.exists():
            print(f"Using default LION PSMA model path: {default_psma_model}")
            return default_psma_model
    return None


def main() -> None:
    args = _parse_args()

    case_dir = args.case_dir.expanduser().resolve()
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    if args.pet_dir is not None:
        pet_dir = args.pet_dir.expanduser().resolve()
        ct_dir = args.ct_dir.expanduser().resolve() if args.ct_dir is not None else None
    else:
        pet_dir, discovered_ct_dir = _discover_case_series(case_dir)
        ct_dir = args.ct_dir.expanduser().resolve() if args.ct_dir is not None else discovered_ct_dir

    if not pet_dir.exists():
        raise FileNotFoundError(f"PET series directory not found: {pet_dir}")
    if ct_dir is None:
        raise FileNotFoundError(
            "CT series was not found. Provide --ct-dir explicitly or verify Case_1 layout."
        )
    if not ct_dir.exists():
        raise FileNotFoundError(f"CT series directory not found: {ct_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _print_series_info(case_dir=case_dir, pet_dir=pet_dir, ct_dir=ct_dir)
    print(f"PET task:       {args.pet_task}")
    print(f"CT task:        {args.ct_task} (mode={args.ct_pipeline_mode})")
    print(f"Output root:    {output_dir}")

    resolved_pet_model_path = _resolve_default_pet_model_path(
        pet_task=args.pet_task,
        pet_model_path=args.pet_model_path,
    )

    pet_config = Config(
        output_dir=output_dir,
        use_gpu=not args.cpu,
        gpu_id=args.gpu_id,
    )

    pet_orchestrator = SegmentationOrchestrator(
        task_name=args.pet_task,
        config=pet_config,
        model_path=resolved_pet_model_path,
    )

    print("\nRunning PET segmentation...")
    try:
        pet_result = pet_orchestrator.segment(
            input_data=str(pet_dir),
            output_path=None,
            return_labels=not args.no_labels,
        )
    except ModelNotFoundError as exc:
        print("\nModel not found in local registry path.")
        print("Provide --pet-model-path or install local models first, for example:")
        print(
            "  nnunetsegmentator install-models --task lion "
            "--model Dataset711_psma=/path/to/clin_pt_psma_2046_25112025.zip"
        )
        raise RuntimeError("PET segmentation cannot proceed without model files.") from exc

    pet_out_root = output_dir / "pet_segmentation"
    pet_seg_series_dir = pet_out_root / "segmentation_series"
    ImageWriter.write(pet_result.segmentation, pet_seg_series_dir, format="dicom")
    print(f"PET main segmentation DICOM series: {pet_seg_series_dir}")

    if not args.no_labels and pet_result.labels:
        pet_labels_root = pet_out_root / "label_series"
        pet_labels_root.mkdir(parents=True, exist_ok=True)
        for label_name, label_img in pet_result.labels.items():
            label_dir = pet_labels_root / label_name
            ImageWriter.write(label_img, label_dir, format="dicom")
            print(f"PET label DICOM series [{label_name}]: {label_dir}")

    ct_config = Config(
        output_dir=output_dir,
        use_gpu=not args.cpu,
        gpu_id=args.gpu_id,
        custom={"pipeline_mode": args.ct_pipeline_mode},
    )
    ct_orchestrator = SegmentationOrchestrator(
        task_name=args.ct_task,
        config=ct_config,
        model_path=args.ct_model_path,
    )

    print("\nRunning CT segmentation...")
    try:
        ct_result = ct_orchestrator.segment(
            input_data=str(ct_dir),
            output_path=None,
            return_labels=not args.no_labels,
        )
    except ModelNotFoundError as exc:
        print("\nCT model not found in local registry path.")
        print("Install local CT models for your selected task, for example:")
        print(
            "  nnunetsegmentator install-models --task total "
            "--model Dataset297_total_fast=/path/to/total_fast.zip"
        )
        raise RuntimeError("CT segmentation cannot proceed without model files.") from exc

    ct_out_root = output_dir / "ct_segmentation"
    ct_seg_series_dir = ct_out_root / "segmentation_series"
    ImageWriter.write(ct_result.segmentation, ct_seg_series_dir, format="dicom")
    print(f"CT main segmentation DICOM series: {ct_seg_series_dir}")

    if not args.no_labels and ct_result.labels:
        ct_labels_root = ct_out_root / "label_series"
        ct_labels_root.mkdir(parents=True, exist_ok=True)
        for label_name, label_img in ct_result.labels.items():
            label_dir = ct_labels_root / label_name
            ImageWriter.write(label_img, label_dir, format="dicom")
            print(f"CT label DICOM series [{label_name}]: {label_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
