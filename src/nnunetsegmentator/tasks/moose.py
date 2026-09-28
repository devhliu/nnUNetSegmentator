"""
MOOSE Task

MOOSE (Multi-Organ Objective SEgmentation).

CT/MR organ, sub-organ, bone, muscle and body-composition segmentation, plus
PET brain/face models.

Model Storage:
    - Default path: ~/.nnunetsegmentator/models/moose/<task_id>/
    - Weights are downloaded from the model releases on first use.

Label handling:
    - Static labels are seeded from the bundled SNOMED mapping CSV
      (mapping/data/moose_snomed_mapping.csv, subset of MOOSE labels).
    - After a model's weights are downloaded, `ensure_runtime_labels()`
      replaces the static set with the authoritative labels from the model
      payload's dataset.json (all labels, including those without a SNOMED
      entry). Static labels may therefore be incomplete until first run.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional

from .base import BaseTask
from ._sources import moose_url
from ..core.registry import TaskRegistry, TaskDefinition, ModelInfo
from ..core.exceptions import ModelNotFoundError
from ..pipeline.base import Pipeline

logger = logging.getLogger(__name__)

SNOMED_CSV_PATH = Path(__file__).parents[1] / "mapping" / "data" / "moose_snomed_mapping.csv"

MOOSE_MODEL_METADATA = {
    "clin_ct_body": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_body_27112023"),
        "folder_name": "Dataset001_body",
    },
    "clin_ct_lungs": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_lungs_24062023"),
        "folder_name": "Dataset333_HMS3dlungs",
    },
    "clin_ct_organs": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_organs_ras_07052025"),
        "folder_name": "Dataset123_Organs",
    },
    "clin_ct_ribs": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_ribs_ras_08052025"),
        "folder_name": "Dataset444_Ribs",
    },
    "clin_ct_muscles": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_muscles_ras_21052025"),
        "folder_name": "Dataset555_Muscles",
    },
    "clin_ct_peripheral_bones": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_peripheral_bones_ras_07052025"),
        "folder_name": "Dataset666_Peripheral-Bones",
    },
    "clin_ct_vertebrae": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_vertebrae_ras_21052025"),
        "folder_name": "Dataset111_Vertebrae",
    },
    "clin_ct_cardiac": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_cardiac_ras_21052025"),
        "folder_name": "Dataset888_Cardiac",
    },
    "clin_ct_digestive": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_digestive_06102024"),
        "folder_name": "Dataset999_Digestive",
    },
    "clin_ct_ALPACA": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_ALPACA"),
        "folder_name": "Dataset080_Alpaca",
    },
    "clin_ct_PUMA": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_PUMA_1k_23052024"),
        "folder_name": "Dataset002_PUMA",
    },
    "clin_ct_PUMA4": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_PUMA4_06032024"),
        "folder_name": "Dataset003_PUMA4",
    },
    "clin_ct_body_composition": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_body_composition_05092024"),
        "folder_name": "Dataset778_Body_composition",
    },
    "clin_ct_fast_organs": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_organs_6_02092024"),
        "folder_name": "Dataset145_Fast_organs",
    },
    "clin_ct_fast_vertebrae": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_vertebrae3_10092024"),
        "folder_name": "Dataset112_FastVertebrae",
    },
    "clin_ct_fast_cardiac": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_cardiac3_10092024"),
        "folder_name": "Dataset890_FastCardiac",
    },
    "clin_ct_all_bones_v1": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_all_bones_25102023"),
        "folder_name": "Dataset600_Original_bones",
    },
    "clin_pt_fdg_brain_v1": {
        "url": moose_url("moosez-v.3.1.3", "clin_fdg_pt_brain_v1_17112023"),
        "folder_name": "Dataset100_Brain_v1",
    },
    "clin_ct_fat_old": {
        "url": moose_url("moosez-v.3.1.3", "clin_ct_fat_31082023"),
        "folder_name": "Dataset777_Fat",
    },
    "clin_ct_dental": {
        # Not a MOOSE GitHub release; hosted on mdforge, kept as-is.
        "url": "https://model.s.mdforge.com/Dataset112_DentalSegmentator_v100_moose.zip",
        "folder_name": "Dataset112_DentalSegmentator_v100",
    },
    "clin_ct_face": {
        "url": moose_url("moosez-v.3.1.7", "clin_ct_face_ear_12052026"),
        "folder_name": "Dataset006_face_ear_v2",
    },
    "clin_pt_fdg_face": {
        "url": moose_url("moosez-v.3.1.5", "clin_pt_face_12012026"),
        "folder_name": "Dataset183_PETface",
    },
    "preclin_mr_all": {
        "url": moose_url("moosez-v.3.1.3", "preclin_mr_all_05122023"),
        "folder_name": "Dataset234_minimoose",
    },
    "preclin_ct_legs": {
        "url": moose_url("moosez-v.3.1.3", "preclin_ct_legs_05122023"),
        "folder_name": "Dataset256_Preclin_leg_muscles",
    },
    "clin_mr_FVM": {
        "url": moose_url("moosez-v.3.2.0", "clin_mr_FVM_30032026"),
        "folder_name": "Dataset501_FVM",
    },
}

# Cascade workflow (from MOOSE WORKFLOW_REGISTRY): only clin_ct_body_composition
# uses a crop_FOV pre-step; every other model is a single-step segmentation.
CROP_FOV_WORKFLOW = {
    "crop_model": "clin_ct_fast_vertebrae",
    "fov_intensities": [20, 24],
    "crop_label": 22,
    "largest_component_only": True,
}

# CSV 'model' short name -> MOOSE model identifier (for static label seeding)
_CSV_MODEL_TO_IDENTIFIER = {
    "body": "clin_ct_body",
    "organs": "clin_ct_organs",
    "ribs": "clin_ct_ribs",
    "cardiac": "clin_ct_cardiac",
    "muscles": "clin_ct_muscles",
    "peripheral_bones": "clin_ct_peripheral_bones",
    "vertebrae": "clin_ct_vertebrae",
    "digestive": "clin_ct_digestive",
    "body_composition": "clin_ct_body_composition",
    "fat_old": "clin_ct_fat_old",
    "face": "clin_ct_face",
}


def _model_modality(identifier: str) -> str:
    """Infer registry modality from a MOOSE model identifier."""
    segments = identifier.split("_")
    modality = segments[1].upper() if len(segments) > 1 else "CT"
    return {"CT": "CT", "PT": "PT", "MR": "MR"}.get(modality, "CT")


def _load_static_labels() -> Dict[str, Dict[str, int]]:
    """
    Seed per-model {label_name: label_id} dictionaries from the SNOMED CSV.

    Only labels covered by the CSV are seeded; `ensure_runtime_labels()`
    provides the complete label set once weights are available.
    """
    labels: Dict[str, Dict[str, int]] = {}
    if not SNOMED_CSV_PATH.exists():
        logger.warning("MOOSE SNOMED CSV not found: %s", SNOMED_CSV_PATH)
        return labels

    with open(SNOMED_CSV_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            identifier = _CSV_MODEL_TO_IDENTIFIER.get((row.get("model") or "").strip())
            label_name = (row.get("label_name") or "").strip()
            label_id = (row.get("label_id") or "").strip()
            if not identifier or not label_name or not label_id:
                continue
            labels.setdefault(identifier, {})[label_name] = int(label_id)
    return labels


def _find_dataset_json(model_path: Path) -> Optional[Path]:
    """Locate dataset.json inside a downloaded model payload."""
    candidates = [model_path / "dataset.json"]
    candidates.extend(sorted(model_path.glob("*/dataset.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_dataset_labels(dataset_json_path: Path) -> Dict[str, int]:
    """Parse nnUNet dataset.json labels into {label_name: label_id}."""
    try:
        with open(dataset_json_path, encoding="utf-8") as f:
            dataset = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read %s: %s", dataset_json_path, exc)
        return {}

    labels: Dict[str, int] = {}
    for name, value in (dataset.get("labels") or {}).items():
        if name.lower() == "background":
            continue
        try:
            label_id = int(value)
        except (TypeError, ValueError):
            continue
        if label_id == 0:
            continue
        labels[name] = label_id
    return labels


def ensure_runtime_labels(model_name: str) -> Dict[str, int]:
    """
    Return authoritative labels for a MOOSE model, enriching from the
    downloaded model payload's dataset.json when available.

    The enriched set is written back into the registered ModelInfo so label
    splitting, the organ-mapping sidecar and the model selector index all use
    the complete label set.
    """
    task = TaskRegistry.get_task("moose")
    model_info = task.models.get(model_name)
    if model_info is None:
        raise ModelNotFoundError(model_name, task_name="moose")

    try:
        model_path = TaskRegistry.get_model_path(model_name)
    except ModelNotFoundError:
        return dict(model_info.labels)

    dataset_json = _find_dataset_json(model_path)
    if dataset_json is None:
        return dict(model_info.labels)

    labels = _parse_dataset_labels(dataset_json)
    if labels and labels != model_info.labels:
        logger.info(
            "Enriched MOOSE labels for '%s' from %s (%d labels)",
            model_name, dataset_json, len(labels),
        )
        model_info.labels = labels
    return dict(model_info.labels)


class MOOSETask(BaseTask):
    """
    MOOSE: Multi-Organ Objective SEgmentation.

    Single task exposing all 25 MOOSE models as separate ModelInfo entries.
    Each model keeps its native MOOSE label naming; the organ-mapping sidecar
    written at output time provides the SNOMED cross-reference.
    """

    name = "moose"
    description = "MOOSE multi-organ/sub-organ segmentation"

    _static_labels = _load_static_labels()

    @classmethod
    def _build_model_infos(cls) -> Dict[str, ModelInfo]:
        models: Dict[str, ModelInfo] = {}
        for identifier, metadata in MOOSE_MODEL_METADATA.items():
            models[identifier] = ModelInfo(
                name=identifier,
                task_id=metadata["folder_name"],
                url=metadata["url"],
                checksum=None,
                labels=dict(cls._static_labels.get(identifier, {})),
                modality=_model_modality(identifier),
                description=f"MOOSE model {identifier} ({metadata['folder_name']})",
                default_preprocessing="default",
                default_postprocessing="default",
            )
        return models

    @classmethod
    def _single_model_steps(cls) -> list:
        """Inference steps for a standard (non-cascade) MOOSE model."""
        return [
            {
                "type": "preserve_original",
                "name": "preserve_intensities",
                "params": {},
            },
            {
                "type": "nnunet_inference",
                "name": "moose_inference",
                "params": {
                    "model_path": "${model_path}",
                    "folds": ("all",),
                    "use_mirroring": True,
                },
            },
        ]

    @classmethod
    def _cascade_model_steps(cls) -> list:
        """Crop-FOV cascade steps (replicates MOOSE Workflow for body composition)."""
        return [
            {
                "type": "preserve_original",
                "name": "preserve_intensities",
                "params": {},
            },
            {
                "type": "crop_fov",
                "name": "crop_fov",
                "params": {
                    "crop_model_name": CROP_FOV_WORKFLOW["crop_model"],
                    "fov_intensities": CROP_FOV_WORKFLOW["fov_intensities"],
                    "folds": ("all",),
                    "use_mirroring": True,
                },
            },
            {
                "type": "nnunet_inference",
                "name": "moose_inference",
                "params": {
                    "model_path": "${model_path}",
                    "folds": ("all",),
                    "use_mirroring": True,
                },
            },
            {
                "type": "fov_restrict",
                "name": "restrict_fov",
                "params": {
                    "crop_label": CROP_FOV_WORKFLOW["crop_label"],
                    "largest_component_only": CROP_FOV_WORKFLOW["largest_component_only"],
                },
            },
        ]

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        models = cls._build_model_infos()
        primary = next(iter(models))
        primary_labels = models[primary].labels

        return TaskDefinition(
            name=cls.name,
            models=models,
            pipeline_config={
                "name": "moose_pipeline",
                "steps": cls._single_model_steps(),
                # Per-model overrides (picked up by the orchestrator for
                # per_model strategy); keys are MOOSE model identifiers.
                "model_steps": {
                    "clin_ct_body_composition": cls._cascade_model_steps(),
                },
            },
            input_requirements={
                "modalities": ["CT", "PT", "MR"],
                "format": "NIfTI or DICOM",
                "description": (
                    "CT for most models; PET (FDG) for brain/face models; "
                    "MR for clin_mr_FVM / preclin_mr_all. Select models matching "
                    "the input modality."
                ),
            },
            output_config={
                "format": "NIfTI",
                "labels": {label_id: name for name, label_id in primary_labels.items()},
                "save_individual_labels": True,
                # Multi-model aware output: label splitting, per-model NIfTIs
                # and the organ-mapping sidecar are driven per selected model.
                "strategy": "per_model",
                "emit_label_mapping": True,
                "runtime_labels": True,
            },
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default processing pipeline (primary model)."""
        pipeline = Pipeline(name="moose_default")

        pipeline.add_step(_make_step("preserve_original", "preserve_intensities", {}))
        pipeline.add_step(_make_step("nnunet_inference", "moose_inference", {
            "folds": ("all",),
            "use_mirroring": True,
        }))

        return pipeline


def _make_step(step_type: str, name: str, params: dict):
    """Instantiate a pipeline step by registry type (local import to avoid cycles)."""
    from ..pipeline.builders import PipelineBuilder

    step_class = PipelineBuilder.STEP_REGISTRY.get(step_type)
    if step_class is None:
        raise ValueError(f"Unknown step type: {step_type}")
    return step_class(name, params)
