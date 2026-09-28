"""
TotalSegmentator Tasks - Unified Definitions

All TotalSegmentator task classes in one module, faithful to the official
v2.5.0-weights release (map_tasks_config.py / nnunet.py / map_to_binary.py):

- ``total`` / ``total_mr``: comprehensive whole-body CT/MR segmentation.
  The full-resolution default workflow runs 5 CT part models (or 2 MR part
  models) whose outputs are concatenated by label name; fast/fastest modes
  run single low-resolution models. The CT task also provides the official
  roi workflow (per-part roi inference).
- Specialized tasks (lung_vessels, body, tissue_types, heartchambers_highres,
  cerebral_bleed, liver_vessels, lung_nodules, vertebrae_body, liver_lesions,
  kidney_cysts, pleural_pericard_effusion, body_mr, tissue_types_mr).
- v2.5.0-weights tasks (vertebrae_mr, breasts, ventricle_parts,
  liver_segments, liver_segments_mr, trunk_cavities, brain_aneurysm,
  vertebrae_pp, abdominal_muscles, craniofacial_structures, teeth).

Workflow conventions shared by the region-crop tasks:
- A coarse "total" model runs first (official crop model selection:
  robust_crop -> 3mm (total_fast / total_mr_fast); otherwise 6mm
  (total_fastest) for CT and 3mm for MR; crop labels containing
  "body_trunc" -> "body_fast"; "teeth" -> "craniofacial_structures"), the
  image is cropped to the mask bounding box expanded by ``crop_addon`` mm,
  the specialized model infers, and the prediction is pasted back into the
  full-frame grid. The crop model runs with tta=False.
- No intensity clipping or explicit resampling is performed: the inference
  step resamples to the model spacing from the model plans, exactly like the
  official nnUNetv2-based predictor.
- Official per-task postprocessing is replicated (body and vertebrae_pp).

All organ label tables are managed by the :mod:`nnunetsegmentator.labels`
sub-module (including the model-label -> SNOMED-standard naming mapping);
this module only wires them into task/model definitions.

Weight URLs come from :mod:`._sources` (official TotalSegmentator GitHub
release folders); the commercial models without public weights use the
``NON_DOWNLOADABLE_URL`` sentinel.
"""

import logging

from ..core.registry import TaskDefinition, ModelInfo
from ..labels import (
    ABDOMINAL_MUSCLES_LABELS,
    BODY_LABELS,
    BRAIN_ANEURYSM_LABELS,
    BREASTS_LABELS,
    CEREBRAL_BLEED_LABELS,
    CRANIOFACIAL_STRUCTURES_LABELS,
    HEART_CHAMBERS_LABELS,
    KIDNEY_CYSTS_LABELS,
    LIVER_LESIONS_LABELS,
    LIVER_SEGMENTS_LABELS,
    LIVER_VESSELS_LABELS,
    LUNG_LOBES,
    LUNG_NODULES_LABELS,
    LUNG_VESSELS_LABELS,
    MR_LABELS,
    MR_MUSCLES_LABELS,
    MR_ORGANS_LABELS,
    PART_CARDIAC_LABELS,
    PART_MUSCLES_LABELS,
    PART_ORGANS_LABELS,
    PART_RIBS_LABELS,
    PART_VERTEBRAE_LABELS,
    PLEURAL_PERICARD_EFFUSION_LABELS,
    TEETH_LABELS,
    TISSUE_TYPES_LABELS_3,
    TISSUE_TYPES_LABELS_4,
    TISSUE_TYPES_MR_LABELS,
    TOTAL_LABELS,
    TRUNK_CAVITIES_LABELS,
    VERTEBRAE_BODY_LABELS,
    VERTEBRAE_MR_LABELS,
    VERTEBRAE_PP_LABELS,
    VENTRICLE_PARTS_LABELS,
)
from ..pipeline.base import Pipeline
from ._helpers import SingleModelTask, build_pipeline, crop_pre, restore_post
from ._sources import (
    BODY_EXTREMITIES_MIN_SIZE_MM3,
    NON_DOWNLOADABLE_URL,
    totalseg_url,
)
from .base import BaseTask

logger = logging.getLogger(__name__)


# Official body task postprocessing (nnunet.py): keep the largest body_trunc
# component and remove body_extremities blobs below 50,000 mm3.
BODY_POSTPROCESSING = [
    {"type": "largest_component", "name": "largest_component",
     "params": {"labels": ["body_trunc"], "keep_top_n": 1}},
    {"type": "remove_small_objects", "name": "remove_small_objects",
     "params": {"labels": ["body_extremities"], "min_size_mm3": BODY_EXTREMITIES_MIN_SIZE_MM3}},
]


class TotalSegmentatorTask(BaseTask):
    """
    TotalSegmentator task for comprehensive whole-body CT segmentation.

    Segments 117 anatomical structures including:
    - Organs (liver, spleen, kidneys, pancreas, etc.)
    - Vertebrae (C1-L5, S1, sacrum)
    - Ribs (left and right, 1-12)
    - Cardiac structures (heart, vessels)
    - Muscles and bones
    - Brain and skull

    Model Storage: ~/.nnunetsegmentator/models/total/
    """

    name = "total"
    description = "Comprehensive whole-body CT segmentation with 117 anatomical structures"

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""

        models = {
            # 5-part model strategy for full resolution.
            # task_ids follow the official TotalSegmentator v2.0.0-weights
            # folder names (TASK_ID_WEIGHTS_CONFIGS in map_tasks_config.py).
            "total_organs": ModelInfo(
                name="total_organs",
                task_id="Dataset291_TotalSegmentator_part1_organs_1559subj",
                url=totalseg_url("Dataset291_TotalSegmentator_part1_organs_1559subj", "v2.0.0-weights"),
                checksum="",
                labels=PART_ORGANS_LABELS,
                modality="CT",
                description="Organ segmentation (24 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            "total_vertebrae": ModelInfo(
                name="total_vertebrae",
                task_id="Dataset292_TotalSegmentator_part2_vertebrae_1532subj",
                url=totalseg_url("Dataset292_TotalSegmentator_part2_vertebrae_1532subj", "v2.0.0-weights"),
                checksum="",
                labels=PART_VERTEBRAE_LABELS,
                modality="CT",
                description="Vertebrae segmentation (26 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            "total_cardiac": ModelInfo(
                name="total_cardiac",
                task_id="Dataset293_TotalSegmentator_part3_cardiac_1559subj",
                url=totalseg_url("Dataset293_TotalSegmentator_part3_cardiac_1559subj", "v2.0.0-weights"),
                checksum="",
                labels=PART_CARDIAC_LABELS,
                modality="CT",
                description="Cardiac and vessel segmentation (18 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            "total_muscles": ModelInfo(
                name="total_muscles",
                task_id="Dataset294_TotalSegmentator_part4_muscles_1559subj",
                url=totalseg_url("Dataset294_TotalSegmentator_part4_muscles_1559subj", "v2.0.0-weights"),
                checksum="",
                labels=PART_MUSCLES_LABELS,
                modality="CT",
                description="Muscle and bone segmentation (23 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            "total_ribs": ModelInfo(
                name="total_ribs",
                task_id="Dataset295_TotalSegmentator_part5_ribs_1559subj",
                url=totalseg_url("Dataset295_TotalSegmentator_part5_ribs_1559subj", "v2.0.0-weights"),
                checksum="",
                labels=PART_RIBS_LABELS,
                modality="CT",
                description="Rib segmentation (26 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            # Fast model (single model, lower resolution)
            "total_fast": ModelInfo(
                name="total_fast",
                task_id="Dataset297_TotalSegmentator_total_3mm_1559subj",
                url=totalseg_url("Dataset297_TotalSegmentator_total_3mm_1559subj", "v2.0.0-weights"),
                checksum="",
                labels=TOTAL_LABELS,
                modality="CT",
                description="Fast whole-body segmentation (3mm resolution)",
                default_preprocessing="ct_fast",
                default_postprocessing="default"
            ),
            # Fastest model (even lower resolution)
            "total_fastest": ModelInfo(
                name="total_fastest",
                task_id="Dataset298_TotalSegmentator_total_6mm_1559subj",
                url=totalseg_url("Dataset298_TotalSegmentator_total_6mm_1559subj", "v2.0.0-weights"),
                checksum="",
                labels=TOTAL_LABELS,
                modality="CT",
                description="Fastest whole-body segmentation (6mm resolution)",
                default_preprocessing="ct_fastest",
                default_postprocessing="default"
            ),
        }

        return TaskDefinition(
            name="total",
            models=models,
            pipeline_config={
                # Faithful to the official TotalSegmentator workflow: no
                # intensity clipping and no generic post-processing; resampling
                # to the model spacing happens inside the inference step.
                "default": {
                    "preprocessing": [],
                    "inference": {
                        "models": ["total_organs", "total_vertebrae", "total_cardiac",
                                   "total_muscles", "total_ribs"],
                        "ensemble_mode": "concatenate",
                    },
                    "postprocessing": [],
                },
                "fast": {
                    "preprocessing": [],
                    "inference": {"models": ["total_fast"]},
                    "postprocessing": []
                },
                "fastest": {
                    "preprocessing": [],
                    "inference": {"models": ["total_fastest"]},
                    "postprocessing": []
                },
                "roi": {
                    "preprocessing": [],
                    "inference": {
                        "type": "roi",
                        "low_res_model": "total_fast",
                        "roi_groups": {
                            "organs": {"model": "total_organs", "labels": list(PART_ORGANS_LABELS.keys())},
                            "vertebrae": {"model": "total_vertebrae", "labels": list(PART_VERTEBRAE_LABELS.keys())},
                            "cardiac": {"model": "total_cardiac", "labels": list(PART_CARDIAC_LABELS.keys())},
                            "muscles": {"model": "total_muscles", "labels": list(PART_MUSCLES_LABELS.keys())},
                            "ribs": {"model": "total_ribs", "labels": list(PART_RIBS_LABELS.keys())},
                        }
                    },
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "orientation": "any",
                "spacing": "any",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": TOTAL_LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """
        Return default processing pipeline.

        Delegates construction to PipelineBuilder so the mode-based
        pipeline_config (pre/inference/post sections) is interpreted in one
        place, including multi-model concat label mappings.

        Args:
            mode: Pipeline mode - "default", "fast", "fastest" or "roi"
        """
        return build_pipeline(cls.get_definition(), mode)


class TotalSegmentatorMRTask(BaseTask):
    """
    TotalSegmentator task for MR imaging.

    Segments 50 anatomical structures from MR images.

    Official workflow (map_tasks_config.py, v2.5.0-weights):
        - default: two part models 850 (organs) + 851 (muscles), concatenated
        - fast: single 3mm model 852
        - fastest: single 6mm model 853
    No MR-specific pre/post-processing is performed.

    Model Storage: ~/.nnunetsegmentator/models/total_mr/
    """

    name = "total_mr"
    description = "Whole-body MR segmentation with 50 anatomical structures"

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""

        models = {
            "total_mr": ModelInfo(
                name="total_mr",
                task_id="Dataset850_TotalSegMRI_part1_organs_1088subj",
                url=totalseg_url("Dataset850_TotalSegMRI_part1_organs_1088subj", "v2.5.0-weights"),
                checksum="",
                labels=MR_ORGANS_LABELS,
                modality="MR",
                description="MR organ segmentation (29 classes)",
                default_preprocessing="mr_standard",
                default_postprocessing="default"
            ),
            "total_mr_part2": ModelInfo(
                name="total_mr_part2",
                task_id="Dataset851_TotalSegMRI_part2_muscles_1088subj",
                url=totalseg_url("Dataset851_TotalSegMRI_part2_muscles_1088subj", "v2.5.0-weights"),
                checksum="",
                labels=MR_MUSCLES_LABELS,
                modality="MR",
                description="MR muscle segmentation (21 classes)",
                default_preprocessing="mr_standard",
                default_postprocessing="default"
            ),
            "total_mr_fast": ModelInfo(
                name="total_mr_fast",
                task_id="Dataset852_TotalSegMRI_total_3mm_1088subj",
                url=totalseg_url("Dataset852_TotalSegMRI_total_3mm_1088subj", "v2.5.0-weights"),
                checksum="",
                labels=MR_LABELS,
                modality="MR",
                description="Fast MR segmentation (3mm)",
                default_preprocessing="mr_fast",
                default_postprocessing="default"
            ),
            "total_mr_fastest": ModelInfo(
                name="total_mr_fastest",
                task_id="Dataset853_TotalSegMRI_total_6mm_1088subj",
                url=totalseg_url("Dataset853_TotalSegMRI_total_6mm_1088subj", "v2.5.0-weights"),
                checksum="",
                labels=MR_LABELS,
                modality="MR",
                description="Fastest MR segmentation (6mm)",
                default_preprocessing="mr_fastest",
                default_postprocessing="default"
            ),
        }

        return TaskDefinition(
            name="total_mr",
            models=models,
            pipeline_config={
                # Faithful to the official workflow: no preprocessing and no
                # post-processing; resampling happens inside the inference step.
                "default": {
                    "preprocessing": [],
                    "inference": {
                        "models": ["total_mr", "total_mr_part2"],
                        "ensemble_mode": "concatenate",
                    },
                    "postprocessing": []
                },
                "fast": {
                    "preprocessing": [],
                    "inference": {"models": ["total_mr_fast"]},
                    "postprocessing": []
                },
                "fastest": {
                    "preprocessing": [],
                    "inference": {"models": ["total_mr_fastest"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["MR"],
                "format": ["nifti", "dicom"],
                "orientation": "any",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": MR_LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """
        Return default processing pipeline.

        Delegates construction to PipelineBuilder; the "default" mode
        automatically becomes a multi_model_concat step with name-based
        label mappings (part models -> global MR_LABELS).

        Args:
            mode: Pipeline mode - "default", "fast" or "fastest"
        """
        return build_pipeline(cls.get_definition(), mode)


class LungVesselsTask(SingleModelTask):
    """
    Lung vessels and airways segmentation.

    Official workflow: robust crop to the lung fields (3mm total model),
    then segment airways/wall/arteries/veins within the cropped region.

    Segments:
    - Lung airways (trachea, bronchi)
    - Airways wall
    - Lung arteries
    - Lung veins
    """

    name = "lung_vessels"
    description = "Lung vessels and airways segmentation"

    TASK_ID = "Dataset117_lung_airways_arteries_veins_282subj"
    VERSION = "v2.5.0-weights"
    LABELS = LUNG_VESSELS_LABELS
    MODEL_DESCRIPTION = "Lung vessels and airways"
    CROP = ("total_fast", LUNG_LOBES)


class BodySegmentationTask(SingleModelTask):
    """
    Body segmentation for radiotherapy planning.

    Official workflow: plain inference followed by the official body
    postprocessing (keep largest body_trunc; drop body_extremities blobs
    below 50,000 mm3).

    Segments:
    - Body trunk (torso)
    - Body extremities (arms, legs)
    """

    name = "body"
    description = "Body segmentation for radiotherapy planning"

    TASK_ID = "Dataset299_body_1559subj"
    VERSION = "v2.0.0-weights"
    LABELS = BODY_LABELS
    MODEL_DESCRIPTION = "Body segmentation (1.5mm)"
    POST = BODY_POSTPROCESSING
    FAST = (
        "body_fast",
        "Dataset300_body_6mm_1559subj",
        "v2.0.0-weights",
        "Fast body segmentation (6mm)",
    )


class TissueTypesTask(BaseTask):
    """
    Tissue type segmentation for body composition analysis.

    Segments:
    - Subcutaneous fat
    - Torso fat (visceral)
    - Skeletal muscle
    - Intermuscular fat (tissue_4_types model)
    """

    name = "tissue_types"
    description = "Tissue type segmentation for body composition"

    LABELS_3 = TISSUE_TYPES_LABELS_3
    LABELS_4 = TISSUE_TYPES_LABELS_4

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        models = {
            "tissue_types": ModelInfo(
                name="tissue_types",
                task_id="Dataset481_tissue_1559subj",
                url=NON_DOWNLOADABLE_URL,
                checksum="",
                labels=cls.LABELS_3,
                modality="CT",
                description="Tissue type segmentation (3 classes)",
            ),
            "tissue_4_types": ModelInfo(
                name="tissue_4_types",
                task_id="Dataset485_tissue_4types_1559subj",
                url=NON_DOWNLOADABLE_URL,
                checksum="",
                labels=cls.LABELS_4,
                modality="CT",
                description="Tissue type segmentation (4 classes)",
            )
        }

        return TaskDefinition(
            name="tissue_types",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [],
                    "inference": {"models": ["tissue_types"]},
                    "postprocessing": []
                },
                "four_types": {
                    "preprocessing": [],
                    "inference": {"models": ["tissue_4_types"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS_3,
            }
        )

    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """Return default pipeline."""
        return build_pipeline(cls.get_definition(), mode)


class HeartChambersTask(SingleModelTask):
    """
    High-resolution heart chamber segmentation.

    Official workflow: robust crop around the heart (3mm total model,
    5mm addon), inference, then zero the prediction outside the dilated
    heart/aorta/inferior_vena_cava mask (remove_outside, 10mm dilation).

    Segments:
    - Myocardium
    - Left atrium
    - Left ventricle
    - Right atrium
    - Right ventricle
    - Aorta
    - Pulmonary artery
    """

    name = "heartchambers_highres"
    description = "High-resolution heart chamber segmentation"

    TASK_ID = "Dataset301_heart_highres_1559subj"
    URL = NON_DOWNLOADABLE_URL
    LABELS = HEART_CHAMBERS_LABELS
    MODEL_DESCRIPTION = "High-resolution heart chambers"
    CROP = ("total_fast", ["heart"], (5, 5, 5))
    POST = [restore_post({
        "labels": ["heart", "aorta", "inferior_vena_cava"],
        "dilation_mm": 10,
    })]


class CerebralBleedTask(SingleModelTask):
    """
    Cerebral hemorrhage segmentation.

    Official workflow: crop around the brain (6mm total model), then segment
    intracerebral hemorrhage within the cropped region.
    """

    name = "cerebral_bleed"
    description = "Cerebral hemorrhage segmentation"

    TASK_ID = "Dataset150_icb_v0"
    VERSION = "v2.0.0-weights"
    LABELS = CEREBRAL_BLEED_LABELS
    MODEL_DESCRIPTION = "Cerebral hemorrhage"
    CROP = ("total_fastest", ["brain"])


class LiverVesselsTask(SingleModelTask):
    """
    Liver vessels and tumor segmentation.

    Official workflow: crop around the liver (6mm total model, 20mm addon),
    then segment vessels and tumors within the cropped region.

    Segments:
    - Liver vessels (portal vein, hepatic veins)
    - Liver tumors
    """

    name = "liver_vessels"
    description = "Liver vessels and tumor segmentation"

    TASK_ID = "Dataset008_HepaticVessel"
    VERSION = "v2.4.0-weights"
    LABELS = LIVER_VESSELS_LABELS
    MODEL_DESCRIPTION = "Liver vessels and tumors"
    CROP = ("total_fastest", ["liver"], (20, 20, 20))


class LungNodulesTask(SingleModelTask):
    """
    Lung nodules segmentation.

    Official workflow: crop to the lung fields (6mm total model, 10mm addon),
    then segment nodules within the cropped region.
    """

    name = "lung_nodules"
    description = "Lung nodules segmentation"

    TASK_ID = "Dataset913_lung_nodules"
    VERSION = "v2.5.0-weights"
    LABELS = LUNG_NODULES_LABELS
    MODEL_DESCRIPTION = "Lung nodules"
    CROP = ("total_fastest", LUNG_LOBES, (10, 10, 10))


class VertebraeBodyTask(SingleModelTask):
    """
    Vertebrae body and intervertebral disc segmentation.

    Official workflow: plain inference (no crop, no postprocessing).
    """

    name = "vertebrae_body"
    description = "Vertebrae body segmentation"

    TASK_ID = "Dataset305_vertebrae_discs_1559subj"
    VERSION = "v2.5.0-weights"
    LABELS = VERTEBRAE_BODY_LABELS


class LiverLesionsTask(BaseTask):
    """
    Liver lesions segmentation.

    Official workflow: robust crop around the liver, then segment lesions
    within the cropped region. The CT model crops with the 3mm total model
    (10mm addon); the MR model crops with the 3mm MR total model (default
    addon).
    """

    name = "liver_lesions"
    description = "Liver lesions segmentation"

    LABELS = LIVER_LESIONS_LABELS

    @classmethod
    def _ct_steps(cls):
        return [
            crop_pre("total_fast", ["liver"], (10, 10, 10)),
            {"type": "nnunet_inference", "name": "inference",
             "params": {"model_name": "liver_lesions"}},
            restore_post(),
        ]

    @classmethod
    def _mr_steps(cls):
        return [
            crop_pre("total_mr_fast", ["liver"]),
            {"type": "nnunet_inference", "name": "inference",
             "params": {"model_name": "liver_lesions_mr"}},
            restore_post(),
        ]

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        models = {
            "liver_lesions": ModelInfo(
                name="liver_lesions",
                task_id="Dataset591_ct_liver_lesions_842subj",
                url=totalseg_url("Dataset591_ct_liver_lesions_842subj", "v2.5.0-weights"),
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Liver lesions (CT)",
            ),
            "liver_lesions_mr": ModelInfo(
                name="liver_lesions_mr",
                task_id="Dataset589_ct_mri_liver_lesions_750subj",
                url=totalseg_url("Dataset589_ct_mri_liver_lesions_750subj", "v2.5.0-weights"),
                checksum="",
                labels=cls.LABELS,
                modality="MR",
                description="Liver lesions (MR)",
            )
        }

        return TaskDefinition(
            name="liver_lesions",
            models=models,
            pipeline_config={
                "name": "liver_lesions_pipeline",
                "steps": cls._ct_steps(),
                # Per-model overrides (picked up by the orchestrator for the
                # per_model strategy); keys are model identifiers.
                "model_steps": {
                    "liver_lesions_mr": cls._mr_steps(),
                },
            },
            input_requirements={
                "modality": ["CT", "MR"],
                "format": ["nifti", "dicom"],
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """Return default pipeline (mode "mr" selects the MR workflow)."""
        definition = cls.get_definition()
        steps = definition.pipeline_config["model_steps"]["liver_lesions_mr"] \
            if mode == "mr" else definition.pipeline_config["steps"]
        return build_pipeline(
            definition,
            config={"name": "liver_lesions_pipeline", "steps": steps},
        )


class KidneyCystsTask(SingleModelTask):
    """
    Kidney cysts segmentation.

    Official workflow: crop around kidneys/liver/spleen/colon (6mm total
    model, 10mm addon), then segment cysts within the cropped region.
    """

    name = "kidney_cysts"
    description = "Kidney cysts segmentation"

    TASK_ID = "Dataset789_kidney_cyst_501subj"
    VERSION = "v2.5.0-weights"
    LABELS = KIDNEY_CYSTS_LABELS
    MODEL_DESCRIPTION = "Kidney cysts"
    CROP = (
        "total_fastest",
        ["kidney_left", "kidney_right", "liver", "spleen", "colon"],
        (10, 10, 10),
    )


class PleuralPericardEffusionTask(SingleModelTask):
    """
    Pleural and pericardial effusion segmentation.

    Official workflow: crop to the lung fields (6mm total model, 50mm addon),
    then segment effusions within the cropped region.
    """

    name = "pleural_pericard_effusion"
    description = "Pleural and pericardial effusion segmentation"

    TASK_ID = "Dataset315_thoraxCT"
    VERSION = "v2.0.0-weights"
    LABELS = PLEURAL_PERICARD_EFFUSION_LABELS
    MODEL_DESCRIPTION = "Pleural and pericardial effusion"
    CROP = ("total_fastest", LUNG_LOBES, (50, 50, 50))


class BodyMRTask(SingleModelTask):
    """
    Body segmentation for MR images.

    Official workflow: plain inference; the MR body task has no
    postprocessing (unlike its CT counterpart).
    """

    name = "body_mr"
    description = "Body segmentation for MR images"

    TASK_ID = "Dataset597_mri_body_139subj"
    VERSION = "v2.5.0-weights"
    LABELS = BODY_LABELS
    MODALITY = "MR"
    MODEL_DESCRIPTION = "Body segmentation (MR, 1.5mm)"
    FAST = (
        "body_mr_fast",
        "Dataset598_mri_body_6mm_139subj",
        "v2.5.0-weights",
        "Fast body segmentation (MR, 6mm)",
    )


class TissueTypesMRTask(SingleModelTask):
    """
    Tissue type segmentation for MR images.

    Segments subcutaneous fat, torso fat and skeletal muscle in MR images.
    Official workflow: plain inference.
    """

    name = "tissue_types_mr"
    description = "Tissue type segmentation for MR images"

    TASK_ID = "Dataset925_MRI_tissue_subset_903subj"
    URL = NON_DOWNLOADABLE_URL
    LABELS = TISSUE_TYPES_MR_LABELS
    MODALITY = "MR"
    MODEL_DESCRIPTION = "Tissue types (MR)"


class VertebraeMRTask(SingleModelTask):
    """
    MR vertebrae segmentation (sacrum, L5..C1).

    Official workflow: plain inference (no crop, no postprocessing).
    """

    name = "vertebrae_mr"
    description = "MR vertebrae segmentation (25 structures)"

    TASK_ID = "Dataset756_mri_vertebrae_1076subj"
    VERSION = "v2.5.0-weights"
    LABELS = VERTEBRAE_MR_LABELS
    MODALITY = "MR"
    MODEL_DESCRIPTION = "MR vertebrae segmentation"


class BreastsTask(SingleModelTask):
    """
    Breast segmentation (CT).

    Official workflow: plain inference (no crop, no postprocessing).
    """

    name = "breasts"
    description = "Breast segmentation"

    TASK_ID = "Dataset527_breasts_1559subj"
    VERSION = "v2.5.0-weights"
    LABELS = BREASTS_LABELS


class VentriclePartsTask(SingleModelTask):
    """
    Brain ventricle parts segmentation (CT).

    Official workflow: crop tightly around the brain (6mm total model,
    0mm addon), then segment the ventricle substructures.
    """

    name = "ventricle_parts"
    description = "Brain ventricle parts segmentation"

    TASK_ID = "Dataset552_ventricle_parts_38subj"
    VERSION = "v2.5.0-weights"
    LABELS = VENTRICLE_PARTS_LABELS
    CROP = ("total_fastest", ["brain"], (0, 0, 0))


class LiverSegmentsTask(SingleModelTask):
    """
    Liver segments segmentation (Couinaud 1-8, CT).

    Official workflow: crop around the liver (6mm total model, 10mm addon),
    then segment the liver segments.
    """

    name = "liver_segments"
    description = "Liver segments segmentation (Couinaud 1-8)"

    TASK_ID = "Dataset570_ct_liver_segments"
    VERSION = "v2.5.0-weights"
    LABELS = LIVER_SEGMENTS_LABELS
    MODEL_DESCRIPTION = "Liver segments segmentation (CT)"
    CROP = ("total_fastest", ["liver"], (10, 10, 10))


class LiverSegmentsMRTask(SingleModelTask):
    """
    Liver segments segmentation (Couinaud 1-8, MR).

    Official workflow: crop around the liver (3mm MR total model, 10mm
    addon), then segment the liver segments.
    """

    name = "liver_segments_mr"
    description = "Liver segments segmentation (MR, Couinaud 1-8)"

    TASK_ID = "Dataset576_mri_liver_segments_120subj"
    VERSION = "v2.5.0-weights"
    LABELS = LIVER_SEGMENTS_LABELS
    MODALITY = "MR"
    MODEL_DESCRIPTION = "Liver segments segmentation (MR)"
    CROP = ("total_mr_fast", ["liver"], (10, 10, 10))


class TrunkCavitiesTask(SingleModelTask):
    """
    Trunk cavities segmentation (CT).

    Official workflow: plain inference (no crop, no postprocessing).
    """

    name = "trunk_cavities"
    description = "Trunk cavities segmentation"

    TASK_ID = "Dataset343_mediastinum_1786subj"
    VERSION = "v2.5.0-weights"
    LABELS = TRUNK_CAVITIES_LABELS


class BrainAneurysmTask(SingleModelTask):
    """
    Brain aneurysm segmentation (TOF MR).

    Official workflow: plain inference (no crop, no postprocessing).
    Only works with TOF MRI images.
    """

    name = "brain_aneurysm"
    description = "Brain aneurysm segmentation (TOF MR)"

    TASK_ID = "Dataset615_MAXIMUS"
    VERSION = "v2.5.0-weights"
    LABELS = BRAIN_ANEURYSM_LABELS
    MODALITY = "MR"
    MODEL_DESCRIPTION = "Brain aneurysm segmentation (TOF MR only)"


class VertebraePPTask(SingleModelTask):
    """
    Vertebrae segmentation with posterior elements (C1..L5, CT).

    Official workflow: plain inference followed by the official
    ``postprocess_vertebrae_pp`` refinement (anatomical relabeling,
    connected-component filtering, volume thresholding and dilation).
    """

    name = "vertebrae_pp"
    description = "Vertebrae segmentation with posterior elements (C1-L5)"

    TASK_ID = "Dataset803_TotalSegmentator_vertebrae_inner_1559subj"
    VERSION = "v2.5.0-weights"
    LABELS = VERTEBRAE_PP_LABELS
    MODEL_DESCRIPTION = "Vertebrae segmentation with posterior elements"
    POST = [{"type": "vertebrae_pp", "name": "vertebrae_pp", "params": {}}]


class AbdominalMusclesTask(SingleModelTask):
    """
    Abdominal muscles segmentation (CT).

    Official workflow: crop to the body trunk (6mm body model, 5mm addon),
    then segment the muscle groups. The model only segments within T4-L4;
    training annotations were restricted to that region.
    """

    name = "abdominal_muscles"
    description = "Abdominal muscles segmentation"

    TASK_ID = "Dataset952_abdominal_muscles_167subj"
    VERSION = "v2.5.0-weights"
    LABELS = ABDOMINAL_MUSCLES_LABELS
    CROP = ("body_fast", ["body_trunc"], (5, 5, 5))


class CraniofacialStructuresTask(SingleModelTask):
    """
    Craniofacial structures segmentation (CT).

    Official workflow: crop around the skull (6mm total model, 20mm addon),
    then segment the craniofacial structures. Also serves as the crop model
    for the "teeth" task.
    """

    name = "craniofacial_structures"
    description = "Craniofacial structures segmentation"

    TASK_ID = "Dataset115_mandible"
    VERSION = "v2.5.0-weights"
    LABELS = CRANIOFACIAL_STRUCTURES_LABELS
    CROP = ("total_fastest", ["skull"], (20, 20, 20))


class TeethTask(SingleModelTask):
    """
    Teeth and jaw segmentation (77 structures, CT/CBCT).

    Official workflow: crop around the lower/upper teeth using the
    "craniofacial_structures" model (10mm addon; mandible excluded from the
    crop mask to keep the FOV narrow as trained on CBCT data), then segment
    teeth, jaws, canals and restorations.
    """

    name = "teeth"
    description = "Teeth and jaw segmentation (77 structures)"

    TASK_ID = "Dataset113_ToothFairy3"
    VERSION = "v2.5.0-weights"
    LABELS = TEETH_LABELS
    CROP = ("craniofacial_structures", ["teeth_lower", "teeth_upper"], (10, 10, 10))