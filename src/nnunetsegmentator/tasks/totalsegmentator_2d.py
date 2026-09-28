"""
TotalSegmentator 2D Task - Fast 2D Projection-Based Segmentation

This module implements the TotalSegmentator 2D (TS2D) task for rapid anatomical
structure segmentation using 2D projections of 3D CT scans.

Model Storage:
    - Default path: ~/.nnunetsegmentator/models/ts2d/
    - Model weights are downloaded on first use

Model Components (each model has 5 sub-models):
    - cardiac: Heart and vessel structures
    - muscles: Muscle and bone structures
    - organs: Organ segmentation
    - ribs: Rib segmentation
    - vertebrae: Vertebrae segmentation

Performance:
    - Inference time: 0.5-0.9 seconds (vs 43-146 seconds for 3D)
    - Best for bone structures (DSC ~0.90)
"""

from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ._helpers import build_pipeline
from ._sources import ts2d_url
from .base import BaseTask
import logging

logger = logging.getLogger(__name__)


# Label mappings for TS2D (117 classes - same as TotalSegmentator v2)
TS2D_LABELS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel",
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right", 25: "sacrum",
    26: "vertebrae_S1", 27: "vertebrae_L5", 28: "vertebrae_L4",
    29: "vertebrae_L3", 30: "vertebrae_L2", 31: "vertebrae_L1",
    32: "vertebrae_T12", 33: "vertebrae_T11", 34: "vertebrae_T10",
    35: "vertebrae_T9", 36: "vertebrae_T8", 37: "vertebrae_T7",
    38: "vertebrae_T6", 39: "vertebrae_T5", 40: "vertebrae_T4",
    41: "vertebrae_T3", 42: "vertebrae_T2", 43: "vertebrae_T1",
    44: "vertebrae_C7", 45: "vertebrae_C6", 46: "vertebrae_C5",
    47: "vertebrae_C4", 48: "vertebrae_C3", 49: "vertebrae_C2",
    50: "vertebrae_C1", 51: "heart", 52: "aorta", 53: "pulmonary_vein",
    54: "brachiocephalic_trunk", 55: "subclavian_artery_right",
    56: "subclavian_artery_left", 57: "common_carotid_artery_right",
    58: "common_carotid_artery_left", 59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right", 61: "atrial_appendage_left",
    62: "superior_vena_cava", 63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein", 65: "iliac_artery_left",
    66: "iliac_artery_right", 67: "iliac_vena_left", 68: "iliac_vena_right",
    69: "humerus_left", 70: "humerus_right", 71: "scapula_left",
    72: "scapula_right", 73: "clavicula_left", 74: "clavicula_right",
    75: "femur_left", 76: "femur_right", 77: "hip_left", 78: "hip_right",
    79: "spinal_cord", 80: "gluteus_maximus_left", 81: "gluteus_maximus_right",
    82: "gluteus_medius_left", 83: "gluteus_medius_right",
    84: "gluteus_minimus_left", 85: "gluteus_minimus_right",
    86: "autochthon_left", 87: "autochthon_right", 88: "iliopsoas_left",
    89: "iliopsoas_right", 90: "brain", 91: "skull", 92: "rib_left_1",
    93: "rib_left_2", 94: "rib_left_3", 95: "rib_left_4",
    96: "rib_left_5", 97: "rib_left_6", 98: "rib_left_7",
    99: "rib_left_8", 100: "rib_left_9", 101: "rib_left_10",
    102: "rib_left_11", 103: "rib_left_12", 104: "rib_right_1",
    105: "rib_right_2", 106: "rib_right_3", 107: "rib_right_4",
    108: "rib_right_5", 109: "rib_right_6", 110: "rib_right_7",
    111: "rib_right_8", 112: "rib_right_9", 113: "rib_right_10",
    114: "rib_right_11", 115: "rib_right_12", 116: "sternum",
    117: "costal_cartilages"
}

# 5-part model decomposition for TS2D (same grouping as 3D)
TS2D_PART_ORGANS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel",
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right"
}

TS2D_PART_VERTEBRAE = {
    1: "sacrum", 2: "vertebrae_S1", 3: "vertebrae_L5", 4: "vertebrae_L4",
    5: "vertebrae_L3", 6: "vertebrae_L2", 7: "vertebrae_L1", 8: "vertebrae_T12",
    9: "vertebrae_T11", 10: "vertebrae_T10", 11: "vertebrae_T9", 12: "vertebrae_T8",
    13: "vertebrae_T7", 14: "vertebrae_T6", 15: "vertebrae_T5", 16: "vertebrae_T4",
    17: "vertebrae_T3", 18: "vertebrae_T2", 19: "vertebrae_T1", 20: "vertebrae_C7",
    21: "vertebrae_C6", 22: "vertebrae_C5", 23: "vertebrae_C4", 24: "vertebrae_C3",
    25: "vertebrae_C2", 26: "vertebrae_C1"
}

TS2D_PART_CARDIAC = {
    1: "heart", 2: "aorta", 3: "pulmonary_vein", 4: "brachiocephalic_trunk",
    5: "subclavian_artery_right", 6: "subclavian_artery_left",
    7: "common_carotid_artery_right", 8: "common_carotid_artery_left",
    9: "brachiocephalic_vein_left", 10: "brachiocephalic_vein_right",
    11: "atrial_appendage_left", 12: "superior_vena_cava", 13: "inferior_vena_cava",
    14: "portal_vein_and_splenic_vein", 15: "iliac_artery_left",
    16: "iliac_artery_right", 17: "iliac_vena_left", 18: "iliac_vena_right"
}

TS2D_PART_MUSCLES = {
    1: "humerus_left", 2: "humerus_right", 3: "scapula_left", 4: "scapula_right",
    5: "clavicula_left", 6: "clavicula_right", 7: "femur_left", 8: "femur_right",
    9: "hip_left", 10: "hip_right", 11: "spinal_cord", 12: "gluteus_maximus_left",
    13: "gluteus_maximus_right", 14: "gluteus_medius_left", 15: "gluteus_medius_right",
    16: "gluteus_minimus_left", 17: "gluteus_minimus_right", 18: "autochthon_left",
    19: "autochthon_right", 20: "iliopsoas_left", 21: "iliopsoas_right",
    22: "brain", 23: "skull"
}

TS2D_PART_RIBS = {
    1: "rib_left_1", 2: "rib_left_2", 3: "rib_left_3", 4: "rib_left_4",
    5: "rib_left_5", 6: "rib_left_6", 7: "rib_left_7", 8: "rib_left_8",
    9: "rib_left_9", 10: "rib_left_10", 11: "rib_left_11", 12: "rib_left_12",
    13: "rib_right_1", 14: "rib_right_2", 15: "rib_right_3", 16: "rib_right_4",
    17: "rib_right_5", 18: "rib_right_6", 19: "rib_right_7", 20: "rib_right_8",
    21: "rib_right_9", 22: "rib_right_10", 23: "rib_right_11", 24: "rib_right_12",
    25: "sternum", 26: "costal_cartilages"
}


class TS2DTask(BaseTask):
    """
    TotalSegmentator 2D task for rapid anatomical structure segmentation.
    
    Uses 2D projections of 3D CT scans for fast inference (< 1 second)
    while maintaining good accuracy for most anatomical structures.

    Model Storage: ~/.nnunetsegmentator/models/ts2d/

    Features:
    - Coronal projection (MIP + AIP) for 2D U-Net input
    - 5-part model strategy for efficient inference
    - 117 anatomical structures (TotalSegmentator v2 labels)
    - Inference time: 0.5-0.9 seconds (vs 43-146 seconds for 3D)
    
    Best for:
    - Bone structures (DSC ~0.90)
    - Large soft-tissue structures
    - Rapid screening and analysis
    """
    
    name = "ts2d"
    description = "TotalSegmentator 2D: Fast projection-based segmentation (117 structures, <1s inference)"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""
        
        # TS2D v2 default weights: five anatomical group models on Zenodo
        # (key ``ts2d-v2-ep4000b2``, record 16985939), one archive per group.
        key = "ts2d-v2-ep4000b2"

        models = {
            # 5-part model strategy for TS2D v2
            "ts2d_v2_organs": ModelInfo(
                name="ts2d_v2_organs",
                task_id="Dataset900_ts2d_v2_organs",
                url=ts2d_url(key, "organs"),
                checksum="",
                labels=TS2D_PART_ORGANS,
                modality="CT",
                description="TS2D v2: Organ segmentation (24 classes)",
                default_preprocessing="projection_ct",
                default_postprocessing="default"
            ),
            "ts2d_v2_vertebrae": ModelInfo(
                name="ts2d_v2_vertebrae",
                task_id="Dataset901_ts2d_v2_vertebrae",
                url=ts2d_url(key, "vertebrae"),
                checksum="",
                labels=TS2D_PART_VERTEBRAE,
                modality="CT",
                description="TS2D v2: Vertebrae segmentation (26 classes)",
                default_preprocessing="projection_ct",
                default_postprocessing="default"
            ),
            "ts2d_v2_cardiac": ModelInfo(
                name="ts2d_v2_cardiac",
                task_id="Dataset902_ts2d_v2_cardiac",
                url=ts2d_url(key, "cardiac"),
                checksum="",
                labels=TS2D_PART_CARDIAC,
                modality="CT",
                description="TS2D v2: Cardiac segmentation (18 classes)",
                default_preprocessing="projection_ct",
                default_postprocessing="default"
            ),
            "ts2d_v2_muscles": ModelInfo(
                name="ts2d_v2_muscles",
                task_id="Dataset903_ts2d_v2_muscles",
                url=ts2d_url(key, "muscles"),
                checksum="",
                labels=TS2D_PART_MUSCLES,
                modality="CT",
                description="TS2D v2: Muscle/bone segmentation (23 classes)",
                default_preprocessing="projection_ct",
                default_postprocessing="default"
            ),
            "ts2d_v2_ribs": ModelInfo(
                name="ts2d_v2_ribs",
                task_id="Dataset904_ts2d_v2_ribs",
                url=ts2d_url(key, "ribs"),
                checksum="",
                labels=TS2D_PART_RIBS,
                modality="CT",
                description="TS2D v2: Rib segmentation (26 classes)",
                default_preprocessing="projection_ct",
                default_postprocessing="default"
            ),
        }
        
        return TaskDefinition(
            name="ts2d",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "projection", "method": "coronal", "channels": ["mip", "aip"]},
                        # Intensity normalization is nnUNetv2-internal (plans scheme);
                        # the upstream pipeline does not clip intensities.
                    ],
                    "inference": {
                        "models": ["ts2d_v2_organs", "ts2d_v2_vertebrae", "ts2d_v2_cardiac",
                                   "ts2d_v2_muscles", "ts2d_v2_ribs"],
                        "ensemble_mode": "concatenate",
                    },
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "orientation": "any",
                "dimensions": 3,  # 3D input that will be projected to 2D
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": TS2D_LABELS,
                "dimensions": 2,  # 2D output
            }
        )
    
    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """
        Return default processing pipeline.

        Delegates construction to PipelineBuilder so the mode-based
        pipeline_config (pre/inference/post sections) is interpreted in one
        place, including the multi-model concat label mappings derived from
        each group model's local labels versus the task's global labels.

        Args:
            mode: Pipeline mode (currently only "default" supported)
        """
        return build_pipeline(cls.get_definition(), mode)


class TSXRTask(BaseTask):
    """
    TotalSegmentator XR (TSXR) task for X-ray image segmentation.
    
    Segments 117 anatomical structures directly from 2D X-ray images
    using models trained on synthetic projections.

    Model Storage: ~/.nnunetsegmentator/models/tsxr/
    """
    
    name = "tsxr"
    description = "TotalSegmentator XR: X-ray image segmentation (117 structures)"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""
        
        # TSXR v2 weights: five anatomical group models on Zenodo
        # (key ``tsxr-v2-ep1000b2``, record 17052912), same group split as TS2D.
        key = "tsxr-v2-ep1000b2"

        models = {
            "tsxr_v2_organs": ModelInfo(
                name="tsxr_v2_organs",
                task_id="Dataset910_tsxr_v2_organs",
                url=ts2d_url(key, "organs"),
                checksum="",
                labels=TS2D_PART_ORGANS,
                modality="XR",  # X-ray
                description="TSXR v2: Organ segmentation (24 classes)",
                default_preprocessing="xr_standard",
                default_postprocessing="default"
            ),
            "tsxr_v2_vertebrae": ModelInfo(
                name="tsxr_v2_vertebrae",
                task_id="Dataset911_tsxr_v2_vertebrae",
                url=ts2d_url(key, "vertebrae"),
                checksum="",
                labels=TS2D_PART_VERTEBRAE,
                modality="XR",  # X-ray
                description="TSXR v2: Vertebrae segmentation (26 classes)",
                default_preprocessing="xr_standard",
                default_postprocessing="default"
            ),
            "tsxr_v2_cardiac": ModelInfo(
                name="tsxr_v2_cardiac",
                task_id="Dataset912_tsxr_v2_cardiac",
                url=ts2d_url(key, "cardiac"),
                checksum="",
                labels=TS2D_PART_CARDIAC,
                modality="XR",  # X-ray
                description="TSXR v2: Cardiac segmentation (18 classes)",
                default_preprocessing="xr_standard",
                default_postprocessing="default"
            ),
            "tsxr_v2_muscles": ModelInfo(
                name="tsxr_v2_muscles",
                task_id="Dataset914_tsxr_v2_muscles",
                url=ts2d_url(key, "muscles"),
                checksum="",
                labels=TS2D_PART_MUSCLES,
                modality="XR",  # X-ray
                description="TSXR v2: Muscle/bone segmentation (23 classes)",
                default_preprocessing="xr_standard",
                default_postprocessing="default"
            ),
            "tsxr_v2_ribs": ModelInfo(
                name="tsxr_v2_ribs",
                task_id="Dataset915_tsxr_v2_ribs",
                url=ts2d_url(key, "ribs"),
                checksum="",
                labels=TS2D_PART_RIBS,
                modality="XR",  # X-ray
                description="TSXR v2: Rib segmentation (26 classes)",
                default_preprocessing="xr_standard",
                default_postprocessing="default"
            ),
        }
        
        return TaskDefinition(
            name="tsxr",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        # Intensity normalization is nnUNetv2-internal (plans scheme).
                    ],
                    "inference": {
                        "models": ["tsxr_v2_organs", "tsxr_v2_vertebrae", "tsxr_v2_cardiac",
                                   "tsxr_v2_muscles", "tsxr_v2_ribs"],
                        "ensemble_mode": "concatenate",
                    },
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["XR"],  # X-ray
                "format": ["nifti", "dicom", "png", "jpg"],
                "orientation": "any",
                "dimensions": 2,  # 2D input
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": TS2D_LABELS,
                "dimensions": 2,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """
        Return default processing pipeline.

        Delegates construction to PipelineBuilder so the five group models are
        concatenated with per-model local->global label mappings derived from
        their local label tables versus the task's global labels.
        """
        return build_pipeline(cls.get_definition(), mode)
