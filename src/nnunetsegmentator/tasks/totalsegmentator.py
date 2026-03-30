"""
TotalSegmentator Task - Comprehensive Whole-Body Segmentation

This module implements the TotalSegmentator task with 104+ anatomical structures
for CT and MR imaging.

Repository: https://github.com/wasserth/TotalSegmentator
Description: Tool for segmentation of most major anatomical structures in any CT or MR image.
            Trained on a wide range of different CT and MR images.

Model Download:
    - Models are automatically downloaded when running TotalSegmentator
    - Default download location: ~/.totalsegmentator/nnunet/results
    - Can also be downloaded from Zenodo
    
Model Storage:
    - Default path: ~/.nnunetsegmentator/models/total/
    - Model weights are downloaded automatically on first use
    
Available Models:
    - total: 117 classes (CT)
    - total_mr: 50 classes (MR)
    - total_fast: 3mm resolution (CT)
    - total_fastest: 6mm resolution (CT)
    - Plus many specialized models (lung_vessels, body, vertebrae_mr, etc.)
    
License: Apache-2.0
Citation: Wasserthal et al., Radiology: Artificial Intelligence (2023). https://doi.org/10.1148/ryai.230024
"""

from typing import Dict, List, Optional
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import ResampleStep, ClipIntensityStep, NormalizeStep
from ..pipeline.steps.inference import (
    nnUNetInferenceStep,
    EnsembleInferenceStep,
    MultiModelConcatStep,
)
from ..pipeline.steps.postprocessing import LargestComponentStep, FillHolesStep
from ..pipeline.steps.roi import ROIProcessingStep
from .base import BaseTask
import logging

logger = logging.getLogger(__name__)


# Label mappings for TotalSegmentator (117 classes)
TOTAL_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
    10: "lung_upper_lobe_left",
    11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right",
    13: "lung_middle_lobe_right",
    14: "lung_lower_lobe_right",
    15: "esophagus",
    16: "trachea",
    17: "thyroid_gland",
    18: "small_bowel",
    19: "duodenum",
    20: "colon",
    21: "urinary_bladder",
    22: "prostate",
    23: "kidney_cyst_left",
    24: "kidney_cyst_right",
    25: "sacrum",
    26: "vertebrae_S1",
    27: "vertebrae_L5",
    28: "vertebrae_L4",
    29: "vertebrae_L3",
    30: "vertebrae_L2",
    31: "vertebrae_L1",
    32: "vertebrae_T12",
    33: "vertebrae_T11",
    34: "vertebrae_T10",
    35: "vertebrae_T9",
    36: "vertebrae_T8",
    37: "vertebrae_T7",
    38: "vertebrae_T6",
    39: "vertebrae_T5",
    40: "vertebrae_T4",
    41: "vertebrae_T3",
    42: "vertebrae_T2",
    43: "vertebrae_T1",
    44: "vertebrae_C7",
    45: "vertebrae_C6",
    46: "vertebrae_C5",
    47: "vertebrae_C4",
    48: "vertebrae_C3",
    49: "vertebrae_C2",
    50: "vertebrae_C1",
    51: "heart",
    52: "aorta",
    53: "pulmonary_vein",
    54: "brachiocephalic_trunk",
    55: "subclavian_artery_right",
    56: "subclavian_artery_left",
    57: "common_carotid_artery_right",
    58: "common_carotid_artery_left",
    59: "brachiocephalic_vein_left",
    60: "brachiocephalic_vein_right",
    61: "atrial_appendage_left",
    62: "superior_vena_cava",
    63: "inferior_vena_cava",
    64: "portal_vein_and_splenic_vein",
    65: "iliac_artery_left",
    66: "iliac_artery_right",
    67: "iliac_vena_left",
    68: "iliac_vena_right",
    69: "humerus_left",
    70: "humerus_right",
    71: "scapula_left",
    72: "scapula_right",
    73: "clavicula_left",
    74: "clavicula_right",
    75: "femur_left",
    76: "femur_right",
    77: "hip_left",
    78: "hip_right",
    79: "spinal_cord",
    80: "gluteus_maximus_left",
    81: "gluteus_maximus_right",
    82: "gluteus_medius_left",
    83: "gluteus_medius_right",
    84: "gluteus_minimus_left",
    85: "gluteus_minimus_right",
    86: "autochthon_left",
    87: "autochthon_right",
    88: "iliopsoas_left",
    89: "iliopsoas_right",
    90: "brain",
    91: "skull",
    92: "rib_left_1",
    93: "rib_left_2",
    94: "rib_left_3",
    95: "rib_left_4",
    96: "rib_left_5",
    97: "rib_left_6",
    98: "rib_left_7",
    99: "rib_left_8",
    100: "rib_left_9",
    101: "rib_left_10",
    102: "rib_left_11",
    103: "rib_left_12",
    104: "rib_right_1",
    105: "rib_right_2",
    106: "rib_right_3",
    107: "rib_right_4",
    108: "rib_right_5",
    109: "rib_right_6",
    110: "rib_right_7",
    111: "rib_right_8",
    112: "rib_right_9",
    113: "rib_right_10",
    114: "rib_right_11",
    115: "rib_right_12",
    116: "sternum",
    117: "costal_cartilages"
}

# 5-part model decomposition for efficient inference
PART_ORGANS_LABELS = {
    1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
    5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
    9: "adrenal_gland_left", 10: "lung_upper_lobe_left", 11: "lung_lower_lobe_left",
    12: "lung_upper_lobe_right", 13: "lung_middle_lobe_right", 14: "lung_lower_lobe_right",
    15: "esophagus", 16: "trachea", 17: "thyroid_gland", 18: "small_bowel",
    19: "duodenum", 20: "colon", 21: "urinary_bladder", 22: "prostate",
    23: "kidney_cyst_left", 24: "kidney_cyst_right"
}

PART_VERTEBRAE_LABELS = {
    1: "sacrum", 2: "vertebrae_S1", 3: "vertebrae_L5", 4: "vertebrae_L4",
    5: "vertebrae_L3", 6: "vertebrae_L2", 7: "vertebrae_L1", 8: "vertebrae_T12",
    9: "vertebrae_T11", 10: "vertebrae_T10", 11: "vertebrae_T9", 12: "vertebrae_T8",
    13: "vertebrae_T7", 14: "vertebrae_T6", 15: "vertebrae_T5", 16: "vertebrae_T4",
    17: "vertebrae_T3", 18: "vertebrae_T2", 19: "vertebrae_T1", 20: "vertebrae_C7",
    21: "vertebrae_C6", 22: "vertebrae_C5", 23: "vertebrae_C4", 24: "vertebrae_C3",
    25: "vertebrae_C2", 26: "vertebrae_C1"
}

PART_CARDIAC_LABELS = {
    1: "heart", 2: "aorta", 3: "pulmonary_vein", 4: "brachiocephalic_trunk",
    5: "subclavian_artery_right", 6: "subclavian_artery_left",
    7: "common_carotid_artery_right", 8: "common_carotid_artery_left",
    9: "brachiocephalic_vein_left", 10: "brachiocephalic_vein_right",
    11: "atrial_appendage_left", 12: "superior_vena_cava", 13: "inferior_vena_cava",
    14: "portal_vein_and_splenic_vein", 15: "iliac_artery_left",
    16: "iliac_artery_right", 17: "iliac_vena_left", 18: "iliac_vena_right"
}

PART_MUSCLES_LABELS = {
    1: "humerus_left", 2: "humerus_right", 3: "scapula_left", 4: "scapula_right",
    5: "clavicula_left", 6: "clavicula_right", 7: "femur_left", 8: "femur_right",
    9: "hip_left", 10: "hip_right", 11: "spinal_cord", 12: "gluteus_maximus_left",
    13: "gluteus_maximus_right", 14: "gluteus_medius_left", 15: "gluteus_medius_right",
    16: "gluteus_minimus_left", 17: "gluteus_minimus_right", 18: "autochthon_left",
    19: "autochthon_right", 20: "iliopsoas_left", 21: "iliopsoas_right",
    22: "brain", 23: "skull"
}

PART_RIBS_LABELS = {
    1: "rib_left_1", 2: "rib_left_2", 3: "rib_left_3", 4: "rib_left_4",
    5: "rib_left_5", 6: "rib_left_6", 7: "rib_left_7", 8: "rib_left_8",
    9: "rib_left_9", 10: "rib_left_10", 11: "rib_left_11", 12: "rib_left_12",
    13: "rib_right_1", 14: "rib_right_2", 15: "rib_right_3", 16: "rib_right_4",
    17: "rib_right_5", 18: "rib_right_6", 19: "rib_right_7", 20: "rib_right_8",
    21: "rib_right_9", 22: "rib_right_10", 23: "rib_right_11", 24: "rib_right_12",
    25: "sternum", 26: "costal_cartilages"
}


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
    
    Repository: https://github.com/wasserth/TotalSegmentator
    Model Download: Automatic download on first use
    Model Path: ~/.nnunetsegmentator/models/total/
    """
    
    name = "total"
    description = "Comprehensive whole-body CT segmentation with 117 anatomical structures"
    
    # Repository and model information
    REPO_URL = "https://github.com/wasserth/TotalSegmentator"
    MODEL_DOWNLOAD_METHOD = "automatic"
    LICENSE = "Apache-2.0"
    DOI = "10.1148/ryai.230024"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""
        
        # Model URLs (these would be actual URLs in production)
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            # 5-part model strategy for full resolution
            "total_organs": ModelInfo(
                name="total_organs",
                task_id="Dataset291_total_organs",
                url=f"{base_url}/total_organs.zip",
                checksum="",
                labels=PART_ORGANS_LABELS,
                modality="CT",
                description="Organ segmentation (24 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="organ_postprocess"
            ),
            "total_vertebrae": ModelInfo(
                name="total_vertebrae",
                task_id="Dataset292_total_vertebrae",
                url=f"{base_url}/total_vertebrae.zip",
                checksum="",
                labels=PART_VERTEBRAE_LABELS,
                modality="CT",
                description="Vertebrae segmentation (26 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="vertebrae_postprocess"
            ),
            "total_cardiac": ModelInfo(
                name="total_cardiac",
                task_id="Dataset293_total_cardiac",
                url=f"{base_url}/total_cardiac.zip",
                checksum="",
                labels=PART_CARDIAC_LABELS,
                modality="CT",
                description="Cardiac and vessel segmentation (18 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="cardiac_postprocess"
            ),
            "total_muscles": ModelInfo(
                name="total_muscles",
                task_id="Dataset294_total_muscles",
                url=f"{base_url}/total_muscles.zip",
                checksum="",
                labels=PART_MUSCLES_LABELS,
                modality="CT",
                description="Muscle and bone segmentation (23 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="muscle_postprocess"
            ),
            "total_ribs": ModelInfo(
                name="total_ribs",
                task_id="Dataset295_total_ribs",
                url=f"{base_url}/total_ribs.zip",
                checksum="",
                labels=PART_RIBS_LABELS,
                modality="CT",
                description="Rib segmentation (26 classes)",
                default_preprocessing="ct_standard",
                default_postprocessing="rib_postprocess"
            ),
            # Fast model (single model, lower resolution)
            "total_fast": ModelInfo(
                name="total_fast",
                task_id="Dataset297_total_fast",
                url=f"{base_url}/total_fast.zip",
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
                task_id="Dataset298_total_fastest",
                url=f"{base_url}/total_fastest.zip",
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
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                    ],
                    "inference": {
                        "models": ["total_organs", "total_vertebrae", "total_cardiac", 
                                   "total_muscles", "total_ribs"],
                        "ensemble_mode": "concatenate",
                    },
                    "postprocessing": [
                        {"type": "largest_component", "labels": ["liver", "spleen"]},
                        {"type": "fill_holes", "labels": ["liver", "kidney_right", "kidney_left"]},
                    ]
                },
                "fast": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [3.0, 3.0, 3.0]},
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                    ],
                    "inference": {
                        "models": ["total_fast"],
                    },
                    "postprocessing": []
                },
                "fastest": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [6.0, 6.0, 6.0]},
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                    ],
                    "inference": {
                        "models": ["total_fastest"],
                    },
                    "postprocessing": []
                },
                "roi": {
                    "preprocessing": [
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                    ],
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
                    "postprocessing": [
                        {"type": "largest_component", "labels": ["liver", "spleen"]},
                        {"type": "fill_holes", "labels": ["liver", "kidney_right", "kidney_left"]},
                    ]
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
        
        Args:
            mode: Pipeline mode - "default", "fast", or "fastest"
        """
        definition = cls.get_definition()
        config = definition.pipeline_config.get(mode, definition.pipeline_config["default"])
        
        # Build preprocessing pipeline
        pipeline = Pipeline()
        
        for step_config in config["preprocessing"]:
            step_type = step_config["type"]
            if step_type == "resample":
                pipeline = pipeline | ResampleStep(
                    name="resample",
                    config={"spacing": step_config["spacing"]}
                )
            elif step_type == "clip_intensity":
                pipeline = pipeline | ClipIntensityStep(
                    name="clip_intensity",
                    config={
                        "lower": step_config.get("min"),
                        "upper": step_config.get("max")
                    }
                )
        
        # Add inference step
        inference_config = config["inference"]
        
        if inference_config.get("type") == "roi":
            # ROI processing step
            pipeline = pipeline | ROIProcessingStep(
                name="total_segmentator_roi",
                config={
                    "low_res_model": inference_config["low_res_model"],
                    "roi_groups": inference_config["roi_groups"],
                    "crop_margin": [20.0, 20.0, 20.0],  # Could be configurable
                    "low_res_spacing": [3.0, 3.0, 3.0]
                }
            )
        elif len(inference_config["models"]) > 1:
            # Multi-model ensemble
            pipeline = pipeline | MultiModelConcatStep(
                name="ensemble_inference",
                config={
                    "model_names": inference_config["models"],
                    "label_mappings": [],
                }
            )
        else:
            # Single model
            pipeline = pipeline | nnUNetInferenceStep(
                name="inference",
                config={"model_name": inference_config["models"][0]}
            )
        
        # Add postprocessing steps
        for step_config in config.get("postprocessing", []):
            step_type = step_config["type"]
            if step_type == "largest_component":
                pipeline = pipeline | LargestComponentStep(
                    name="largest_component",
                    config={"labels": step_config.get("labels")}
                )
            elif step_type == "fill_holes":
                pipeline = pipeline | FillHolesStep(
                    name="fill_holes",
                    config={"labels": step_config.get("labels")}
                )
        
        return pipeline


class TotalSegmentatorMRTask(BaseTask):
    """
    TotalSegmentator task for MR imaging.
    
    Segments 50 anatomical structures from MR images.
    
    Repository: https://github.com/wasserth/TotalSegmentator
    Model Download: Automatic download on first use
    Model Path: ~/.nnunetsegmentator/models/total_mr/
    """
    
    name = "total_mr"
    description = "Whole-body MR segmentation with 50 anatomical structures"
    
    # Repository and model information
    REPO_URL = "https://github.com/wasserth/TotalSegmentator"
    MODEL_DOWNLOAD_METHOD = "automatic"
    LICENSE = "Apache-2.0"
    DOI = "10.1148/ryai.230024"
    
    # MR label mapping (50 classes)
    MR_LABELS = {
        1: "spleen", 2: "kidney_right", 3: "kidney_left", 4: "gallbladder",
        5: "liver", 6: "stomach", 7: "pancreas", 8: "adrenal_gland_right",
        9: "adrenal_gland_left", 10: "lung_left", 11: "lung_right",
        12: "esophagus", 13: "small_bowel", 14: "duodenum", 15: "colon",
        16: "urinary_bladder", 17: "prostate", 18: "sacrum", 19: "vertebrae",
        20: "intervertebral_discs", 21: "spinal_cord", 22: "heart", 23: "aorta",
        24: "inferior_vena_cava", 25: "portal_vein_and_splenic_vein",
        26: "iliac_artery_left", 27: "iliac_artery_right", 28: "iliac_vena_left",
        29: "iliac_vena_right", 30: "humerus_left", 31: "humerus_right",
        32: "scapula_left", 33: "scapula_right", 34: "clavicula_left",
        35: "clavicula_right", 36: "femur_left", 37: "femur_right",
        38: "hip_left", 39: "hip_right", 40: "gluteus_maximus_left",
        41: "gluteus_maximus_right", 42: "gluteus_medius_left",
        43: "gluteus_medius_right", 44: "gluteus_minimus_left",
        45: "gluteus_minimus_right", 46: "autochthon_left", 47: "autochthon_right",
        48: "iliopsoas_left", 49: "iliopsoas_right", 50: "brain"
    }
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""
        
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "total_mr_organs": ModelInfo(
                name="total_mr_organs",
                task_id="Dataset850_total_mr_organs",
                url=f"{base_url}/total_mr_organs.zip",
                checksum="",
                labels=cls.MR_LABELS,
                modality="MR",
                description="MR organ segmentation",
                default_preprocessing="mr_standard",
                default_postprocessing="default"
            ),
            "total_mr_fast": ModelInfo(
                name="total_mr_fast",
                task_id="Dataset852_total_mr_fast",
                url=f"{base_url}/total_mr_fast.zip",
                checksum="",
                labels=cls.MR_LABELS,
                modality="MR",
                description="Fast MR segmentation (3mm)",
                default_preprocessing="mr_fast",
                default_postprocessing="default"
            ),
        }
        
        return TaskDefinition(
            name="total_mr",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                        {"type": "normalize", "method": "zscore"},
                    ],
                    "inference": {"models": ["total_mr_organs"]},
                    "postprocessing": []
                },
                "fast": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [3.0, 3.0, 3.0]},
                        {"type": "normalize", "method": "zscore"},
                    ],
                    "inference": {"models": ["total_mr_fast"]},
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
                "labels": cls.MR_LABELS,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """Return default processing pipeline."""
        definition = cls.get_definition()
        config = definition.pipeline_config.get(mode, definition.pipeline_config["default"])
        
        pipeline = Pipeline()
        
        for step_config in config["preprocessing"]:
            step_type = step_config["type"]
            if step_type == "resample":
                pipeline = pipeline | ResampleStep(
                    name="resample",
                    config={"spacing": step_config["spacing"]}
                )
            elif step_type == "normalize":
                pipeline = pipeline | NormalizeStep(
                    name="normalize",
                    config={"method": step_config["method"]}
                )
        
        inference_config = config["inference"]
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": inference_config["models"][0]}
        )
        
        return pipeline
