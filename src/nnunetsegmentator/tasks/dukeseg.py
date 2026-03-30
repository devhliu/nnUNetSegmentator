"""
DukeSeg (XCAT 3.0) Task - Comprehensive Anatomical Structure Segmentation

This module implements the DukeSeg task for segmenting up to 140 anatomical
structures from CT images, as described in XCAT-3.0 paper.

Repository: https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git
Description: Comprehensive anatomical structure segmentation for CT images.
            Segments up to 140 structures using a multi-model approach.

Model Download:
    - Git clone from: https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git
    - Models are included in the repository
    
Model Storage:
    - Default path: ~/.nnunetsegmentator/models/dukeseg/
    - Model weights are downloaded and extracted on first use
    
Model Components:
    - Skeleton (Task 1004): 62 bone structures
    - Model2 (Task 1001): 61 head, neck, thorax, and muscle structures
    - Model3 (Task 1002): 17 abdominal organ structures
    - Body Composition (Task 1005): 5 tissue types
    
Reference:
    Dahal, L., et al. (2024). XCAT-3.0: A Comprehensive Library of Personalized
    Digital Twins Derived from CT Scans. arXiv preprint arXiv:2405.11133.
"""

from typing import Dict, List, Optional
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import ResampleStep, ClipIntensityStep, NormalizeStep
from ..pipeline.steps.inference import nnUNetInferenceStep, MultiModelConcatStep
from .base import BaseTask
import logging

logger = logging.getLogger(__name__)


# Label mappings for DukeSeg v1 (140 classes - combined from 3 models)
DUKESEG_V1_LABELS = {
    # Skeleton (62 classes from task 1004)
    1: "skull", 2: "mandible", 3: "sternum",
    4: "vertebrae_L5", 5: "vertebrae_L4", 6: "vertebrae_L3",
    7: "vertebrae_L2", 8: "vertebrae_L1", 9: "vertebrae_T12",
    10: "vertebrae_T11", 11: "vertebrae_T10", 12: "vertebrae_T9",
    13: "vertebrae_T8", 14: "vertebrae_T7", 15: "vertebrae_T6",
    16: "vertebrae_T5", 17: "vertebrae_T4", 18: "vertebrae_T3",
    19: "vertebrae_T2", 20: "vertebrae_T1", 21: "vertebrae_C7",
    22: "vertebrae_C6", 23: "vertebrae_C5", 24: "vertebrae_C4",
    25: "vertebrae_C3", 26: "vertebrae_C2", 27: "vertebrae_C1",
    28: "rib_left_1", 29: "rib_left_2", 30: "rib_left_3",
    31: "rib_left_4", 32: "rib_left_5", 33: "rib_left_6",
    34: "rib_left_7", 35: "rib_left_8", 36: "rib_left_9",
    37: "rib_left_10", 38: "rib_left_11", 39: "rib_left_12",
    40: "rib_right_1", 41: "rib_right_2", 42: "rib_right_3",
    43: "rib_right_4", 44: "rib_right_5", 45: "rib_right_6",
    46: "rib_right_7", 47: "rib_right_8", 48: "rib_right_9",
    49: "rib_right_10", 50: "rib_right_11", 51: "rib_right_12",
    52: "humerus_left", 53: "humerus_right", 54: "scapula_left",
    55: "scapula_right", 56: "clavicula_left", 57: "clavicula_right",
    58: "femur_left", 59: "femur_right", 60: "hip_left",
    61: "hip_right", 62: "sacrum",
    
    # Model2 (61 classes from task 1001) - Head, Neck, Thorax, Muscles
    63: "brain", 64: "brainstem", 65: "face",
    66: "eye_globe_right", 67: "eye_globe_left", 68: "optic_nerve_right",
    69: "optic_nerve_left", 70: "glottis", 71: "supraglottic_larynx",
    72: "submandibular_gland_right", 73: "submandibular_gland_left",
    74: "lips", 75: "oral_cavity", 76: "parotid_gland_right",
    77: "parotid_gland_left", 78: "brachial_plexus_right",
    79: "brachial_plexus_left", 80: "spinal_cord", 81: "trachea",
    82: "lung_upper_lobe_left", 83: "lung_lower_lobe_left",
    84: "lung_upper_lobe_right", 85: "lung_middle_lobe_right",
    86: "lung_lower_lobe_right", 87: "aorta", 88: "inferior_vena_cava",
    89: "portal_vein_and_splenic_vein", 90: "heart_myocardium",
    91: "heart_atrium_left", 92: "heart_ventricle_left",
    93: "heart_atrium_right", 94: "heart_ventricle_right",
    95: "pulmonary_artery", 96: "iliac_artery_left",
    97: "iliac_artery_right", 98: "iliac_vena_left",
    99: "iliac_vena_right", 100: "gluteus_maximus_left",
    101: "gluteus_maximus_right", 102: "gluteus_medius_left",
    103: "gluteus_medius_right", 104: "gluteus_minimus_left",
    105: "gluteus_minimus_right", 106: "autochthon_left",
    107: "autochthon_right", 108: "iliopsoas_left",
    109: "iliopsoas_right", 110: "breast_right",
    111: "breast_left", 112: "ln_common_iliac_right",
    113: "ln_common_iliac_left", 114: "ln_external_iliac_right",
    115: "ln_external_iliac_left", 116: "ln_internal_iliac_right",
    117: "ln_internal_iliac_left", 118: "ln_obturator_right",
    119: "ln_obturator_left", 120: "ln_presacral",
    121: "lens_right", 122: "lens_left", 123: "optic_chiasm",
    
    # Model3 (17 classes from task 1002) - Abdominal Organs
    124: "esophagus", 125: "liver", 126: "adrenal_gland_left",
    127: "adrenal_gland_right", 128: "kidney_left",
    129: "kidney_right", 130: "spleen", 131: "pancreas",
    132: "gallbladder", 133: "urinary_bladder", 134: "stomach",
    135: "prostate", 136: "small_bowel", 137: "duodenum",
    138: "colon", 139: "rectum", 140: "seminal_vesicles",
}

# Part models for DukeSeg v1
DUKESEG_SKELETON = {
    1: "skull", 2: "mandible", 3: "sternum",
    4: "vertebrae_L5", 5: "vertebrae_L4", 6: "vertebrae_L3",
    7: "vertebrae_L2", 8: "vertebrae_L1", 9: "vertebrae_T12",
    10: "vertebrae_T11", 11: "vertebrae_T10", 12: "vertebrae_T9",
    13: "vertebrae_T8", 14: "vertebrae_T7", 15: "vertebrae_T6",
    16: "vertebrae_T5", 17: "vertebrae_T4", 18: "vertebrae_T3",
    19: "vertebrae_T2", 20: "vertebrae_T1", 21: "vertebrae_C7",
    22: "vertebrae_C6", 23: "vertebrae_C5", 24: "vertebrae_C4",
    25: "vertebrae_C3", 26: "vertebrae_C2", 27: "vertebrae_C1",
    28: "rib_left_1", 29: "rib_left_2", 30: "rib_left_3",
    31: "rib_left_4", 32: "rib_left_5", 33: "rib_left_6",
    34: "rib_left_7", 35: "rib_left_8", 36: "rib_left_9",
    37: "rib_left_10", 38: "rib_left_11", 39: "rib_left_12",
    40: "rib_right_1", 41: "rib_right_2", 42: "rib_right_3",
    43: "rib_right_4", 44: "rib_right_5", 45: "rib_right_6",
    46: "rib_right_7", 47: "rib_right_8", 48: "rib_right_9",
    49: "rib_right_10", 50: "rib_right_11", 51: "rib_right_12",
    52: "humerus_left", 53: "humerus_right", 54: "scapula_left",
    55: "scapula_right", 56: "clavicula_left", 57: "clavicula_right",
    58: "femur_left", 59: "femur_right", 60: "hip_left",
    61: "hip_right", 62: "sacrum",
}

DUKESEG_MODEL2 = {
    1: "brain", 2: "brainstem", 3: "face",
    4: "eye_globe_right", 5: "eye_globe_left", 6: "lens_right",
    7: "lens_left", 8: "optic_chiasm", 9: "optic_nerve_right",
    10: "optic_nerve_left", 11: "glottis", 12: "supraglottic_larynx",
    13: "submandibular_gland_right", 14: "submandibular_gland_left",
    15: "lips", 16: "oral_cavity", 17: "parotid_gland_right",
    18: "parotid_gland_left", 19: "brachial_plexus_right",
    20: "brachial_plexus_left", 21: "spinal_cord", 22: "trachea",
    23: "lung_upper_lobe_left", 24: "lung_lower_lobe_left",
    25: "lung_upper_lobe_right", 26: "lung_middle_lobe_right",
    27: "lung_lower_lobe_right", 28: "aorta", 29: "inferior_vena_cava",
    30: "portal_vein_and_splenic_vein", 31: "heart_myocardium",
    32: "heart_atrium_left", 33: "heart_ventricle_left",
    34: "heart_atrium_right", 35: "heart_ventricle_right",
    36: "pulmonary_artery", 37: "iliac_artery_left",
    38: "iliac_artery_right", 39: "iliac_vena_left",
    40: "iliac_vena_right", 41: "gluteus_maximus_left",
    42: "gluteus_maximus_right", 43: "gluteus_medius_left",
    44: "gluteus_medius_right", 45: "gluteus_minimus_left",
    46: "gluteus_minimus_right", 47: "autochthon_left",
    48: "autochthon_right", 49: "iliopsoas_left",
    50: "iliopsoas_right", 51: "breast_right",
    52: "breast_left", 53: "ln_common_iliac_right",
    54: "ln_common_iliac_left", 55: "ln_external_iliac_right",
    56: "ln_external_iliac_left", 57: "ln_internal_iliac_right",
    58: "ln_internal_iliac_left", 59: "ln_obturator_right",
    60: "ln_obturator_left", 61: "ln_presacral",
}

DUKESEG_MODEL3 = {
    1: "esophagus", 2: "liver", 3: "adrenal_gland_left",
    4: "adrenal_gland_right", 5: "kidney_left",
    6: "kidney_right", 7: "spleen", 8: "pancreas",
    9: "gallbladder", 10: "urinary_bladder", 11: "stomach",
    12: "prostate", 13: "small_bowel", 14: "duodenum",
    15: "colon", 16: "rectum", 17: "seminal_vesicles",
}

DUKESEG_BODY_COMPOSITION = {
    1: "body", 2: "subcutaneous_fat", 3: "muscles",
    4: "bones", 5: "visceral_fat",
}


class DukeSegTask(BaseTask):
    """
    DukeSeg (XCAT 3.0) task for comprehensive anatomical structure segmentation.
    
    Segments up to 140 anatomical structures from CT images using a multi-model
    approach with specialized models for different anatomical regions.
    
    Repository: https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git
    Model Download: Git clone from repository
    Model Path: ~/.nnunetsegmentator/models/dukeseg/
    
    Features:
    - 140 anatomical structures (DukeSeg v1)
    - 3-part model strategy (skeleton, model2, model3)
    - Specialized models for different anatomical regions
    - High accuracy for comprehensive whole-body segmentation
    
    Models:
    - Skeleton (Task 1004): 62 bone structures
    - Model2 (Task 1001): 61 head, neck, thorax, and muscle structures
    - Model3 (Task 1002): 17 abdominal organ structures
    
    Best for:
    - Comprehensive whole-body segmentation
    - Digital twin generation
    - Treatment planning
    - Anatomical studies
    """
    
    name = "dukeseg"
    description = "DukeSeg (XCAT 3.0): Comprehensive segmentation (140 structures)"
    
    # Repository and model information
    REPO_URL = "https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git"
    MODEL_DOWNLOAD_METHOD = "git"
    REFERENCE = "arXiv:2405.11133"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""
        
        base_url = "https://zenodo.org/api/files/"  # Would be actual Zenodo URLs
        
        models = {
            # 3-part model strategy for DukeSeg v1
            "dukeseg_skeleton": ModelInfo(
                name="dukeseg_skeleton",
                task_id="Dataset1004_dukeseg_skeleton",
                url=f"{base_url}/dukeseg_skeleton.zip",
                checksum="",
                labels=DUKESEG_SKELETON,
                modality="CT",
                description="DukeSeg: Skeleton segmentation (62 bone structures)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            "dukeseg_model2": ModelInfo(
                name="dukeseg_model2",
                task_id="Dataset1001_dukeseg_model2",
                url=f"{base_url}/dukeseg_model2.zip",
                checksum="",
                labels=DUKESEG_MODEL2,
                modality="CT",
                description="DukeSeg: Head/Neck/Thorax/Muscles (61 structures)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
            "dukeseg_model3": ModelInfo(
                name="dukeseg_model3",
                task_id="Dataset1002_dukeseg_model3",
                url=f"{base_url}/dukeseg_model3.zip",
                checksum="",
                labels=DUKESEG_MODEL3,
                modality="CT",
                description="DukeSeg: Abdominal organs (17 structures)",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
        }
        
        return TaskDefinition(
            name="dukeseg",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                        {"type": "normalize", "method": "ct"},
                    ],
                    "inference": {
                        "models": ["dukeseg_skeleton", "dukeseg_model2", "dukeseg_model3"],
                        "ensemble_mode": "concatenate",
                    },
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "orientation": "any",
                "dimensions": 3,
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": DUKESEG_V1_LABELS,
                "dimensions": 3,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """
        Return default processing pipeline.
        
        Args:
            mode: Pipeline mode (currently only "default" supported)
        """
        definition = cls.get_definition()
        config = definition.pipeline_config.get(mode, definition.pipeline_config["default"])
        
        pipeline = Pipeline()
        
        # Add preprocessing
        for step_config in config["preprocessing"]:
            if step_config["type"] == "clip_intensity":
                pipeline = pipeline | ClipIntensityStep(
                    name="clip_intensity",
                    config={
                        "lower": step_config.get("min"),
                        "upper": step_config.get("max")
                    }
                )
            elif step_config["type"] == "normalize":
                pipeline = pipeline | NormalizeStep(
                    name="normalize",
                    config={"method": step_config["method"]}
                )
        
        # Add ensemble inference
        inference_config = config["inference"]
        pipeline = pipeline | MultiModelConcatStep(
            name="dukeseg_ensemble",
            config={
                "model_names": inference_config["models"],
                "label_mappings": [],
            }
        )
        
        return pipeline


class BodyCompositionTask(BaseTask):
    """
    Body composition task for tissue type segmentation.
    
    Segments 5 tissue types for body composition analysis:
    - Body
    - Subcutaneous fat
    - Muscles
    - Bones
    - Visceral fat
    
    Repository: https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git
    Model Download: Git clone from repository
    Model Path: ~/.nnunetsegmentator/models/body_composition/
    """
    
    name = "body_composition"
    description = "Body composition: Tissue type segmentation (5 classes)"
    
    # Repository and model information
    REPO_URL = "https://gitlab.oit.duke.edu/cvit-public/dukeseg_public.git"
    MODEL_DOWNLOAD_METHOD = "git"
    REFERENCE = "arXiv:2405.11133"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition for registry."""
        
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "body_composition": ModelInfo(
                name="body_composition",
                task_id="Dataset1005_body_composition",
                url=f"{base_url}/body_composition.zip",
                checksum="",
                labels=DUKESEG_BODY_COMPOSITION,
                modality="CT",
                description="Body composition: 5 tissue types",
                default_preprocessing="ct_standard",
                default_postprocessing="default"
            ),
        }
        
        return TaskDefinition(
            name="body_composition",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                        {"type": "normalize", "method": "ct"},
                    ],
                    "inference": {"models": ["body_composition"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "orientation": "any",
                "dimensions": 3,
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": DUKESEG_BODY_COMPOSITION,
                "dimensions": 3,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """Return default processing pipeline."""
        definition = cls.get_definition()
        config = definition.pipeline_config.get(mode, definition.pipeline_config["default"])
        
        pipeline = Pipeline()
        
        # Add preprocessing
        for step_config in config["preprocessing"]:
            if step_config["type"] == "clip_intensity":
                pipeline = pipeline | ClipIntensityStep(
                    name="clip_intensity",
                    config={
                        "lower": step_config.get("min"),
                        "upper": step_config.get("max")
                    }
                )
            elif step_config["type"] == "normalize":
                pipeline = pipeline | NormalizeStep(
                    name="normalize",
                    config={"method": step_config["method"]}
                )
        
        # Add inference
        inference_config = config["inference"]
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": inference_config["models"][0]}
        )
        
        return pipeline
