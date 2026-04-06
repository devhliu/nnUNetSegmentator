"""
MRSegmentator Task - Multi-Modality Segmentation of 40 Classes in MRI and CT

This module implements the MRSegmentator task with 40 anatomical structures
for MRI and CT imaging of abdominal, pelvic, and thorax regions.

Repository: https://github.com/hhaentze/MRSegmentator
Description: Multi-modality segmentation of 40 classes in MRI and CT.
            Works well on different sequence types including T1- and T2-weighted,
            Dixon sequences and even CT images.

Model Download:
    - Models are automatically downloaded from GitHub releases
    - Default download location: ~/.conda/envs/<name>/lib/python3.11/site-packages/mrsegmentator/weights
    - Can be customized via MRSEG_WEIGHTS_PATH environment variable
    
Model Storage:
    - Default path: ~/.nnunetsegmentator/models/mrsegmentator/
    - Model weights are downloaded automatically on first use
    
Available Models:
    - mrsegmentator: 40 classes (MRI and CT)
    
License: Apache-2.0
Citation: Häntze et al., Radiology: Artificial Intelligence (2024). https://doi.org/10.1148/ryai.240777
"""

from typing import Dict, List, Optional
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import ResampleStep, NormalizeStep
from ..pipeline.steps.inference import nnUNetInferenceStep
from ..pipeline.steps.postprocessing import LargestComponentStep
from .base import BaseTask
import logging

logger = logging.getLogger(__name__)


# Label mappings for MRSegmentator (40 classes)
MRSEGMENTATOR_LABELS = {
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "right_adrenal_gland",
    9: "left_adrenal_gland",
    10: "left_lung",
    11: "right_lung",
    12: "heart",
    13: "aorta",
    14: "inferior_vena_cava",
    15: "portal_vein_and_splenic_vein",
    16: "left_iliac_artery",
    17: "right_iliac_artery",
    18: "left_iliac_vena",
    19: "right_iliac_vena",
    20: "esophagus",
    21: "small_bowel",
    22: "duodenum",
    23: "colon",
    24: "urinary_bladder",
    25: "spine",
    26: "sacrum",
    27: "left_hip",
    28: "right_hip",
    29: "left_femur",
    30: "right_femur",
    31: "left_autochthonous_muscle",
    32: "right_autochthonous_muscle",
    33: "left_iliopsoas_muscle",
    34: "right_iliopsoas_muscle",
    35: "left_gluteus_maximus",
    36: "right_gluteus_maximus",
    37: "left_gluteus_medius",
    38: "right_gluteus_medius",
    39: "left_gluteus_minimus",
    40: "right_gluteus_minimus",
}


class MRSegmentatorTask(BaseTask):
    """
    MRSegmentator: Multi-Modality Segmentation of 40 Classes in MRI and CT.
    
    This task segments 40 organs and structures in human MRI scans of the
    abdominal, pelvic and thorax regions. Works well on different sequence
    types including T1- and T2-weighted, Dixon sequences and even CT images.
    
    Repository: https://github.com/hhaentze/MRSegmentator
    Model Download: Automatic from GitHub releases
    Model Path: ~/.nnunetsegmentator/models/mrsegmentator/
    """
    
    name = "mrsegmentator"
    description = "Multi-modality segmentation of 40 classes in MRI and CT"
    
    # Repository and model information
    REPO_URL = "https://github.com/hhaentze/MRSegmentator"
    MODEL_DOWNLOAD_METHOD = "pip"
    WEIGHTS_URL = "https://github.com/hhaentze/MRSegmentator/releases/download/v1.2.0/weights.zip"
    PAPER_URL = "https://doi.org/10.1148/ryai.240777"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        return TaskDefinition(
            name=cls.name,
            models={
                'mrsegmentator': ModelInfo(
                    name='mrsegmentator',
                    task_id='Dataset1500_mrsegmentator',
                    url='https://github.com/hhaentze/MRSegmentator',
                    checksum=None,
                    labels=MRSEGMENTATOR_LABELS,
                    modality='MR',  # Works on both MR and CT
                    description='MRSegmentator: 40-class multi-modality segmentation (MRI/CT)',
                    default_preprocessing='mr_default',
                    default_postprocessing='organ_default',
                )
            },
            pipeline_config={
                'name': 'mrsegmentator_pipeline',
                'steps': [
                    {
                        'type': 'resample',
                        'name': 'resample_mr',
                        'params': {'spacing': (1.5, 1.5, 1.5)}
                    },
                    {
                        'type': 'normalize',
                        'name': 'normalize_mr',
                        'params': {'method': 'zscore'}
                    },
                    {
                        'type': 'nnunet_inference',
                        'name': 'mrsegmentator_inference',
                        'params': {
                            'model_path': '${model_path}',
                            'folds': [0, 1, 2, 3, 4],
                            'use_mirroring': True,
                            'tile_step_size': 0.5,
                            'use_gaussian': True
                        }
                    },
                    {
                        'type': 'largest_component',
                        'name': 'keep_largest_organs',
                        'params': {'apply_to_labels': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
                    }
                ]
            },
            input_requirements={
                'modalities': ['MR', 'CT'],
                'format': 'NIfTI, DICOM, MHA, or NRRD',
                'sequences': ['T1-weighted', 'T2-weighted', 'Dixon', 'CT'],
                'regions': ['abdominal', 'pelvic', 'thorax'],
                'description': 'Works on various MRI sequences and CT images'
            },
            output_config={
                'format': 'NIfTI',
                'labels': MRSEGMENTATOR_LABELS,
                'num_classes': 40,
                'save_individual_labels': False
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default processing pipeline"""
        pipeline = Pipeline(name="mrsegmentator_default")
        
        # Preprocessing
        pipeline.add_step(ResampleStep(
            'resample',
            {'spacing': (1.5, 1.5, 1.5)}
        ))
        
        pipeline.add_step(NormalizeStep(
            'normalize',
            {'method': 'zscore'}
        ))
        
        # Inference
        pipeline.add_step(nnUNetInferenceStep(
            'inference',
            {
                'folds': [0, 1, 2, 3, 4],
                'use_mirroring': True,
                'tile_step_size': 0.5,
                'use_gaussian': True
            }
        ))
        
        # Postprocessing
        pipeline.add_step(LargestComponentStep(
            'largest_component',
            {'apply_to_labels': [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        ))
        
        return pipeline
    
    @classmethod
    def get_label_info(cls) -> Dict[int, str]:
        """Return label mappings"""
        return MRSEGMENTATOR_LABELS
    
    @classmethod
    def get_class_count(cls) -> int:
        """Return number of segmentation classes"""
        return 40
