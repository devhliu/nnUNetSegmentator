"""
DEEP-PSMA Task

DEEP-PSMA: Deep learning for PSMA PET segmentation in prostate cancer.

Repository: https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA
Description: DEEP-PSMA Grand Challenge model for PSMA PET segmentation.
            Specialized model for prostate cancer lesion detection and segmentation.

Model Download:
    - Git LFS: https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA.git
    - Models are stored in Git LFS and will be automatically downloaded when cloning
    
Model Storage:
    - Default path: ~/.nnunetsegmentator/models/deep_psma/
    - Model weights are extracted from Git LFS repository
    
Reference:
    - Part of GTRC-Net family for prostate cancer imaging
    - Sample data available at: https://zenodo.org/records/18150034
"""

from .base import BaseTask
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import (
    ResampleStep,
    ClipIntensityStep,
    NormalizeStep,
    SUVThresholdStep,
)
from ..pipeline.steps.inference import nnUNetInferenceStep, CascadeInferenceStep
from ..pipeline.steps.postprocessing import (
    LargestComponentStep,
    MorphologicalOpsStep,
    FillHolesStep,
)


class DEEPPSMATask(BaseTask):
    """
    DEEP-PSMA: PSMA PET Segmentation.
    
    This task segments PSMA-avid lesions from PSMA PET/CT images
    for prostate cancer assessment.
    
    Repository: https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA
    Model Download: Git LFS clone from repository
    Model Path: ~/.nnunetsegmentator/models/deep_psma/
    """
    
    name = "deep_psma"
    description = "PSMA PET lesion segmentation for prostate cancer"
    
    # Repository and model information
    REPO_URL = "https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA"
    MODEL_DOWNLOAD_METHOD = "git_lfs"
    SAMPLE_DATA_URL = "https://zenodo.org/records/18150034"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        return TaskDefinition(
            name=cls.name,
            models={
                'deep_psma': ModelInfo(
                    name='deep_psma',
                    task_id='Dataset881_PSMA_PET',
                    url='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-DEEP-PSMA.git',
                    checksum=None,  # Git LFS handles integrity
                    labels={
                        'psma_avid_lesion': 1,
                        'prostate': 2,
                        'lymph_node': 3,
                        'bone_lesion': 4
                    },
                    modality='PET',
                    description=cls.description,
                    default_preprocessing='pet_default',
                    default_postprocessing='lesion_default',
                )
            },
            pipeline_config={
                'name': 'deep_psma_pipeline',
                'steps': [
                    {
                        'type': 'resample',
                        'name': 'resample_pet',
                        'params': {'spacing': (2.0, 2.0, 2.0)}
                    },
                    {
                        'type': 'suv_threshold',
                        'name': 'suv_threshold',
                        'params': {'threshold': 3.0}
                    },
                    {
                        'type': 'normalize',
                        'name': 'normalize_pet',
                        'params': {'method': 'minmax'}
                    },
                    {
                        'type': 'nnunet_inference',
                        'name': 'deep_psma_inference',
                        'params': {
                            'model_path': '${model_path}',
                            'folds': [0, 1, 2, 3, 4],
                            'use_mirroring': True
                        }
                    },
                    {
                        'type': 'fill_holes',
                        'name': 'fill_holes',
                        'params': {}
                    },
                    {
                        'type': 'largest_component',
                        'name': 'keep_largest',
                        'params': {'keep_top_n': 5}
                    },
                    {
                        'type': 'morphological_ops',
                        'name': 'smooth_boundary',
                        'params': {'operation': 'close', 'radius': 1}
                    }
                ]
            },
            input_requirements={
                'modalities': ['PET'],
                'format': 'NIfTI or DICOM',
                'description': 'Requires PSMA PET (preferably with SUV normalization)'
            },
            output_config={
                'format': 'NIfTI',
                'labels': {
                    1: 'psma_avid_lesion',
                    2: 'prostate',
                    3: 'lymph_node',
                    4: 'bone_lesion'
                },
                'save_individual_labels': True
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default processing pipeline"""
        pipeline = Pipeline(name="deep_psma_default")
        
        # Preprocessing
        pipeline.add_step(ResampleStep(
            'resample',
            {'spacing': (2.0, 2.0, 2.0)}
        ))
        
        pipeline.add_step(SUVThresholdStep(
            'suv_threshold',
            {'threshold': 3.0}
        ))
        
        pipeline.add_step(NormalizeStep(
            'normalize',
            {'method': 'minmax'}
        ))
        
        # Inference
        pipeline.add_step(nnUNetInferenceStep(
            'inference',
            {'folds': [0, 1, 2, 3, 4], 'use_mirroring': True}
        ))
        
        # Postprocessing
        pipeline.add_step(FillHolesStep(
            'fill_holes',
            {}
        ))
        
        pipeline.add_step(LargestComponentStep(
            'largest_component',
            {'keep_top_n': 5}
        ))
        
        pipeline.add_step(MorphologicalOpsStep(
            'morphological_close',
            {'operation': 'close', 'radius': 1}
        ))
        
        return pipeline
