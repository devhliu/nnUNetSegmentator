"""
DEEP-PSMA Task

DEEP-PSMA: Deep learning for PSMA PET segmentation in prostate cancer.

Model Storage:
    - Default path: ~/.nnunetsegmentator/models/deep_psma/
    - Model weights are extracted from a Git LFS repository
"""

from .base import BaseTask
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import SUVThresholdStep
from ..pipeline.steps.inference import nnUNetInferenceStep
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
    """
    
    name = "deep_psma"
    description = "PSMA PET lesion segmentation for prostate cancer"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        return TaskDefinition(
            name=cls.name,
            models={
                'deep_psma': ModelInfo(
                    name='deep_psma',
                    task_id='Dataset881_deep_psma',
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
                    # Resampling to the model spacing happens inside the
                    # inference step; intensity normalization is nnUNetv2-internal.
                    {
                        'type': 'suv_threshold',
                        'name': 'suv_threshold',
                        'params': {'threshold': 3.0}
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

        # Preprocessing (resampling to the model spacing happens inside the
        # inference step; intensity normalization is nnUNetv2-internal)
        pipeline.add_step(SUVThresholdStep(
            'suv_threshold',
            {'threshold': 3.0}
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
