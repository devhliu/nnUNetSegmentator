"""
GTRC-Net Task

GTRC-Net: segmentation of total tumor burden (TTB) in metastatic prostate
cancer imaging with PSMA PET/CT, FDG PET/CT, and LuPSMA SPECT/CT.

Model Storage:
    - Default path: ~/.nnunetsegmentator/models/gtrc/
    - Model weights are extracted from a Git LFS repository
"""

from .base import BaseTask
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import (
    SUVThresholdStep,
    MultiChannelStackStep,
)
from ..pipeline.steps.inference import nnUNetInferenceStep
from ..pipeline.steps.postprocessing import (
    LargestComponentStep,
    MorphologicalOpsStep,
)


class GTRCTask(BaseTask):
    """
    GTRC-Net: total tumor burden segmentation.
    
    This task segments total tumor burden from combined PET/CT images.
    """
    
    name = "gtrc"
    description = "Tumor burden segmentation using PET/CT"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        return TaskDefinition(
            name=cls.name,
            models={
                'gtrc_psma': ModelInfo(
                    name='gtrc_psma',
                    task_id='Dataset881_PSMA_PET',
                    url='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained.git',
                    checksum=None,  # Git LFS handles integrity
                    labels={
                        'tumor_burden': 1
                    },
                    modality='PETCT',
                    description='GTRC-Net PSMA PET: Total tumor burden segmentation',
                    default_preprocessing='petct_default',
                    default_postprocessing='tumor_default',
                ),
                'gtrc_fdg': ModelInfo(
                    name='gtrc_fdg',
                    task_id='Dataset882_FDG_PET',
                    url='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained.git',
                    checksum=None,  # Git LFS handles integrity
                    labels={
                        'tumor_burden': 1
                    },
                    modality='PETCT',
                    description='GTRC-Net FDG PET: Total tumor burden segmentation',
                    default_preprocessing='petct_default',
                    default_postprocessing='tumor_default',
                ),
                'gtrc_lupsma': ModelInfo(
                    name='gtrc_lupsma',
                    task_id='Dataset883_LUPSMA_SPECT',
                    url='https://github.com/Peter-MacCallum-Cancer-Centre/GTRC-Net-Pretrained.git',
                    checksum=None,  # Git LFS handles integrity
                    labels={
                        'tumor_burden': 1
                    },
                    modality='SPECTCT',
                    description='GTRC-Net LuPSMA SPECT: Total tumor burden segmentation',
                    default_preprocessing='spectct_default',
                    default_postprocessing='tumor_default',
                )
            },
            pipeline_config={
                'name': 'gtrc_pipeline',
                'steps': [
                    # Resampling to the model spacing happens inside the
                    # inference step; CT intensity handling is nnUNetv2-internal.
                    {
                        'type': 'suv_threshold',
                        'name': 'suv_pet',
                        'params': {'threshold': 2.5}
                    },
                    {
                        'type': 'multi_channel_stack',
                        'name': 'stack_petct',
                        'params': {'order': ['pet', 'ct']}
                    },
                    {
                        'type': 'nnunet_inference',
                        'name': 'gtrc_inference',
                        'params': {
                            'model_path': '${model_path}',
                            'folds': [0, 1, 2, 3, 4],
                            'use_mirroring': True
                        }
                    },
                    {
                        'type': 'largest_component',
                        'name': 'keep_largest',
                        'params': {'keep_top_n': 1}
                    },
                    {
                        'type': 'morphological_ops',
                        'name': 'smooth_boundary',
                        'params': {'operation': 'close', 'radius': 1}
                    }
                ]
            },
            input_requirements={
                'modalities': ['PET', 'CT'],
                'format': 'NIfTI or DICOM',
                'alignment': 'PET and CT must be co-registered',
                'description': 'Requires co-registered PET and CT images'
            },
            output_config={
                'format': 'NIfTI',
                'labels': {
                    1: 'tumor_burden'
                },
                'save_individual_labels': True
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default processing pipeline"""
        pipeline = Pipeline(name="gtrc_default")

        # Preprocessing (resampling to the model spacing happens inside the
        # inference step; CT intensity handling is nnUNetv2-internal)
        pipeline.add_step(SUVThresholdStep(
            'suv_pet',
            {'threshold': 2.5}
        ))
        
        pipeline.add_step(MultiChannelStackStep(
            'stack_modalities',
            {'order': ['pet', 'ct']}
        ))
        
        # Inference
        pipeline.add_step(nnUNetInferenceStep(
            'inference',
            {'folds': [0, 1, 2, 3, 4], 'use_mirroring': True}
        ))
        
        # Postprocessing
        pipeline.add_step(LargestComponentStep(
            'largest_component',
            {'keep_top_n': 1}
        ))
        
        pipeline.add_step(MorphologicalOpsStep(
            'morphological_close',
            {'operation': 'close', 'radius': 1}
        ))
        
        return pipeline
