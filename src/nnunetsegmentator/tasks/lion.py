"""
LION Task

LION: Lesion Identification and Oncology Network for PET lesion segmentation.
Updated with latest models from ENHANCE-PET/LION (v1.0.0).

Repository: https://github.com/ENHANCE-PET/LION
Description: Fully automated tumor segmentation for FDG and PSMA PET scans.
            Part of the ENHANCE.PET initiative. Version 1.0.0 with significantly 
            larger training datasets.

Model Download:
    - FDG Model: https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_fdg_5235_17122025.zip
    - PSMA Model: https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_psma_2046_25112025.zip
    - Models are automatically downloaded to ~/.lionz/ or can be downloaded from GitHub releases
    
Model Storage:
    - Default path: ~/.nnunetsegmentator/models/lion/
    - Model weights are downloaded from GitHub releases and extracted
    
Training Data:
    - FDG: Trained on 5,235 patients
    - PSMA: Trained on 2,046 patients
    
License: Apache 2.0
DOI: 10.5281/zenodo.12626789
"""

from .base import BaseTask
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import (
    ResampleStep,
    ClipIntensityStep,
    NormalizeStep,
    PreserveOriginalStep,
)
from ..pipeline.steps.inference import nnUNetInferenceStep
from ..pipeline.steps.postprocessing import (
    LargestComponentStep,
    MorphologicalOpsStep,
    RemoveSmallObjectsStep,
    PetThresholdStep,
)
from ..pipeline.steps.visualization import MIPGenerationStep
from ..pipeline.steps.quantification import TumorMetricsStep


class LIONTask(BaseTask):
    """
    LION: Lesion Identification and Oncology Network.
    
    This task segments tumors from PET images.
    Supports FDG and PSMA PET models trained on large datasets.
    
    Repository: https://github.com/ENHANCE-PET/LION
    Model Download: GitHub releases (automatic download)
    Model Path: ~/.nnunetsegmentator/models/lion/
    
    Updated models (v1.0.0):
    - FDG: Trained on 5,235 patients
    - PSMA: Trained on 2,046 patients
    """
    
    name = "lion"
    description = "PET lesion segmentation using LION network (v1.0.0)"
    
    # Repository and model information
    REPO_URL = "https://github.com/ENHANCE-PET/LION"
    MODEL_DOWNLOAD_METHOD = "github_releases"
    DOI = "10.5281/zenodo.12626789"
    LICENSE = "Apache 2.0"
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        return TaskDefinition(
            name=cls.name,
            models={
                'fdg': ModelInfo(
                    name='fdg',
                    task_id='Dataset789_fdg',
                    url='https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_fdg_5235_17122025.zip',
                    checksum=None,
                    labels={
                        'tumor': 11
                    },
                    modality='PT',
                    description="FDG PET Tumor Segmentation (5,235 patients)",
                    default_preprocessing='pet_default',
                    default_postprocessing='lesion_default',
                ),
                'psma': ModelInfo(
                    name='psma',
                    task_id='Dataset711_psma',
                    url='https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_psma_2046_25112025.zip',
                    checksum=None,
                    labels={
                        'tumor': 6
                    },
                    modality='PT',
                    description="PSMA PET Tumor Segmentation (2,046 patients)",
                    default_preprocessing='pet_default',
                    default_postprocessing='lesion_default',
                )
            },
            pipeline_config={
                'name': 'lion_pipeline',
                'steps': [
                    {
                        'type': 'resample',
                        'name': 'resample_pet',
                        'params': {'spacing': (1.5, 1.5, 1.5)} # LION default spacing
                    },
                    {
                        'type': 'preserve_original',
                        'name': 'preserve_intensities',
                        'params': {}
                    },
                    {
                        'type': 'normalize',
                        'name': 'normalize_pet',
                        'params': {'method': 'minmax'} # Simple normalization
                    },
                    {
                        'type': 'nnunet_inference',
                        'name': 'lion_inference',
                        'params': {
                            'model_path': '${model_path}',
                            'folds': ('all',),
                            'use_mirroring': True
                        }
                    },
                    {
                        'type': 'pet_threshold',
                        'name': 'threshold_tumor',
                        'params': {'threshold': '${threshold}'}
                    },
                    {
                        'type': 'morphological_ops',
                        'name': 'remove_noise',
                        'params': {'operation': 'open', 'radius': 1}
                    },
                    {
                        'type': 'largest_component',
                        'name': 'keep_largest',
                        'params': {'keep_top_n': 3}
                    },
                    {
                        'type': 'tumor_metrics',
                        'name': 'calculate_metrics',
                        'params': {'output_csv': 'tumor_metrics.csv'}
                    },
                    {
                        'type': 'mip_generation',
                        'name': 'generate_mip',
                        'params': {'output_filename': 'mip.gif'}
                    }
                ]
            },
            input_requirements={
                'modalities': ['PT'],
                'format': 'NIfTI or DICOM',
                'description': 'Requires PET images (FDG or PSMA). CT is optional.'
            },
            output_config={
                'format': 'NIfTI',
                'labels': {
                    1: 'tumor'
                },
                'save_individual_labels': True
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default processing pipeline"""
        pipeline = Pipeline(name="lion_default")
        
        # Preprocessing
        pipeline.add_step(ResampleStep(
            'resample',
            {'spacing': (1.5, 1.5, 1.5)}
        ))
        
        pipeline.add_step(PreserveOriginalStep(
            'preserve_intensities',
            {}
        ))
        
        # LION normalizes PET
        pipeline.add_step(NormalizeStep(
            'normalize',
            {'method': 'minmax'}
        ))
        
        # Inference
        pipeline.add_step(nnUNetInferenceStep(
            'inference',
            {'folds': ('all',), 'use_mirroring': True}
        ))
        
        # Postprocessing
        pipeline.add_step(PetThresholdStep(
            'threshold_tumor',
            {'threshold': None} # Default no threshold
        ))

        pipeline.add_step(MorphologicalOpsStep(
            'morphological_open',
            {'operation': 'open', 'radius': 1}
        ))
        
        pipeline.add_step(LargestComponentStep(
            'largest_component',
            {'keep_top_n': 3}
        ))
        
        pipeline.add_step(TumorMetricsStep(
            'calculate_metrics',
            {'output_csv': 'tumor_metrics.csv'}
        ))

        pipeline.add_step(MIPGenerationStep(
            'generate_mip',
            {'output_filename': 'mip.gif'}
        ))
        
        return pipeline
