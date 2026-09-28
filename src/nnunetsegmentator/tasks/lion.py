"""
LION Task

LION: Lesion Identification and Oncology Network for PET lesion segmentation
for FDG and PSMA PET scans.

Model Storage:
    - Default path: ~/.nnunetsegmentator/models/lion/
    - Model weights are downloaded from GitHub releases and extracted

Pipeline:
    The image is B-spline resampled to the model spacing inside the inference
    step, predictions are restored to the input grid, and only the optional
    user threshold is applied afterwards (no minmax normalization, morphology
    or largest-component filtering).

Training Data:
    - FDG: Trained on 5,209 patients (v1.0.3)
    - PSMA: Trained on 2,046 patients (v1.0.0)
"""

from .base import BaseTask
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import PreserveOriginalStep
from ..pipeline.steps.inference import nnUNetInferenceStep
from ..pipeline.steps.postprocessing import PetThresholdStep
from ..pipeline.steps.visualization import MIPGenerationStep
from ..pipeline.steps.quantification import TumorMetricsStep


class LIONTask(BaseTask):
    """
    LION: Lesion Identification and Oncology Network.

    This task segments tumors from PET images.
    Supports FDG and PSMA PET models trained on large datasets.

    Models:
    - FDG: Trained on 5,209 patients (v1.0.3, Dataset789_Tumors)
    - PSMA: Trained on 2,046 patients (v1.0.0, Dataset711_PSMA)
    """

    name = "lion"
    description = "PET lesion segmentation using LION network"

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition"""
        return TaskDefinition(
            name=cls.name,
            models={
                'fdg': ModelInfo(
                    name='fdg',
                    task_id='Dataset789_Tumors',
                    url='https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.3/clin_pt_fdg_5209_27032026.zip',
                    checksum=None,
                    # Tumor label id from upstream lionz/models.py
                    # MODEL_METADATA["fdg"][TUMOR_LABEL].
                    labels={
                        'tumor': 11
                    },
                    modality='PT',
                    description="FDG PET Tumor Segmentation (5,209 patients, v1.0.3)",
                    default_preprocessing='pet_default',
                    default_postprocessing='lesion_default',
                ),
                'psma': ModelInfo(
                    name='psma',
                    task_id='Dataset711_PSMA',
                    url='https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_psma_2046_25112025.zip',
                    checksum=None,
                    # Tumor label id from upstream lionz/models.py
                    # MODEL_METADATA["psma"][TUMOR_LABEL].
                    labels={
                        'tumor': 6
                    },
                    modality='PT',
                    description="PSMA PET Tumor Segmentation (2,046 patients, v1.0.0)",
                    default_preprocessing='pet_default',
                    default_postprocessing='lesion_default',
                )
            },
            pipeline_config={
                'name': 'lion_pipeline',
                'steps': [
                    {
                        'type': 'preserve_original',
                        'name': 'preserve_intensities',
                        'params': {}
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

        pipeline.add_step(PreserveOriginalStep(
            'preserve_intensities',
            {}
        ))

        # Inference: B-spline resampling to the model spacing and the
        # nearest-neighbour restore to the input grid happen inside the step.
        pipeline.add_step(nnUNetInferenceStep(
            'inference',
            {'folds': ('all',), 'use_mirroring': True}
        ))

        # Postprocessing: only the optional user threshold
        pipeline.add_step(PetThresholdStep(
            'threshold_tumor',
            {'threshold': None} # Default no threshold
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
