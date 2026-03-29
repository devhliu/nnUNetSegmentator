"""
Additional TotalSegmentator Tasks

This module implements additional segmentation tasks from TotalSegmentator:
- Lung vessels segmentation
- Body segmentation
- Tissue type segmentation
- And more specialized tasks
"""

from typing import Dict
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.steps.preprocessing import ResampleStep, ClipIntensityStep, NormalizeStep
from ..pipeline.steps.inference import nnUNetInferenceStep
from ..pipeline.steps.postprocessing import LargestComponentStep, FillHolesStep
from .base import BaseTask
import logging

logger = logging.getLogger(__name__)


class LungVesselsTask(BaseTask):
    """
    Lung vessels and airways segmentation.
    
    Segments:
    - Lung airways (trachea, bronchi)
    - Airways wall
    - Lung arteries
    - Lung veins
    """
    
    name = "lung_vessels"
    description = "Lung vessels and airways segmentation"
    
    LABELS = {
        1: "lung_airways",
        2: "lung_airways_wall",
        3: "lung_arteries",
        4: "lung_veins"
    }
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "lung_vessels": ModelInfo(
                name="lung_vessels",
                task_id="117",
                url=f"{base_url}/lung_vessels.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Lung vessels and airways",
                default_preprocessing="ct_lung",
                default_postprocessing="vessel_postprocess"
            )
        }
        
        return TaskDefinition(
            name="lung_vessels",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [0.703125, 0.703125, 1.0]},
                        {"type": "clip_intensity", "min": -1024, "max": 3071},
                    ],
                    "inference": {"models": ["lung_vessels"]},
                    "postprocessing": [
                        {"type": "largest_component", "labels": ["lung_airways"]},
                    ]
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "roi": "lung",  # Requires lung region
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [0.703125, 0.703125, 1.0]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -1024, "upper": 3071}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "lung_vessels"}
        )
        return pipeline


class BodySegmentationTask(BaseTask):
    """
    Body segmentation for radiotherapy planning.
    
    Segments:
    - Body trunk (torso)
    - Body extremities (arms, legs)
    """
    
    name = "body"
    description = "Body segmentation for radiotherapy planning"
    
    LABELS = {
        1: "body_trunc",
        2: "body_extremities"
    }
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "body": ModelInfo(
                name="body",
                task_id="299",
                url=f"{base_url}/body.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Body segmentation (1.5mm)",
            ),
            "body_fast": ModelInfo(
                name="body_fast",
                task_id="300",
                url=f"{base_url}/body_fast.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Fast body segmentation (6mm)",
            )
        }
        
        return TaskDefinition(
            name="body",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                    ],
                    "inference": {"models": ["body"]},
                    "postprocessing": []
                },
                "fast": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [6.0, 6.0, 6.0]},
                    ],
                    "inference": {"models": ["body_fast"]},
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
                "labels": cls.LABELS,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """Return default pipeline."""
        definition = cls.get_definition()
        config = definition.pipeline_config.get(mode, definition.pipeline_config["default"])
        
        pipeline = Pipeline()
        spacing = config["preprocessing"][0]["spacing"]
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": spacing}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": config["inference"]["models"][0]}
        )
        return pipeline


class TissueTypesTask(BaseTask):
    """
    Tissue type segmentation for body composition analysis.
    
    Segments:
    - Subcutaneous fat
    - Torso fat (visceral)
    - Skeletal muscle
    - Intermuscular fat (optional)
    """
    
    name = "tissue_types"
    description = "Tissue type segmentation for body composition"
    
    LABELS_3 = {
        1: "subcutaneous_fat",
        2: "torso_fat",
        3: "skeletal_muscle"
    }
    
    LABELS_4 = {
        1: "subcutaneous_fat",
        2: "torso_fat",
        3: "skeletal_muscle",
        4: "intermuscular_fat"
    }
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "tissue_types": ModelInfo(
                name="tissue_types",
                task_id="481",
                url=f"{base_url}/tissue_types.zip",
                checksum="",
                labels=cls.LABELS_3,
                modality="CT",
                description="Tissue type segmentation (3 classes)",
            ),
            "tissue_4_types": ModelInfo(
                name="tissue_4_types",
                task_id="485",
                url=f"{base_url}/tissue_4_types.zip",
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
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                        {"type": "clip_intensity", "min": -200, "max": 200},  # Fat/muscle range
                    ],
                    "inference": {"models": ["tissue_types"]},
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
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.5, 1.5, 1.5]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -200, "upper": 200}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "tissue_types"}
        )
        return pipeline


class HeartChambersTask(BaseTask):
    """
    High-resolution heart chamber segmentation.
    
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
    
    LABELS = {
        1: "heart_myocardium",
        2: "heart_atrium_left",
        3: "heart_ventricle_left",
        4: "heart_atrium_right",
        5: "heart_ventricle_right",
        6: "aorta",
        7: "pulmonary_artery"
    }
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "heartchambers_highres": ModelInfo(
                name="heartchambers_highres",
                task_id="301",
                url=f"{base_url}/heartchambers_highres.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="High-resolution heart chambers",
                default_preprocessing="ct_cardiac",
                default_postprocessing="cardiac_postprocess"
            )
        }
        
        return TaskDefinition(
            name="heartchambers_highres",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [0.5, 0.5, 0.5]},  # High resolution
                        {"type": "clip_intensity", "min": -200, "max": 500},
                    ],
                    "inference": {"models": ["heartchambers_highres"]},
                    "postprocessing": [
                        {"type": "fill_holes", "labels": ["heart_atrium_left", "heart_ventricle_left"]},
                    ]
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "roi": "cardiac",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [0.5, 0.5, 0.5]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -200, "upper": 500}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "heartchambers_highres"}
        )
        pipeline = pipeline | FillHolesStep(name="fill_holes")
        return pipeline


class CerebralBleedTask(BaseTask):
    """
    Cerebral hemorrhage segmentation.
    
    Segments intracerebral hemorrhage in brain CT.
    """
    
    name = "cerebral_bleed"
    description = "Cerebral hemorrhage segmentation"
    
    LABELS = {
        1: "intracerebral_hemorrhage"
    }
    
    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"
        
        models = {
            "cerebral_bleed": ModelInfo(
                name="cerebral_bleed",
                task_id="150",
                url=f"{base_url}/cerebral_bleed.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Cerebral hemorrhage",
            )
        }
        
        return TaskDefinition(
            name="cerebral_bleed",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "clip_intensity", "min": 0, "max": 100},  # Blood range
                    ],
                    "inference": {"models": ["cerebral_bleed"]},
                    "postprocessing": [
                        {"type": "largest_component"},
                    ]
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "roi": "brain",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )
    
    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": 0, "upper": 100}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "cerebral_bleed"}
        )
        pipeline = pipeline | LargestComponentStep(name="largest_component")
        return pipeline


class LiverVesselsTask(BaseTask):
    """
    Liver vessels and tumor segmentation.

    Segments:
    - Liver vessels (portal vein, hepatic veins)
    - Liver tumors
    """

    name = "liver_vessels"
    description = "Liver vessels and tumor segmentation"

    LABELS = {
        1: "liver_vessels",
        2: "liver_tumor"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "liver_vessels": ModelInfo(
                name="liver_vessels",
                task_id="XXX",  # Would have actual task ID
                url=f"{base_url}/liver_vessels.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Liver vessels and tumors",
            )
        }

        return TaskDefinition(
            name="liver_vessels",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.0, 1.0, 1.0]},
                        {"type": "clip_intensity", "min": -100, "max": 400},
                    ],
                    "inference": {"models": ["liver_vessels"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "roi": "liver",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.0, 1.0, 1.0]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -100, "upper": 400}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "liver_vessels"}
        )
        return pipeline


class LungNodulesTask(BaseTask):
    """
    Lung nodules segmentation.

    Segments lung nodules in CT images.
    """

    name = "lung_nodules"
    description = "Lung nodules segmentation"

    LABELS = {
        1: "lung_nodule"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "lung_nodules": ModelInfo(
                name="lung_nodules",
                task_id="504",
                url=f"{base_url}/lung_nodules.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Lung nodules",
            )
        }

        return TaskDefinition(
            name="lung_nodules",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.0, 1.0, 1.0]},
                        {"type": "clip_intensity", "min": -1000, "max": 400},
                    ],
                    "inference": {"models": ["lung_nodules"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "roi": "lung",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.0, 1.0, 1.0]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -1000, "upper": 400}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "lung_nodules"}
        )
        return pipeline


class VertebraeBodyTask(BaseTask):
    """
    Vertebrae body segmentation.

    Segments individual vertebrae bodies.
    """

    name = "vertebrae_body"
    description = "Vertebrae body segmentation"

    LABELS = {
        i: f"vertebrae_{region}{level}"
        for i, (region, level) in enumerate([
            ("C", "1"), ("C", "2"), ("C", "3"), ("C", "4"), ("C", "5"), ("C", "6"), ("C", "7"),
            ("T", "1"), ("T", "2"), ("T", "3"), ("T", "4"), ("T", "5"), ("T", "6"),
            ("T", "7"), ("T", "8"), ("T", "9"), ("T", "10"), ("T", "11"), ("T", "12"),
            ("L", "1"), ("L", "2"), ("L", "3"), ("L", "4"), ("L", "5"),
        ], start=1)
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "vertebrae_body": ModelInfo(
                name="vertebrae_body",
                task_id="257",
                url=f"{base_url}/vertebrae_body.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Vertebrae body segmentation",
            )
        }

        return TaskDefinition(
            name="vertebrae_body",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                    ],
                    "inference": {"models": ["vertebrae_body"]},
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
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.5, 1.5, 1.5]}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "vertebrae_body"}
        )
        return pipeline


class LiverLesionsTask(BaseTask):
    """
    Liver lesions segmentation.

    Segments liver lesions in CT or MR images.
    """

    name = "liver_lesions"
    description = "Liver lesions segmentation"

    LABELS = {
        1: "liver_lesion"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "liver_lesions": ModelInfo(
                name="liver_lesions",
                task_id="XXX",
                url=f"{base_url}/liver_lesions.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Liver lesions (CT)",
            ),
            "liver_lesions_mr": ModelInfo(
                name="liver_lesions_mr",
                task_id="XXX",
                url=f"{base_url}/liver_lesions_mr.zip",
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
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.0, 1.0, 1.0]},
                        {"type": "clip_intensity", "min": -100, "max": 400},
                    ],
                    "inference": {"models": ["liver_lesions"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT", "MR"],
                "format": ["nifti", "dicom"],
                "roi": "liver",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.0, 1.0, 1.0]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -100, "upper": 400}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "liver_lesions"}
        )
        return pipeline


class KidneyCystsTask(BaseTask):
    """
    Kidney cysts segmentation.

    Segments kidney cysts in CT images.
    """

    name = "kidney_cysts"
    description = "Kidney cysts segmentation"

    LABELS = {
        1: "kidney_cyst_left",
        2: "kidney_cyst_right"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "kidney_cysts": ModelInfo(
                name="kidney_cysts",
                task_id="XXX",
                url=f"{base_url}/kidney_cysts.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Kidney cysts",
            )
        }

        return TaskDefinition(
            name="kidney_cysts",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.0, 1.0, 1.0]},
                        {"type": "clip_intensity", "min": -100, "max": 400},
                    ],
                    "inference": {"models": ["kidney_cysts"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["CT"],
                "format": ["nifti", "dicom"],
                "roi": "kidney",
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.0, 1.0, 1.0]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -100, "upper": 400}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "kidney_cysts"}
        )
        return pipeline


class PleuralPericardEffusionTask(BaseTask):
    """
    Pleural and pericardial effusion segmentation.

    Segments fluid accumulations in pleural and pericardial spaces.
    """

    name = "pleural_pericard_effusion"
    description = "Pleural and pericardial effusion segmentation"

    LABELS = {
        1: "pleural_effusion_left",
        2: "pleural_effusion_right",
        3: "pericardial_effusion"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "pleural_pericard_effusion": ModelInfo(
                name="pleural_pericard_effusion",
                task_id="XXX",
                url=f"{base_url}/pleural_pericard_effusion.zip",
                checksum="",
                labels=cls.LABELS,
                modality="CT",
                description="Pleural and pericardial effusion",
            )
        }

        return TaskDefinition(
            name="pleural_pericard_effusion",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                        {"type": "clip_intensity", "min": -100, "max": 100},
                    ],
                    "inference": {"models": ["pleural_pericard_effusion"]},
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
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.5, 1.5, 1.5]}
        )
        pipeline = pipeline | ClipIntensityStep(
            name="clip_intensity",
            config={"lower": -100, "upper": 100}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "pleural_pericard_effusion"}
        )
        return pipeline


class BodyMRTask(BaseTask):
    """
    Body segmentation for MR images.

    Segments body trunk and extremities in MR images.
    """

    name = "body_mr"
    description = "Body segmentation for MR images"

    LABELS = {
        1: "body_trunc",
        2: "body_extremities"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "body_mr": ModelInfo(
                name="body_mr",
                task_id="XXX",
                url=f"{base_url}/body_mr.zip",
                checksum="",
                labels=cls.LABELS,
                modality="MR",
                description="Body segmentation (MR)",
            )
        }

        return TaskDefinition(
            name="body_mr",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                    ],
                    "inference": {"models": ["body_mr"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["MR"],
                "format": ["nifti", "dicom"],
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.5, 1.5, 1.5]}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "body_mr"}
        )
        return pipeline


class TissueTypesMRTask(BaseTask):
    """
    Tissue type segmentation for MR images.

    Segments fat and muscle in MR images.
    """

    name = "tissue_types_mr"
    description = "Tissue type segmentation for MR images"

    LABELS = {
        1: "subcutaneous_fat",
        2: "visceral_fat",
        3: "skeletal_muscle"
    }

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        """Return task definition."""
        base_url = "https://zenodo.org/api/files/"

        models = {
            "tissue_types_mr": ModelInfo(
                name="tissue_types_mr",
                task_id="XXX",
                url=f"{base_url}/tissue_types_mr.zip",
                checksum="",
                labels=cls.LABELS,
                modality="MR",
                description="Tissue types (MR)",
            )
        }

        return TaskDefinition(
            name="tissue_types_mr",
            models=models,
            pipeline_config={
                "default": {
                    "preprocessing": [
                        {"type": "resample", "spacing": [1.5, 1.5, 1.5]},
                    ],
                    "inference": {"models": ["tissue_types_mr"]},
                    "postprocessing": []
                }
            },
            input_requirements={
                "modality": ["MR"],
                "format": ["nifti", "dicom"],
            },
            output_config={
                "format": "nifti",
                "multilabel": True,
                "labels": cls.LABELS,
            }
        )

    @classmethod
    def get_default_pipeline(cls) -> Pipeline:
        """Return default pipeline."""
        pipeline = Pipeline()
        pipeline = pipeline | ResampleStep(
            name="resample",
            config={"spacing": [1.5, 1.5, 1.5]}
        )
        pipeline = pipeline | nnUNetInferenceStep(
            name="inference",
            config={"model_name": "tissue_types_mr"}
        )
        return pipeline

