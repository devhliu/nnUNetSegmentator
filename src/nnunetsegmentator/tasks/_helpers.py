"""
Shared task construction helpers.

Centralises the boilerplate that single-model tasks used to repeat: building
the ``ModelInfo``/``TaskDefinition`` pair, the crop/restore step pairs and the
pipeline construction through :class:`PipelineBuilder`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
from ..pipeline.builders import PipelineBuilder
from ._sources import totalseg_url
from .base import BaseTask

__all__ = [
    "crop_pre",
    "restore_post",
    "build_pipeline",
    "single_model_definition",
    "SingleModelTask",
]


def crop_pre(crop_model_name: str, crop_labels: Sequence[str], crop_addon=None) -> dict:
    """Region-crop preprocessing step (coarse model inference + crop)."""
    params = {"crop_model_name": crop_model_name, "crop_labels": list(crop_labels)}
    if crop_addon is not None:
        params["crop_addon"] = list(crop_addon)
    return {"type": "region_crop", "name": "region_crop", "params": params}


def restore_post(remove_outside=None) -> dict:
    """Region-restore postprocessing step (undo crop [+ remove outside])."""
    params = {}
    if remove_outside is not None:
        params["remove_outside"] = remove_outside
    return {"type": "region_restore", "name": "region_restore", "params": params}


def build_pipeline(
    definition: TaskDefinition,
    mode: str = "default",
    config: Optional[dict] = None,
) -> Pipeline:
    """Build a pipeline via PipelineBuilder with task metadata variables."""
    return PipelineBuilder(
        config or definition.pipeline_config,
        variables={
            "_task_models": definition.models,
            "_output_labels": definition.output_config.get("labels", {}),
        },
        mode=mode,
    ).build()


def single_model_definition(
    *,
    name: str,
    description: str,
    model_name: str,
    task_id: str,
    url: str,
    labels: Dict[str, int],
    modality: str = "CT",
    model_description: str = "",
    crop: Optional[Tuple] = None,
    pre: Optional[List[dict]] = None,
    post: Optional[List[dict]] = None,
    inference: Optional[dict] = None,
    fast: Optional[Tuple[str, str, str, str]] = None,
    pipeline_name: Optional[str] = None,
) -> TaskDefinition:
    """
    Build a standard one-or-two-model task definition.

    ``crop`` is ``(crop_model_name, crop_labels, crop_addon)``; ``pre`` holds
    extra preprocessing steps appended after the crop step; ``post`` defaults to
    a region-restore step when ``crop`` is set and to no postprocessing
    otherwise. ``inference`` holds extra inference parameters (e.g. ``folds``)
    merged into the default-mode inference section. ``fast`` is
    ``(model_name, task_id, url, description)`` and adds a ``fast`` pipeline
    mode backed by a second model.
    """
    models = {
        model_name: ModelInfo(
            name=model_name,
            task_id=task_id,
            url=url,
            checksum="",
            labels=labels,
            modality=modality,
            description=model_description or description,
        )
    }
    if fast is not None:
        fast_name, fast_task_id, fast_url, fast_description = fast
        models[fast_name] = ModelInfo(
            name=fast_name,
            task_id=fast_task_id,
            url=fast_url,
            checksum="",
            labels=labels,
            modality=modality,
            description=fast_description,
        )

    if post is None:
        post = [restore_post()] if crop is not None else []

    preprocessing: List[dict] = []
    if crop is not None:
        preprocessing.append(crop_pre(*crop))
    preprocessing.extend(pre or [])

    pipeline_config: Dict[str, dict] = {
        "default": {
            "preprocessing": preprocessing,
            "inference": {"models": [model_name], **(inference or {})},
            "postprocessing": post,
        }
    }
    if fast is not None:
        pipeline_config["fast"] = {
            "preprocessing": [],
            "inference": {"models": [fast[0]]},
            "postprocessing": post,
        }

    if pipeline_name is not None:
        pipeline_config["name"] = pipeline_name

    return TaskDefinition(
        name=name,
        models=models,
        pipeline_config=pipeline_config,
        input_requirements={"modality": [modality], "format": ["nifti", "dicom"]},
        output_config={"format": "nifti", "multilabel": True, "labels": labels},
    )


class SingleModelTask(BaseTask):
    """
    Base class for the common single-model TotalSegmentator task shape.

    Subclasses declare their identity and workflow as class attributes instead
    of repeating ``ModelInfo``/``TaskDefinition`` construction::

        class BreastsTask(SingleModelTask):
            name = "breasts"
            description = "Breast segmentation"
            TASK_ID = "Dataset527_breasts_1559subj"
            VERSION = "v2.5.0-weights"
            LABELS = BREASTS_LABELS

    Supported workflow knobs: ``CROP`` (crop model, labels, addon), ``POST``
    (explicit postprocessing steps), ``FAST`` (second low-resolution model) and
    ``MODEL_NAME`` when the registry model name differs from the task name.
    """

    name: Optional[str] = None
    description = ""
    LABELS: Dict[str, int] = {}
    MODEL_NAME: Optional[str] = None
    TASK_ID: str = ""
    VERSION: str = ""
    URL: Optional[str] = None
    MODALITY = "CT"
    MODEL_DESCRIPTION = ""
    CROP: Optional[Tuple] = None
    PRE: Optional[List[dict]] = None
    POST: Optional[List[dict]] = None
    INFERENCE: Optional[dict] = None
    FAST: Optional[Tuple[str, str, str, str]] = None

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        model_name = cls.MODEL_NAME or cls.name
        task_id = cls.TASK_ID or model_name
        fast = None
        if cls.FAST is not None:
            fast_name, fast_task_id, fast_version, fast_description = cls.FAST
            fast = (fast_name, fast_task_id, totalseg_url(fast_task_id, fast_version), fast_description)
        return single_model_definition(
            name=cls.name,
            description=cls.description,
            model_name=model_name,
            task_id=task_id,
            url=cls.URL or totalseg_url(task_id, cls.VERSION),
            labels=cls.LABELS,
            modality=cls.MODALITY,
            model_description=cls.MODEL_DESCRIPTION,
            crop=cls.CROP,
            pre=cls.PRE,
            post=cls.POST,
            inference=cls.INFERENCE,
            fast=fast,
        )

    @classmethod
    def get_default_pipeline(cls, mode: str = "default") -> Pipeline:
        """Return default pipeline for ``mode`` ("default" or "fast")."""
        return build_pipeline(cls.get_definition(), mode)