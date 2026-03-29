# Task–Model Workflows Guide

This guide explains how task-model wiring works at runtime and how to add or extend tasks safely.

## 1. Runtime flow

| Step | File | Behavior |
|---|---|---|
| Task discovery | `src/nnunetsegmentator/tasks/__init__.py` | Imports task modules, finds `BaseTask` subclasses, and registers them with `TaskRegistry`. |
| Task retrieval | `src/nnunetsegmentator/core/orchestrator.py` | `SegmentationOrchestrator` resolves task via `TaskRegistry.get_task(task_name)`. |
| Primary model path resolution | `src/nnunetsegmentator/core/orchestrator.py` | Checks `{model_root}/{task_name}/{task_id}` first, then falls back to `TaskRegistry.get_model_path(model_name)`. |
| Pipeline execution | `src/nnunetsegmentator/pipeline/builders.py` + steps | Inference steps use `model_name`; registry resolves the actual path. |
| Model acquisition | `src/nnunetsegmentator/core/registry.py` | Download path: `get_model_path`; local path install: `install_task_models_from_local`. |

## 2. Add a new built-in task (inside this repo)

| Item | Requirement |
|---|---|
| Task file | Create `src/nnunetsegmentator/tasks/<your_task>.py` |
| Base class | Subclass `BaseTask` |
| Definition | Implement `get_definition()` returning `TaskDefinition` |
| Models | Add entries in `TaskDefinition.models` (`model_name -> ModelInfo`) |
| Pipeline | Ensure every inference `model_name` exists in `models` |
| Registration | Automatic (module auto-discovery); no manual register call needed for built-ins |

### Minimal skeleton

```python
from nnunetsegmentator.tasks.base import BaseTask
from nnunetsegmentator.core.registry import TaskDefinition, ModelInfo


class MyTask(BaseTask):
    name = "my_task"
    description = "My segmentation task"

    @classmethod
    def get_definition(cls) -> TaskDefinition:
        return TaskDefinition(
            name=cls.name,
            models={
                "my_model": ModelInfo(
                    name="my_model",
                    task_id="999",
                    url="https://example.com/my_model.zip",
                    checksum="",
                    labels={"target": 1},
                    modality="CT",
                )
            },
            pipeline_config={
                "name": "my_task_pipeline",
                "steps": [
                    {
                        "type": "nnunet_inference",
                        "name": "inference",
                        "params": {"model_name": "my_model"},
                    }
                ],
            },
            input_requirements={"modality": ["CT"], "format": ["nifti", "dicom"]},
            output_config={"format": "nifti", "labels": {1: "target"}},
        )
```

## 3. Extend after package release (without editing package source)

| Pattern | Use case | How |
|---|---|---|
| External task registration (recommended) | Add custom tasks in downstream app | Build `TaskDefinition`/`ModelInfo` and call `TaskRegistry.register(task_def)` at startup. |
| Local model installation | Use private or pre-downloaded artifacts | Call `TaskRegistry.install_task_models_from_local(task_name=..., model_sources=...)`. |
| Explicit model path per run | One-off execution | Pass model path at orchestrator/step level (less reusable). |

## 4. Local model install interfaces

| Interface | Example | Key format |
|---|---|---|
| Python API | `TaskRegistry.install_task_models_from_local(task_name="total", model_sources={"total_organs": "/local/total_organs.zip", "292": "/local/total_vertebrae/"}, force=True)` | Model name or task ID |
| CLI | `nnunetsegmentator install-models --task total --model total_organs=/local/total_organs.zip --model 292=/local/total_vertebrae/ --force` | Model name or task ID |
| Script | `python scripts/model_manager.py --install-local --task total --model total_organs=/local/total_organs.zip` | Model name or task ID |

## 5. Validation checklist

| Check | Expected result |
|---|---|
| Task visibility | New task appears in `TaskRegistry.list_tasks()` |
| Model consistency | Every inference `model_name` exists in `TaskDefinition.models` |
| Install location | Artifacts appear in `{model_root}/{task_name}/{task_id}` |
| Runtime resolution | `TaskRegistry.get_model_path(model_name)` resolves without missing-model errors |
| Naming hygiene | No cross-task `model_name` collisions |
