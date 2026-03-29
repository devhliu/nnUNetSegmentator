"""
Task and Model Registry System

This module provides a centralized registry for managing segmentation tasks and models.
"""

from typing import Dict, Type, Any, Callable, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import shutil
import tarfile
import zipfile
import logging

from .exceptions import (
    ConfigurationError,
    TaskNotFoundError,
    ModelNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Model metadata container"""
    name: str
    task_id: str
    url: str
    checksum: str
    labels: Dict[str, int]
    modality: str  # 'CT', 'PET', 'MR', 'PETCT'
    description: str = ""
    default_preprocessing: str = "default"
    default_postprocessing: str = "default"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'task_id': self.task_id,
            'url': self.url,
            'checksum': self.checksum,
            'labels': self.labels,
            'modality': self.modality,
            'description': self.description,
            'default_preprocessing': self.default_preprocessing,
            'default_postprocessing': self.default_postprocessing,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelInfo':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class TaskDefinition:
    """Complete task definition"""
    name: str
    models: Dict[str, ModelInfo]
    pipeline_config: dict
    input_requirements: dict
    output_config: dict
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'models': {k: v.to_dict() for k, v in self.models.items()},
            'pipeline_config': self.pipeline_config,
            'input_requirements': self.input_requirements,
            'output_config': self.output_config,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskDefinition':
        """Create from dictionary"""
        models = {k: ModelInfo.from_dict(v) for k, v in data['models'].items()}
        return cls(
            name=data['name'],
            models=models,
            pipeline_config=data['pipeline_config'],
            input_requirements=data['input_requirements'],
            output_config=data['output_config'],
        )


class TaskRegistry:
    """
    Central registry for all segmentation tasks.
    
    This is a singleton class that manages task definitions and model paths.
    """
    
    _instance = None
    _tasks: Dict[str, TaskDefinition] = {}
    _model_paths: Dict[str, Path] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, task: TaskDefinition) -> None:
        """Register a new segmentation task"""
        cls._tasks[task.name] = task
        logger.info(f"Registered task: {task.name}")
        
    @classmethod
    def unregister(cls, task_name: str) -> None:
        """Unregister a task"""
        if task_name in cls._tasks:
            del cls._tasks[task_name]
            logger.info(f"Unregistered task: {task_name}")
    
    @classmethod
    def get_task(cls, name: str) -> TaskDefinition:
        """Retrieve task definition"""
        if name not in cls._tasks:
            raise TaskNotFoundError(name, list(cls._tasks.keys()))
        return cls._tasks[name]
    
    @classmethod
    def list_tasks(cls) -> Dict[str, str]:
        """List all registered tasks with descriptions"""
        return {name: task.models[list(task.models.keys())[0]].description 
                for name, task in cls._tasks.items()}
    
    @classmethod
    def register_model_path(cls, model_name: str, path: Path) -> None:
        """Register local model path"""
        cls._model_paths[model_name] = Path(path)
        logger.info(f"Registered model path: {model_name} -> {path}")
        
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get model path, download if necessary"""
        if model_name not in cls._model_paths:
            cls._download_model(model_name)
        return cls._model_paths[model_name]
    
    @classmethod
    def _download_model(cls, model_name: str) -> None:
        """Download model if not available locally"""
        # Find model info from registered tasks
        from ..utils.model_download import ModelDownloader

        for task in cls._tasks.values():
            if model_name in task.models:
                model_info = task.models[model_name]
                model_root = Path(
                    os.environ.get(
                        "NNUNETSEGMENTATOR_MODEL_ROOTPATH",
                        str(Path.home() / ".nnunetsegmentator" / "models"),
                    )
                )
                conventional_path = model_root / task.name / model_info.task_id
                if conventional_path.exists():
                    cls.register_model_path(model_name, conventional_path)
                    return

                if model_info.url.endswith(".git"):
                    raise ModelNotFoundError(
                        model_name,
                        task.name,
                    )

                if "xxxxx" in model_info.url or "zenodo.org/api/files/" in model_info.url:
                    raise ModelNotFoundError(
                        model_name,
                        task.name,
                    )

                logger.info(f"Model {model_name} needs to be downloaded from {model_info.url}")
                downloader = ModelDownloader(model_dir=model_root / task.name)
                model_path = downloader.download_model(
                    model_name=model_name,
                    url=model_info.url,
                    checksum=model_info.checksum,
                )
                
                # Register the path
                cls.register_model_path(model_name, model_path)
                return
        
        raise ModelNotFoundError(model_name)

    @classmethod
    def _resolve_model_name_in_task(cls, task: TaskDefinition, model_identifier: str) -> str:
        """
        Resolve a model identifier within a task.

        The identifier can be either:
        - model name (e.g. "total_organs")
        - task ID (e.g. "Dataset291")
        """
        if model_identifier in task.models:
            return model_identifier

        matched = [
            model_name
            for model_name, model_info in task.models.items()
            if model_info.task_id == model_identifier
        ]
        if len(matched) == 1:
            return matched[0]
        if len(matched) > 1:
            raise ConfigurationError(
                f"Ambiguous model identifier '{model_identifier}' for task '{task.name}'. "
                f"Matched model names: {', '.join(sorted(matched))}"
            )
        raise ModelNotFoundError(model_identifier, task_name=task.name)

    @classmethod
    def install_task_models_from_local(
        cls,
        task_name: str,
        model_sources: Dict[str, Union[str, Path]],
        force: bool = False,
    ) -> Dict[str, Path]:
        """
        Install task model(s) from local archive/directory sources.

        Args:
            task_name: Registered task name.
            model_sources: Mapping of model identifiers to local paths.
                Keys can be model names or task IDs.
                Values can be directories (unzipped) or archives (.zip/.tar/.tar.gz/.tgz).
            force: Overwrite existing installed model directories.

        Returns:
            Mapping of resolved model names to installed model directories.
        """
        if not model_sources:
            raise ConfigurationError("model_sources cannot be empty")

        task = cls.get_task(task_name)
        model_root = Path(
            os.environ.get(
                "NNUNETSEGMENTATOR_MODEL_ROOTPATH",
                str(Path.home() / ".nnunetsegmentator" / "models"),
            )
        )
        task_root = model_root / task.name
        task_root.mkdir(parents=True, exist_ok=True)

        from ..utils.model_download import ModelDownloader

        downloader = ModelDownloader(model_dir=task_root)
        installed: Dict[str, Path] = {}

        for identifier, source in model_sources.items():
            model_name = cls._resolve_model_name_in_task(task, identifier)
            model_info = task.models[model_name]
            source_path = Path(source).expanduser().resolve()

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Local model source does not exist for '{identifier}': {source_path}"
                )

            target_path = task_root / model_info.task_id
            if target_path.exists():
                if not force:
                    raise FileExistsError(
                        f"Target model path already exists for '{model_name}': {target_path}. "
                        "Use force=True to overwrite."
                    )
                shutil.rmtree(target_path)

            if source_path.is_dir():
                shutil.copytree(source_path, target_path)
            elif source_path.is_file():
                if zipfile.is_zipfile(source_path) or tarfile.is_tarfile(source_path):
                    downloader.extract_archive(source_path, target_path)
                else:
                    raise ValueError(
                        f"Unsupported local model file format for '{identifier}': {source_path}. "
                        "Provide a directory or supported archive (.zip, .tar, .tar.gz, .tgz)."
                    )
            else:
                raise ValueError(f"Invalid local model source for '{identifier}': {source_path}")

            cls.register_model_path(model_name, target_path)
            installed[model_name] = target_path
            logger.info(
                "Installed local model '%s' (identifier: %s) to %s",
                model_name,
                identifier,
                target_path,
            )

        return installed
    
    @classmethod
    def save_to_file(cls, filepath: Path) -> None:
        """Save registry to JSON file"""
        data = {
            'tasks': {k: v.to_dict() for k, v in cls._tasks.items()},
            'model_paths': {k: str(v) for k, v in cls._model_paths.items()},
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Registry saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> None:
        """Load registry from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load tasks
        for task_data in data['tasks'].values():
            task = TaskDefinition.from_dict(task_data)
            cls.register(task)
        
        # Load model paths
        for model_name, path_str in data['model_paths'].items():
            cls.register_model_path(model_name, Path(path_str))
        
        logger.info(f"Registry loaded from {filepath}")
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks and model paths"""
        cls._tasks.clear()
        cls._model_paths.clear()
        logger.info("Registry cleared")
