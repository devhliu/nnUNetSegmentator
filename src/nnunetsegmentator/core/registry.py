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
from ..utils.model_layout import (
    dataset_prefix,
    find_model_payload_root,
    has_model_payload,
    is_canonical_model_key,
    is_canonical_task_id,
    normalize_model_directory,
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
        for model_name, model_info in task.models.items():
            if not is_canonical_task_id(model_info.task_id):
                raise ConfigurationError(
                    f"Invalid task_id '{model_info.task_id}' for model '{model_name}' in task "
                    f"'{task.name}'. task_id must use canonical format "
                    "'Dataset<数字>_<model_name>'."
                )
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
    def _has_model_payload(cls, path: Path) -> bool:
        """Check whether a directory looks like an nnUNet model root."""
        return has_model_payload(path)

    @classmethod
    def _normalize_model_directory(cls, target_path: Path, task_id: str) -> Path:
        """
        Normalize installed layout so `{task_id}` directly contains model artifacts.

        Example normalized target:
            .../{task_name}/{task_id}/nnUNetTrainer__.../
            .../{task_name}/{task_id}/dataset.json
        """
        return normalize_model_directory(target_path, task_id)
        
    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """Get model path, download if necessary"""
        if model_name in cls._model_paths:
            # Verify that the path actually contains dataset.json or plans.json
            path = cls._model_paths[model_name]
            if cls._has_model_payload(path):
                return path
            # If still not valid, re-download or re-resolve
            del cls._model_paths[model_name]
        
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
                
                task_dir = model_root / task.name
                target_path = task_dir / model_info.task_id
                legacy_path = task_dir / dataset_prefix(model_info.task_id)
                if not target_path.exists() and legacy_path.exists():
                    legacy_payload_root = find_model_payload_root(legacy_path, model_info.task_id)
                    if legacy_payload_root:
                        logger.info(
                            "Migrating legacy model dir for '%s': %s -> %s",
                            model_name,
                            legacy_payload_root,
                            target_path,
                        )
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        if legacy_payload_root == legacy_path:
                            shutil.move(str(legacy_path), str(target_path))
                        else:
                            if target_path.exists():
                                shutil.rmtree(target_path)
                            shutil.copytree(legacy_payload_root, target_path)
                        cls._normalize_model_directory(target_path, model_info.task_id)

                if target_path.exists():
                    cls._normalize_model_directory(target_path, model_info.task_id)
                    if cls._has_model_payload(target_path):
                        cls.register_model_path(model_name, target_path)
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
                downloader = ModelDownloader(model_dir=task_dir)
                model_path = downloader.download_model(
                    model_name=model_name,
                    target_dir_name=model_info.task_id,
                    url=model_info.url,
                    checksum=model_info.checksum,
                )

                final_path = model_path
                cls._normalize_model_directory(final_path, model_info.task_id)
                if not cls._has_model_payload(final_path):
                    raise ModelNotFoundError(model_name, task.name)

                cls.register_model_path(model_name, final_path)
                return
        
        raise ModelNotFoundError(model_name)

    @classmethod
    def _canonical_model_key(cls, model_name: str, model_info: ModelInfo) -> str:
        """Build canonical key (same as canonical task_id)."""
        return model_info.task_id

    @classmethod
    def _resolve_model_name_in_task(cls, task: TaskDefinition, model_identifier: str) -> str:
        """
        Resolve a model identifier within a task.

        Supported identifier format is canonical key only:
        - Dataset<数字>_<model_name> (e.g. Dataset291_total_organs), exactly equal to task_id
        """
        canonical_to_model = {
            cls._canonical_model_key(model_name, model_info): model_name
            for model_name, model_info in task.models.items()
        }
        if model_identifier in canonical_to_model:
            return canonical_to_model[model_identifier]

        if model_identifier in task.models or any(
            dataset_prefix(model_info.task_id) == model_identifier
            for model_info in task.models.values()
        ):
            raise ConfigurationError(
                f"Identifier '{model_identifier}' is not supported. "
                "Use canonical key format 'Dataset<数字>_<model_name>', e.g. "
                f"'{next(iter(sorted(canonical_to_model)))}'."
            )

        if is_canonical_model_key(model_identifier):
            raise ModelNotFoundError(model_identifier, task_name=task.name)

        raise ConfigurationError(
            f"Invalid model identifier '{model_identifier}'. "
            "Expected canonical key format 'Dataset<数字>_<model_name>'."
        )

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
                Keys must be canonical keys: Dataset<数字>_<model_name> (same as task_id).
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

            cls._normalize_model_directory(target_path, model_info.task_id)
            if not cls._has_model_payload(target_path):
                detected = find_model_payload_root(target_path, model_info.task_id)
                if detected and detected != target_path:
                    cls._normalize_model_directory(target_path, model_info.task_id)
            if not cls._has_model_payload(target_path):
                raise ConfigurationError(
                    f"Installed directory for '{identifier}' does not contain nnUNet payload "
                    f"(expected under {target_path})."
                )
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
