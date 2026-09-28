"""
Task and Model Registry System

This module provides a centralized registry for managing segmentation tasks and models.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
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
    is_downloadable_url,
    iter_model_candidates,
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
    def find_model(cls, model_name: str):
        """
        Find a registered model by name across all tasks.

        Returns:
            Tuple (task, model_info) or None when the model is unknown.
        """
        for task in cls._tasks.values():
            model_info = task.models.get(model_name)
            if model_info is not None:
                return task, model_info
        return None
    
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
    def model_root(cls) -> Path:
        """Root directory holding all installed models."""
        return Path(
            os.environ.get(
                "NNUNETSEGMENTATOR_MODEL_ROOTPATH",
                str(Path.home() / ".nnunetsegmentator" / "models"),
            )
        )

    @classmethod
    def canonical_model_dir(cls, task: TaskDefinition, model_name: str) -> Path:
        """Canonical install location for a model: {model_root}/{task_name}/{task_id}."""
        return cls.model_root() / task.name / task.models[model_name].task_id

    @classmethod
    def get_model_path(cls, model_name: str) -> Path:
        """
        Resolve the location of an installed model.

        Pure lookup — never downloads. An explicitly registered path wins;
        otherwise the canonical location `{model_root}/{task}/{task_id}` is
        checked. Raises ModelNotFoundError when the model is not installed,
        in which case call `download_model()` explicitly.
        """
        if model_name in cls._model_paths:
            # Verify that the path actually contains dataset.json or plans.json
            path = cls._model_paths[model_name]
            if cls._has_model_payload(path):
                return path
            del cls._model_paths[model_name]

        for task in cls._tasks.values():
            if model_name not in task.models:
                continue
            target_path = cls.canonical_model_dir(task, model_name)
            if cls._has_model_payload(target_path):
                cls.register_model_path(model_name, target_path)
                return target_path
            raise ModelNotFoundError(model_name, task.name)

        raise ModelNotFoundError(model_name)

    @classmethod
    def download_model(cls, model_name: str, force: bool = False) -> Path:
        """
        Download a registered model into canonical storage.

        Explicit entry point — inference never triggers this. Normalizes the
        installed directory so `{task_id}` directly contains the nnUNet payload.

        Args:
            model_name: Name of a model registered by a task.
            force: Remove an existing installation and re-download.

        Returns:
            Path to the installed model directory.
        """
        # Find model info from registered tasks
        from ..utils.model_download import ModelDownloader

        for task in cls._tasks.values():
            if model_name in task.models:
                model_info = task.models[model_name]
                task_dir = cls.model_root() / task.name
                target_path = cls.canonical_model_dir(task, model_name)

                if force and target_path.exists():
                    logger.info(f"Removing existing installation for '{model_name}': {target_path}")
                    shutil.rmtree(target_path)
                    cls._model_paths.pop(model_name, None)

                if target_path.exists():
                    cls._normalize_model_directory(target_path, model_info.task_id)
                    if cls._has_model_payload(target_path):
                        cls.register_model_path(model_name, target_path)
                        return target_path

                if not is_downloadable_url(model_info.url):
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
                    force=force,
                )

                final_path = model_path
                cls._normalize_model_directory(final_path, model_info.task_id)
                if not cls._has_model_payload(final_path):
                    raise ModelNotFoundError(model_name, task.name)

                cls.register_model_path(model_name, final_path)
                return final_path

        raise ModelNotFoundError(model_name)

    @classmethod
    def download_task_models(
        cls,
        task_name: str,
        model_names: Optional[Union[str, list]] = None,
        force: bool = False,
    ) -> Dict[str, Path]:
        """
        Download one, several, or all models of a task from their registered URLs.

        Args:
            task_name: Registered task name.
            model_names: Single model name or list of names; None means all
                models of the task.
            force: Remove existing installations and re-download.

        Returns:
            Mapping of model names to installed model directories.
        """
        task = cls.get_task(task_name)
        if model_names is None:
            names = list(task.models)
        elif isinstance(model_names, str):
            names = [model_names]
        else:
            names = list(model_names)

        if not names:
            raise ConfigurationError("model_names cannot be empty")

        unknown = [name for name in names if name not in task.models]
        if unknown:
            raise ConfigurationError(
                f"Unknown model(s) for task '{task_name}': {', '.join(unknown)}. "
                f"Available: {', '.join(task.models)}"
            )

        downloaded: Dict[str, Path] = {}
        for name in names:
            downloaded[name] = cls.download_model(name, force=force)
        return downloaded

    @classmethod
    def _canonical_model_key(cls, model_info: ModelInfo) -> str:
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
            cls._canonical_model_key(model_info): model_name
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
        model_root = cls.model_root()
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
    def _registered_model_index(
        cls, task_name: Optional[str] = None
    ) -> tuple:
        """
        Build lookup tables for matching discovered candidates to registered models.

        Returns (by_task_id, by_prefix) where by_task_id maps task_id to
        (task_name, model_name) and by_prefix maps Dataset<数字> prefixes to the
        list of (task_name, model_name, task_id) entries sharing that prefix.
        """
        task_names = [task_name] if task_name else list(cls._tasks.keys())
        by_task_id: Dict[str, tuple] = {}
        by_prefix: Dict[str, list] = {}
        for name in task_names:
            task = cls.get_task(name)
            for model_name, model_info in task.models.items():
                by_task_id[model_info.task_id] = (task.name, model_name)
                by_prefix.setdefault(dataset_prefix(model_info.task_id), []).append(
                    (task.name, model_name, model_info.task_id)
                )
        return by_task_id, by_prefix

    @staticmethod
    def _candidate_name_tokens(candidate: Path, root: Path) -> list:
        """Collect the candidate's own name plus ancestor directory names up to root."""
        name = candidate.name
        if candidate.is_file() and name.lower().endswith(".tar.gz"):
            name = name[: -len(".tar.gz")]
        elif candidate.is_file():
            name = candidate.stem
        tokens = [name]
        for parent in candidate.parents:
            tokens.append(parent.name)
            if parent == root:
                break
        return tokens

    @staticmethod
    def _match_candidate(tokens: list, by_task_id: Dict[str, tuple], by_prefix: Dict[str, list]):
        """
        Match candidate name tokens against registered models.

        Returns (entry, None) on a unique match, (None, info) when ambiguous, and
        (None, None) when nothing matches. Deterministic order: exact task_id,
        exact Dataset prefix, contains task_id, contains Dataset prefix.
        """
        for token in tokens:
            if token in by_task_id:
                return by_task_id[token], None

        for token in tokens:
            if token in by_prefix:
                entries = by_prefix[token]
                if len({(task, model) for task, model, _ in entries}) == 1:
                    task, model, _ = entries[0]
                    return (task, model), None
                return None, token

        container = [(task_id, entry) for token in tokens
                     for task_id, entry in by_task_id.items() if task_id in token]
        if container:
            return max(container, key=lambda item: len(item[0]))[1], None

        for token in tokens:
            for prefix, entries in by_prefix.items():
                position = token.find(prefix)
                if position == -1:
                    continue
                end = position + len(prefix)
                if end < len(token) and token[end].isdigit():
                    continue
                if len({(task, model) for task, model, _ in entries}) == 1:
                    task, model, _ = entries[0]
                    return (task, model), None
                return None, prefix

        return None, None

    @classmethod
    def install_models_from_root(
        cls,
        root: Union[str, Path],
        task_name: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Discover model weights under a root directory and install them.

        Candidates are payload directories (`dataset.json`/`plans.json` present)
        and weight archives (.zip/.tar/.tar.gz/.tgz). Each candidate is matched to
        a registered model by canonical task_id or Dataset<数字> prefix found in
        the candidate or ancestor directory names; ambiguous candidates are skipped.

        Args:
            root: Directory to search recursively.
            task_name: Restrict matching to a single registered task.
            force: Overwrite existing installed model directories.
            dry_run: Only compute the plan without writing anything.

        Returns:
            Mapping with keys 'installed', 'planned', 'unmatched', 'ambiguous',
            'skipped'.
        """
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists() or not root_path.is_dir():
            raise ConfigurationError(f"Model source root is not a directory: {root_path}")

        by_task_id, by_prefix = cls._registered_model_index(task_name)

        planned: Dict[tuple, tuple] = {}
        unmatched: list = []
        ambiguous: list = []
        skipped: list = []

        for candidate in iter_model_candidates(root_path):
            tokens = cls._candidate_name_tokens(candidate, root_path)
            entry, conflict = cls._match_candidate(tokens, by_task_id, by_prefix)
            if entry is None and conflict is not None:
                ambiguous.append((str(candidate), conflict))
                continue
            if entry is None:
                unmatched.append(str(candidate))
                continue
            task, model = entry
            model_info = cls.get_task(task).models[model]
            if entry in planned:
                skipped.append((str(candidate), model_info.task_id))
                continue
            planned[entry] = (model_info.task_id, candidate)

        result: Dict[str, Any] = {
            "installed": {},
            "planned": {
                model: {"task": task, "task_id": task_id, "source": str(source)}
                for (task, model), (task_id, source) in planned.items()
            },
            "unmatched": unmatched,
            "ambiguous": ambiguous,
            "skipped": skipped,
        }
        if dry_run:
            return result

        per_task: Dict[str, Dict[str, Path]] = {}
        for (task, _model), (task_id, source) in planned.items():
            per_task.setdefault(task, {})[task_id] = source

        installed: Dict[str, Path] = {}
        for task, sources in per_task.items():
            installed.update(
                cls.install_task_models_from_local(task, sources, force=force)
            )
        result["installed"] = installed
        return result

    @classmethod
    def clear(cls) -> None:
        """Clear all registered tasks and model paths"""
        cls._tasks.clear()
        cls._model_paths.clear()
        logger.info("Registry cleared")
