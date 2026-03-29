"""Task definitions for the nnUNet framework."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Dict, List, Type

from ..core.exceptions import RegistrationError
from ..core.registry import TaskRegistry
from .base import BaseTask

logger = logging.getLogger(__name__)

_DISCOVERY_DONE = False
_DISCOVERED_TASK_CLASSES: Dict[str, Type[BaseTask]] = {}


def _import_task_modules() -> None:
    """Import all task modules so BaseTask subclasses are discoverable."""
    package_name = __name__
    for module_info in pkgutil.iter_modules(__path__):
        module_name = module_info.name
        if module_name.startswith("_") or module_name == "base":
            continue
        importlib.import_module(f"{package_name}.{module_name}")


def discover_task_classes(refresh: bool = False) -> List[Type[BaseTask]]:
    """
    Discover all built-in task classes.

    Any new task module that defines a BaseTask subclass under this package
    is picked up automatically.
    """
    global _DISCOVERY_DONE
    if refresh or not _DISCOVERY_DONE:
        _DISCOVERED_TASK_CLASSES.clear()
        _import_task_modules()

        for task_class in BaseTask.__subclasses__():
            if task_class.__module__.startswith(f"{__name__}."):
                key = task_class.name or task_class.__name__
                _DISCOVERED_TASK_CLASSES[key] = task_class

        _DISCOVERY_DONE = True

    return [_DISCOVERED_TASK_CLASSES[name] for name in sorted(_DISCOVERED_TASK_CLASSES)]


def register_all_tasks(refresh: bool = False) -> List[Type[BaseTask]]:
    """Register all discovered built-in tasks in the global registry."""
    task_classes = discover_task_classes(refresh=refresh)

    failures = {}
    for task_class in task_classes:
        try:
            TaskRegistry.register(task_class.get_definition())
        except (ValueError, TypeError, RuntimeError, OSError) as exc:
            failures[task_class.__name__] = str(exc)
            logger.error("Failed to register %s: %s", task_class.__name__, exc)

    if failures:
        details = "; ".join(f"{task}: {error}" for task, error in failures.items())
        raise RegistrationError(f"Task registration failed for {len(failures)} task(s): {details}")

    return task_classes


for _task_class in discover_task_classes():
    globals()[_task_class.__name__] = _task_class

__all__ = [
    "BaseTask",
    "discover_task_classes",
    "register_all_tasks",
    *[task_class.__name__ for task_class in discover_task_classes()],
]

# Register on import
register_all_tasks()
