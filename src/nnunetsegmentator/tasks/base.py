"""
Base Task Class

This module provides the base class for defining segmentation tasks.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..core.registry import TaskDefinition, ModelInfo
from ..pipeline.base import Pipeline
import logging

logger = logging.getLogger(__name__)


class BaseTask(ABC):
    """
    Base class for segmentation tasks.
    
    Each task defines a specific segmentation problem with its models,
    pipeline configuration, and input/output requirements.
    """
    
    name: str = None
    description: str = None
    
    @classmethod
    @abstractmethod
    def get_definition(cls) -> TaskDefinition:
        """
        Return task definition for registry.
        
        Returns:
            TaskDefinition object
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_default_pipeline(cls) -> Pipeline:
        """
        Return default processing pipeline.
        
        Returns:
            Pipeline object
        """
        pass
    
    @classmethod
    def get_info(cls) -> Dict[str, Any]:
        """
        Get task information.
        
        Returns:
            Dictionary with task information
        """
        definition = cls.get_definition()
        
        return {
            'name': definition.name,
            'description': cls.description,
            'models': {k: {
                'task_id': v.task_id,
                'modality': v.modality,
                'labels': v.labels,
            } for k, v in definition.models.items()},
            'input_requirements': definition.input_requirements,
        }
