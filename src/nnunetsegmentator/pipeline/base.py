"""
Pipeline Base Classes

This module provides the base classes for building composable processing pipelines.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from .. import image as sitk
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    """
    Shared context through pipeline execution.
    
    This object is passed through each pipeline step and accumulates
    results and metadata.
    """
    input_image: sitk.Image
    input_array: np.ndarray
    metadata: Dict[str, Any]
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.intermediate_results is None:
            self.intermediate_results = {}
    
    def get_array(self) -> np.ndarray:
        """Get current working array"""
        return self.input_array
    
    def set_array(self, array: np.ndarray) -> None:
        """Set current working array"""
        self.input_array = array
    
    def get_image(self) -> sitk.Image:
        """Get current working image"""
        return self.input_image
    
    def set_image(self, image: sitk.Image) -> None:
        """Set current working image"""
        self.input_image = image


class PipelineStep(ABC):
    """
    Base class for pipeline steps.
    
    Each step represents a single operation in the processing pipeline.
    """
    
    def __init__(self, name: str, config: dict = None):
        """
        Initialize pipeline step.
        
        Args:
            name: Step name for logging and debugging
            config: Configuration dictionary for the step
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute the pipeline step.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Modified pipeline context
        """
        pass
    
    def __call__(self, context: PipelineContext) -> PipelineContext:
        """Make the step callable"""
        logger.debug(f"Executing step: {self.name}")
        return self.execute(context)
    
    def validate_config(self) -> bool:
        """
        Validate step configuration.
        
        Override this method to add custom validation logic.
        
        Returns:
            True if configuration is valid
        """
        return True


class Pipeline:
    """
    Composable processing pipeline.
    
    A pipeline consists of multiple steps that are executed in sequence.
    Pipelines can be combined using the | operator.
    """
    
    def __init__(self, steps: List[PipelineStep] = None, name: str = "pipeline"):
        """
        Initialize pipeline.
        
        Args:
            steps: List of pipeline steps
            name: Pipeline name for logging
        """
        self.steps = steps or []
        self.name = name
        self._hooks = {'pre': [], 'post': []}
    
    def add_step(self, step: PipelineStep) -> 'Pipeline':
        """
        Add a step to the pipeline (fluent interface).
        
        Args:
            step: Pipeline step to add
            
        Returns:
            Self for chaining
        """
        self.steps.append(step)
        return self
    
    def add_hook(self, stage: str, hook: Callable) -> 'Pipeline':
        """
        Add pre/post execution hook.
        
        Args:
            stage: 'pre' or 'post'
            hook: Hook function that takes and returns PipelineContext
            
        Returns:
            Self for chaining
        """
        if stage not in self._hooks:
            raise ValueError(f"Invalid hook stage: {stage}. Must be 'pre' or 'post'")
        self._hooks[stage].append(hook)
        return self
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute all pipeline steps.
        
        Args:
            context: Initial pipeline context
            
        Returns:
            Final pipeline context after all steps
        """
        logger.info(f"Starting pipeline: {self.name}")
        
        # Run pre-hooks
        for hook in self._hooks['pre']:
            context = hook(context)
        
        # Execute steps
        for i, step in enumerate(self.steps):
            try:
                logger.debug(f"Step {i+1}/{len(self.steps)}: {step.name}")
                context = step.execute(context)
            except (ValueError, RuntimeError, OSError, TypeError) as exc:
                from ..core.exceptions import PipelineError
                logger.exception("Error in step %s", step.name)
                raise PipelineError(self.name, step.name, cause=exc) from exc
        
        # Run post-hooks
        for hook in self._hooks['post']:
            context = hook(context)
        
        logger.info(f"Pipeline completed: {self.name}")
        return context
    
    def __call__(self, context: PipelineContext) -> PipelineContext:
        """Make the pipeline callable"""
        return self.execute(context)
    
    def __or__(self, other: Any) -> 'Pipeline':
        """
        Combine pipelines using | operator.
        
        Args:
            other: Another pipeline or step to combine with
            
        Returns:
            New pipeline with steps from both
        """
        if isinstance(other, Pipeline):
            other_steps = other.steps
            other_name = other.name
        elif isinstance(other, PipelineStep):
            other_steps = [other]
            other_name = other.name
        else:
            raise TypeError(f"Cannot combine Pipeline with {type(other)}")
            
        combined = Pipeline(
            steps=self.steps + other_steps,
            name=f"{self.name}_{other_name}"
        )
        return combined
    
    def __len__(self) -> int:
        """Return number of steps"""
        return len(self.steps)
    
    def __repr__(self) -> str:
        """String representation"""
        step_names = [step.name for step in self.steps]
        return f"Pipeline({self.name}): {' -> '.join(step_names)}"
    
    def validate(self) -> bool:
        """
        Validate all steps in the pipeline.
        
        Returns:
            True if all steps are valid
        """
        for step in self.steps:
            if not step.validate_config():
                logger.error(f"Invalid configuration for step: {step.name}")
                return False
        return True
