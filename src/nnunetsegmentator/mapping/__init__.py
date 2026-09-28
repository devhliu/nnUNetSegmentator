"""
Central Label Mapping Package

Provides unified mapping between model-specific organ/sub-organ label names
and standard anatomical terminology (SNOMED CT), plus model selection based
on required organs.
"""

from .label_mapper import (
    LabelMapper,
    LabelMappingEntry,
    SnomedCode,
    normalize_label_name,
)
from .model_selector import ModelSelector, SelectionPlan

__all__ = [
    "LabelMapper",
    "LabelMappingEntry",
    "SnomedCode",
    "ModelSelector",
    "SelectionPlan",
    "normalize_label_name",
]
