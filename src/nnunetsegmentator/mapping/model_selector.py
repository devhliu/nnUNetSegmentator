"""
Model Selector

Selects the segmentation model(s) that cover a requested set of organs,
based on the central label mapping (SNOMED / canonical names).

The selector builds an index over all registered tasks:
    canonical organ name -> [(task, model, label)]

and resolves requests in either direction:
    - standard names / SNOMED codes / aliases ("liver", "SCT:10200004")
    - any model's native label names ("kidney_left", "Spleen")
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..core.registry import TaskRegistry
from .label_mapper import LabelMapper, LabelMappingEntry

logger = logging.getLogger(__name__)

_SNOMED_CODE_PATTERN = re.compile(r"^(?:SCT:)?(\d{6,8})$", re.IGNORECASE)


@dataclass
class OrganProvider:
    """One (task, model, label) triple able to produce a requested organ."""
    task_name: str
    model_name: str
    label_name: str
    label_id: int
    canonical_name: str
    snomed_code: Optional[str] = None
    modality: str = ""

    def to_dict(self) -> Dict:
        return {
            "task": self.task_name,
            "model": self.model_name,
            "label_name": self.label_name,
            "label_id": self.label_id,
            "canonical_name": self.canonical_name,
            "snomed_code": self.snomed_code,
            "modality": self.modality,
        }


@dataclass
class SelectionPlan:
    """Result of model selection for a set of requested organs."""
    task_name: str
    models: List[str]
    organ_to_provider: Dict[str, OrganProvider] = field(default_factory=dict)
    unmatched_organs: List[str] = field(default_factory=list)
    strategy: str = "min_models"

    @property
    def matched_organs(self) -> List[str]:
        return list(self.organ_to_provider.keys())

    def to_dict(self) -> Dict:
        return {
            "task": self.task_name,
            "models": self.models,
            "organs": {
                organ: provider.to_dict()
                for organ, provider in self.organ_to_provider.items()
            },
            "unmatched_organs": self.unmatched_organs,
            "strategy": self.strategy,
        }


class ModelSelector:
    """
    Automatic model selection based on required organs.

    Usage:
        selector = ModelSelector.get_instance()
        plan = selector.select(["liver", "left kidney"], modality="CT")
        # plan.task_name == 'moose', plan.models == ['clin_ct_organs']
    """

    _instance: Optional["ModelSelector"] = None

    def __init__(self, mapper: Optional[LabelMapper] = None):
        self.mapper = mapper or LabelMapper.get_instance()
        self._index: Dict[str, List[OrganProvider]] = {}
        self._model_label_counts: Dict[tuple, int] = {}

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #
    def build_index(self, refresh: bool = False) -> Dict[str, List[OrganProvider]]:
        """
        Build (or return cached) index: canonical organ -> provider list.

        Models with empty label sets (e.g. not yet downloaded/enriched) are
        skipped until their labels become available.
        """
        if self._index and not refresh:
            return self._index

        index: Dict[str, List[OrganProvider]] = {}
        label_counts: Dict[tuple, int] = {}

        for task_name, task in TaskRegistry._tasks.items():
            for model_name, model_info in task.models.items():
                if not model_info.labels:
                    continue
                key = (task_name, model_name)
                label_counts[key] = len(model_info.labels)
                for label_key, label_value in model_info.labels.items():
                    # Some tasks store {label_id: name} instead of {name: label_id}
                    if isinstance(label_key, int) or (
                        isinstance(label_key, str) and label_key.isdigit()
                    ):
                        label_id, label_name = int(label_key), str(label_value)
                    else:
                        label_name, label_id = str(label_key), int(label_value)
                    entry = self._resolve_or_fallback(label_name)
                    index.setdefault(entry.canonical_name, []).append(
                        OrganProvider(
                            task_name=task_name,
                            model_name=model_name,
                            label_name=label_name,
                            label_id=label_id,
                            canonical_name=entry.canonical_name,
                            snomed_code=entry.type.code if entry.type else None,
                            modality=model_info.modality,
                        )
                    )

        self._index = index
        self._model_label_counts = label_counts
        logger.info(
            "Model selector index built: %d canonical organs, %d models",
            len(index),
            len(label_counts),
        )
        return index

    def _resolve_or_fallback(self, label_name: str) -> LabelMappingEntry:
        """Resolve through the mapper; fall back to a native-name entry."""
        try:
            return self.mapper.resolve(label_name, add_fallback=False)
        except KeyError:
            return LabelMappingEntry(canonical_name=label_name, source="native")

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select(
        self,
        organs: List[str],
        modality: Optional[str] = None,
        strategy: str = "min_models",
        task_filter: Optional[str] = None,
    ) -> SelectionPlan:
        """
        Select model(s) covering the requested organs.

        Args:
            organs: Organ names (standard names, SNOMED codes, aliases or
                any model's native label names).
            modality: Optional modality filter ('CT', 'PT', 'MR', ...).
            strategy: 'min_models' (single model preferred, then greedy
                minimal set cover). 
            task_filter: Optional task name restriction.

        Returns:
            SelectionPlan with the chosen task, model list and per-organ
            provider mapping. Organs that cannot be produced by any
            registered model are listed in `unmatched_organs`.
        """
        if not organs:
            raise ValueError("organs must be a non-empty list")
        if strategy != "min_models":
            raise ValueError(f"Unknown selection strategy: '{strategy}'")

        index = self.build_index()

        # Resolve each requested organ to provider candidates
        requests: Dict[str, List[OrganProvider]] = {}
        unmatched: List[str] = []
        for organ in organs:
            canonical, candidates = self._resolve_organ(organ, index)
            if task_filter:
                candidates = [c for c in candidates if c.task_name == task_filter]
            if modality:
                candidates = [
                    c for c in candidates
                    if c.modality.upper() == modality.upper()
                ]
            if candidates:
                requests[canonical] = candidates
            else:
                unmatched.append(organ)

        organ_to_provider: Dict[str, OrganProvider] = {}
        chosen_models: List[str] = []
        task_name = ""

        if requests:
            # Prefer a single model covering all requested organs
            best = self._find_covering_model(requests)
            if best is not None:
                task_name, model_name = best
                chosen_models = [model_name]
            else:
                # Greedy minimal set cover across tasks
                chosen, task_name = self._greedy_cover(requests)
                chosen_models = chosen

            for canonical, candidates in requests.items():
                for provider in candidates:
                    if provider.task_name == task_name and provider.model_name in chosen_models:
                        organ_to_provider[canonical] = provider
                        break

        plan = SelectionPlan(
            task_name=task_name,
            models=chosen_models,
            organ_to_provider=organ_to_provider,
            unmatched_organs=unmatched,
            strategy=strategy,
        )
        logger.info(
            "Selected task '%s' models %s for %d organ(s) (%d unmatched)",
            task_name or "<none>",
            chosen_models,
            len(organ_to_provider),
            len(unmatched),
        )
        return plan

    def _resolve_organ(
        self,
        organ: str,
        index: Dict[str, List[OrganProvider]],
    ) -> tuple:
        """Resolve one requested organ to (canonical_name, candidates)."""
        # SNOMED code input (e.g. '10200004' or 'SCT:10200004')
        code_match = _SNOMED_CODE_PATTERN.match(organ.strip())
        if code_match:
            entries = self.mapper.get_by_snomed(code_match.group(1))
            for entry in entries:
                if entry.canonical_name in index:
                    return entry.canonical_name, index[entry.canonical_name]

        canonical = self._resolve_or_fallback(organ).canonical_name
        if canonical in index:
            return canonical, index[canonical]

        # Fall back to the raw input as canonical key (native naming)
        return organ.strip(), index.get(organ.strip(), [])

    def _model_key(self, provider: OrganProvider) -> tuple:
        return (provider.task_name, provider.model_name)

    def _find_covering_model(
        self,
        requests: Dict[str, List[OrganProvider]],
    ) -> Optional[tuple]:
        """Find a single model covering all requested organs (fewest labels wins)."""
        model_organs: Dict[tuple, set] = {}
        for canonical, candidates in requests.items():
            for provider in candidates:
                model_organs.setdefault(self._model_key(provider), set()).add(canonical)

        covering = [
            key for key, organs in model_organs.items()
            if organs == set(requests.keys())
        ]
        if not covering:
            return None

        return min(
            covering,
            key=lambda key: (self._model_label_counts.get(key, 0), key),
        )

    def _greedy_cover(
        self,
        requests: Dict[str, List[OrganProvider]],
    ) -> tuple:
        """Greedy minimal set cover; returns (model_names, task_name)."""
        uncovered = set(requests.keys())
        model_organs: Dict[tuple, set] = {}
        task_of_model: Dict[tuple, str] = {}
        for canonical, candidates in requests.items():
            for provider in candidates:
                key = self._model_key(provider)
                model_organs.setdefault(key, set()).add(canonical)
                task_of_model[key] = provider.task_name

        chosen: List[tuple] = []
        while uncovered:
            key = max(
                model_organs,
                key=lambda k: (
                    len(model_organs[k] & uncovered),
                    -self._model_label_counts.get(k, 0),
                    k,
                ),
            )
            covered = model_organs[key] & uncovered
            if not covered:
                break
            chosen.append(key)
            uncovered -= covered

        if not chosen:
            return [], ""

        task_name = task_of_model[chosen[0]]
        return [model for _, model in chosen], task_name

    # ------------------------------------------------------------------ #
    # Singleton management
    # ------------------------------------------------------------------ #
    @classmethod
    def get_instance(cls) -> "ModelSelector":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> "ModelSelector":
        cls._instance = None
        return cls._instance
