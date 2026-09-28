"""
Central Label Mapping (SNOMED)

This module provides a unified mapping between model-specific organ/sub-organ
label names and standard anatomical terminology (SNOMED CT codes), in a
DICOM-SEG-compatible coding schema.

The mapping data ships as:
    data/moose_snomed_mapping.csv  - curated label -> SNOMED coding table,
                                     including the optional ``merge_group``
                                     column used to collapse fine-grained
                                     labels into a coarser concept
    data/label_aliases.json        - synonyms and model-specific aliases that
                                     resolve onto canonical label names

Labels without a curated SNOMED entry fall back to a synthesized entry whose
canonical name is the label name itself, so every registered task label can
participate in mapping and model selection. Entries can be extended via the
aliases JSON without code changes.
"""

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
CSV_FILENAME = "moose_snomed_mapping.csv"
ALIASES_FILENAME = "label_aliases.json"

_CODE_SEQUENCE_PREFIXES = (
    "SegmentedPropertyCategoryCodeSequence",
    "SegmentedPropertyTypeCodeSequence",
    "SegmentedPropertyTypeModifierCodeSequence",
    "AnatomicRegionSequence",
    "AnatomicRegionModifierSequence",
)

_LATERALITY_SUFFIXES = {
    "left": "left",
    "right": "right",
    "l": "left",
    "r": "right",
}


@dataclass(frozen=True)
class SnomedCode:
    """A single coded concept (coding scheme + code + meaning)."""
    scheme: str
    code: str
    meaning: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "CodingSchemeDesignator": self.scheme,
            "CodeValue": self.code,
            "CodeMeaning": self.meaning,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, str]]) -> Optional["SnomedCode"]:
        if not data:
            return None
        return cls(
            scheme=data.get("CodingSchemeDesignator", ""),
            code=data.get("CodeValue", ""),
            meaning=data.get("CodeMeaning", ""),
        )


@dataclass
class LabelMappingEntry:
    """
    Unified label mapping entry for one canonical organ/sub-organ label.

    Field structure follows the DICOM SEG coding schema so entries map directly
    onto DICOM SEG metadata.
    """
    canonical_name: str
    category: Optional[SnomedCode] = None
    type: Optional[SnomedCode] = None
    type_modifier: Optional[SnomedCode] = None
    anatomic_region: Optional[SnomedCode] = None
    anatomic_region_modifier: Optional[SnomedCode] = None
    rgb: Optional[List[int]] = None
    merge_group: Optional[str] = None
    source: str = "fallback"

    @property
    def snomed_code(self) -> Optional[SnomedCode]:
        """Primary SNOMED concept for this label (the property type)."""
        return self.type

    @property
    def laterality(self) -> Optional[str]:
        """Laterality derived from the type modifier, if any ('left'/'right')."""
        if self.type_modifier is None:
            return None
        meaning = (self.type_modifier.meaning or "").strip().lower()
        if meaning in ("left", "right"):
            return meaning
        return None

    def to_dict(self) -> Dict:
        return {
            "canonical_name": self.canonical_name,
            "SegmentedPropertyCategoryCodeSequence": self.category.to_dict() if self.category else None,
            "SegmentedPropertyTypeCodeSequence": self.type.to_dict() if self.type else None,
            "SegmentedPropertyTypeModifierCodeSequence": self.type_modifier.to_dict() if self.type_modifier else None,
            "AnatomicRegionSequence": self.anatomic_region.to_dict() if self.anatomic_region else None,
            "AnatomicRegionModifierSequence": self.anatomic_region_modifier.to_dict() if self.anatomic_region_modifier else None,
            "recommendedDisplayRGBValue": self.rgb,
            "merge_group": self.merge_group,
            "source": self.source,
        }


def normalize_label_name(name: str) -> str:
    """Normalize a label name for lookup (case, whitespace, separators)."""
    normalized = (name or "").strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    return normalized


def _parse_rgb(raw: str) -> Optional[List[int]]:
    raw = (raw or "").strip().strip("[]")
    if not raw:
        return None
    try:
        values = [int(part) for part in raw.split(",")]
    except ValueError:
        return None
    return values if len(values) == 3 else None


class LabelMapper:
    """
    Central bidirectional mapper between model label names and SNOMED codes.

    Usage:
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("left kidney")          # -> kidney_left entry
        entry = mapper.resolve("kidney_l", laterality="left")
        entry = mapper.get_by_snomed("10200004")       # -> liver entry
        mapping = mapper.get_model_label_mapping({'liver': 1, 'spleen': 2})
    """

    _instance: Optional["LabelMapper"] = None

    def __init__(self, data_dir: Union[str, Path] = None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._by_name: Dict[str, LabelMappingEntry] = {}
        self._by_normalized: Dict[str, str] = {}
        self._by_snomed: Dict[str, List[str]] = {}
        self._aliases: Dict[str, str] = {}
        self._load()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        self._load_csv()
        self._load_aliases()

    def _load_csv(self) -> None:
        csv_path = self.data_dir / CSV_FILENAME
        if not csv_path.exists():
            logger.warning("Label mapping CSV not found: %s", csv_path)
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                label_name = (row.get("label_name") or "").strip()
                if not label_name:
                    continue

                codes = {}
                for prefix in _CODE_SEQUENCE_PREFIXES:
                    value = (row.get(f"{prefix}.CodeValue") or "").strip()
                    codes[prefix] = SnomedCode(
                        scheme=(row.get(f"{prefix}.CodingSchemeDesignator") or "").strip(),
                        code=value,
                        meaning=(row.get(f"{prefix}.CodeMeaning") or "").strip(),
                    ) if value else None

                entry = LabelMappingEntry(
                    canonical_name=label_name,
                    category=codes["SegmentedPropertyCategoryCodeSequence"],
                    type=codes["SegmentedPropertyTypeCodeSequence"],
                    type_modifier=codes["SegmentedPropertyTypeModifierCodeSequence"],
                    anatomic_region=codes["AnatomicRegionSequence"],
                    anatomic_region_modifier=codes["AnatomicRegionModifierSequence"],
                    rgb=_parse_rgb(row.get("recommendedDisplayRGBValue")),
                    merge_group=(row.get("merge_group") or "").strip() or None,
                    source="moose_csv",
                )

                existing = self._by_name.get(label_name)
                if existing is not None:
                    if existing.to_dict() != entry.to_dict():
                        logger.warning(
                            "Conflicting SNOMED coding for label '%s' in %s; keeping first entry",
                            label_name, csv_path.name,
                        )
                    continue

                self._by_name[label_name] = entry
                self._by_normalized[normalize_label_name(label_name)] = label_name
                if entry.type is not None:
                    self._by_snomed.setdefault(entry.type.code, []).append(label_name)

        logger.info("Loaded %d SNOMED label mappings from %s", len(self._by_name), csv_path.name)

    def _load_aliases(self) -> None:
        aliases_path = self.data_dir / ALIASES_FILENAME
        if not aliases_path.exists():
            return

        try:
            with open(aliases_path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load label aliases from %s: %s", aliases_path, exc)
            return

        for alias, target in (data.get("aliases") or {}).items():
            normalized = normalize_label_name(alias)
            if normalized in self._aliases and self._aliases[normalized] != target:
                logger.warning("Conflicting alias '%s'; keeping first target", alias)
                continue
            self._aliases[normalized] = target

        # Optional curated entries for labels that lack CSV coverage
        for name, entry_data in (data.get("entries") or {}).items():
            if name in self._by_name:
                continue
            entry = LabelMappingEntry(
                canonical_name=name,
                category=SnomedCode.from_dict(entry_data.get("category")),
                type=SnomedCode.from_dict(entry_data.get("type")),
                type_modifier=SnomedCode.from_dict(entry_data.get("type_modifier")),
                anatomic_region=SnomedCode.from_dict(entry_data.get("anatomic_region")),
                anatomic_region_modifier=SnomedCode.from_dict(entry_data.get("anatomic_region_modifier")),
                rgb=entry_data.get("rgb"),
                source="aliases_json",
            )
            self._by_name[name] = entry
            self._by_normalized[normalize_label_name(name)] = name
            if entry.type is not None:
                self._by_snomed.setdefault(entry.type.code, []).append(name)

        logger.info("Loaded %d label aliases from %s", len(self._aliases), aliases_path.name)

    # ------------------------------------------------------------------ #
    # Lookup
    # ------------------------------------------------------------------ #
    def resolve(
        self,
        name: str,
        laterality: Optional[str] = None,
        add_fallback: bool = True,
    ) -> LabelMappingEntry:
        """
        Resolve a label name (any model's naming) to a LabelMappingEntry.

        Resolution order:
            1. exact canonical name
            2. normalized name (case/separator-insensitive)
            3. alias table (e.g. 'left kidney' -> 'kidney_left')
            4. laterality suffix variants ('kidney' + left -> 'kidney_left')
            5. fallback entry (canonical = input name, no SNOMED codes)

        Args:
            name: Label name in any model's vocabulary.
            laterality: Optional 'left'/'right' hint to disambiguate
                paired structures.
            add_fallback: Register the fallback entry so repeated lookups and
                the SNOMED index stay consistent.
        """
        if not name:
            raise ValueError("Label name must be non-empty")

        normalized = normalize_label_name(name)

        # 1-2: direct and normalized canonical hits
        if name in self._by_name:
            return self._by_name[name]
        canonical = self._by_normalized.get(normalized)
        if canonical is not None:
            return self._by_name[canonical]

        # 3: alias table
        alias_target = self._aliases.get(normalized)
        if alias_target is not None:
            target = self._by_name.get(alias_target)
            if target is not None:
                return target

        # 4: laterality suffix variants ('kidney' + left -> 'kidney_left',
        #    'kidney_l' -> 'kidney_left')
        lat = (laterality or "").strip().lower()
        lat = _LATERALITY_SUFFIXES.get(lat, lat)
        base = normalized
        for suffix in ("_left", "_right", "_l", "_r"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        candidates = []
        if lat in ("left", "right"):
            candidates.append(f"{base}_{lat}")
        for suffix in ("left", "right"):
            candidates.append(f"{base}_{suffix}")
        for candidate in candidates:
            canonical = self._by_normalized.get(normalize_label_name(candidate))
            if canonical is not None:
                return self._by_name[canonical]

        # 5: fallback
        if add_fallback:
            entry = LabelMappingEntry(canonical_name=name, source="fallback")
            self._by_name[name] = entry
            self._by_normalized[normalized] = name
            logger.debug("No SNOMED mapping for label '%s'; using fallback entry", name)
            return entry

        raise KeyError(f"Cannot resolve label name: '{name}'")

    def get_by_snomed(self, code: str, scheme: str = "SCT") -> List[LabelMappingEntry]:
        """Reverse lookup: SNOMED code -> matching label entries (optionally filtered by scheme)."""
        results = []
        for canonical in self._by_snomed.get(code, []):
            entry = self._by_name.get(canonical)
            if entry is None:
                continue
            if entry.type is not None and scheme and entry.type.scheme != scheme:
                continue
            results.append(entry)
        return results

    def get_entry(self, canonical_name: str) -> Optional[LabelMappingEntry]:
        """Return the curated entry for a canonical name, if present."""
        return self._by_name.get(canonical_name)

    def get_merge_group(self, label_name: str) -> Optional[str]:
        """
        Return the coarser label name this label collapses into, if any.

        Labels sharing the same non-empty merge group are combined into a
        single label by the ``merge_labels`` pipeline step (e.g. the left lung
        lobes merge into 'Left Lung'). Labels without a merge group, or whose
        name cannot be resolved, return ``None``.
        """
        try:
            entry = self.resolve(label_name, add_fallback=False)
        except (KeyError, ValueError):
            return None
        return entry.merge_group

    def list_entries(self) -> List[LabelMappingEntry]:
        """All curated (non-fallback) entries."""
        return [e for e in self._by_name.values() if e.source != "fallback"]

    def get_model_label_mapping(
        self,
        labels: Dict[str, int],
        model_name: str = "",
    ) -> Dict[int, LabelMappingEntry]:
        """
        Map a model's {label_name: label_id} dictionary to
        {label_id: LabelMappingEntry}.
        """
        mapping: Dict[int, LabelMappingEntry] = {}
        for label_name, label_id in labels.items():
            try:
                mapping[int(label_id)] = self.resolve(label_name)
            except (ValueError, TypeError):
                logger.warning("Skipping invalid label id for '%s' in model '%s'", label_name, model_name)
        return mapping

    # ------------------------------------------------------------------ #
    # Singleton management
    # ------------------------------------------------------------------ #
    @classmethod
    def get_instance(cls) -> "LabelMapper":
        """Return the shared LabelMapper instance (lazy creation)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls, data_dir: Union[str, Path] = None) -> "LabelMapper":
        """Reset the shared instance (mainly for tests / custom data dirs)."""
        cls._instance = cls(data_dir=data_dir) if data_dir else None
        return cls._instance
