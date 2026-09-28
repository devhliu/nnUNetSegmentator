"""Tests for the central SNOMED label mapper"""

import pytest

from nnunetsegmentator.mapping.label_mapper import (
    LabelMapper,
    normalize_label_name,
)


@pytest.fixture(autouse=True)
def fresh_mapper():
    """Isolate the shared mapper instance for every test."""
    LabelMapper.reset_instance()
    yield
    LabelMapper.reset_instance()


class TestNormalizeLabelName:
    def test_case_and_separators(self):
        assert normalize_label_name("  Left  Kidney ") == "left_kidney"
        assert normalize_label_name("Vertebrae-L1") == "vertebrae_l1"
        assert normalize_label_name("SPLEEN") == "spleen"

    def test_empty(self):
        assert normalize_label_name(None) == ""
        assert normalize_label_name("") == ""


class TestResolve:
    def test_exact_canonical(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("liver")
        assert entry.canonical_name == "liver"
        assert entry.source == "moose_csv"
        assert entry.type is not None
        assert entry.type.code == "10200004"
        assert entry.type.meaning == "Liver"

    def test_normalized_lookup(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("Liver")
        assert entry.canonical_name == "liver"

    def test_alias_lookup(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("left kidney")
        assert entry.canonical_name == "kidney_left"

    def test_laterality_suffix_variants(self):
        mapper = LabelMapper.get_instance()
        assert mapper.resolve("kidney_l").canonical_name == "kidney_left"
        assert mapper.resolve("kidney_right").canonical_name == "kidney_right"

    def test_laterality_hint(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("kidney", laterality="left")
        assert entry.canonical_name == "kidney_left"

    def test_fallback_unknown_label(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("totally_unknown_organ")
        assert entry.canonical_name == "totally_unknown_organ"
        assert entry.source == "fallback"
        assert entry.type is None
        # Fallback entries are registered for repeat lookups
        assert mapper.resolve("totally_unknown_organ") is entry

    def test_empty_name_raises(self):
        mapper = LabelMapper.get_instance()
        with pytest.raises(ValueError):
            mapper.resolve("")
        with pytest.raises(ValueError):
            mapper.resolve(None)

    def test_resolve_without_fallback_raises(self):
        mapper = LabelMapper.get_instance()
        with pytest.raises(KeyError):
            mapper.resolve("totally_unknown_organ", add_fallback=False)


class TestReverseLookup:
    def test_get_by_snomed(self):
        mapper = LabelMapper.get_instance()
        entries = mapper.get_by_snomed("10200004")
        assert any(e.canonical_name == "liver" for e in entries)

    def test_get_by_snomed_unknown(self):
        mapper = LabelMapper.get_instance()
        assert mapper.get_by_snomed("99999999") == []

    def test_get_model_label_mapping(self):
        mapper = LabelMapper.get_instance()
        mapping = mapper.get_model_label_mapping({"liver": 1, "spleen": 2})
        assert set(mapping.keys()) == {1, 2}
        assert mapping[1].canonical_name == "liver"
        assert mapping[2].canonical_name == "spleen"


class TestEntryStructure:
    def test_laterality_from_type_modifier(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("kidney_left")
        assert entry.laterality == "left"
        assert entry.type_modifier is not None
        assert entry.type_modifier.code == "7771000"

    def test_rgb(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("liver")
        assert entry.rgb == [221, 130, 101]

    def test_to_dict_round_trip(self):
        mapper = LabelMapper.get_instance()
        entry = mapper.resolve("liver")
        data = entry.to_dict()
        assert data["canonical_name"] == "liver"
        assert data["SegmentedPropertyTypeCodeSequence"]["CodeValue"] == "10200004"


class TestSingleton:
    def test_get_instance_shared(self):
        assert LabelMapper.get_instance() is LabelMapper.get_instance()

    def test_reset_instance(self):
        first = LabelMapper.get_instance()
        LabelMapper.reset_instance()
        assert LabelMapper.get_instance() is not first
