"""Tests for IPC/CrPC -> BNS/BNSS mapping and section prediction."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).parent.parent))

from src.legal_sections.section_predictor import (
    LegalSectionDatabase,
    LegalSectionPredictor,
    NEW_CODE_MAPPING,
)


@pytest.fixture(scope="module")
def db():
    return LegalSectionDatabase()


@pytest.mark.parametrize("ipc,bns", [
    ("302", "BNS 103"), ("307", "BNS 109"), ("420", "BNS 318(4)"),
    ("498A", "BNS 85"), ("376", "BNS 64"),
])
def test_ipc_maps_to_bns(db, ipc, bns):
    assert db.get_section(ipc).new_code_section == bns


def test_crpc_maps_to_bnss(db):
    assert db.get_section("197").new_code_section == "BNSS 218"
    assert db.get_section("41").new_code_section == "BNSS 35"


def test_every_mapping_key_exists(db):
    for (code, number) in NEW_CODE_MAPPING:
        section = db.get_section(number)
        assert section is not None and section.code.name == code


def test_display_name_puts_current_law_first(db):
    assert db.get_section("302").display_name == "BNS 103 (formerly IPC 302)"
    assert db.get_section("14").display_name == "CONSTITUTION 14"  # no new-code equivalent


def test_find_by_new_code(db):
    assert db.find_by_new_code("bns 103").section_number == "302"
    assert db.find_by_new_code("BNS 999") is None


def test_murder_facts_predict_bns_103():
    rec = LegalSectionPredictor().predict_sections(
        "The accused killed the victim with premeditation; the victim died at the scene.")
    numbers = [p.section.section_number for p in rec.primary_sections]
    assert "302" in numbers
    assert rec.primary_sections[0].section.new_code_section == "BNS 103"


def test_explicit_bns_citation_gives_high_confidence(db):
    predictor = LegalSectionPredictor()
    conf = predictor._calculate_confidence("Charged under BNS 103 for the death.", db.get_section("302"))
    assert conf >= 0.9
