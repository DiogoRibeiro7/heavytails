"""Tests for repository Zenodo metadata."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.validate_zenodo_metadata import validate_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_METADATA = REPO_ROOT / ".zenodo.json"
CITATION_METADATA = REPO_ROOT / "CITATION.cff"


def test_zenodo_metadata_is_valid() -> None:
    """The Zenodo metadata should satisfy repository policy checks."""
    with ZENODO_METADATA.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    assert validate_metadata(metadata) == []


def test_zenodo_metadata_matches_citation_cff_core_fields() -> None:
    """Keep Zenodo and CITATION.cff aligned on stable citation fields."""
    with ZENODO_METADATA.open(encoding="utf-8") as metadata_file:
        zenodo = json.load(metadata_file)

    citation = CITATION_METADATA.read_text(encoding="utf-8")

    assert "title: heavytails" in citation
    assert zenodo["creators"][0]["name"] == "Ribeiro, Diogo"
    assert zenodo["creators"][0]["orcid"] in citation
    assert zenodo["publication_date"] in citation
    assert zenodo["license"] == "mit"
    assert "license: MIT" in citation
