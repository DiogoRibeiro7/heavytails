"""Tests for repository Zenodo metadata."""

from __future__ import annotations

import json
from pathlib import Path
import re

from scripts.validate_zenodo_metadata import (
    validate_against_citation,
    validate_metadata,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ZENODO_METADATA = REPO_ROOT / ".zenodo.json"
CITATION_METADATA = REPO_ROOT / "CITATION.cff"


def test_zenodo_metadata_is_valid() -> None:
    """The Zenodo metadata should satisfy repository policy checks."""
    with ZENODO_METADATA.open(encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    assert validate_metadata(metadata) == []
    assert validate_against_citation(metadata) == []


def test_zenodo_metadata_matches_citation_cff_core_fields() -> None:
    """Keep Zenodo and CITATION.cff aligned on stable citation fields."""
    with ZENODO_METADATA.open(encoding="utf-8") as metadata_file:
        zenodo = json.load(metadata_file)

    citation = CITATION_METADATA.read_text(encoding="utf-8")

    # Read the title out of CITATION.cff and require Zenodo to match it,
    # rather than writing it here a third time. The literal that used to sit
    # in this test was the reason a one-word change to the title broke two
    # tests that were supposed to be checking the two files against each
    # other: a transcription cannot notice that it has gone stale.
    cff_title = re.search(r'^title:\s*"(.+)"$', citation, re.MULTILINE)
    assert cff_title is not None, "CITATION.cff has no quoted title"
    assert zenodo["title"] == cff_title.group(1)
    assert zenodo["creators"][0]["name"] == "Ribeiro, Diogo"
    assert zenodo["creators"][0]["orcid"] in citation
    assert zenodo["publication_date"] in citation
    assert zenodo["license"] == "mit"
    assert "license: MIT" in citation


def test_zenodo_metadata_rejects_citation_drift() -> None:
    """Zenodo validation should fail when citation metadata drifts."""
    with ZENODO_METADATA.open(encoding="utf-8") as metadata_file:
        zenodo = json.load(metadata_file)

    zenodo["publication_date"] = "2025-01-01"

    errors = validate_against_citation(zenodo)

    assert "Zenodo publication_date must match CITATION.cff date-released." in errors
