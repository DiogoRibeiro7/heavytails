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


def _cff_field(name: str) -> str:
    """Read a top-level scalar out of CITATION.cff."""
    citation = CITATION_METADATA.read_text(encoding="utf-8")
    match = re.search(rf'^{name}:\s*"?([^"\n]+)"?\s*$', citation, re.MULTILINE)
    assert match is not None, f"CITATION.cff has no {name}"
    return match.group(1).strip()


def test_the_citation_version_is_the_package_version() -> None:
    """CITATION.cff describes a release, so it must name the one it ships with."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match is not None, "pyproject.toml has no version"
    assert _cff_field("version") == match.group(1)


def test_the_citation_docs_cite_that_same_version() -> None:
    """Every version named in the citation guidance is the current one.

    This is here because they drifted. CITATION.cff was advanced to 0.4.0 at
    release and the citation page was not, so for a while the documentation
    handed people APA, IEEE, MLA, Chicago and BibTeX entries for 0.3.0 and the
    version DOI of a release they were not running -- while the file the
    "Cite this repository" button reads said something else.

    Nothing checked the two against each other, so nothing said so. Bumping the
    version now fails here until the guidance is bumped with it.
    """
    expected = _cff_field("version")
    for name in ("docs/about/citation.md", "README.md"):
        text = (REPO_ROOT / name).read_text(encoding="utf-8")
        # DOIs come out first. A registrant's suffix can look exactly like a
        # version -- 10.1080/00401706.1993.10485040 reads as one to any
        # three-part pattern -- and citing a method's paper is the whole point
        # of half this page.
        text = re.sub(r"10\.\d{4,}/\S+", "", text)
        found = set(re.findall(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])", text))
        stale = found - {expected}
        assert not stale, f"{name} cites {sorted(stale)}, but the release is {expected}"
