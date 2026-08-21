"""Validate repository-level Zenodo metadata."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA_PATH = REPO_ROOT / ".zenodo.json"

ALLOWED_TOP_LEVEL_FIELDS = {
    "$schema",
    "access_conditions",
    "access_right",
    "communities",
    "contributors",
    "creators",
    "description",
    "doi",
    "embargo_date",
    "grants",
    "imprint_isbn",
    "imprint_place",
    "imprint_publisher",
    "journal_issue",
    "journal_pages",
    "journal_title",
    "journal_volume",
    "keywords",
    "license",
    "conference_acronym",
    "conference_dates",
    "conference_place",
    "conference_session",
    "conference_session_part",
    "conference_title",
    "conference_url",
    "notes",
    "partof_pages",
    "partof_title",
    "publication_date",
    "references",
    "related_identifiers",
    "upload_type",
    "publication_type",
    "image_type",
    "openaire_type",
    "subjects",
    "thesis_supervisors",
    "thesis_university",
    "title",
}

REQUIRED_FIELDS = {
    "access_right",
    "creators",
    "description",
    "keywords",
    "license",
    "publication_date",
    "related_identifiers",
    "title",
    "upload_type",
}

VALID_ACCESS_RIGHTS = {"open", "embargoed", "restricted", "closed"}
VALID_UPLOAD_TYPES = {
    "publication",
    "poster",
    "presentation",
    "dataset",
    "image",
    "video",
    "software",
    "lesson",
    "workflow",
    "physicalobject",
    "other",
}
VALID_IDENTIFIER_SCHEMES = {
    "ads",
    "ark",
    "arrayexpress_array",
    "arrayexpress_experiment",
    "arxiv",
    "ascl",
    "bioproject",
    "biosample",
    "doi",
    "ean13",
    "ean8",
    "ensembl",
    "genome",
    "geo",
    "gnd",
    "hal",
    "handle",
    "isbn",
    "isni",
    "issn",
    "istc",
    "lsid",
    "orcid",
    "pmcid",
    "pmid",
    "purl",
    "refseq",
    "ror",
    "sra",
    "swh",
    "uniprot",
    "url",
    "urn",
}
VALID_RELATIONS = {
    "isCitedBy",
    "cites",
    "isSupplementTo",
    "isSupplementedBy",
    "isContinuedBy",
    "continues",
    "isDescribedBy",
    "describes",
    "hasMetadata",
    "isMetadataFor",
    "isNewVersionOf",
    "isPreviousVersionOf",
    "isPartOf",
    "hasPart",
    "isReferencedBy",
    "references",
    "isDocumentedBy",
    "documents",
    "isCompiledBy",
    "compiles",
    "isVariantFormOf",
    "isOrignialFormOf",
    "isIdenticalTo",
    "isAlternateIdentifier",
    "isReviewedBy",
    "reviews",
    "isDerivedFrom",
    "isSourceOf",
    "requires",
    "isRequiredBy",
    "isObsoletedBy",
    "obsoletes",
    "isPublishedIn",
}
ORCID_PATTERN = re.compile(r"^\d{4}-\d{4}-\d{4}-[\dX]{4}$")
PLACEHOLDER_PATTERN = re.compile(r"\b(TBD|TODO|XXXX+)\b|zenodo\.XXXX+", re.IGNORECASE)


def load_metadata(path: Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Load Zenodo metadata from JSON."""
    with path.open(encoding="utf-8") as metadata_file:
        data = json.load(metadata_file)
    if not isinstance(data, dict):
        raise TypeError("Zenodo metadata must be a JSON object.")
    return data


def validate_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return validation errors for repository Zenodo metadata."""
    errors: list[str] = []

    unknown_fields = sorted(set(metadata) - ALLOWED_TOP_LEVEL_FIELDS)
    if unknown_fields:
        errors.append(f"Unsupported Zenodo metadata fields: {unknown_fields}")

    missing_fields = sorted(REQUIRED_FIELDS - set(metadata))
    if missing_fields:
        errors.append(f"Missing required Zenodo metadata fields: {missing_fields}")

    _validate_string(metadata, "title", errors, min_length=12)
    _validate_string(metadata, "description", errors, min_length=80)
    _validate_string(metadata, "license", errors)
    _validate_string(metadata, "publication_date", errors)

    if metadata.get("upload_type") not in VALID_UPLOAD_TYPES:
        errors.append("upload_type must be a valid Zenodo upload type.")
    if metadata.get("upload_type") != "software":
        errors.append("upload_type must be 'software' for this repository.")
    if metadata.get("access_right") not in VALID_ACCESS_RIGHTS:
        errors.append("access_right must be a valid Zenodo access right.")
    if metadata.get("access_right") != "open":
        errors.append("access_right must be 'open' for public releases.")
    if metadata.get("license") != "mit":
        errors.append("license must use Zenodo's current MIT identifier: 'mit'.")

    _validate_publication_date(metadata.get("publication_date"), errors)
    _validate_creators(metadata.get("creators"), errors)
    _validate_keywords(metadata.get("keywords"), errors)
    _validate_related_identifiers(metadata.get("related_identifiers"), errors)
    _validate_references(metadata.get("references"), errors)
    _validate_no_placeholders(metadata, errors)

    return errors


def _validate_string(
    metadata: dict[str, Any], field: str, errors: list[str], *, min_length: int = 1
) -> None:
    value = metadata.get(field)
    if not isinstance(value, str) or len(value.strip()) < min_length:
        errors.append(
            f"{field} must be a string with at least {min_length} characters."
        )


def _validate_publication_date(value: object, errors: list[str]) -> None:
    if not isinstance(value, str):
        return
    try:
        date.fromisoformat(value)
    except ValueError:
        errors.append("publication_date must use ISO YYYY-MM-DD format.")


def _validate_creators(value: object, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("creators must be a non-empty list.")
        return

    for index, creator in enumerate(value):
        if not isinstance(creator, dict):
            errors.append(f"creators[{index}] must be an object.")
            continue

        name = creator.get("name")
        if not isinstance(name, str) or "," not in name:
            errors.append(f"creators[{index}].name must use 'Family, Given' format.")

        affiliation = creator.get("affiliation")
        if not isinstance(affiliation, str) or not affiliation.strip():
            errors.append(f"creators[{index}].affiliation must be set.")

        orcid = creator.get("orcid")
        if not isinstance(orcid, str) or ORCID_PATTERN.fullmatch(orcid) is None:
            errors.append(f"creators[{index}].orcid must be a bare ORCID iD.")


def _validate_keywords(value: object, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) < 8:
        errors.append("keywords must contain at least eight terms.")
        return

    normalized: set[str] = set()
    for index, keyword in enumerate(value):
        if not isinstance(keyword, str) or not keyword.strip():
            errors.append(f"keywords[{index}] must be a non-empty string.")
            continue
        key = keyword.casefold()
        if key in normalized:
            errors.append(f"Duplicate keyword: {keyword}")
        normalized.add(key)

    required_keywords = {"heavy tails", "extreme value theory", "python"}
    missing = sorted(required_keywords - normalized)
    if missing:
        errors.append(f"Missing core Zenodo keywords: {missing}")


def _validate_related_identifiers(value: object, errors: list[str]) -> None:
    if not isinstance(value, list) or not value:
        errors.append("related_identifiers must be a non-empty list.")
        return

    relations_by_identifier: dict[str, set[str]] = {}
    for index, related_identifier in enumerate(value):
        if not isinstance(related_identifier, dict):
            errors.append(f"related_identifiers[{index}] must be an object.")
            continue

        identifier = related_identifier.get("identifier")
        relation = related_identifier.get("relation")
        scheme = related_identifier.get("scheme")
        if not isinstance(identifier, str) or not identifier.strip():
            errors.append(f"related_identifiers[{index}].identifier must be set.")
        if relation not in VALID_RELATIONS:
            errors.append(f"related_identifiers[{index}].relation is invalid.")
        if scheme not in VALID_IDENTIFIER_SCHEMES:
            errors.append(f"related_identifiers[{index}].scheme is invalid.")

        if isinstance(identifier, str) and isinstance(relation, str):
            relations_by_identifier.setdefault(identifier, set()).add(relation)

    github_url = "https://github.com/DiogoRibeiro7/heavytails"
    docs_url = "https://diogoribeiro7.github.io/heavytails"
    if "isDerivedFrom" not in relations_by_identifier.get(github_url, set()):
        errors.append("GitHub source URL must be related with isDerivedFrom.")
    if "isDocumentedBy" not in relations_by_identifier.get(docs_url, set()):
        errors.append("Documentation URL must be related with isDocumentedBy.")


def _validate_references(value: object, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) < 3:
        errors.append("references must contain at least three scholarly references.")
        return

    if not all(isinstance(reference, str) and reference.strip() for reference in value):
        errors.append("references must be non-empty strings.")

    joined_references = "\n".join(value)
    if "10.1214/aos/1176343247" not in joined_references:
        errors.append("references must include the Hill estimator DOI.")


def _validate_no_placeholders(
    value: object, errors: list[str], path: str = "$"
) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _validate_no_placeholders(child, errors, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_no_placeholders(child, errors, f"{path}[{index}]")
    elif isinstance(value, str) and PLACEHOLDER_PATTERN.search(value):
        errors.append(f"Placeholder value found at {path}.")


def main() -> int:
    """Run Zenodo metadata validation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Path to .zenodo.json",
    )
    args = parser.parse_args()

    metadata = load_metadata(args.path)
    errors = validate_metadata(metadata)
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"{args.path} is valid Zenodo metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
