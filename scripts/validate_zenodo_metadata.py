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
DEFAULT_CITATION_PATH = REPO_ROOT / "CITATION.cff"

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


def validate_against_citation(
    metadata: dict[str, Any], citation_path: Path = DEFAULT_CITATION_PATH
) -> list[str]:
    """Return errors for mismatches between Zenodo metadata and CITATION.cff."""
    citation = _parse_citation_cff(citation_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if metadata.get("title") != citation.get("title"):
        errors.append("Zenodo title must match CITATION.cff title.")

    citation_abstract = citation.get("abstract")
    if isinstance(citation_abstract, str) and _normalized_text(
        metadata.get("description")
    ) != _normalized_text(citation_abstract):
        errors.append("Zenodo description must match CITATION.cff abstract.")

    if metadata.get("publication_date") != citation.get("date-released"):
        errors.append("Zenodo publication_date must match CITATION.cff date-released.")

    if metadata.get("license") != _zenodo_license_id(citation.get("license")):
        errors.append("Zenodo license must match CITATION.cff license.")

    _validate_creator_against_citation(metadata, citation, errors)
    _validate_keywords_against_citation(metadata, citation, errors)
    _validate_related_identifiers_against_citation(metadata, citation, errors)
    _validate_references_against_citation(metadata, citation, errors)

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


def _validate_creator_against_citation(
    metadata: dict[str, Any], citation: dict[str, Any], errors: list[str]
) -> None:
    creators = metadata.get("creators")
    cff_author = citation.get("first_author")
    if (
        not isinstance(creators, list)
        or not creators
        or not isinstance(cff_author, dict)
    ):
        return

    creator = creators[0]
    if not isinstance(creator, dict):
        return

    expected_name = f"{cff_author.get('family-names')}, {cff_author.get('given-names')}"
    if creator.get("name") != expected_name:
        errors.append("Zenodo first creator must match CITATION.cff first author.")

    expected_orcid = _bare_orcid(cff_author.get("orcid"))
    if creator.get("orcid") != expected_orcid:
        errors.append(
            "Zenodo first creator ORCID must match CITATION.cff first author."
        )

    affiliation = cff_author.get("affiliation")
    creator_affiliation = creator.get("affiliation")
    if (
        isinstance(affiliation, str)
        and isinstance(creator_affiliation, str)
        and _ascii_fold(creator_affiliation) != _ascii_fold(affiliation)
    ):
        errors.append(
            "Zenodo first creator affiliation must match CITATION.cff first author."
        )


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


def _validate_keywords_against_citation(
    metadata: dict[str, Any], citation: dict[str, Any], errors: list[str]
) -> None:
    zenodo_keywords = metadata.get("keywords")
    cff_keywords = citation.get("keywords")
    if not isinstance(zenodo_keywords, list) or not isinstance(cff_keywords, list):
        return

    zenodo_normalized = {
        keyword.casefold() for keyword in zenodo_keywords if isinstance(keyword, str)
    }
    cff_normalized = {
        keyword.casefold() for keyword in cff_keywords if isinstance(keyword, str)
    }
    missing = sorted(cff_normalized - zenodo_normalized)
    if missing:
        errors.append(f"Zenodo keywords must include CITATION.cff keywords: {missing}")


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


def _validate_related_identifiers_against_citation(
    metadata: dict[str, Any], citation: dict[str, Any], errors: list[str]
) -> None:
    related_identifiers = metadata.get("related_identifiers")
    if not isinstance(related_identifiers, list):
        return

    relations_by_identifier: dict[str, set[str]] = {}
    for related_identifier in related_identifiers:
        if not isinstance(related_identifier, dict):
            continue
        identifier = related_identifier.get("identifier")
        relation = related_identifier.get("relation")
        if isinstance(identifier, str) and isinstance(relation, str):
            relations_by_identifier.setdefault(identifier, set()).add(relation)

    repository_code = citation.get("repository-code")
    if isinstance(repository_code, str) and "isDerivedFrom" not in (
        relations_by_identifier.get(repository_code, set())
    ):
        errors.append(
            "Zenodo related_identifiers must include CITATION.cff repository-code."
        )

    documentation_url = citation.get("url")
    if isinstance(documentation_url, str) and "isDocumentedBy" not in (
        relations_by_identifier.get(documentation_url, set())
    ):
        errors.append("Zenodo related_identifiers must include CITATION.cff url.")


def _validate_references(value: object, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) < 3:
        errors.append("references must contain at least three scholarly references.")
        return

    if not all(isinstance(reference, str) and reference.strip() for reference in value):
        errors.append("references must be non-empty strings.")

    joined_references = "\n".join(value)
    if "10.1214/aos/1176343247" not in joined_references:
        errors.append("references must include the Hill estimator DOI.")


def _validate_references_against_citation(
    metadata: dict[str, Any], citation: dict[str, Any], errors: list[str]
) -> None:
    zenodo_references = metadata.get("references")
    cff_reference_titles = citation.get("reference_titles")
    if not isinstance(zenodo_references, list) or not isinstance(
        cff_reference_titles, list
    ):
        return

    joined_references = "\n".join(
        reference for reference in zenodo_references if isinstance(reference, str)
    ).casefold()
    missing_titles = [
        title
        for title in cff_reference_titles
        if isinstance(title, str) and title.casefold() not in joined_references
    ]
    if missing_titles:
        errors.append(
            f"Zenodo references must include CITATION.cff references: {missing_titles}"
        )


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


def _parse_citation_cff(text: str) -> dict[str, Any]:
    citation: dict[str, Any] = {
        "keywords": _parse_list_block(text, "keywords"),
        "reference_titles": _parse_reference_titles(text),
    }

    for field in ("title", "date-released", "license", "repository-code", "url"):
        value = _parse_scalar(text, field)
        if value is not None:
            citation[field] = value

    abstract = _parse_folded_block(text, "abstract")
    if abstract is not None:
        citation["abstract"] = abstract

    first_author = _parse_first_author(text)
    if first_author:
        citation["first_author"] = first_author

    return citation


def _parse_scalar(text: str, field: str) -> str | None:
    match = re.search(rf"^{re.escape(field)}:\s*(.+?)\s*$", text, re.MULTILINE)
    if match is None:
        return None
    return _strip_yaml_quotes(match.group(1).strip())


def _parse_folded_block(text: str, field: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith(f"{field}:"):
            block_lines: list[str] = []
            for block_line in lines[index + 1 :]:
                if block_line and not block_line.startswith((" ", "\t")):
                    break
                stripped = block_line.strip()
                if stripped:
                    block_lines.append(stripped)
            return " ".join(block_lines)
    return None


def _parse_list_block(text: str, field: str) -> list[str]:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == f"{field}:":
            values: list[str] = []
            for item_line in lines[index + 1 :]:
                if item_line.startswith("  - "):
                    values.append(_strip_yaml_quotes(item_line[4:].strip()))
                    continue
                if item_line and not item_line.startswith((" ", "\t")):
                    break
            return values
    return []


def _parse_first_author(text: str) -> dict[str, str]:
    author_block = _parse_first_list_item_block(text, "authors")
    if author_block is None:
        return {}

    author: dict[str, str] = {}
    for field in ("family-names", "given-names", "orcid", "affiliation"):
        value = _parse_scalar(author_block, field)
        if value is not None:
            author[field] = value
    return author


def _parse_first_list_item_block(text: str, field: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == f"{field}:":
            block_lines: list[str] = []
            in_first_item = False
            for item_line in lines[index + 1 :]:
                if item_line.startswith("  - "):
                    if in_first_item:
                        break
                    in_first_item = True
                    block_lines.append(item_line[4:])
                    continue
                if in_first_item:
                    if item_line.startswith("    "):
                        block_lines.append(item_line[4:])
                        continue
                    if item_line and not item_line.startswith((" ", "\t")):
                        break
            return "\n".join(block_lines)
    return None


def _parse_reference_titles(text: str) -> list[str]:
    references_block = _parse_section_block(text, "references")
    if references_block is None:
        return []

    titles: list[str] = []
    for line in references_block.splitlines():
        stripped = line.strip()
        if stripped.startswith("title: "):
            titles.append(_strip_yaml_quotes(stripped.removeprefix("title: ").strip()))
    return titles


def _parse_section_block(text: str, field: str) -> str | None:
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line == f"{field}:":
            block_lines: list[str] = []
            for block_line in lines[index + 1 :]:
                if block_line and not block_line.startswith((" ", "\t")):
                    break
                block_lines.append(block_line)
            return "\n".join(block_lines)
    return None


def _strip_yaml_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _normalized_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())


def _zenodo_license_id(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if value == "MIT":
        return "mit"
    return value.casefold()


def _bare_orcid(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.removeprefix("https://orcid.org/")


def _ascii_fold(value: str) -> str:
    return value.replace("Média", "Media")


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
    if DEFAULT_CITATION_PATH.exists():
        errors.extend(validate_against_citation(metadata, DEFAULT_CITATION_PATH))
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"{args.path} is valid Zenodo metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
