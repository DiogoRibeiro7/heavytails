# Releasing

This checklist keeps PyPI, GitHub, documentation, citation metadata, and Zenodo
aligned for each public release.

--------------------------------------------------------------------------------

## Before Tagging

1. Update the package version in `pyproject.toml`.
2. Update `CITATION.cff` with the same version and release date.
3. Update `CITATION.cff` when the title, authors, affiliations, keywords,
   references, license, or release date change.
4. Update `.zenodo.json` only when Zenodo-specific fields change. The validator
   checks that shared citation fields stay aligned with `CITATION.cff`.
5. Run the metadata validator:

```bash
python scripts/validate_zenodo_metadata.py
```

6. Run the regular quality checks:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy heavytails/ scripts/
poetry run pytest
poetry run mkdocs build --strict
```

--------------------------------------------------------------------------------

## Zenodo Setup

Zenodo archives GitHub releases after the repository is connected in Zenodo's
GitHub integration. The repository-level `.zenodo.json` file overrides the
metadata Zenodo would otherwise infer from GitHub, while `CITATION.cff` remains
the source of truth for shared citation fields.

For the first archived release:

1. Sign in to Zenodo with the maintainer account.
2. Enable GitHub integration for `DiogoRibeiro7/heavytails`.
3. Create and publish a GitHub release from an annotated version tag.
4. Wait for Zenodo to archive the release.
5. Copy the concept DOI and version DOI from Zenodo.
6. Replace DOI placeholders in citation documentation with the real DOI values.
7. Add a Zenodo DOI badge to `README.md` once the concept DOI exists.

Do not add a fake DOI before Zenodo has minted one.

--------------------------------------------------------------------------------

## Release Notes

Each GitHub release should include:

- Version number and date.
- New distributions, estimators, or diagnostics.
- Bug fixes and numerical accuracy changes.
- Backward-incompatible changes.
- Documentation and citation metadata changes.
- The Zenodo DOI once the archive is available.

--------------------------------------------------------------------------------

## After Release

1. Confirm the GitHub release triggered the PyPI publish workflow.
2. Confirm Zenodo created a new archived version.
3. Confirm the Zenodo record uses the `.zenodo.json` title, creator ORCID,
   license, references, keywords, and related identifiers.
4. Update documentation and badges with the minted DOI if this was the first
   archived release.
