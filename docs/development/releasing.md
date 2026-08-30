# Releasing

This checklist keeps PyPI, GitHub, documentation, citation metadata, and Zenodo
aligned for each public release.

--------------------------------------------------------------------------------

## Before Tagging

Five files carry the release identity. The version appears in
`pyproject.toml`, `CITATION.cff`, `CHANGELOG.md`, `docs/about/citation.md` and
`README.md`; the date appears in `CITATION.cff`, `.zenodo.json` and
`CHANGELOG.md`. Nothing writes them together, so update all of them:

1. Update the package version in `pyproject.toml`.
2. Update `CITATION.cff` with the same version and release date.
3. Roll `CHANGELOG.md`: turn `## [Unreleased]` into a dated
   `## [<version>] - <date>` section, open a fresh empty `## [Unreleased]`
   above it, add the `[<version>]` link reference, and repoint the
   `[Unreleased]` comparison at `v<version>`.
4. Update the citation guidance in `docs/about/citation.md` and `README.md`.
   Both name the version in running text and in the BibTeX, APA, IEEE, MLA and
   Chicago entries.
5. Update `CITATION.cff` when the title, authors, affiliations, keywords,
   references, license, or release date change.
6. Update `.zenodo.json` only when Zenodo-specific fields change. The validator
   checks that shared citation fields stay aligned with `CITATION.cff`.
7. Confirm every file agrees and the tag is still free:

```bash
make release-preflight
```

   Do not tag until this passes. It reports every problem at once, so there is
   no need to fix one and run it again.

   This step exists because skipping it is expensive. The same couplings are
   enforced by the CI `test` job, and `publish` waits on five jobs -- `test`,
   `coverage`, `lint-and-type-check`, `security` and `build` -- so a release
   commit that misses one of these files still tags cleanly, still cuts a
   GitHub release, and simply never reaches PyPI. Nothing fails loudly at the
   moment anyone is watching. Since PyPI will not accept a re-upload of a version,
   recovering costs a whole patch release. That is how 0.6.0 was lost: it
   missed steps 3 and 4, which this checklist did not previously mention.

8. Dispatch the coverage job and wait for it to pass:

```bash
gh workflow run CI --ref main
```

   `coverage` runs on a release or a manual dispatch and on nothing else --
   not on pushes, not on pull requests. But `publish` waits on it, so on an
   ordinary release the first time it ever runs against the code being shipped
   is *after* the tag is public. If it fails there, for any reason including a
   tooling change rather than a real coverage drop, the tag and the GitHub
   release are already spent and the recovery is another version number. That
   is the same shape of failure as 0.6.0. Dispatching it here moves the one
   unexercised gate in front of the irreversible step.

9. Run the regular quality checks:

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

1. Confirm the GitHub release triggered the PyPI publish workflow, and that
   it ran rather than being skipped. `publish` waits on `test`, `coverage`,
   `lint-and-type-check`, `security` and `build`; if any of them failed, the
   release is tagged and public but not on PyPI.
2. Confirm Zenodo created a new archived version.
3. Confirm the Zenodo record uses the `.zenodo.json` title, creator ORCID,
   license, references, keywords, and related identifiers.
4. Copy the new version DOI into `CITATION.cff` and `docs/about/citation.md`.
   Zenodo mints it only on archiving, so it cannot be filled in before the
   release, and `make release-check` fails while the two files disagree. The
   "Citing release X exactly" block must carry the *version* DOI: the concept
   DOI resolves to whatever is newest, so quoting it there pins nothing.
5. Update badges with the minted DOI if this was the first archived release.
