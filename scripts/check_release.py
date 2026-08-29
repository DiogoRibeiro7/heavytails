"""Check that the repository is internally consistent about which release it is.

Release identity is spread across five files. The version appears in
``pyproject.toml``, ``CITATION.cff``, ``CHANGELOG.md``, ``docs/about/citation.md``
and ``README.md``; the date appears in ``CITATION.cff``, ``.zenodo.json`` and
``CHANGELOG.md``. Nothing writes them together, so a release commit is seven
opportunities to leave one behind.

Most of those couplings already have tests. The problem was never that they went
unchecked -- it was *when* they were checked. They run in the CI ``test`` job,
one of the five ``publish`` waits on, so a release commit that misses one file
still tags cleanly, still cuts a GitHub release, and simply never reaches
PyPI. The failure is silent at the only moment anyone is watching, and by the
time it is visible the tag is already spent: PyPI will not accept a re-upload of
a version, so the fix costs a whole patch release. That is how 0.6.0 was lost.

So this runs the same couplings *before* the release commit, adds the two the
tests do not cover (the changelog, and whether the tag is already taken), and
reports every problem at once rather than the first -- fixing one and
rediscovering the next is the same slow loop through CI.

Run ``make release-check`` before tagging anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):  # allow `python scripts/check_release.py`
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validate_zenodo_metadata import (  # noqa: E402
    validate_against_citation,
    validate_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

FINAL_VERSION = re.compile(r"\d+\.\d+\.\d+")

#: Files whose citation guidance must name the released version. The test suite
#: checks these two; they are named here as well because this script has to run
#: on a working tree that may not be the one the tests will run on.
CITATION_DOCS = ("docs/about/citation.md", "README.md")


def collect_problems(root: Path = REPO_ROOT, *, pre_tag: bool = False) -> list[str]:
    """Return every release-consistency problem found, in reporting order."""
    problems: list[str] = []

    package_version = _pyproject_version(root, problems)
    citation_version = _cff_field(root, "version", problems)
    citation_date = _cff_field(root, "date-released", problems)

    if package_version is None or citation_version is None:
        return problems  # nothing below can say anything useful

    _check_version_agreement(package_version, citation_version, problems)
    _check_citation_docs(root, citation_version, problems)
    _check_version_doi(root, citation_version, problems)
    _check_changelog(root, citation_version, citation_date, problems)
    _check_zenodo(root, problems)

    if pre_tag:
        _check_tag_is_free(root, citation_version, problems)

    return problems


def _check_version_agreement(package: str, citation: str, problems: list[str]) -> None:
    """Apply the release-versus-development rule.

    At a release the two must be equal. Between releases they must not be:
    ``pyproject.toml`` moves to the next version's ``.devN`` while
    ``CITATION.cff`` goes on naming the last thing anyone can actually cite.
    """
    if FINAL_VERSION.fullmatch(package):
        if citation != package:
            problems.append(
                f"pyproject.toml is at {package}, a final version, so "
                f"CITATION.cff must name it; it names {citation}."
            )
        return

    if not FINAL_VERSION.fullmatch(citation):
        problems.append(f"CITATION.cff must name a released version, not {citation}.")
        return

    base = re.match(r"(\d+)\.(\d+)\.(\d+)", package)
    if base is None:
        problems.append(f"Cannot read a version out of pyproject.toml: {package}.")
        return

    if tuple(int(p) for p in citation.split(".")) > tuple(
        int(g) for g in base.groups()
    ):
        problems.append(f"CITATION.cff names {citation}, which is ahead of {package}.")


def _check_citation_docs(root: Path, version: str, problems: list[str]) -> None:
    """Every version named in the citation guidance must be the current one."""
    for name in CITATION_DOCS:
        path = root / name
        if not path.exists():
            problems.append(f"{name} is missing.")
            continue

        text = path.read_text(encoding="utf-8")
        # DOIs first: a registrant's suffix can read as a version to any
        # three-part pattern, and this guidance cites other people's papers.
        text = re.sub(r"10\.\d{4,}/\S+", "", text)
        stale = sorted(
            set(re.findall(r"(?<![\w.])\d+\.\d+\.\d+(?![\w.])", text)) - {version}
        )
        if stale:
            problems.append(f"{name} cites {stale}, but the release is {version}.")


def _check_version_doi(root: Path, version: str, problems: list[str]) -> None:
    """The citation guidance must quote the version DOI CITATION.cff names.

    Zenodo mints two kinds of DOI, and the difference is the whole point of
    that page: a concept DOI resolves to whatever is newest, a version DOI to
    one release. So a citation block that pins a version while quoting the
    concept DOI pins nothing at all.

    That is exactly what the block titled "Citing release X exactly" did until
    0.6.1 -- the failure its own page warns about two sections earlier. It went
    unnoticed because the version DOI does not exist yet when the release is
    cut; Zenodo mints it on archiving, and updating the docs afterwards is a
    step nothing enforced.

    This does not require the DOI to correspond to the current version. It
    cannot: between a release and Zenodo archiving it, CITATION.cff still
    holds the previous release's version DOI, correctly. What it requires is
    that the two files agree, which is the part that silently drifts.
    """
    citation_path = root / "CITATION.cff"
    doc_path = root / CITATION_DOCS[0]
    if not citation_path.exists() or not doc_path.exists():
        return

    text = citation_path.read_text(encoding="utf-8")
    concept = re.search(r'^doi:\s*"([^"]+)"', text, re.MULTILINE)
    listed = re.findall(r'^\s+value:\s*"(10\.\d{4,}/[^"]+)"', text, re.MULTILINE)
    if concept is None or not listed:
        return

    version_dois = {doi for doi in listed if doi != concept.group(1)}
    if len(version_dois) != 1:
        return  # no single version DOI to check against

    version_doi = version_dois.pop()
    if version_doi not in doc_path.read_text(encoding="utf-8"):
        problems.append(
            f"CITATION.cff names version DOI {version_doi}, but "
            f"{CITATION_DOCS[0]} does not cite it. An exact-release citation "
            f"that quotes the concept DOI resolves to whatever is newest."
        )

    _check_versioned_citations(root, version, version_doi, problems)


def _check_versioned_citations(
    root: Path, version: str, version_doi: str, problems: list[str]
) -> None:
    """Any citation that names a version must cite that version's DOI.

    Checking that the version DOI appears *somewhere* is too weak. It passed
    on a page where one BibTeX block was correct and the APA, IEEE, MLA and
    Chicago renderings below it all said "Version 0.6.1" beside the concept
    DOI -- a DOI that, as the same page explains, resolves to whatever is
    newest. A reader following it in a year gets a different release than the
    one the citation names.

    The rule is the one the page already teaches, applied mechanically:

        names a version  -> cite that version's DOI
        cites the concept DOI -> claim no particular version
    """
    for name in CITATION_DOCS:
        path = root / name
        if not path.exists():
            continue

        for line_number, block in _citation_blocks(path.read_text(encoding="utf-8")):
            if version not in block:
                continue
            cited = set(re.findall(r"10\.\d{4,}/zenodo\.\d+", block))
            if not cited or version_doi in cited:
                continue
            problems.append(
                f"{name}:{line_number} names {version} but cites "
                f"{sorted(cited)}; a citation naming a version must use that "
                f"version's DOI, {version_doi}."
            )


def _citation_blocks(text: str) -> Iterator[tuple[int, str]]:
    """Yield ``(line number, text)`` for each unit a citation can occupy.

    A citation is not a line. The APA rendering puts the version on one line
    and the DOI on the next, and BibTeX spreads both over a fenced block, so
    line-by-line reading would miss exactly the cases that matter. Fenced
    blocks and runs of blockquote lines are therefore kept whole, while
    everything else stays a line -- which keeps the rows of a Markdown table
    separate, so the row explaining the concept DOI is not read as part of the
    row naming the version.
    """
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]

        if line.startswith("```"):
            start = index
            block = [line]
            index += 1
            while index < len(lines):
                block.append(lines[index])
                if lines[index].startswith("```"):
                    index += 1
                    break
                index += 1
            yield start + 1, "\n".join(block)
            continue

        if line.lstrip().startswith(">"):
            start = index
            block = []
            while index < len(lines) and lines[index].lstrip().startswith(">"):
                block.append(lines[index])
                index += 1
            yield start + 1, "\n".join(block)
            continue

        yield index + 1, line
        index += 1


def _check_changelog(
    root: Path, version: str, date: str | None, problems: list[str]
) -> None:
    """The changelog must have a dated entry and link refs for the release.

    Nothing else checks this file, and it carries the third copy of the date.
    """
    path = root / "CHANGELOG.md"
    if not path.exists():
        problems.append("CHANGELOG.md is missing.")
        return

    text = path.read_text(encoding="utf-8")

    heading = re.search(
        rf"^## \[{re.escape(version)}\] - (\d{{4}}-\d{{2}}-\d{{2}})\s*$",
        text,
        re.MULTILINE,
    )
    if heading is None:
        problems.append(f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' entry.")
    elif date is not None and heading.group(1) != date:
        problems.append(
            f"CHANGELOG.md dates {version} at {heading.group(1)}, but "
            f"CITATION.cff date-released is {date}."
        )

    if not re.search(r"^## \[Unreleased\]\s*$", text, re.MULTILINE):
        problems.append("CHANGELOG.md has no '## [Unreleased]' section.")

    if not re.search(rf"^\[{re.escape(version)}\]: \S+", text, re.MULTILINE):
        problems.append(f"CHANGELOG.md has no link reference for [{version}].")

    unreleased_ref = re.search(r"^\[Unreleased\]: (\S+)\s*$", text, re.MULTILINE)
    if unreleased_ref is None:
        problems.append("CHANGELOG.md has no link reference for [Unreleased].")
    elif f"v{version}...HEAD" not in unreleased_ref.group(1):
        problems.append(
            f"CHANGELOG.md compares [Unreleased] against something other than "
            f"v{version}: {unreleased_ref.group(1)}"
        )


def _check_zenodo(root: Path, problems: list[str]) -> None:
    """Delegate to the Zenodo validator rather than restating its rules here.

    A transcription cannot notice that it has gone stale, which is the same
    reason the metadata tests read CITATION.cff instead of hardcoding it.
    """
    metadata_path = root / ".zenodo.json"
    citation_path = root / "CITATION.cff"
    if not metadata_path.exists():
        problems.append(".zenodo.json is missing.")
        return

    try:
        metadata: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        problems.append(f".zenodo.json is not valid JSON: {error}")
        return

    if not isinstance(metadata, dict):
        problems.append(".zenodo.json must contain a JSON object.")
        return

    problems.extend(validate_metadata(metadata))
    if citation_path.exists():
        problems.extend(validate_against_citation(metadata, citation_path))


def _check_tag_is_free(root: Path, version: str, problems: list[str]) -> None:
    """A tag that already exists cannot be reused to publish.

    The publish job checks out the tag, and PyPI refuses a re-upload of a
    version. If ``v0.6.0`` is already spent, the only way forward is 0.6.1 --
    better to learn that here than after cutting the release.
    """
    tag = f"v{version}"
    if tag_sha(root, tag) is not None:
        problems.append(
            f"Tag {tag} already exists. PyPI will not accept a re-upload of "
            f"{version}, so this release needs a new version number."
        )


def tag_sha(root: Path, tag: str) -> str | None:
    """Return what ``tag`` points at, reading refs rather than shelling out.

    ``scripts/_provenance.py`` reads git state off the filesystem for the same
    reasons: it needs no subprocess, it works when git is not on PATH, and it
    cannot be talked into executing anything. Both subtleties that helper
    documents apply here too -- ``.git`` is a *file* in a linked worktree, and
    a worktree shares its refs with the repository it names in ``commondir``
    -- and a tag may be packed rather than loose.
    """
    git_entry = root / ".git"
    try:
        if git_entry.is_file():
            content = git_entry.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir:"):
                return None
            git_dir = (root / content.removeprefix("gitdir:").strip()).resolve()
        elif git_entry.is_dir():
            git_dir = git_entry
        else:
            return None  # not a git checkout; nothing to say

        directories = [git_dir]
        commondir_file = git_dir / "commondir"
        if commondir_file.is_file():
            shared = commondir_file.read_text(encoding="utf-8").strip()
            directories.append((git_dir / shared).resolve())

        ref = f"refs/tags/{tag}"
        for directory in directories:
            ref_file = directory / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip() or None
            packed = directory / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(f" {ref}"):
                        return line.split()[0]
    except OSError:
        return None
    return None


def _pyproject_version(root: Path, problems: list[str]) -> str | None:
    path = root / "pyproject.toml"
    if not path.exists():
        problems.append("pyproject.toml is missing.")
        return None

    match = re.search(
        r'^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"), re.MULTILINE
    )
    if match is None:
        problems.append("pyproject.toml has no version.")
        return None
    return match.group(1)


def _cff_field(root: Path, name: str, problems: list[str]) -> str | None:
    path = root / "CITATION.cff"
    if not path.exists():
        problems.append("CITATION.cff is missing.")
        return None

    match = re.search(
        rf'^{re.escape(name)}:\s*"?([^"\n]+)"?\s*$',
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        problems.append(f"CITATION.cff has no {name}.")
        return None
    return match.group(1).strip()


def main() -> int:
    """Run the release preflight from the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--pre-tag",
        action="store_true",
        help="also require that the release tag does not exist yet",
    )
    args = parser.parse_args()

    problems = collect_problems(pre_tag=args.pre_tag)
    if problems:
        print("Release metadata is inconsistent:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1

    version = _cff_field(REPO_ROOT, "version", [])
    print(f"Release metadata is consistent for {version}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
