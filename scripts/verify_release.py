"""Check that a released version is actually, publicly, everywhere it claims.

``check_release.py`` is a preflight: it reads local files and asks whether they
agree before a release is cut. It deliberately stops there, because half of a
release does not exist yet at that point. The version DOI in particular is
minted by Zenodo only on archiving, so no preflight can require it to match the
version being released.

This is the other half. It runs *after* a release and asks a different
question -- not "do these files agree" but "is this version real":

    tag  =  GitHub release  =  PyPI  =  Zenodo archive

Each of those is a separate system that can silently lag or fail. 0.6.0 was
tagged, released on GitHub, and never reached PyPI; nothing said so, because
nothing was asking. The four had drifted apart and the only thing that would
have noticed was somebody typing ``pip install`` a week later.

Run ``make verify-release VERSION=0.6.1`` after a release, once Zenodo has
archived it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any, NamedTuple
import urllib.error
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]

if __package__ in (None, ""):  # allow `python scripts/verify_release.py`
    sys.path.insert(0, str(REPO_ROOT))

from scripts.check_release import CITATION_DOCS, tag_sha  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable

GITHUB_REPO = "DiogoRibeiro7/heavytails"
PACKAGE = "heavytails"
TIMEOUT = 30


class Check(NamedTuple):
    """One question asked of the released version.

    ``ok`` is None when the question does not apply. The local files describe
    one release, so asking whether they cite some *other* version's DOI has no
    answer -- and reporting a pass there would be worse than useless: running
    this against 0.6.0 said "README cites the version DOI" while the DOI in
    question was 0.6.1's.
    """

    ok: bool | None
    label: str
    detail: str = ""


def fetch_json(url: str) -> Any | None:
    """Return decoded JSON from ``url``, or None if it cannot be read.

    None means "could not answer", not "answered no". The caller reports that
    as a failed check either way -- an unreachable index is not evidence that
    a release is fine.
    """
    request = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": PACKAGE}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return None


def verify(
    version: str,
    *,
    root: Path = REPO_ROOT,
    fetch: Callable[[str], Any | None] = fetch_json,
) -> list[Check]:
    """Return one Check per question, in the order a release goes public."""
    tag = f"v{version}"
    named = _cff_version(root)
    # The local metadata describes one release. Against any other version the
    # public surfaces still mean something; the local ones do not.
    current = named == version
    version_doi = _version_doi(root) if current else None

    checks = [
        _check_local_tag(root, tag),
        *_check_github(fetch, root, tag),
        _check_pypi(fetch, version),
        Check(
            named == version, "CITATION.cff names this release", named or "not found"
        ),
    ]

    if not current:
        stale = f"CITATION.cff describes {named or '?'}, so this cannot be asked"
        checks.append(Check(None, "Zenodo version DOI resolves", stale))
        checks.extend(
            Check(None, f"{name} cites the version DOI", stale)
            for name in CITATION_DOCS
        )
        return checks

    checks.append(_check_zenodo(fetch, version_doi, version))
    checks.extend(_check_local_metadata(root, version_doi))
    return checks


def _check_local_tag(root: Path, tag: str) -> Check:
    sha = tag_sha(root, tag)
    if sha is None:
        return Check(False, f"tag {tag} exists locally", "not found in this checkout")
    return Check(True, f"tag {tag} exists locally", sha[:12])


def _check_github(
    fetch: Callable[[str], Any | None], root: Path, tag: str
) -> list[Check]:
    release = fetch(f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}")
    if not isinstance(release, dict) or "tag_name" not in release:
        return [
            Check(False, f"GitHub release {tag} exists", "no release for that tag"),
            Check(False, "GitHub tag matches the local tag", "release not found"),
        ]

    checks = [
        Check(True, f"GitHub release {tag} exists", str(release.get("html_url", "")))
    ]

    if release.get("draft"):
        checks.append(
            Check(
                False,
                "GitHub release is published",
                "still a draft, so it never triggered publish",
            )
        )

    ref = fetch(f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/tags/{tag}")
    remote = ref.get("object", {}).get("sha") if isinstance(ref, dict) else None
    local = tag_sha(root, tag)
    if remote is None:
        checks.append(Check(False, "GitHub tag matches the local tag", "no remote ref"))
    elif local is None:
        checks.append(
            Check(False, "GitHub tag matches the local tag", "no local tag to compare")
        )
    elif remote != local:
        checks.append(
            Check(
                False,
                "GitHub tag matches the local tag",
                f"local {local[:12]} != remote {remote[:12]}",
            )
        )
    else:
        checks.append(Check(True, "GitHub tag matches the local tag", remote[:12]))
    return checks


def _check_pypi(fetch: Callable[[str], Any | None], version: str) -> Check:
    """PyPI is the one that failed silently for 0.6.0."""
    data = fetch(f"https://pypi.org/pypi/{PACKAGE}/{version}/json")
    if not isinstance(data, dict):
        return Check(False, f"PyPI serves {version}", "no such release on PyPI")

    urls = data.get("urls")
    if not isinstance(urls, list) or not urls:
        return Check(False, f"PyPI serves {version}", "release exists but has no files")

    kinds = sorted({str(u.get("packagetype")) for u in urls if isinstance(u, dict)})
    return Check(True, f"PyPI serves {version}", ", ".join(kinds))


def _check_zenodo(
    fetch: Callable[[str], Any | None], version_doi: str | None, version: str
) -> Check:
    if version_doi is None:
        return Check(
            False, "Zenodo version DOI resolves", "CITATION.cff names no version DOI"
        )

    record = re.search(r"zenodo\.(\d+)$", version_doi)
    if record is None:
        return Check(
            False,
            "Zenodo version DOI resolves",
            f"cannot read a record from {version_doi}",
        )

    data = fetch(f"https://zenodo.org/api/records/{record.group(1)}")
    if not isinstance(data, dict):
        return Check(
            False, "Zenodo version DOI resolves", f"{version_doi} not reachable"
        )

    archived = str(data.get("metadata", {}).get("version", ""))
    if archived.lstrip("v") != version:
        return Check(
            False,
            "Zenodo version DOI resolves",
            f"{version_doi} archives {archived or '?'}, not {version}",
        )
    return Check(True, "Zenodo version DOI resolves", f"{version_doi} -> {archived}")


def _cff_version(root: Path) -> str | None:
    """Return the release CITATION.cff describes."""
    cff = root / "CITATION.cff"
    if not cff.exists():
        return None
    match = re.search(
        r'^version:\s*"?([^"\n]+)"?\s*$',
        cff.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def _check_local_metadata(root: Path, version_doi: str | None) -> list[Check]:
    checks: list[Check] = []
    for name in CITATION_DOCS:
        path = root / name
        if version_doi is None or not path.exists():
            checks.append(Check(False, f"{name} cites the version DOI", "cannot check"))
            continue
        cites = version_doi in path.read_text(encoding="utf-8")
        checks.append(
            Check(
                cites,
                f"{name} cites the version DOI",
                version_doi if cites else "absent",
            )
        )
    return checks


def _version_doi(root: Path) -> str | None:
    """The identifier in CITATION.cff that is not the concept DOI."""
    cff = root / "CITATION.cff"
    if not cff.exists():
        return None
    text = cff.read_text(encoding="utf-8")
    concept = re.search(r'^doi:\s*"([^"]+)"', text, re.MULTILINE)
    listed = re.findall(r'^\s+value:\s*"(10\.\d{4,}/[^"]+)"', text, re.MULTILINE)
    if concept is None:
        return None
    version_dois = {doi for doi in listed if doi != concept.group(1)}
    if len(version_dois) != 1:
        return None
    return version_dois.pop()


def main() -> int:
    """Verify a released version from the command line."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="the released version, e.g. 0.6.1")
    args = parser.parse_args()

    checks = verify(args.version)
    for check in checks:
        mark = "SKIP" if check.ok is None else ("PASS" if check.ok else "FAIL")
        detail = f"  ({check.detail})" if check.detail else ""
        print(f"[{mark}] {check.label}{detail}")

    failed = [check for check in checks if check.ok is False]
    if failed:
        print(f"\n{len(failed)} of {len(checks)} checks failed.", file=sys.stderr)
        return 1

    print(f"\n{args.version} is published everywhere it claims to be.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
