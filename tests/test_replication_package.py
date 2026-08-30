"""The replication package, tested as an artifact rather than as source.

The package is excluded from Ruff, is not type-checked, and is not imported by
the library, so the ordinary suite went green while executable defects lived in
it: a supplement builder that could not run from the deposit it shipped in, a
reproducibility instruction that compared nothing, a provenance CSV with a
five-field row under a four-field header. Every one was found by reading, not
by CI.

Linting frozen bytes would be the wrong fix -- reformatting an archive breaks
the checksums that are its point. What is testable is the archive's behaviour:
does it verify against its own manifest, and do the scripts it ships still run
from where they are deposited.

These are deliberately cheap. Rebuilding the manuscript needs a TeX
installation and reconstructing the tables needs the 213 MB replicate export,
so neither belongs in a suite that runs on every push; `REPRODUCE.md` covers
those and the release checklist runs them by hand.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import re
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "research" / "sparse_contamination" / "replication_package"

pytestmark = pytest.mark.skipif(
    not (PACKAGE / "MANIFEST.txt").is_file(),
    reason="replication package not present in this checkout",
)


def _manifest_entries() -> list[tuple[str, str, int]]:
    entries = []
    for raw in (PACKAGE / "MANIFEST.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        path, digest, size = line.split()
        entries.append((path, digest, int(size)))
    return entries


def test_every_manifested_file_is_present_and_unchanged() -> None:
    """The manifest is the archive's entire integrity claim.

    Line-ending normalisation broke 26 of these once, silently: the files were
    all there and every checksum was wrong.
    """
    entries = _manifest_entries()
    assert entries, "manifest lists nothing"

    wrong = []
    for path, digest, size in entries:
        target = PACKAGE / path
        if not target.is_file():
            wrong.append(f"{path}: missing")
            continue
        data = target.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            wrong.append(f"{path}: checksum")
        elif len(data) != size:
            wrong.append(f"{path}: size {len(data)} != {size}")
    assert not wrong, wrong


def test_nothing_in_the_package_is_missing_from_the_manifest() -> None:
    """A file nobody checksums is a file nobody notices changing."""
    listed = {path for path, _, _ in _manifest_entries()}
    present = {
        p.relative_to(PACKAGE).as_posix()
        for p in PACKAGE.rglob("*")
        if p.is_file() and p.name != "MANIFEST.txt"
    }
    assert present - listed == set(), "present but unmanifested"


def test_the_supplement_builder_runs_from_where_it_is_deposited(
    tmp_path: Path,
) -> None:
    """It is archived inside the package it reads.

    Its package root came from the script's own directory, which resolves to
    ``replication_package/scripts/replication_package`` once deposited. The
    script that generates the journal supplement could not be run from the
    archive supplying it.
    """
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "scripts" / "build_online_resource.py"),
            "--doi",
            "10.5281/zenodo.0000000",
            "--email",
            "someone@example.org",
            "--outdir",
            str(tmp_path / "or1"),
        ],
        cwd=PACKAGE / "scripts",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "or1" / "README.md").is_file()


def test_the_supplement_builder_refuses_unresolved_placeholders(
    tmp_path: Path,
) -> None:
    """It writes a journal artifact, so an incomplete one must not exit 0.

    It used to print a note and succeed, having already written a
    complete-looking bundle containing "[REPLICATION DOI]".
    """
    outdir = tmp_path / "or1"
    result = subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "scripts" / "build_online_resource.py"),
            "--outdir",
            str(outdir),
        ],
        cwd=PACKAGE / "scripts",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert not outdir.exists(), "refused, but left a partial bundle behind"


def test_the_object_provenance_csv_parses(tmp_path: Path) -> None:
    """One description contains a comma.

    Built by joining on commas, its row had five fields under a four-field
    header -- in a file supplied to a journal and parsed by nobody.
    """
    outdir = tmp_path / "or1"
    subprocess.run(
        [
            sys.executable,
            str(PACKAGE / "scripts" / "build_online_resource.py"),
            "--doi",
            "10.5281/zenodo.0000000",
            "--email",
            "someone@example.org",
            "--outdir",
            str(outdir),
        ],
        cwd=PACKAGE / "scripts",
        check=True,
        capture_output=True,
    )

    with (outdir / "object_provenance.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))

    assert rows[0] == ["object", "description", "script", "artifact"]
    assert len(rows) > 1
    assert all(len(row) == 4 for row in rows[1:]), [r for r in rows if len(r) != 4]
    assert any("," in row[1] for row in rows[1:]), (
        "the row that motivated this test is gone; the check is now vacuous"
    )


def test_the_expected_manuscript_text_digest_is_shipped_and_usable() -> None:
    """REPRODUCE.md tells the reader to check a rebuild against this file.

    It claimed the extracted text was byte-identical while shipping nothing to
    compare against, and the command it gave hashed the rebuild itself.
    """
    expected = PACKAGE / "paper" / "main.txt.sha256"
    assert expected.is_file(), "no expected text digest shipped"

    raw = expected.read_bytes()
    assert b"\r" not in raw, "CRLF makes sha256sum -c look for 'rebuilt.txt\\r'"

    digest, name = raw.decode("utf-8").split()
    assert len(digest) == 64
    assert name == "rebuilt.txt", "must name the file REPRODUCE.md creates"


def test_the_frozen_run_provenance_survives_a_software_version_bump() -> None:
    """The archive must not change because the library moved on.

    The packaging record read `heavytails` from the current pyproject, so
    bumping the tree to a development version rewrote a supposedly frozen
    archive with no result, analysis or manuscript having changed.
    """
    record = (PACKAGE / "PACKAGING_ENVIRONMENT.txt").read_text(encoding="utf-8")
    run_section = record.split("# What assembled")[0]
    assert "heavytails  0.5.0" in run_section, (
        "the run section must name the version that produced the results"
    )

    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    current = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert current is not None
    if current.group(1) != "0.5.0":
        assert current.group(1) not in run_section, (
            "the current software version leaked into the frozen run record"
        )
