"""Tests for the release preflight.

The cases below are not hypothetical. Every one of the first three is a way a
release commit has actually broken, and the cost each time was a version number:
the CI test job failed, the publish job that depends on it never ran, and the
tag was already spent because PyPI will not accept a re-upload.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess

import pytest
from scripts.check_release import collect_problems

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Everything that carries a version or a date.
RELEASE_FILES = (
    "pyproject.toml",
    "CITATION.cff",
    ".zenodo.json",
    "CHANGELOG.md",
    "docs/about/citation.md",
    "README.md",
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A copy of the real release metadata, safe to corrupt."""
    for name in RELEASE_FILES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, destination)
    return tmp_path


def _edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"{path.name} does not contain {old!r}"
    path.write_text(text.replace(old, new), encoding="utf-8")


def _version(repo: Path) -> str:
    match = re.search(
        r'^version:\s*"?([^"\n]+)"?\s*$',
        (repo / "CITATION.cff").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1).strip()


def test_the_repository_is_release_consistent() -> None:
    """The tree as committed must pass its own preflight."""
    assert collect_problems(REPO_ROOT) == []


def test_the_fixture_starts_clean(repo: Path) -> None:
    """Otherwise the drift tests below would pass for the wrong reason."""
    assert collect_problems(repo) == []


def test_it_catches_citation_guidance_left_on_the_old_version(repo: Path) -> None:
    """0.6.0's second failure: the docs still offered 0.5.0 entries."""
    version = _version(repo)
    _edit(repo / "docs/about/citation.md", version, "0.0.1")

    problems = collect_problems(repo)

    assert any(
        "docs/about/citation.md cites ['0.0.1']" in problem for problem in problems
    ), problems


def test_it_catches_a_zenodo_date_that_drifted(repo: Path) -> None:
    """0.6.0's first failure: CITATION.cff moved and .zenodo.json did not."""
    _edit(repo / ".zenodo.json", _cff_date(repo), "2020-01-01")

    problems = collect_problems(repo)

    assert (
        "Zenodo publication_date must match CITATION.cff date-released." in problems
    ), problems


def test_it_catches_a_changelog_date_that_drifted(repo: Path) -> None:
    """The third copy of the date, which no other check reads."""
    version, date = _version(repo), _cff_date(repo)
    _edit(
        repo / "CHANGELOG.md",
        f"## [{version}] - {date}",
        f"## [{version}] - 2020-01-01",
    )

    problems = collect_problems(repo)

    assert any(
        f"CHANGELOG.md dates {version} at 2020-01-01" in problem for problem in problems
    ), problems


def test_it_catches_a_missing_changelog_entry(repo: Path) -> None:
    """Tagging a version the changelog never mentions."""
    version, date = _version(repo), _cff_date(repo)
    _edit(repo / "CHANGELOG.md", f"## [{version}] - {date}", "## [9.9.9] - 2020-01-01")

    problems = collect_problems(repo)

    assert f"CHANGELOG.md has no '## [{version}] - YYYY-MM-DD' entry." in problems


def test_it_catches_a_citation_ahead_of_the_package(repo: Path) -> None:
    """CITATION.cff must never name a version that does not exist yet."""
    _edit(repo / "pyproject.toml", f'version = "{_version(repo)}"', 'version = "0.1.0"')

    problems = collect_problems(repo)

    assert any("is ahead of 0.1.0" in problem for problem in problems) or any(
        "CITATION.cff must name it" in problem for problem in problems
    ), problems


def test_a_development_version_may_lead_the_citation(repo: Path) -> None:
    """Between releases the two are *supposed* to differ.

    This is the case the first version of the equality test got wrong: it
    would have gone red on the first commit after every release.
    """
    _edit(
        repo / "pyproject.toml",
        f'version = "{_version(repo)}"',
        'version = "9.9.9.dev0"',
    )

    assert collect_problems(repo) == []


def test_it_reports_every_problem_at_once(repo: Path) -> None:
    """Fixing one and rediscovering the next is a slow loop through CI.

    This is the whole reason the script accumulates instead of raising: the
    0.6.0 recovery took two patch attempts because the date and the docs were
    found one release apart.
    """
    version = _version(repo)
    _edit(repo / ".zenodo.json", _cff_date(repo), "2020-01-01")
    _edit(repo / "docs/about/citation.md", version, "0.0.1")

    problems = collect_problems(repo)

    assert len(problems) >= 2
    assert any("publication_date" in problem for problem in problems)
    assert any("citation.md" in problem for problem in problems)


def test_pre_tag_catches_a_tag_that_is_already_spent(repo: Path) -> None:
    """A tag that exists means the version is gone; PyPI will not take it twice."""
    version = _version(repo)
    for command in (
        ["git", "init", "--quiet"],
        [
            "git",
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "seed",
        ],
        ["git", "tag", f"v{version}"],
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)

    assert collect_problems(repo, pre_tag=True) != []
    assert any(
        f"Tag v{version} already exists" in problem
        for problem in collect_problems(repo, pre_tag=True)
    )
    # Without --pre-tag the same tree is fine: the check is only for the
    # moment before tagging, not for every run after it.
    assert collect_problems(repo) == []


def _cff_date(repo: Path) -> str:
    match = re.search(
        r"^date-released:\s*(\S+)\s*$",
        (repo / "CITATION.cff").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1)
