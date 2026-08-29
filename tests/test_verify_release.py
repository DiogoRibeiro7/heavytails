"""Tests for the post-release verifier.

These never touch the network. Every case injects a ``fetch`` that answers
from a dictionary, so the suite tests what the verifier concludes rather than
whether PyPI happened to be reachable.
"""

from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import pytest
from scripts.verify_release import GITHUB_REPO, PACKAGE, verify

REPO_ROOT = Path(__file__).resolve().parents[1]

RELEASE_FILES = ("CITATION.cff", "docs/about/citation.md", "README.md")

VERSION = "9.9.9"
TAG = f"v{VERSION}"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A checkout that describes release 9.9.9 and has the tag to match."""
    for name in RELEASE_FILES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / name, destination)

    # Rewrite the copies to describe a version that will never exist, so the
    # tests cannot accidentally pass by agreeing with the real repository.
    real = _cff_version(REPO_ROOT)
    for name in RELEASE_FILES:
        path = tmp_path / name
        path.write_text(
            path.read_text(encoding="utf-8").replace(real, VERSION), encoding="utf-8"
        )

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
        ["git", "tag", TAG],
    ):
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def _cff_version(root: Path) -> str:
    match = re.search(
        r'^version:\s*"?([^"\n]+)"?\s*$',
        (root / "CITATION.cff").read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    assert match is not None
    return match.group(1).strip()


def _tag_object(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", TAG], cwd=repo, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _world(repo: Path, *, on_pypi: bool = True) -> dict[str, Any]:
    """Everything published, unless a case says otherwise."""
    world = {
        f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{TAG}": {
            "tag_name": TAG,
            "html_url": f"https://github.com/{GITHUB_REPO}/releases/tag/{TAG}",
            "draft": False,
        },
        f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/tags/{TAG}": {
            "object": {"sha": _tag_object(repo)}
        },
        "https://zenodo.org/api/records/22166257": {"metadata": {"version": TAG}},
    }
    if on_pypi:
        world[f"https://pypi.org/pypi/{PACKAGE}/{VERSION}/json"] = {
            "urls": [{"packagetype": "bdist_wheel"}, {"packagetype": "sdist"}]
        }
    return world


def _fetch(world: dict[str, Any]):
    return lambda url: world.get(url)


def _by_label(checks) -> dict[str, Any]:
    return {check.label: check for check in checks}


def test_a_fully_published_release_passes(repo: Path) -> None:
    checks = verify(VERSION, root=repo, fetch=_fetch(_world(repo)))

    assert [check for check in checks if check.ok is False] == []
    assert [check for check in checks if check.ok is None] == []


def test_it_catches_the_0_6_0_failure(repo: Path) -> None:
    """Tagged, released on GitHub, absent from PyPI -- and nothing said so.

    This is the shape of the incident the verifier exists for: every public
    marker of a release present except the one that matters to a user.
    """
    checks = _by_label(
        verify(VERSION, root=repo, fetch=_fetch(_world(repo, on_pypi=False)))
    )

    assert checks[f"GitHub release {TAG} exists"].ok
    assert checks[f"tag {TAG} exists locally"].ok
    assert checks[f"PyPI serves {VERSION}"].ok is False


def test_local_checks_are_skipped_for_another_release_not_passed(repo: Path) -> None:
    """A pass here would be a lie, and a worse one than a failure.

    The first version reported "README cites the version DOI" when asked about
    0.6.0, because the README cites 0.6.1's DOI. The question does not apply,
    so it gets no answer rather than a favourable one.
    """
    checks = _by_label(verify("0.0.1", root=repo, fetch=_fetch(_world(repo))))

    assert checks["CITATION.cff names this release"].ok is False
    for label in ("README.md cites the version DOI", "Zenodo version DOI resolves"):
        assert checks[label].ok is None, label


def test_an_unreachable_index_is_not_a_pass(repo: Path) -> None:
    """Silence is not evidence that a release is fine."""
    checks = _by_label(verify(VERSION, root=repo, fetch=lambda url: None))

    assert checks[f"PyPI serves {VERSION}"].ok is False
    assert checks[f"GitHub release {TAG} exists"].ok is False


def test_a_draft_release_is_reported(repo: Path) -> None:
    """A draft never fires the publish job, which is why 0.6.1 was not one."""
    world = _world(repo)
    world[f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{TAG}"][
        "draft"
    ] = True

    checks = _by_label(verify(VERSION, root=repo, fetch=_fetch(world)))

    assert checks["GitHub release is published"].ok is False


def test_a_tag_that_moved_after_release_is_caught(repo: Path) -> None:
    """What was tagged must be what is public."""
    world = _world(repo)
    world[f"https://api.github.com/repos/{GITHUB_REPO}/git/ref/tags/{TAG}"] = {
        "object": {"sha": "0" * 40}
    }

    checks = _by_label(verify(VERSION, root=repo, fetch=_fetch(world)))

    assert checks["GitHub tag matches the local tag"].ok is False


def test_zenodo_archiving_the_wrong_version_is_caught(repo: Path) -> None:
    world = _world(repo)
    world["https://zenodo.org/api/records/22166257"] = {
        "metadata": {"version": "v1.2.3"}
    }

    checks = _by_label(verify(VERSION, root=repo, fetch=_fetch(world)))

    assert checks["Zenodo version DOI resolves"].ok is False
