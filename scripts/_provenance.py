"""Shared provenance helpers for repository research scripts."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING, Any

import numpy

import heavytails

if TYPE_CHECKING:
    from pathlib import Path


def _git_commit(repo_root: Path) -> str | None:
    """Return the checked-out commit, or None outside a git checkout."""
    git_entry = repo_root / ".git"
    try:
        if git_entry.is_file():
            content = git_entry.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                git_dir = (
                    repo_root / content.removeprefix("gitdir:").strip()
                ).resolve()
            else:
                return None
        else:
            git_dir = git_entry

        # A linked worktree has its own HEAD but shares refs with the
        # repository it was created from, which it names in `commondir`.
        # Looking only in the worktree's git directory finds the HEAD and then
        # no ref, and the whole helper returns None -- which is how two result
        # files came to be written with `"git_commit": null` while sitting in
        # a perfectly ordinary checkout.
        common_dir = git_dir
        commondir_file = git_dir / "commondir"
        if commondir_file.is_file():
            common_dir = (
                git_dir / commondir_file.read_text(encoding="utf-8").strip()
            ).resolve()

        head = git_dir / "HEAD"
        content = head.read_text(encoding="utf-8").strip()
        if not content.startswith("ref:"):
            return content or None
        ref = content.removeprefix("ref:").strip()

        for directory in (git_dir, common_dir):
            ref_file = directory / ref
            if ref_file.is_file():
                return ref_file.read_text(encoding="utf-8").strip() or None
        for directory in (git_dir, common_dir):
            packed = directory / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(f" {ref}"):
                        return line.split()[0]
    except OSError:
        return None
    return None


def _package_version(repo_root: Path) -> tuple[str, str]:
    """Return the package version and the source that supplied it."""
    pyproject = repo_root / "pyproject.toml"
    if pyproject.is_file():
        try:
            text = pyproject.read_text(encoding="utf-8")
        except OSError:
            text = None
        if text is not None:
            try:
                import tomllib  # noqa: PLC0415

                version = tomllib.loads(text).get("project", {}).get("version")
            except ModuleNotFoundError:
                match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
                version = match.group(1) if match else None
            except ValueError:
                version = None
            if isinstance(version, str) and version:
                return version, "pyproject.toml"
    return heavytails.__version__, "installed distribution metadata"


def base_provenance(repo_root: Path) -> dict[str, Any]:
    """Describe code and runtime versions shared by research result artifacts."""
    version, version_source = _package_version(repo_root)
    return {
        "heavytails_version": version,
        "heavytails_version_source": version_source,
        "version_source": version_source,
        "git_commit": _git_commit(repo_root),
        "python_version": sys.version.split()[0],
        "numpy_version": numpy.__version__,
    }
