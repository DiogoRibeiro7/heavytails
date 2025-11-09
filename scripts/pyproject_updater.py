#!/usr/bin/env python
"""pyproject_updater.py

Update dependency constraints in `pyproject.toml` to the **latest** versions from PyPI.

Features
--------
- Reads both Poetry ([tool.poetry]) and PEP 621 ([project]) layouts.
- Preserves formatting and comments using tomlkit.
- Fetches the latest version from PyPI (skips yanked releases; pre-releases optional).
- Multiple update strategies: exact | caret | tilde | floor.
- By default respects the current major version unless --allow-major is set.
- Can target specific groups (Poetry group or PEP 621 optional-dependencies) and/or specific packages.
- Dry-run `--check` prints a unified diff without writing.

Usage
-----
python scripts/pyproject_updater.py upgrade \
    --strategy caret \
    --groups main,dev \
    --only numpy,pandas \
    --respect-major \
    --no-prerelease \
    --check

Requirements
------------
pip install tomlkit packaging
(Uses only stdlib for HTTP via urllib.request.)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import tomlkit
from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any


@dataclass(frozen=True)
class Options:
    strategy: str  # one of: exact, caret, tilde, floor
    allow_major: bool  # if False, keep within current major
    include_prerelease: bool  # if True, accept pre-releases
    groups: list[str]  # e.g., ["main", "dev"]
    only: list[str] | None  # package name filters (normalized)
    check: bool  # dry-run
    file: Path  # pyproject.toml path
    timeout: float  # HTTP timeout


# ---------- TOML helpers ----------


def _read_doc(path: Path) -> tuple[str, Any]:
    text = path.read_text(encoding="utf-8")
    return text, tomlkit.parse(text)


def _write_or_diff(path: Path, before: str, after: str, check: bool) -> int:
    if check:
        diff = "".join(
            unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile=str(path),
                tofile=str(path),
            )
        )
        sys.stdout.write(diff)
        return 0
    path.write_text(after, encoding="utf-8")
    return 0


def _layout(doc: Any) -> str:
    # Prefer Poetry if both exist
    if "tool" in doc and isinstance(doc["tool"], dict) and "poetry" in doc["tool"]:
        return "poetry"
    if "project" in doc:
        return "pep621"
    raise ValueError(
        "Unsupported pyproject: neither [tool.poetry] nor [project] found."
    )


# ---------- PyPI version lookup ----------


def _normalize_pkg_name(name: str) -> str:
    """Normalize per PEP 503 for PyPI URLs: lowercase and replace `_`/`.` with `-`.
    Keep extras separate (e.g., 'foo[bar]') — we strip extras for lookup.
    """
    base = name.split("[", 1)[0]
    return base.lower().replace("_", "-").replace(".", "-")


def _fetch_pypi_versions(name: str, timeout: float) -> dict[str, bool]:
    """Return ``{version_str: is_yanked}`` for package *name* from PyPI."""
    url = f"https://pypi.org/pypi/{_normalize_pkg_name(name)}/json"
    try:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        TimeoutError,
        json.JSONDecodeError,
    ):
        return {}

    versions: dict[str, bool] = {}
    releases = data.get("releases", {}) or {}
    for ver_str, files in releases.items():
        # files is a list of distributions; consider version yanked if all files are yanked
        if not isinstance(files, list) or len(files) == 0:
            continue
        all_yanked = all(
            bool(f.get("yanked", False)) for f in files if isinstance(f, dict)
        )
        versions[ver_str] = not all_yanked
    return versions


def _select_latest_version(
    versions: dict[str, bool], include_prerelease: bool
) -> Version | None:
    """Pick the highest non-yanked Version. If include_prerelease=False, prefer finals."""
    valid: list[Version] = []
    for ver_str, not_yanked in versions.items():
        if not not_yanked:
            continue
        try:
            v = Version(ver_str)
        except InvalidVersion:
            continue
        if (not include_prerelease) and v.is_prerelease:
            continue
        valid.append(v)
    if not valid:
        return None
    return max(valid)


# ---------- Constraint mapping ----------


def _poetry_string_for_strategy(v: Version, strategy: str) -> str:
    if strategy == "exact":
        return f"=={v}"
    if strategy == "caret":
        return f"^{v}"
    if strategy == "tilde":
        return f"~{v}"
    if strategy == "floor":
        return f">={v}"
    raise ValueError(f"Unknown strategy: {strategy}")


def _pep440_string_for_strategy(v: Version, strategy: str) -> str:
    if strategy == "exact":
        return f"=={v}"
    if strategy == "floor":
        return f">={v}"
    if strategy == "caret":
        # compatible with current MAJOR; upper bound next major
        upper = f"{v.major + 1}.0.0"
        return f">={v},<{upper}"
    if strategy == "tilde":
        # compatible with current MINOR; upper bound next minor
        upper = f"{v.major}.{v.minor + 1}.0"
        return f">={v},<{upper}"
    raise ValueError(f"Unknown strategy: {strategy}")


def _respect_major_allowed(
    current_spec: str | None, latest: Version, allow_major: bool
) -> bool:
    """If allow_major is False and current_spec indicates a major cap, avoid bumping across majors.
    Heuristic: extract existing max major from spec if present; otherwise compare against any pinned/ranged major.
    """
    if allow_major:
        return True
    if not current_spec:
        return latest.major == latest.major  # trivial True
    # Try to parse the requirement to see any existing max
    try:
        req = Requirement(f"pkg {current_spec}")
    except Exception:
        # Fallback: if spec starts with ^ or ~ (Poetry), infer major from latest string of spec if present
        if current_spec.startswith("^") or current_spec.startswith("~"):
            # Keep within the current major implied by the spec's base number
            try:
                s = current_spec.removeprefix("^").removeprefix("~")
                # If there's a leading comparison operator like >=, ==, etc., remove it once.
                for op in ("==", ">=", "<=", "!=", "~=", ">", "<", "="):
                    if s.startswith(op):
                        s = s[len(op) :]
                        break
                s = s.lstrip()
                base = Version(s)
            except Exception:
                return True
            else:
                return latest.major <= base.major
        return True

    # If there is an upper bound like <2.0.0, prevent crossing it.
    for sp in req.specifier:
        if sp.operator in ("<", "<="):
            try:
                upper = Version(sp.version)
                if upper.major <= latest.major:
                    return False
            except InvalidVersion:
                pass
    return True


# ---------- Dependency iteration & rewriting ----------


@dataclass
class DepRef:
    layout: str  # "poetry" or "pep621"
    group: str  # "main" or group name
    name: str  # raw name as in file (may include extras)
    current_spec: str | None  # string spec; None if path/git or table
    location: tuple[Any, ...]  # references for writing (table/array and key/index)


def _iter_poetry_deps(doc: Any, groups: Iterable[str]) -> Iterable[DepRef]:
    tool = doc.setdefault("tool", tomlkit.table())
    poetry = tool.setdefault("poetry", tomlkit.table())

    def emit_from_table(tbl: Any, group: str) -> Iterable[DepRef]:
        if not isinstance(tbl, dict):
            return
        for k, v in list(tbl.items()):
            # Skip python pseudo-dep
            if k == "python":
                continue
            if isinstance(v, str):
                yield DepRef("poetry", group, k, v, (tbl, k))
            elif isinstance(v, dict):
                # Support {version="..."}; skip non-PyPI (path, git)
                ver = v.get("version")
                if isinstance(ver, str):
                    yield DepRef("poetry", group, k, ver, (v, "version"))
                else:
                    continue  # non-versioned (git/path) -> skip
            else:
                continue

    wanted = set(groups)
    if "main" in wanted or not wanted:
        deps = poetry.get("dependencies", {})
        yield from emit_from_table(deps, "main")

    if "dev" in wanted:
        dev = poetry.get("dev-dependencies", {})
        yield from emit_from_table(dev, "dev")

    # named groups
    group_tbl = poetry.get("group", {})
    if isinstance(group_tbl, dict):
        for gname, gtbl in group_tbl.items():
            if wanted and gname not in wanted:
                continue
            if isinstance(gtbl, dict):
                deps = gtbl.get("dependencies", {})
                yield from emit_from_table(deps, gname)


def _iter_pep621_deps(doc: Any, groups: Iterable[str]) -> Iterable[DepRef]:
    project = doc.setdefault("project", tomlkit.table())
    groups_set = set(groups)

    def emit_from_array(arr: Any, group: str) -> Iterable[DepRef]:
        if not isinstance(arr, tomlkit.items.Array):
            return
        for idx, item in enumerate(list(arr)):
            if not isinstance(item, str):
                continue
            try:
                req = Requirement(item)
            except Exception:
                continue
            spec = str(req.specifier) if req.specifier else None
            # Keep original text shape; we'll overwrite the whole string at index
            yield DepRef("pep621", group, req.name, spec, (arr, idx, req))

    # main deps
    if not groups_set or "main" in groups_set:
        arr = project.setdefault("dependencies", tomlkit.array())
        emit = list(emit_from_array(arr, "main"))
        yield from emit

    # optional groups
    opt = project.setdefault("optional-dependencies", tomlkit.table())
    if isinstance(opt, dict):
        for gname, arr in opt.items():
            if groups_set and gname not in groups_set:
                continue
            emit = list(emit_from_array(arr, gname))
            yield from emit


def _set_dep_spec(dep: DepRef, new_spec: str) -> None:
    """Write back new spec to the TOML document at the stored location."""
    if dep.layout == "poetry":
        tbl, key = dep.location
        if isinstance(tbl, dict):
            # If it was a string spec: overwrite with string.
            if isinstance(tbl.get(key), str):
                tbl[key] = new_spec
            elif isinstance(tbl.get(key), dict):
                tbl[key]["version"] = new_spec
            else:
                tbl[key] = new_spec
    else:
        arr, idx, req = dep.location
        if isinstance(arr, tomlkit.items.Array):
            # Rebuild requirement string with new spec; keep extras/markers
            name = req.name
            extras = f"[{','.join(sorted(req.extras))}]" if req.extras else ""
            markers = f"; {req.marker}" if req.marker else ""
            arr[idx] = f"{name}{extras} {new_spec}{markers}".strip()


# ---------- Main upgrade routine ----------


def upgrade(pyproject: Path, opts: Options) -> int:
    before_text, doc = _read_doc(pyproject)
    layout = _layout(doc)

    # Which groups to consider by default
    groups = opts.groups or ["main", "dev"]  # include Poetry dev by default
    only_norm = {_normalize_pkg_name(n) for n in (opts.only or [])}

    # Iterate deps
    iterator = _iter_poetry_deps if layout == "poetry" else _iter_pep621_deps
    changed = 0

    for dep in iterator(doc, groups):
        base_norm = _normalize_pkg_name(dep.name)
        if only_norm and base_norm not in only_norm:
            continue
        # Skip obviously non-PyPI
        if dep.current_spec is None:
            continue

        # Respect-major check (heuristic against crossing major caps)
        # We perform check after we fetch latest.
        versions = _fetch_pypi_versions(dep.name, opts.timeout)
        latest = _select_latest_version(versions, opts.include_prerelease)
        if latest is None:
            continue

        if not _respect_major_allowed(dep.current_spec, latest, opts.allow_major):
            # If not allowed, try to keep within current major by taking the max version < next major.
            target_major = None
            # Guess current allowed major from current spec or dep.current_spec base
            try:
                if dep.current_spec.startswith(("^", "~")):
                    target_major = Version(dep.current_spec.lstrip("^~ =><")).major
            except Exception:
                pass
            if target_major is not None:
                # pick highest < target_major+1.0.0
                candidates = [Version(v) for v, ok in versions.items() if ok]
                within = [
                    v
                    for v in candidates
                    if (
                        v.major == target_major
                        and (opts.include_prerelease or not v.is_prerelease)
                    )
                ]
                if within:
                    latest = max(within)

        # Compute new spec string according to layout/strategy
        if layout == "poetry":
            new_spec = _poetry_string_for_strategy(latest, opts.strategy)
        else:
            new_spec = _pep440_string_for_strategy(latest, opts.strategy)

        # If spec already implies >= latest (exact match for exact), skip writing.
        if dep.current_spec and dep.current_spec.strip() == new_spec.strip():
            continue

        _set_dep_spec(dep, new_spec)
        changed += 1

    after_text = tomlkit.dumps(doc)
    if before_text == after_text:
        # Nothing to do; still honor --check by printing empty diff (no output).
        return 0
    return _write_or_diff(pyproject, before_text, after_text, opts.check)


def parse_args(argv: list[str] | None = None) -> Options:
    p = argparse.ArgumentParser(
        description="Upgrade pyproject dependency constraints to latest from PyPI."
    )
    p.add_argument("--file", default="pyproject.toml", help="Path to pyproject.toml")
    p.add_argument(
        "--strategy",
        choices=["exact", "caret", "tilde", "floor"],
        default="caret",
        help="How to express the updated constraint.",
    )
    p.add_argument(
        "--allow-major",
        action="store_true",
        help="Allow bumping to a new MAJOR version.",
    )
    p.add_argument(
        "--respect-major",
        dest="allow_major",
        action="store_false",
        help="(default) Keep within the current major if possible.",
    )
    p.set_defaults(allow_major=False)
    p.add_argument(
        "--pre",
        "--include-prerelease",
        dest="include_prerelease",
        action="store_true",
        help="Allow pre-releases when picking the latest.",
    )
    p.add_argument(
        "--no-prerelease",
        dest="include_prerelease",
        action="store_false",
        help="(default) Exclude pre-releases.",
    )
    p.set_defaults(include_prerelease=False)
    p.add_argument(
        "--groups",
        default="main,dev",
        help="Comma-separated groups: e.g., main,dev or analytics,docs",
    )
    p.add_argument(
        "--only",
        default="",
        help="Comma-separated package names to update (normalized). Empty=all.",
    )
    p.add_argument(
        "--check", action="store_true", help="Dry-run: show unified diff, do not write."
    )
    p.add_argument("--timeout", type=float, default=8.0, help="HTTP timeout (seconds).")

    args = p.parse_args(argv)
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    only = [n.strip() for n in args.only.split(",") if n.strip()] or None

    return Options(
        strategy=args.strategy,
        allow_major=args.allow_major,
        include_prerelease=args.include_prerelease,
        groups=groups,
        only=only,
        check=args.check,
        file=Path(args.file),
        timeout=args.timeout,
    )


# ---------- Advanced Dependency Management Features ----------


def _extract_current_versions(doc: Any) -> dict[str, str]:
    """Extract all current package versions from the TOML document."""
    versions: dict[str, str] = {}
    layout = _layout(doc)

    if layout == "poetry":
        tool = doc.get("tool", {})
        poetry = tool.get("poetry", {})

        # Main dependencies
        for name, spec in poetry.get("dependencies", {}).items():
            if name == "python":
                continue
            if isinstance(spec, str):
                versions[name] = spec
            elif isinstance(spec, dict) and "version" in spec:
                versions[name] = spec["version"]

        # Dev dependencies
        for name, spec in poetry.get("dev-dependencies", {}).items():
            if isinstance(spec, str):
                versions[name] = spec
            elif isinstance(spec, dict) and "version" in spec:
                versions[name] = spec["version"]

        # Group dependencies
        groups = poetry.get("group", {})
        if isinstance(groups, dict):
            for group_data in groups.values():
                if isinstance(group_data, dict):
                    deps = group_data.get("dependencies", {})
                    for name, spec in deps.items():
                        if isinstance(spec, str):
                            versions[name] = spec
                        elif isinstance(spec, dict) and "version" in spec:
                            versions[name] = spec["version"]

    elif layout == "pep621":
        project = doc.get("project", {})

        # Main dependencies
        deps_array = project.get("dependencies", [])
        if isinstance(deps_array, list):
            for dep_str in deps_array:
                if isinstance(dep_str, str):
                    try:
                        req = Requirement(dep_str)
                        if req.specifier:
                            versions[req.name] = str(req.specifier)
                    except Exception:
                        continue

        # Optional dependencies
        opt_deps = project.get("optional-dependencies", {})
        if isinstance(opt_deps, dict):
            for group_deps in opt_deps.values():
                if isinstance(group_deps, list):
                    for dep_str in group_deps:
                        if isinstance(dep_str, str):
                            try:
                                req = Requirement(dep_str)
                                if req.specifier:
                                    versions[req.name] = str(req.specifier)
                            except Exception:
                                continue

    return versions


def _check_dep_vulnerabilities(package_name: str) -> list[str]:
    """Check single package for vulnerabilities.

    This is a demonstration using a static database. In production,
    integrate with:
    - pip-audit API
    - Safety DB API (https://pyup.io/safety/)
    - OSV (Open Source Vulnerabilities) API
    - GitHub Advisory Database
    """
    # Known vulnerable packages (subset for demonstration)
    # In production, fetch from a real vulnerability database
    known_vulnerabilities: dict[str, list[str]] = {
        "requests": ["CVE-2023-32681: Unintended leak of Proxy-Authorization header"],
        "pillow": [
            "CVE-2023-44271: Buffer overflow in _getexif",
            "CVE-2023-50447: Arbitrary code execution via crafted font",
        ],
        "flask": ["CVE-2023-30861: Cookie parsing vulnerability"],
        "django": [
            "CVE-2023-43665: Denial-of-service in file uploads",
            "CVE-2023-41164: Potential denial of service in django.utils.encoding.uri_to_iri",
        ],
        "urllib3": [
            "CVE-2023-43804: Cookie request header isn't stripped on cross-origin redirects"
        ],
        "cryptography": [
            "CVE-2023-49083: NULL-dereference when loading PKCS7 certificates"
        ],
        "pyyaml": ["CVE-2020-14343: Arbitrary code execution via unsafe yaml.load"],
        "jinja2": ["CVE-2020-28493: ReDoS vulnerability"],
    }

    return known_vulnerabilities.get(package_name.lower(), [])


def check_vulnerabilities(pyproject: Path) -> dict[str, list[str]]:
    """Check dependencies for known security vulnerabilities.

    Parameters
    ----------
    pyproject : Path
        Path to pyproject.toml file

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping package names to lists of vulnerability descriptions
    """
    _, doc = _read_doc(pyproject)
    vulnerabilities: dict[str, list[str]] = {}

    # Extract all dependencies
    all_deps: set[str] = set()

    try:
        layout = _layout(doc)

        if layout == "poetry":
            tool = doc.get("tool", {})
            poetry = tool.get("poetry", {})

            # Main dependencies
            deps = poetry.get("dependencies", {})
            if isinstance(deps, dict):
                all_deps.update(deps.keys())

            # Dev dependencies
            dev_deps = poetry.get("dev-dependencies", {})
            if isinstance(dev_deps, dict):
                all_deps.update(dev_deps.keys())

            # Group dependencies
            groups = poetry.get("group", {})
            if isinstance(groups, dict):
                for group_data in groups.values():
                    if isinstance(group_data, dict):
                        group_deps = group_data.get("dependencies", {})
                        if isinstance(group_deps, dict):
                            all_deps.update(group_deps.keys())

        elif layout == "pep621":
            project = doc.get("project", {})

            # Main dependencies
            deps_array = project.get("dependencies", [])
            if isinstance(deps_array, list):
                for dep_str in deps_array:
                    if isinstance(dep_str, str):
                        try:
                            req = Requirement(dep_str)
                            all_deps.add(req.name)
                        except Exception:
                            continue

            # Optional dependencies
            opt_deps = project.get("optional-dependencies", {})
            if isinstance(opt_deps, dict):
                for group_deps in opt_deps.values():
                    if isinstance(group_deps, list):
                        for dep_str in group_deps:
                            if isinstance(dep_str, str):
                                try:
                                    req = Requirement(dep_str)
                                    all_deps.add(req.name)
                                except Exception:
                                    continue

    except ValueError:
        # Unsupported layout
        return vulnerabilities

    # Check each dependency against vulnerability database
    for dep_name in all_deps:
        if dep_name == "python":
            continue

        try:
            vuln_list = _check_dep_vulnerabilities(dep_name)
            if vuln_list:
                vulnerabilities[dep_name] = vuln_list
        except Exception:
            continue  # Skip if vulnerability check fails

    return vulnerabilities


def detect_conflicts(pyproject: Path) -> dict[str, str]:
    """Detect potential dependency conflicts.

    Parameters
    ----------
    pyproject : Path
        Path to pyproject.toml file

    Returns
    -------
    dict[str, str]
        Dictionary mapping conflict identifiers to conflict descriptions
    """
    _, doc = _read_doc(pyproject)
    conflicts: dict[str, str] = {}

    # Extract version constraints
    constraints: dict[str, str] = {}

    try:
        layout = _layout(doc)

        if layout == "poetry":
            tool = doc.get("tool", {})
            poetry = tool.get("poetry", {})

            # Main dependencies
            for name, spec in poetry.get("dependencies", {}).items():
                if isinstance(spec, str) and name != "python":
                    constraints[name] = spec
                elif isinstance(spec, dict) and "version" in spec:
                    constraints[name] = spec["version"]

            # Group dependencies
            groups = poetry.get("group", {})
            if isinstance(groups, dict):
                for group_name, group_data in groups.items():
                    if isinstance(group_data, dict):
                        deps = group_data.get("dependencies", {})
                        if isinstance(deps, dict):
                            for name, spec in deps.items():
                                if isinstance(spec, str):
                                    # Check for conflicting constraints
                                    if (
                                        name in constraints
                                        and constraints[name] != spec
                                    ):
                                        conflicts[name] = (
                                            f"Conflict: main has {constraints[name]}, "
                                            f"{group_name} has {spec}"
                                        )
                                    constraints[name] = spec
                                elif isinstance(spec, dict) and "version" in spec:
                                    ver_spec = spec["version"]
                                    if (
                                        name in constraints
                                        and constraints[name] != ver_spec
                                    ):
                                        conflicts[name] = (
                                            f"Conflict: main has {constraints[name]}, "
                                            f"{group_name} has {ver_spec}"
                                        )
                                    constraints[name] = ver_spec

        elif layout == "pep621":
            project = doc.get("project", {})

            # Main dependencies
            deps_array = project.get("dependencies", [])
            if isinstance(deps_array, list):
                for dep_str in deps_array:
                    if isinstance(dep_str, str):
                        try:
                            req = Requirement(dep_str)
                            if req.specifier:
                                constraints[req.name] = str(req.specifier)
                        except Exception:
                            continue

            # Optional dependencies
            opt_deps = project.get("optional-dependencies", {})
            if isinstance(opt_deps, dict):
                for group_name, group_deps in opt_deps.items():
                    if isinstance(group_deps, list):
                        for dep_str in group_deps:
                            if isinstance(dep_str, str):
                                try:
                                    req = Requirement(dep_str)
                                    if req.specifier:
                                        spec_str = str(req.specifier)
                                        if (
                                            req.name in constraints
                                            and constraints[req.name] != spec_str
                                        ):
                                            conflicts[req.name] = (
                                                f"Conflict: main has {constraints[req.name]}, "
                                                f"{group_name} has {spec_str}"
                                            )
                                        constraints[req.name] = spec_str
                                except Exception:
                                    continue

    except ValueError:
        # Unsupported layout
        return conflicts

    # Known package ecosystem conflict rules
    conflict_rules: dict[tuple[str, str], str] = {
        ("requests", "urllib3"): (
            "requests includes urllib3; explicit urllib3 version may conflict"
        ),
        ("pandas", "numpy"): "Ensure numpy version compatibility with pandas",
        ("matplotlib", "numpy"): "Matplotlib requires specific numpy versions",
        ("scikit-learn", "numpy"): "scikit-learn has strict numpy version requirements",
        ("tensorflow", "numpy"): "TensorFlow requires specific numpy versions",
        ("torch", "numpy"): "PyTorch may have numpy version constraints",
    }

    for (pkg1, pkg2), message in conflict_rules.items():
        if pkg1 in constraints and pkg2 in constraints:
            conflicts[f"{pkg1}+{pkg2}"] = message

    return conflicts


if TYPE_CHECKING:
    from typing import Any


def upgrade_with_analysis(pyproject: Path, opts: Options) -> dict[str, Any]:
    """Upgrade dependencies with comprehensive impact analysis.

    Parameters
    ----------
    pyproject : Path
        Path to pyproject.toml file
    opts : Options
        Upgrade options

    Returns
    -------
    dict[str, Any]
        Comprehensive analysis including pre/post upgrade state and impacts
    """
    _before_text, doc = _read_doc(pyproject)

    # Pre-upgrade analysis
    pre_analysis: dict[str, Any] = {
        "vulnerabilities": check_vulnerabilities(pyproject),
        "conflicts": detect_conflicts(pyproject),
        "current_versions": _extract_current_versions(doc),
    }

    # Perform upgrade (but restore the original file content first for proper comparison)
    upgrade_result = upgrade(pyproject, opts)

    # Post-upgrade analysis
    _, updated_doc = _read_doc(pyproject)
    post_analysis: dict[str, Any] = {
        "vulnerabilities": check_vulnerabilities(pyproject),
        "conflicts": detect_conflicts(pyproject),
        "new_versions": _extract_current_versions(updated_doc),
    }

    # Calculate changes and impacts
    version_changes: dict[str, dict[str, str]] = {}
    current_versions = pre_analysis["current_versions"]
    if isinstance(current_versions, dict):
        for pkg in current_versions:
            old_ver = current_versions[pkg]
            new_versions = post_analysis["new_versions"]
            if isinstance(new_versions, dict):
                new_ver = new_versions.get(pkg, old_ver)
            else:
                new_ver = old_ver
            # Ensure both old_ver and new_ver are strings
            if (
                isinstance(old_ver, str)
                and isinstance(new_ver, str)
                and old_ver != new_ver
            ):
                version_changes[pkg] = {"from": old_ver, "to": new_ver}

    # Impact assessment
    pre_vulns = pre_analysis["vulnerabilities"]
    post_vulns = post_analysis["vulnerabilities"]
    pre_conflicts = pre_analysis["conflicts"]
    post_conflicts = post_analysis["conflicts"]

    impact_assessment: dict[str, Any] = {
        "version_changes": version_changes,
        "vulnerabilities_fixed": (
            len(pre_vulns) - len(post_vulns)
            if isinstance(pre_vulns, dict) and isinstance(post_vulns, dict)
            else 0
        ),
        "new_conflicts": [
            conflict
            for conflict in post_conflicts
            if isinstance(post_conflicts, dict)
            and isinstance(pre_conflicts, dict)
            and conflict not in pre_conflicts
        ],
        "resolved_conflicts": [
            conflict
            for conflict in pre_conflicts
            if isinstance(pre_conflicts, dict)
            and isinstance(post_conflicts, dict)
            and conflict not in post_conflicts
        ],
    }

    return {
        "upgrade_result": upgrade_result,
        "pre_analysis": pre_analysis,
        "post_analysis": post_analysis,
        "impact_assessment": impact_assessment,
    }


def _security_only_upgrade(pyproject: Path, opts: Options) -> int:
    """Upgrade only packages with known security vulnerabilities.

    Parameters
    ----------
    pyproject : Path
        Path to pyproject.toml file
    opts : Options
        Upgrade options (will be modified to target only vulnerable packages)

    Returns
    -------
    int
        Exit code
    """
    vulnerabilities = check_vulnerabilities(pyproject)

    if not vulnerabilities:
        print("No security vulnerabilities found. No updates needed.")
        return 0

    # Create a modified opts to only update vulnerable packages
    vulnerable_packages = list(vulnerabilities.keys())

    print(f"Security update: Upgrading {len(vulnerable_packages)} vulnerable packages")
    for pkg, vulns in vulnerabilities.items():
        print(f"  {pkg}: {len(vulns)} vulnerabilities")
        for vuln in vulns:
            print(f"    - {vuln}")

    # Create new Options with only vulnerable packages
    security_opts = replace(opts, only=vulnerable_packages)

    result = upgrade(pyproject, security_opts)

    return result


def upgrade_with_prerelease_policy(
    pyproject: Path, opts: Options, policy: str = "conservative"
) -> int:
    """Upgrade with configurable pre-release policies.

    Parameters
    ----------
    pyproject : Path
        Path to pyproject.toml file
    opts : Options
        Base upgrade options
    policy : str
        Policy name: 'conservative', 'moderate', 'aggressive', or 'security_only'

    Returns
    -------
    int
        Exit code

    Raises
    ------
    ValueError
        If unknown policy is specified
    """
    # Define pre-release policies
    policies = {
        "conservative": {"include_prerelease": False, "allow_major": False},
        "moderate": {"include_prerelease": False, "allow_major": True},
        "aggressive": {"include_prerelease": True, "allow_major": True},
        "security_only": {
            "include_prerelease": False,
            "allow_major": False,
            "security_only": True,
        },
    }

    if policy not in policies:
        raise ValueError(
            f"Unknown policy: {policy}. Available: {list(policies.keys())}"
        )

    policy_settings = policies[policy]

    # Create modified Options with policy settings
    modified_opts = replace(
        opts,
        include_prerelease=policy_settings["include_prerelease"],
        allow_major=policy_settings["allow_major"],
    )

    # Special handling for security-only updates
    if policy_settings.get("security_only", False):
        return _security_only_upgrade(pyproject, modified_opts)
    else:
        return upgrade(pyproject, modified_opts)


def main(argv: list[str] | None = None) -> int:
    opts = parse_args(argv)
    return upgrade(opts.file, opts)


if __name__ == "__main__":
    raise SystemExit(main())
