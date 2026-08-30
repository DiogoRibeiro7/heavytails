"""Build Online Resource 1: the compact reproduction guide for the journal.

The full replication archive is large, mostly because of the per-replicate
export, so it is not what travels with the manuscript.  It lives in the
project's public repository and is archived with each release, which is what
gives it a citable identifier.  What the journal carries alongside the paper is
this small bundle: the manifest of that archive, the run configuration and
provenance, a map from each table and figure to the file that produces it, and
the archive's identifier.

Usage::

    python build_online_resource.py
    python build_online_resource.py --doi 10.5281/zenodo.XXXXXXX
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
DOI_PLACEHOLDER = "[REPLICATION DOI]"
EMAIL_PLACEHOLDER = "[CORRESPONDING E-MAIL]"

# Which script writes each numbered object in the manuscript.
PROVENANCE = [
    ("Table 1", "clean control", "analyze_frozen_run.py", "primary_summary.csv"),
    ("Table 2", "detection transition", "analyze_frozen_run.py", "primary_summary.csv"),
    ("Table 3", "intervention scale", "analyze_frozen_run.py", "primary_summary.csv"),
    (
        "Table 4",
        "bounded-influence family",
        "analyze_frozen_run.py",
        "primary_summary.csv",
    ),
    (
        "Table 5",
        "paired beta contrasts",
        "analyze_frozen_run.py",
        "primary_replicates.csv",
    ),
    (
        "Table 6",
        "detection transition, fine grid",
        "analyze_stress.py",
        "stress_summary.csv",
    ),
    ("Table 7", "contamination profiles", "analyze_stress.py", "stress_summary.csv"),
    ("Table 8", "second-order tails", "analyze_stress.py", "stress_summary.csv"),
    (
        "Table 9",
        "pre-asymptotic stress test",
        "analyze_frozen_run.py",
        "primary_summary.csv",
    ),
    ("Table 10", "beta sweep over r", "analyze_frozen_run.py", "primary_summary.csv"),
    (
        "Table 11",
        "full beta contrasts",
        "analyze_beta_grid.py",
        "primary_replicates.csv",
    ),
    ("Fig. 1", "the three scales", "make_scale_figure.py", "primary_summary.csv"),
]

README = """# Online Resource 1 --- Reproduction Guide

**Article** Sparse Contamination in Tail-Index Estimation: Detectability,
Negligibility, and Risk

**Journal** Extremes

**Author** Diogo Ribeiro, Faculty of Media Arts and Design, Technical
University of Porto, Porto, Portugal

**Corresponding author** {email}

This bundle is the compact companion to the paper.  The **complete replication
archive** --- simulation drivers, analysis-only scripts, the frozen summary and
per-replicate losses, and the manuscript sources --- is archived at

    {doi}

which is the software release that carries it; the per-replicate export alone is
{replicates_mb:.0f} MB, which is why it is not attached to the article.  This bundle
contains what a reader needs to check the archive's integrity and to find the
file behind any number in the paper.

## Contents

- `README.md` --- this guide
- `MANIFEST.txt` --- SHA-256 and byte size of every file in the replication
  archive
- `configuration.json` --- the frozen run's design, seed and software
  provenance
- `stress_configuration.json` --- the same for the post-specified stress runs
- `object_provenance.csv` --- which script and which artifact produce each
  table and figure

## The two runs

The primary experiment is **pre-specified**: its design was fixed before any
outcome was inspected, and no result in the paper revises it.  The stress
experiments of Section 13 are **post-specified**, added after the primary
results were seen, and are labelled as such throughout the paper.  They use a
separate driver, a separate seed and separate artifacts.

| run | trials per cell | seed | driver |
|---|---|---|---|
| primary (pre-specified) | {primary_trials} | {primary_seed} | `sparse_experiment.py` |
| stress (post-specified) | {stress_trials} | {stress_seed} | `stress_experiments.py` |

## Reproducing the paper

Full instructions are in `REPRODUCE.md` inside the replication archive.  In
short: unpack the archive, run the four analysis-only scripts, and build the
manuscript with `latexmk -pdf -cd paper/main.tex`.  None of this re-runs the
simulation; the archived artifacts are the frozen record.

Verify the archive against `MANIFEST.txt` before use.
"""


def _pending_placeholders(args: argparse.Namespace) -> list[str]:
    """Which required values are still placeholders."""
    return [
        f"{name} ({flag})"
        for name, value, flag in (
            ("DOI", args.doi, "--doi"),
            ("corresponding e-mail", args.email, "--email"),
        )
        if value.startswith("[")
    ]


def _locate_package() -> Path | None:
    """Find the replication package from wherever this script is running.

    This script is archived *inside* the package it reads, at
    ``replication_package/scripts/``, so the working-tree assumption --- that
    the package sits next to the script --- resolves to
    ``replication_package/scripts/replication_package`` once deposited, which
    does not exist. The script that generates the journal supplement could not
    be run from the deposit it was supplied in.

    Both layouts are checked, identified by the manifest rather than by name.
    """
    for candidate in (HERE / "replication_package", HERE.parent):
        if (candidate / "MANIFEST.txt").is_file():
            return candidate
    return None


def main() -> None:
    """Assemble the Online Resource bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doi",
        default=DOI_PLACEHOLDER,
        help="replication archive DOI, once deposited",
    )
    parser.add_argument(
        "--email",
        default=EMAIL_PLACEHOLDER,
        help="corresponding author address, as it appears on the title page",
    )
    parser.add_argument("--outdir", type=Path, default=HERE / "online_resource_1")
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="build a draft even with an unresolved DOI or e-mail",
    )
    parser.add_argument(
        "--package-root",
        type=Path,
        default=None,
        help="the replication package to read (default: locate it)",
    )
    args = parser.parse_args()

    # Fail closed, and before anything is written. This builds the supplement
    # that goes to the journal; it used to print a note and exit 0, so an
    # unresolved DOI produced a complete-looking five-file bundle that reported
    # success. Checking after the write would still leave that bundle on disk.
    pending = _pending_placeholders(args)
    if pending and not args.allow_placeholders:
        raise SystemExit(
            "refusing to build a supplement with unresolved "
            + ", ".join(pending)
            + "\npass --allow-placeholders to build a draft anyway"
        )

    package = args.package_root or _locate_package()
    if package is None or not package.is_dir():
        raise SystemExit(
            "cannot find the replication package; run build_replication_package.py "
            "first, or pass --package-root"
        )

    if args.outdir.exists():
        shutil.rmtree(args.outdir)
    args.outdir.mkdir(parents=True)

    shutil.copy2(package / "MANIFEST.txt", args.outdir / "MANIFEST.txt")
    shutil.copy2(
        package / "results" / "primary_report.json", args.outdir / "configuration.json"
    )
    shutil.copy2(
        package / "results" / "stress_report.json",
        args.outdir / "stress_configuration.json",
    )

    rows = ["object,description,script,artifact"]
    rows += [",".join(r) for r in PROVENANCE]
    (args.outdir / "object_provenance.csv").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
    )

    primary = json.loads((package / "results" / "primary_report.json").read_text())
    stress = json.loads((package / "results" / "stress_report.json").read_text())
    replicates_mb = (
        package / "results" / "primary_replicates.csv.gz"
    ).stat().st_size / 1e6
    (args.outdir / "README.md").write_text(
        README.format(
            doi=args.doi,
            email=args.email,
            replicates_mb=replicates_mb,
            primary_trials=primary["configuration"]["trials"],
            primary_seed=primary["configuration"]["seed"],
            stress_trials=stress["configuration"]["trials"],
            stress_seed=stress["configuration"]["seed"],
        ),
        encoding="utf-8",
    )

    total = sum(p.stat().st_size for p in args.outdir.iterdir())
    print(f"wrote {args.outdir}")
    print(f"{len(list(args.outdir.iterdir()))} files, {total / 1024:.0f} KB")
    if _pending_placeholders(args):
        print("NOTE: draft only, placeholders remain")


if __name__ == "__main__":
    main()
