"""Assemble the frozen replication package for the sparse-contamination paper.

The paper's credibility rests on the design having been fixed before any
outcome was inspected, so the archive ships the driver, the analysis-only
scripts, the frozen result artifacts and the provenance record together.  It is
meant to travel with the paper as supplementary material.

Usage::

    python build_replication_package.py
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent

# (source, destination inside the package).  Repository-relative paths are
# preserved in the REPRODUCE notes rather than in the layout.
PLAIN_FILES = [
    (HERE / "sparse_experiment.py", "scripts/sparse_experiment.py"),
    (HERE / "analyze_frozen_run.py", "scripts/analyze_frozen_run.py"),
    (HERE / "analyze_beta_grid.py", "scripts/analyze_beta_grid.py"),
    (HERE / "make_scale_figure.py", "scripts/make_scale_figure.py"),
    (HERE / "stress_experiments.py", "scripts/stress_experiments.py"),
    (HERE / "analyze_stress.py", "scripts/analyze_stress.py"),
    (HERE / "build_replication_package.py", "scripts/build_replication_package.py"),
    (HERE / "build_online_resource.py", "scripts/build_online_resource.py"),
    (HERE / "primary_summary.csv", "results/primary_summary.csv"),
    (HERE / "primary_report.json", "results/primary_report.json"),
    (HERE / "frozen_run_analysis.json", "results/frozen_run_analysis.json"),
    (HERE / "beta_grid_analysis.json", "results/beta_grid_analysis.json"),
    (HERE / "stress_summary.csv", "results/stress_summary.csv"),
    (HERE / "stress_report.json", "results/stress_report.json"),
    (HERE / "stress_analysis.json", "results/stress_analysis.json"),
    (HERE / "paper" / "main.tex", "paper/main.tex"),
    (HERE / "paper" / "main.pdf", "paper/main.pdf"),
]
GZIP_FILES = [
    (HERE / "primary_replicates.csv", "results/primary_replicates.csv.gz"),
]
GENERATED_DIR = HERE / "paper" / "generated"

REPRODUCE = """# Sparse Contamination Replication Package

Frozen replication package for *Sparse Contamination in Tail-Index Estimation:
Detectability, Negligibility, and Risk*.

The primary Monte Carlo run is **frozen**.  The design in Section 11 of the
manuscript was fixed before any outcome was inspected, and every table and
figure in Section 12 is rendered from the artifacts in `results/` by
analysis-only scripts that never simulate.  Reproducing the paper's reported
numbers therefore does not require re-running the simulation.

## Provenance of the frozen run

| field | value |
|---|---|
| `heavytails` version | {heavytails} |
| Python | {python} |
| NumPy | {numpy} |
| repository commit | `{commit}` |
| seed | {seed} |
| replications per cell | {trials} |
| summary rows | {rows} |
| wall clock | {seconds:.0f} s |

Design grid: gamma = {gamma}, n in {n_values}, r in {r_values}, h = {h},
beta in {betas}, with the signal grid fixed in normalized-scale units and
near-duplicate signal points collapsed at tolerance {tolerance}.

## Contents

- `scripts/sparse_experiment.py` — the simulation driver that produced the
  frozen artifacts.  Running it is **not** needed to reproduce the reported
  numbers, and re-running it with a different seed or grid produces a different
  experiment, not this one.
- `scripts/analyze_frozen_run.py` — analysis only.  Reads the frozen artifacts
  and writes every LaTeX table fragment plus `frozen_run_analysis.json`, the
  digest holding the numbers quoted in the manuscript prose.
- `scripts/make_scale_figure.py` — analysis only.  Renders the three-scale
  figure from the frozen summary.
- `scripts/analyze_beta_grid.py` — analysis only.  Renders the paired
  bounded-influence contrasts at every contamination count.
- `scripts/stress_experiments.py` — the **post-specified** stress driver.  These
  runs were specified after the primary results were seen and are reported as
  such; they use their own seed and write their own artifacts, and they never
  read or modify the frozen ones.
- `scripts/analyze_stress.py` — analysis only, for the stress artifacts.
- `scripts/build_replication_package.py` — this packaging script.
- `results/primary_summary.csv` — one row per design cell and estimator, with
  recovery, detection and paired MSE contrasts and their Monte Carlo standard
  errors.
- `results/primary_replicates.csv.gz` — per-replicate losses for every
  estimator, needed only for the paired estimator-versus-estimator contrasts
  and for independent auditing.
- `results/primary_report.json` — configuration, seed and provenance of the run.
- `results/frozen_run_analysis.json` — the digest of quoted numbers.
- `paper/main.tex`, `paper/main.pdf` and `paper/generated/` — the manuscript,
  the build it produces, and the generated table and figure fragments it
  inputs.

## Reproducing the reported numbers

From this directory, with Python {python}, `pandas` and `matplotlib` available:

```bash
gunzip -k results/primary_replicates.csv.gz
python scripts/analyze_frozen_run.py \\
    --summary results/primary_summary.csv \\
    --replicates results/primary_replicates.csv \\
    --outdir paper/generated --digest-dir rebuilt
python scripts/analyze_beta_grid.py \\
    --summary results/primary_summary.csv \\
    --replicates results/primary_replicates.csv \\
    --outdir paper/generated --digest-dir rebuilt
python scripts/make_scale_figure.py \\
    --summary results/primary_summary.csv \\
    --outdir paper/generated --preview-dir rebuilt
python scripts/analyze_stress.py \\
    --summary results/stress_summary.csv \\
    --outdir paper/generated --digest-dir rebuilt
```

All four are needed: together they write the twelve fragments that
`paper/main.tex` inputs.  Tables 5 and 11 are the only outputs that need the
replicate file, so `analyze_frozen_run.py --skip-replicates` reproduces the
remaining primary tables from the summary alone and `analyze_stress.py` never
reads it.

`--digest-dir rebuilt` sends the regenerated JSON digests to a fresh directory
so they can be compared against the shipped originals, which hold the numbers
quoted in the manuscript prose:

```bash
for f in frozen_run_analysis beta_grid_analysis stress_analysis; do
    diff results/$f.json rebuilt/$f.json && echo "$f matches"
done
```

Then build the manuscript.  Pass `-cd` so that latexmk changes into `paper/`
and the `\\input{{generated/...}}` paths resolve:

```bash
latexmk -pdf -cd paper/main.tex
```

The result should match the shipped `paper/main.pdf`.  PDF bytes are not
reproducible across runs, so compare the extracted text instead; it is
byte-identical:

```bash
pdftotext paper/main.pdf - | sha256sum
```

## Re-running the simulation itself

Only for auditing the driver, not for reproducing the paper:

```bash
python scripts/sparse_experiment.py --trials {trials} --seed {seed}
```

This requires the `heavytails` package at version {heavytails}; the adaptive
trimming scan and the harmonic-moment estimators come from it, so a different
version may not reproduce the frozen artifacts bit for bit.
"""


def sha256(path: Path) -> str:
    """Hash a file in chunks, so the large replicate export stays streamable."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Build the package directory and its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / "replication_package")
    args = parser.parse_args()

    if args.outdir.exists():
        shutil.rmtree(args.outdir)

    written: list[Path] = []
    for source, relative in PLAIN_FILES:
        if not source.exists():
            raise SystemExit(f"missing artifact: {source}")
        target = args.outdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written.append(target)

    for source, relative in GZIP_FILES:
        if not source.exists():
            raise SystemExit(f"missing artifact: {source}")
        target = args.outdir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as raw, gzip.open(target, "wb", compresslevel=6) as out:
            shutil.copyfileobj(raw, out, length=1 << 20)
        written.append(target)

    for fragment in sorted(GENERATED_DIR.iterdir()):
        target = args.outdir / "paper" / "generated" / fragment.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fragment, target)
        written.append(target)

    report = json.loads((HERE / "primary_report.json").read_text(encoding="utf-8"))
    provenance, config = report["provenance"], report["configuration"]
    readme = args.outdir / "REPRODUCE.md"
    readme.write_text(
        REPRODUCE.format(
            heavytails=provenance["heavytails_version"],
            python=provenance["python_version"],
            numpy=provenance["numpy_version"],
            commit=provenance["git_commit"][:7],
            seed=config["seed"],
            trials=config["trials"],
            gamma=config["gamma"],
            n_values=config["n_values"],
            r_values=config["r_values"],
            h=config["h"],
            betas=config["betas"],
            tolerance=config["duplicate_signal_tolerance"],
            rows=report["rows"],
            seconds=report["seconds"],
        ),
        encoding="utf-8",
    )
    written.append(readme)

    lines = ["# path  sha256  bytes", ""]
    for path in sorted(written, key=lambda p: p.relative_to(args.outdir).as_posix()):
        relative = path.relative_to(args.outdir).as_posix()
        lines.append(f"{relative}  {sha256(path)}  {path.stat().st_size}")
    manifest = args.outdir / "MANIFEST.txt"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(p.stat().st_size for p in written) + manifest.stat().st_size
    print(f"wrote {args.outdir}")
    print(f"{len(written) + 1} files, {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
