# Sparse Contamination Replication Package

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
| `heavytails` version | 0.5.0 |
| Python | 3.13.5 |
| NumPy | 2.3.5 |
| repository commit | `755e6ad` |
| seed | 20260826 |
| replications per cell | 5000 |
| summary rows | 498 |
| wall clock | 330 s |

Design grid: gamma = 0.5, n in [1000, 10000, 50000], r in [1, 3, 5, 10], h = 10,
beta in [0.5, 1.0, 2.0], with the signal grid fixed in normalized-scale units and
near-duplicate signal points collapsed at tolerance 1e-10.

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
- `ENVIRONMENT.txt` — the versions the analysis and the manuscript build
  were run with, including the TeX tooling.
- `paper/main.tex`, `paper/main.pdf` and `paper/generated/` — the manuscript,
  the build it produces, and the generated table and figure fragments it
  inputs.

### What the stress artifacts do not carry

Two asymmetries between the primary and the stress evidence, stated here rather
than left for a reader to discover.

`results/stress_report.json` records the configuration, the self-checks, the row
count and the runtime, but **no software provenance** --- no Python, NumPy or
`heavytails` version and no commit. The primary run has always recorded those.
The stress driver writes them now, but the artifact shipped here predates that
change and the values were not captured when it ran. They are not reconstructed
after the fact: a provenance record assembled later would assert something
nobody observed. Reproducing the stress rows exactly therefore depends on an
environment this archive does not pin, and the numbers are reported on that
basis.

There is also no stress replicate-level export. The primary study ships
`primary_replicates.csv.gz`, so its paired standard errors can be recomputed
from the archived data; the stress layer ships summaries only, so the Monte
Carlo standard errors in `stress_summary.csv` can be read but not independently
reconstructed. The stress results are post-specified and secondary to the
paper's claims, but they are not audited to the same depth as the primary run.

## Reproducing the reported numbers

From this directory, with the versions recorded in `ENVIRONMENT.txt` ---
Python, NumPy, pandas, matplotlib and the TeX tooling the archived PDF was
built with:

```bash
gunzip -k results/primary_replicates.csv.gz
python scripts/analyze_frozen_run.py \
    --summary results/primary_summary.csv \
    --replicates results/primary_replicates.csv \
    --outdir paper/generated --digest-dir rebuilt
python scripts/analyze_beta_grid.py \
    --summary results/primary_summary.csv \
    --replicates results/primary_replicates.csv \
    --outdir paper/generated --digest-dir rebuilt
python scripts/make_scale_figure.py \
    --summary results/primary_summary.csv \
    --outdir paper/generated --preview-dir rebuilt
python scripts/analyze_stress.py \
    --summary results/stress_summary.csv \
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
and the `\input{generated/...}` paths resolve:

```bash
latexmk -pdf -cd paper/main.tex
```

PDF bytes are not reproducible across runs --- timestamps and font subset
identifiers differ --- so the extracted text is what is compared.  Note that
the build above has already overwritten `paper/main.pdf`, so hashing that file
now would only hash the rebuild.  Compare against the expected value shipped
with the archive instead:

```bash
pdftotext paper/main.pdf - > rebuilt.txt
sha256sum -c paper/main.txt.sha256
```

`paper/main.txt.sha256` names `rebuilt.txt`, so this succeeds only if the text
your build produces matches the text the archived PDF was made from.  The
manuscript carries a fixed `\date`, so the comparison does not decay with the
calendar.

## Re-running the simulation itself

Only for auditing the driver, not for reproducing the paper:

```bash
python scripts/sparse_experiment.py --trials 5000 --seed 20260826
```

This requires the `heavytails` package at version 0.5.0; the adaptive
trimming scan and the harmonic-moment estimators come from it, so a different
version may not reproduce the frozen artifacts bit for bit.
