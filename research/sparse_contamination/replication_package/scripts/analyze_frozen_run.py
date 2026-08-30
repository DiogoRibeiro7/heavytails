"""Report tables for the first frozen exact-Pareto sparse-contamination run.

The primary run is frozen: this script only reads the artifacts it produced and
renders the reporting objects the manuscript needs.  It never simulates, so it
cannot silently change the design.

Outputs LaTeX fragments into ``paper/generated`` plus a JSON digest of the
numbers quoted in the prose.

Usage::

    python analyze_frozen_run.py
    python analyze_frozen_run.py --skip-replicates   # summary-only, faster
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

HERE = Path(__file__).resolve().parent
TRIALS = 5000
Z = 1.959963984540054

# Deterministic display order for the frozen signal grid.
SIGNAL_ORDER = ["0", "0.5I", "1.5I", "1.5I|0.5R", "0.5D", "1.5D", "0.5R", "1.5R"]
SIGNAL_TEX = {
    "0": r"$0$",
    "0.5I": r"$0.5I$",
    "1.5I": r"$1.5I$",
    "1.5I|0.5R": r"$1.5I{=}0.5R$",
    "0.5D": r"$0.5D$",
    "1.5D": r"$1.5D$",
    "0.5R": r"$0.5R$",
    "1.5R": r"$1.5R$",
}
N_TEX = {1000: r"$10^3$", 10000: r"$10^4$", 50000: r"$5{\times}10^4$"}

ADAPTIVE = "adaptive_trimmed_hill"
ORACLE = "oracle_trimmed_hill"
# The whole pre-specified bounded-influence family, in frozen order.  beta = 1
# is the named t-Hill estimator.
HARMONICS = [
    ("harmonic_moment_beta_0.5", r"$\beta{=}0.5$"),
    ("harmonic_moment_beta_1", r"$\beta{=}1$"),
    ("harmonic_moment_beta_2", r"$\beta{=}2$"),
]
CROSSOVER_SIGNALS = ["0.5I", "1.5I", "0.5D", "1.5D", "0.5R", "1.5R"]


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def wilson(successes: int, trials: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if trials == 0:
        return math.nan, math.nan
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denom
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denom
    return centre - half, centre + half


def fmt_rate(rate: float, trials: int) -> str:
    """Rate with a Wilson 95% interval."""
    lo, hi = wilson(round(rate * trials), trials)
    return f"${rate:.4f}$ $[{lo:.4f},{hi:.4f}]$"


def fmt_pm(value: float, se: float) -> str:
    """Estimate with Monte Carlo standard error on a shared power of ten."""
    if value == 0.0 and se == 0.0:
        return r"$0$ (exact)"
    exp = math.floor(math.log10(max(abs(value), abs(se))))
    mantissa, mantissa_se = value / 10**exp, se / 10**exp
    return rf"$({mantissa:.2f}\pm{mantissa_se:.2f})10^{{{exp}}}$"


def fmt_row_pm(pairs: list[tuple[float, float]]) -> tuple[str, list[str]]:
    """Render several estimates on one shared power of ten.

    Comparing five estimators across a row is much easier when they carry a
    common unit, and it keeps the row narrow enough to typeset.
    """
    scale = max(max(abs(v), abs(se)) for v, se in pairs)
    exp = math.floor(math.log10(scale))
    cells = [rf"${v / 10**exp:.2f}\pm{se / 10**exp:.2f}$" for v, se in pairs]
    return rf"$10^{{{exp}}}$", cells


def tstat(value: float, se: float) -> float:
    """Return the paired t statistic, or zero for a degenerate cell."""
    return value / se if se > 0 else 0.0


def fmt_t(value: float, se: float) -> str:
    """Render a paired t statistic, or a dash when the contrast is exact."""
    return "---" if se == 0 else f"${tstat(value, se):.1f}$"


def table(
    body: str,
    header: str,
    colspec: str,
    caption: str,
    label: str,
    *,
    size: str = "small",
    colsep_pt: float | None = None,
    placement: str = "t",
) -> str:
    """Assemble a booktabs table fragment at the given font size."""
    return "\n".join(
        [
            rf"\begin{{table}}[{placement}]",
            r"\centering",
            rf"\{size}",
            *(
                [rf"\setlength{{\tabcolsep}}{{{colsep_pt}pt}}"]
                if colsep_pt is not None
                else []
            ),
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            header,
            r"\midrule",
            body,
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}",
            "",
        ]
    )


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_summary(path: Path) -> pd.DataFrame:
    """Load the frozen summary artifact in deterministic display order."""
    df = pd.read_csv(path)
    df["signal_rank"] = df["signal_label"].map(
        {label: i for i, label in enumerate(SIGNAL_ORDER)}
    )
    if df["signal_rank"].isna().any():
        unknown = sorted(set(df.loc[df["signal_rank"].isna(), "signal_label"]))
        raise SystemExit(f"unexpected signal labels in summary: {unknown}")
    return df.sort_values(["n", "signal_rank", "nominal_r"])


def paired_contrasts(path: Path, pairs: list[tuple[str, str]]) -> dict:
    """Paired mean/SE of ``a_loss - b_loss`` per cell, from the replicate export.

    Both benefits in the summary share the contaminated-Hill baseline, so the
    difference between two robust estimators is a paired contrast that the
    summary artifact does not carry.
    """
    needed = {"n", "nominal_r", "signal_label"}
    for a, b in pairs:
        needed |= {f"{a}_loss", f"{b}_loss"}
    frame = pd.read_csv(path, usecols=sorted(needed))
    out: dict[tuple[int, str, int, str, str], dict[str, float]] = {}
    for key, group in frame.groupby(["n", "signal_label", "nominal_r"], sort=False):
        n_val, label, r_val = key
        for a, b in pairs:
            d = group[f"{a}_loss"].to_numpy() - group[f"{b}_loss"].to_numpy()
            mean = float(d.mean())
            se = float(d.std(ddof=1) / math.sqrt(d.size)) if d.size > 1 else 0.0
            out[(int(n_val), str(label), int(r_val), a, b)] = {
                "mean": mean,
                "se": se,
                "t": tstat(mean, se),
            }
    return out


def cell(df: pd.DataFrame, n: int, label: str, r_val: int, estimator: str):
    """Select the single summary row for one design cell and estimator."""
    match = df[
        (df["n"] == n)
        & (df["signal_label"] == label)
        & (df["nominal_r"] == r_val)
        & (df["estimator"] == estimator)
    ]
    return match.iloc[0]


def benefit(row) -> tuple[float, float]:
    """Return the paired benefit against contaminated Hill and its MCSE."""
    return (
        float(row["benefit_vs_contaminated_hill"]),
        float(row["benefit_vs_contaminated_hill_se"]),
    )


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def clean_cell_table(df: pd.DataFrame, digest: dict) -> str:
    """Canonical clean control: r=0, S=0, Delta=1, pooled over the four draws."""
    rows, records = [], {}
    for n in sorted(df["n"].unique()):
        cells = df[
            (df["n"] == n) & (df["signal_label"] == "0") & (df["estimator"] == ADAPTIVE)
        ]
        if not (cells["effective_r"] == 0).all():
            raise SystemExit(f"S=0 cells at n={n} do not all have r=0")
        successes = round(cells["trim_recovery_rate"].sum() * TRIALS)
        pooled = TRIALS * len(cells)
        rate = successes / pooled
        lo, hi = wilson(successes, pooled)
        # On the clean control the contaminated and clean samples coincide, so
        # each benefit is exactly the estimator's efficiency cost against Hill.
        costs = []
        for est in [ADAPTIVE, *(e for e, _ in HARMONICS)]:
            draws = df[
                (df["n"] == n) & (df["signal_label"] == "0") & (df["estimator"] == est)
            ]
            mean = float(draws["benefit_vs_contaminated_hill"].mean())
            se = float(
                math.sqrt((draws["benefit_vs_contaminated_hill_se"] ** 2).sum())
                / len(draws)
            )
            costs.append((mean, se))
        unit, cost_cells = fmt_row_pm(costs)
        rows.append(
            " & ".join(
                [
                    N_TEX[n],
                    f"${pooled}$",
                    f"${rate:.4f}$ $[{lo:.4f},{hi:.4f}]$",
                    unit,
                    *cost_cells,
                ]
            )
            + r"\\"
        )
        records[str(n)] = {
            "per_draw": [round(v, 4) for v in cells["trim_recovery_rate"]],
            "pooled_trials": pooled,
            "rate": rate,
            "wilson": [lo, hi],
            "clean_cost": {
                est: {"mean": m, "se": s}
                for est, (m, s) in zip(
                    [ADAPTIVE, *(e for e, _ in HARMONICS)], costs, strict=True
                )
            },
        }
    digest["clean_cell"] = records
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & reps & $\Pp(\widehat r=0)$ $[95\%]$ & unit & adaptive & "
            + " & ".join(tex for _, tex in HARMONICS)
            + r"\\"
        ),
        colspec="cccccccc",
        caption=(
            r"Clean control.  At $S=0$ the contamination factor is $\Delta=1$, so "
            r"the true contamination count is $r=0$ and the four nominal $r$ values "
            r"of the frozen grid describe one design point drawn four times "
            r"independently.  The draws are pooled into a single canonical control.  "
            r"The reported rate is the correct no-trim rate $\Pp(\widehat r=0)$ with "
            r"a Wilson $95\%$ interval.  The remaining columns give each estimator's "
            r"paired benefit against Hill on this uncontaminated sample, in the unit "
            r"shown, which is exactly the efficiency cost it pays for its "
            r"robustness: every value is negative, and they order as the family's "
            r"aggressiveness does.  Oracle trimming is degenerate here and is "
            r"omitted, since with nothing to remove it coincides with contaminated "
            r"Hill in every replicate."
        ),
        label="tab:clean-cell",
        size="footnotesize",
        colsep_pt=4,
        # Pinned in place: floating it splits the paragraph that introduces it.
        placement="H",
    )


def detection_table(df: pd.DataFrame, digest: dict) -> str:
    """Render the detection transition, one row per (n, S, r), with uncertainty."""
    rows, records = [], []
    for n in (10000, 50000):
        if rows:
            rows.append(r"\midrule")
        for label in ["0.5D", "1.5D", "1.5I|0.5R", "0.5R", "1.5R"]:
            cells = df[
                (df["n"] == n)
                & (df["signal_label"] == label)
                & (df["estimator"] == ADAPTIVE)
            ]
            for _, row in cells.iterrows():
                b, se = benefit(row)
                rows.append(
                    " & ".join(
                        [
                            N_TEX[n],
                            SIGNAL_TEX[label],
                            f"${int(row['nominal_r'])}$",
                            f"${row['signal_over_D']:.3f}$",
                            fmt_rate(row["detection_rate"], TRIALS),
                            fmt_rate(row["trim_recovery_rate"], TRIALS),
                            fmt_pm(b, se),
                        ]
                    )
                    + r"\\"
                )
                records.append(
                    {
                        "n": int(n),
                        "signal": label,
                        "r": int(row["nominal_r"]),
                        "signal_over_D": float(row["signal_over_D"]),
                        "detection": float(row["detection_rate"]),
                        "recovery": float(row["trim_recovery_rate"]),
                        "benefit": b,
                        "benefit_se": se,
                    }
                )
    digest["detection"] = records
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & $S$ & $r$ & $S/D_n$ & detection $[95\%]$ & "
            r"recovery $[95\%]$ & $\widehat B_n^{(A)}$\\"
        ),
        colspec="ccccccc",
        caption=(
            r"Detection transition for adaptive trimmed Hill at the two sample sizes "
            r"where the intended ordering $D_n<R_n$ holds.  Detection is "
            r"$\Pp(\widehat r>0)$ and recovery is $\Pp(\widehat r=r)$, each with a "
            r"Wilson $95\%$ interval.  The benefit "
            r"$\widehat B_n^{(A)}=\operatorname{MSE}\{\ghat_H^\star\}"
            r"-\operatorname{MSE}\{\ghat_A^\star\}$ is reported with the Monte Carlo "
            r"standard error of the paired replicate losses "
            r"$d_i=(\ghat_{H,i}^\star-\gamma)^2-(\ghat_{A,i}^\star-\gamma)^2$; "
            r"positive values favour adaptive trimming.  The clean control is "
            r"reported separately in Table \ref{tab:clean-cell}.  At $n=10^4,r=10$ "
            r"the deterministic tolerance rule collapses the duplicate signal "
            r"$1.5I=0.5R$ into the single cell shown."
        ),
        label="tab:first-run-adaptive",
    )


def intervention_table(df: pd.DataFrame, digest: dict) -> str:
    """Proposition 3 finite-sample check: oracle benefit changes sign at I."""
    rows, records = [], []
    for n in sorted(df["n"].unique()):
        if rows:
            rows.append(r"\midrule")
        for label in ["0.5I", "1.5I", "1.5I|0.5R"]:
            cells = df[
                (df["n"] == n)
                & (df["signal_label"] == label)
                & (df["estimator"] == ORACLE)
            ]
            for _, row in cells.iterrows():
                b, se = benefit(row)
                rows.append(
                    " & ".join(
                        [
                            N_TEX[n],
                            SIGNAL_TEX[label],
                            f"${int(row['nominal_r'])}$",
                            f"${row['signal_over_D']:.3f}$",
                            f"${row['detection_rate']:.4f}$",
                            fmt_pm(b, se),
                            fmt_t(b, se),
                        ]
                    )
                    + r"\\"
                )
                records.append(
                    {
                        "n": int(n),
                        "signal": label,
                        "r": int(row["nominal_r"]),
                        "signal_over_I": float(row["signal_over_I"]),
                        "signal_over_D": float(row["signal_over_D"]),
                        "detection": float(row["detection_rate"]),
                        "benefit": b,
                        "benefit_se": se,
                        "t": tstat(b, se),
                    }
                )
    below = [r for r in records if r["signal_over_I"] < 1]
    above = [r for r in records if r["signal_over_I"] > 1]
    digest["intervention"] = {
        "cells": records,
        "below_I_total": len(below),
        "below_I_negative": sum(r["benefit"] < 0 for r in below),
        "below_I_negative_significant": sum(r["t"] < -2 for r in below),
        "above_I_total": len(above),
        "above_I_positive": sum(r["benefit"] > 0 for r in above),
        "above_I_positive_significant": sum(r["t"] > 2 for r in above),
        "max_signal_over_D": max(r["signal_over_D"] for r in records),
        "max_detection": max(r["detection"] for r in records),
        "max_signal_over_D_large_n": max(
            r["signal_over_D"] for r in records if r["n"] >= 10000
        ),
        "max_detection_large_n": max(
            r["detection"] for r in records if r["n"] >= 10000
        ),
    }
    return table(
        body="\n".join(rows),
        header=r"$n$ & $S$ & $r$ & $S/D_n$ & detection & $\widehat B_n^{(T)}$ & $t$\\",
        colspec="ccccccc",
        caption=(
            r"Finite-sample check of Proposition \ref{prop:oracle-intervention}.  The "
            r"oracle benefit $\widehat B_n^{(T)}=\operatorname{MSE}\{\ghat_H^\star\}"
            r"-\operatorname{MSE}\{\ghat_T(r,k)\}$ is reported with the Monte Carlo "
            r"standard error of the paired replicate losses, and $t$ is the "
            r"corresponding paired statistic.  The proposition predicts "
            r"$\widehat B_n^{(T)}<0$ below $I^{T}_{r,k}$ and $\widehat B_n^{(T)}>0$ "
            r"above it.  Every $I$ cell of the frozen grid lies below the detection "
            r"scale, so adaptive detection is nearly inactive throughout this table "
            r"and the intervention boundary is visible only to the oracle."
        ),
        label="tab:intervention",
        # Pinned: floating it splits the paragraph that introduces it.
        placement="H",
    )


def estimator_table(df: pd.DataFrame, digest: dict, r_show: int = 5) -> str:
    """Detection-and-removal against the whole bounded-influence family."""
    rows, records = [], []
    for n in (10000, 50000):
        if rows:
            rows.append(r"\midrule")
        for label in CROSSOVER_SIGNALS:
            sub = df[(df["n"] == n) & (df["signal_label"] == label)]
            if sub[sub["nominal_r"] == r_show].empty:
                continue
            adaptive = cell(df, n, label, r_show, ADAPTIVE)
            oracle = cell(df, n, label, r_show, ORACLE)
            harmonics = [cell(df, n, label, r_show, est) for est, _ in HARMONICS]
            unit, cells = fmt_row_pm(
                [benefit(adaptive), *(benefit(h) for h in harmonics), benefit(oracle)]
            )
            rows.append(
                " & ".join(
                    [
                        N_TEX[n],
                        SIGNAL_TEX[label],
                        f"${adaptive['signal_over_D']:.3f}$",
                        unit,
                        *cells,
                    ]
                )
                + r"\\"
            )
            best = max(harmonics, key=lambda h: benefit(h)[0])
            records.append(
                {
                    "n": int(n),
                    "signal": label,
                    "r": r_show,
                    "signal_over_D": float(adaptive["signal_over_D"]),
                    "adaptive": benefit(adaptive)[0],
                    **{
                        est: benefit(h)[0]
                        for (est, _), h in zip(HARMONICS, harmonics, strict=True)
                    },
                    "oracle": benefit(oracle)[0],
                    "best_harmonic": str(best["estimator"]),
                    "adaptive_beats_all_harmonics": bool(
                        benefit(adaptive)[0] > benefit(best)[0]
                    ),
                }
            )
    digest["estimators"] = records
    above = [r for r in records if r["signal_over_D"] > 1]
    digest["estimator_summary"] = {
        "above_D_total": len(above),
        "above_D_adaptive_beats_all_harmonics": sum(
            r["adaptive_beats_all_harmonics"] for r in above
        ),
        "above_D_exceptions": [
            {
                "n": r["n"],
                "signal": r["signal"],
                "signal_over_D": r["signal_over_D"],
                "best_harmonic": r["best_harmonic"],
            }
            for r in above
            if not r["adaptive_beats_all_harmonics"]
        ],
    }
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & $S$ & $S/D_n$ & unit & adaptive & "
            + " & ".join(tex for _, tex in HARMONICS)
            + r" & oracle\\"
        ),
        colspec="ccccccccc",
        caption=(
            r"Detection-and-removal against the full pre-specified bounded-influence "
            r"family at $r=5$.  Entries are paired benefits against contaminated Hill "
            r"with Monte Carlo standard errors, each row written in the unit given in "
            r"its fourth column; positive values beat contaminated "
            r"Hill.  The harmonic-moment columns are the three frozen values of "
            r"$\beta$, with $\beta=1$ the named t-Hill estimator.  Increasing $\beta$ "
            r"bounds influence more aggressively but carries a larger clean-sample "
            r"efficiency cost: "
            r"$\beta=2$ is the weakest column almost everywhere, and no single "
            r"$\beta$ is best across the grid.  Paired tests of each $\beta$ against "
            r"adaptive trimming are in Table \ref{tab:beta-contrasts}."
        ),
        label="tab:estimator-crossover",
        size="footnotesize",
        colsep_pt=3,
        # Pinned: floating it left an orphaned word after the table.
        placement="H",
    )


def beta_contrast_table(
    df: pd.DataFrame, contrasts: dict | None, digest: dict, r_show: int = 5
) -> str:
    """Paired tests of each frozen beta against adaptive trimming."""
    rows, records = [], []
    for n in (10000, 50000):
        if rows:
            rows.append(r"\midrule")
        for label in CROSSOVER_SIGNALS:
            sub = df[(df["n"] == n) & (df["signal_label"] == label)]
            if sub[sub["nominal_r"] == r_show].empty:
                continue
            adaptive = cell(df, n, label, r_show, ADAPTIVE)
            # d_i = adaptive loss - harmonic loss, so t > 0 favours bounded
            # influence and the sign agrees with the benefit columns.
            found = [
                contrasts.get((n, label, r_show, ADAPTIVE, est)) if contrasts else None
                for est, _ in HARMONICS
            ]
            rows.append(
                " & ".join(
                    [
                        N_TEX[n],
                        SIGNAL_TEX[label],
                        f"${adaptive['signal_over_D']:.3f}$",
                        *(f"${c['t']:.1f}$" if c else "---" for c in found),
                    ]
                )
                + r"\\"
            )
            records.append(
                {
                    "n": int(n),
                    "signal": label,
                    "signal_over_D": float(adaptive["signal_over_D"]),
                    **{
                        est: (c["t"] if c else None)
                        for (est, _), c in zip(HARMONICS, found, strict=True)
                    },
                }
            )
    digest["beta_contrasts"] = records
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & $S$ & $S/D_n$ & " + " & ".join(tex for _, tex in HARMONICS) + r"\\"
        ),
        colspec="cccccc",
        caption=(
            r"Paired comparison of each frozen bounded-influence member against "
            r"adaptive trimming at $r=5$.  Entries are paired $t$ statistics for the "
            r"replicate loss differences "
            r"$d_i=(\ghat_{A,i}^\star-\gamma)^2-(\ghat_{\beta,i}^\star-\gamma)^2$, so "
            r"$t>0$ favours bounded influence and $t<0$ favours adaptive trimming.  "
            r"The advantages claimed for bounded influence in the undetectable "
            r"region exceed two Monte Carlo standard errors at some cells and not at "
            r"others, which is why "
            r"Section \ref{sec:first-run} claims only that bounded influence "
            r"\emph{can} pay there rather than that it dominates."
        ),
        label="tab:beta-contrasts",
        size="small",
    )


def beta_r_sweep_table(df: pd.DataFrame, digest: dict) -> str:
    """How each frozen beta fares against adaptive trimming across every r.

    Tables \\ref{tab:estimator-crossover} and \\ref{tab:beta-contrasts} display
    one contamination count, so claims about the whole family need this sweep
    to be checkable.
    """
    rows, records = [], []
    for n in sorted(df["n"].unique()):
        if rows:
            rows.append(r"\midrule")
        for label in SIGNAL_ORDER:
            if label == "0":
                continue
            sub = df[(df["n"] == n) & (df["signal_label"] == label)]
            if sub.empty:
                continue
            r_values = sorted(sub["nominal_r"].unique())
            counts = []
            for est, _ in HARMONICS:
                wins = sum(
                    benefit(cell(df, n, label, r_val, est))[0]
                    > benefit(cell(df, n, label, r_val, ADAPTIVE))[0]
                    for r_val in r_values
                )
                counts.append(wins)
            rows.append(
                " & ".join(
                    [
                        N_TEX[n],
                        SIGNAL_TEX[label],
                        f"${len(r_values)}$",
                        *(f"${c}$" for c in counts),
                    ]
                )
                + r"\\"
            )
            records.append(
                {
                    "n": int(n),
                    "signal": label,
                    "r_cells": len(r_values),
                    **{est: c for (est, _), c in zip(HARMONICS, counts, strict=True)},
                }
            )
    digest["beta_r_sweep"] = records
    large = [r for r in records if r["n"] >= 10000]
    digest["beta_r_sweep_totals"] = {
        "large_n_cells": sum(r["r_cells"] for r in large),
        "all_cells": sum(r["r_cells"] for r in records),
        "large_n_wins": {est: sum(r[est] for r in large) for est, _ in HARMONICS},
        "all_wins": {est: sum(r[est] for r in records) for est, _ in HARMONICS},
    }
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & $S$ & $r$ cells & "
            + " & ".join(tex for _, tex in HARMONICS)
            + r"\\"
        ),
        colspec="cccccc",
        caption=(
            r"Bounded influence against adaptive trimming across the whole frozen "
            r"grid.  For each design point, the last three columns count the "
            r"contamination counts $r\in\{1,3,5,10\}$ at which that "
            r"harmonic-moment member has the larger paired benefit against "
            r"contaminated Hill.  This is the evidence behind the family-level "
            r"statements of Section \ref{sec:first-run}, which Tables "
            r"\ref{tab:estimator-crossover} and \ref{tab:beta-contrasts} report at "
            r"$r=5$ only.  At $n\ge10^4$ the most aggressively bounded member, "
            r"$\beta=2$, never beats adaptive trimming; the four cells where it "
            r"does are all in the pre-asymptotic $n=10^3$ row."
        ),
        label="tab:beta-r-sweep",
    )


def stress_table(df: pd.DataFrame, digest: dict) -> str:
    """Pre-asymptotic stress test at n=1000, where D_n > R_n."""
    rows, records = [], []
    for label in ["0.5I", "1.5I", "0.5D", "1.5D", "0.5R", "1.5R"]:
        cells = df[
            (df["n"] == 1000)
            & (df["signal_label"] == label)
            & (df["estimator"] == ADAPTIVE)
        ]
        for _, row in cells.iterrows():
            r_val = int(row["nominal_r"])
            oracle = cell(df, 1000, label, r_val, ORACLE)
            rows.append(
                " & ".join(
                    [
                        SIGNAL_TEX[label],
                        f"${r_val}$",
                        f"${row['signal_over_I']:.3f}$",
                        f"${row['signal_over_D']:.3f}$",
                        f"${row['signal_over_R']:.3f}$",
                        fmt_rate(row["detection_rate"], TRIALS),
                        fmt_pm(*benefit(row)),
                        fmt_pm(*benefit(oracle)),
                    ]
                )
                + r"\\"
            )
            b_oracle, se_oracle = benefit(oracle)
            records.append(
                {
                    "signal": label,
                    "r": r_val,
                    "signal_over_I": float(row["signal_over_I"]),
                    "signal_over_D": float(row["signal_over_D"]),
                    "signal_over_R": float(row["signal_over_R"]),
                    "detection": float(row["detection_rate"]),
                    "adaptive": benefit(row)[0],
                    "oracle": b_oracle,
                    "oracle_se": se_oracle,
                }
            )
    digest["stress_n1000"] = {
        "cells": records,
        "oracle_significantly_negative": [
            {
                "signal": c["signal"],
                "r": c["r"],
                "signal_over_I": c["signal_over_I"],
                "oracle": c["oracle"],
            }
            for c in records
            if c["oracle"] + 2 * c["oracle_se"] < 0
        ],
        "cells_below_I_at_D_or_R": [
            {"signal": c["signal"], "r": c["r"], "signal_over_I": c["signal_over_I"]}
            for c in records
            if c["signal"] in {"0.5D", "1.5D", "0.5R", "1.5R"}
            and c["signal_over_I"] < 1
        ],
    }
    return table(
        body="\n".join(rows),
        header=(
            r"$S$ & $r$ & $S/I^{T}_{r,k}$ & $S/D_n$ & $S/R_n$ & detection $[95\%]$ & "
            r"$\widehat B_n^{(A)}$ & $\widehat B_n^{(T)}$\\"
        ),
        colspec="cccccccc",
        caption=(
            r"Pre-asymptotic stress test at $n=10^3$, the pre-specified row where "
            r"$D_n>R_n$ and the intended ordering of the scales does not hold.  "
            r"Detection is $\Pp(\widehat r>0)$ with a Wilson $95\%$ interval; "
            r"$\widehat B_n^{(A)}$ and $\widehat B_n^{(T)}$ are the paired adaptive "
            r"and oracle benefits against contaminated Hill.  The $S/I^{T}_{r,k}$ column "
            r"shows why this row is a stress test: with $k_n=31$ and $r=10$ the "
            r"intervention scale exceeds the detection and risk grid points "
            r"($S/I^{T}_{r,k}=0.80$ at $0.5D$ and $0.72$ at $0.5R$), so the frozen "
            r"signal grid no longer brackets the three scales in the intended order. "
            r"The clearly negative oracle benefits in those cells are what "
            r"Proposition \ref{prop:oracle-intervention} predicts below $I^{T}_{r,k}$, "
            r"not a failure of the oracle."
        ),
        label="tab:stress-n1000",
    )


def identity_checks(df: pd.DataFrame, digest: dict) -> None:
    """Check the two exact identities the figure plots, and the scale ordering.

    Proposition 2 predicts the contaminated Hill MSE inflation in units of the
    clean Hill MSE, and Proposition 3 rearranges to a single curve in the
    normalized signal.  Both are exact, so the run can be checked against them
    rather than only against a predicted sign.
    """
    contaminated = df[df["signal"] > 0]
    clean_mse = df["gamma"] ** 2 / df["k"]

    hill = contaminated[contaminated["estimator"] == "hill"]
    risk_ratio = (hill["contamination_cost"] / clean_mse[hill.index]) / (
        hill["signal_over_R"] ** 2
    )

    oracle = contaminated[contaminated["estimator"] == "oracle_trimmed_hill"]
    r_eff = oracle["effective_r"]
    normalized = (
        oracle["benefit_vs_contaminated_hill"]
        * oracle["k"]
        * (oracle["k"] - r_eff)
        / (oracle["gamma"] ** 2 * r_eff)
    )
    oracle_residual = normalized - (oracle["signal_over_I"] ** 2 - 1)

    scales = df.drop_duplicates("n").set_index("n")
    digest["identities"] = {
        "risk_observed_over_predicted_median": {
            str(n): round(float(v), 4)
            for n, v in risk_ratio.groupby(hill["n"]).median().items()
        },
        "oracle_normalized_residual_median": {
            str(n): round(float(v), 4)
            for n, v in oracle_residual.groupby(oracle["n"]).median().items()
        },
        "D_over_R": {
            str(n): round(float(scales.loc[n, "D"] / scales.loc[n, "R"]), 4)
            for n in scales.index
        },
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    """Render every reporting fragment from the frozen artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "primary_summary.csv")
    parser.add_argument(
        "--replicates", type=Path, default=HERE / "primary_replicates.csv"
    )
    parser.add_argument("--outdir", type=Path, default=HERE / "paper" / "generated")
    parser.add_argument(
        "--digest-dir",
        type=Path,
        default=HERE,
        help="where to write the JSON digest of the numbers quoted in the prose",
    )
    parser.add_argument(
        "--skip-replicates",
        action="store_true",
        help="skip the paired estimator contrasts, which need the large export",
    )
    args = parser.parse_args()

    df = load_summary(args.summary)
    contrasts = None
    if not args.skip_replicates and args.replicates.exists():
        contrasts = paired_contrasts(
            args.replicates, [(ADAPTIVE, est) for est, _ in HARMONICS]
        )

    digest: dict[str, Any] = {
        "source": {
            "summary": args.summary.name,
            "replicates": args.replicates.name if contrasts else None,
            "trials_per_cell": TRIALS,
        }
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    fragments = {
        "tab_clean_cell.tex": clean_cell_table(df, digest),
        "tab_detection.tex": detection_table(df, digest),
        "tab_intervention.tex": intervention_table(df, digest),
        "tab_estimator_crossover.tex": estimator_table(df, digest),
        "tab_beta_contrasts.tex": beta_contrast_table(df, contrasts, digest),
        "tab_beta_r_sweep.tex": beta_r_sweep_table(df, digest),
        "tab_stress_n1000.tex": stress_table(df, digest),
    }
    identity_checks(df, digest)
    for name, text in fragments.items():
        (args.outdir / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.outdir / name}")
    args.digest_dir.mkdir(parents=True, exist_ok=True)
    digest_path = args.digest_dir / "frozen_run_analysis.json"
    digest_path.write_text(json.dumps(digest, indent=2), encoding="utf-8")
    print(f"wrote {digest_path}")

    iv = digest["intervention"]
    print(
        f"\nProposition 3 check: "
        f"{iv['below_I_negative']}/{iv['below_I_total']} cells negative below I "
        f"({iv['below_I_negative_significant']} at |t|>2); "
        f"{iv['above_I_positive']}/{iv['above_I_total']} positive above I "
        f"({iv['above_I_positive_significant']} at |t|>2)."
    )
    print(f"max S/D over all I cells: {iv['max_signal_over_D']:.3f}")
    print(f"max detection over all I cells: {iv['max_detection']:.3f}")
    ident = digest["identities"]
    print(
        f"risk identity, observed/predicted median: {ident['risk_observed_over_predicted_median']}"
    )
    print(
        f"oracle identity, residual median: {ident['oracle_normalized_residual_median']}"
    )
    print(f"D_n/R_n: {ident['D_over_R']}")


if __name__ == "__main__":
    main()
