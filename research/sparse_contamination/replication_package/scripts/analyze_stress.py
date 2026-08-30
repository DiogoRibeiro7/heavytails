"""Render the post-specified stress-test tables from ``stress_summary.csv``.

Like ``analyze_frozen_run.py`` this script only reads artifacts; it never
simulates.  It writes LaTeX fragments into ``paper/generated`` and a JSON
digest of the numbers quoted in the manuscript prose.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analyze_frozen_run import N_TEX, fmt_row_pm, table
import pandas as pd

HERE = Path(__file__).resolve().parent


def fmt_sci(value: float) -> str:
    """Render a positive quantity as a LaTeX power of ten."""
    exponent = math.floor(math.log10(value))
    return rf"${value / 10**exponent:.2f}{{\times}}10^{{{exponent}}}$"


def detection_table(df: pd.DataFrame, digest: dict) -> str:
    """Detection against S/D_n on a fine grid, under both level regimes."""
    d = df[df["study"] == "detection_transition"].copy()
    d["ratio"] = d["signal_over_D"].round(2)
    rows, records = [], []
    for ratio in sorted(d["ratio"].unique()):
        cells = []
        for n in (10_000, 50_000):
            for regime in ("vanishing", "fixed"):
                sub = d[
                    (d["n"] == n)
                    & (d["level_regime"] == regime)
                    & (d["ratio"] == ratio)
                    & (d["r"] == 5)
                ]
                cells.append(
                    f"${sub.iloc[0]['detection_rate']:.3f}$" if len(sub) else "---"
                )
        rows.append(f"${ratio:.1f}$ & " + " & ".join(cells) + r"\\")
        records.append({"signal_over_D": float(ratio), "detection": cells})
    digest["detection_transition"] = records
    return table(
        body="\n".join(rows),
        header=(
            r"$S/D_n$ & \multicolumn{2}{c}{$n=10^4$} & "
            r"\multicolumn{2}{c}{$n=5{\times}10^4$}\\"
            "\n"
            r" & vanishing & fixed & vanishing & fixed\\"
        ),
        colspec="ccccc",
        caption=(
            r"Post-specified detection transition at $r=5$.  Entries are "
            r"$\Pp(\widehat r>0)$ on a grid in $S/D_n$, under the vanishing "
            r"family-wise level $\alpha_n=\min\{0.05,1/\log^2n\}$ of the primary "
            r"design and under a fixed $\alpha=0.05$.  Each column normalizes by "
            r"its own $D_n$.  The transition is centred slightly below "
            r"$S/D_n=1$ and is complete by $S/D_n\approx1.2$ in both regimes, so "
            r"the normalized finite-sample transition is similar under fixed and "
            r"vanishing levels.  The vanishing level additionally supplies the "
            r"asymptotic $I^{T}_{r,k}\ll D_n$ separation and the vanishing "
            r"false-rejection probability that Theorem "
            r"\ref{thm:exact-recovery} assumes."
        ),
        label="tab:stress-detection",
        # Pinned: floating it splits the paragraph that introduces it.
        placement="H",
    )


def profile_table(df: pd.DataFrame, digest: dict) -> str:
    """Equal-factor against spread contamination at matched Hill bias."""
    d = df[df["study"] == "unequal_factors"].copy()
    d["ratio"] = d["signal_over_D"].round(2)
    rows, records = [], []
    for n in (10_000, 50_000):
        if rows:
            rows.append(r"\midrule")
        for r_val in (3, 5, 10):
            for ratio in sorted(d["ratio"].unique()):
                sub = d[(d["n"] == n) & (d["r"] == r_val) & (d["ratio"] == ratio)]
                eq = sub[sub["profile"] == "equal"]
                sp = sub[sub["profile"] == "spread"]
                if eq.empty or sp.empty:
                    continue
                eq, sp = eq.iloc[0], sp.iloc[0]
                unit, cells = fmt_row_pm(
                    [
                        (
                            eq["adaptive_trimmed_hill_benefit"],
                            eq["adaptive_trimmed_hill_benefit_se"],
                        ),
                        (
                            sp["adaptive_trimmed_hill_benefit"],
                            sp["adaptive_trimmed_hill_benefit_se"],
                        ),
                    ]
                )
                rows.append(
                    " & ".join(
                        [
                            N_TEX[n],
                            f"${r_val}$",
                            f"${ratio:.1f}$",
                            f"${eq['detection_rate']:.3f}$",
                            f"${sp['detection_rate']:.3f}$",
                            unit,
                            *cells,
                        ]
                    )
                    + r"\\"
                )
                records.append(
                    {
                        "n": int(n),
                        "r": int(r_val),
                        "signal_over_D": float(ratio),
                        "detection_equal": float(eq["detection_rate"]),
                        "detection_spread": float(sp["detection_rate"]),
                        "cost_equal": float(eq["contamination_cost"]),
                        "cost_spread": float(sp["contamination_cost"]),
                        "benefit_equal": float(eq["adaptive_trimmed_hill_benefit"]),
                        "benefit_spread": float(sp["adaptive_trimmed_hill_benefit"]),
                    }
                )
    ratios = [c for c in records if c["signal_over_D"] >= 1.5]
    digest["profiles"] = records
    digest["profile_summary"] = {
        "cells_at_or_above_1.5D": len(ratios),
        "max_cost_discrepancy": max(
            abs(c["cost_equal"] - c["cost_spread"]) / c["cost_equal"] for c in ratios
        ),
        "min_spread_detection_at_1.5D": min(
            c["detection_spread"]
            for c in records
            if abs(c["signal_over_D"] - 1.5) < 0.01
        ),
    }
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & $r$ & $S/D_n$ & \multicolumn{2}{c}{detection} & unit & "
            r"\multicolumn{2}{c}{$\widehat B_n^{(A)}$}\\"
            "\n"
            r" & & & equal & spread & & equal & spread\\"
        ),
        colspec="cccccccc",
        caption=(
            r"Post-specified comparison of contamination profiles at matched Hill "
            r"bias.  The equal-factor profile places the whole signal on spacing "
            r"$r$; the spread profile uses unequal factors chosen so that, by "
            r"Lemma \ref{lem:unequal}, the shifts $jc$ fall on all of "
            r"$j=1,\ldots,r$ and the total, hence the contaminated-Hill MSE, is "
            r"unchanged.  The two profiles damage Hill identically but are not "
            r"equally detectable: at $S=1.5D_n$ the equal-factor signal is "
            r"detected in every replicate while the spread signal is detected in "
            r"a few percent of them, and the adaptive benefit falls with it.  "
            r"$D_n$ is therefore a detection scale for spacing-localized "
            r"contamination, not for contamination of a given magnitude."
        ),
        label="tab:stress-profile",
        size="footnotesize",
        colsep_pt=4,
        # Pinned: floating it split the central localization paragraph.
        placement="H",
    )


def second_order_table(df: pd.DataFrame, digest: dict) -> str:
    """Burr tails, where the spacings are only approximately exponential."""
    d = df[df["study"] == "second_order"].copy()
    d["ratio"] = d["signal_over_D"].round(2)
    rows, records = [], []
    for n in (10_000, 50_000):
        if rows:
            rows.append(r"\midrule")
        for tau in (2.0, 1.0, 0.5):
            for ratio in (0.0, 0.5, 1.5):
                sub = d[
                    (d["n"] == n)
                    & (d["tau"] == tau)
                    & (d["ratio"] == ratio)
                    & (d["r"] == 5)
                ]
                if sub.empty:
                    continue
                row = sub.iloc[0]
                unit, cells = fmt_row_pm(
                    [
                        (
                            row["adaptive_trimmed_hill_benefit"],
                            row["adaptive_trimmed_hill_benefit_se"],
                        ),
                        (
                            row["oracle_trimmed_hill_benefit"],
                            row["oracle_trimmed_hill_benefit_se"],
                        ),
                    ]
                )
                rows.append(
                    " & ".join(
                        [
                            N_TEX[n],
                            f"${-tau * 0.5:.2f}$",
                            f"${ratio:.1f}$",
                            fmt_sci(row["hill_mse"]),
                            f"${row['detection_rate']:.3f}$",
                            unit,
                            *cells,
                        ]
                    )
                    + r"\\"
                )
                records.append(
                    {
                        "n": int(n),
                        "rho": -tau * 0.5,
                        "signal_over_D": float(ratio),
                        "hill_mse": float(row["hill_mse"]),
                        "detection": float(row["detection_rate"]),
                        "benefit_adaptive": float(row["adaptive_trimmed_hill_benefit"]),
                        "benefit_oracle": float(row["oracle_trimmed_hill_benefit"]),
                        "contamination_cost": float(row["contamination_cost"]),
                    }
                )
    digest["second_order"] = records
    return table(
        body="\n".join(rows),
        header=(
            r"$n$ & $\rho$ & $S/D_n$ & $\operatorname{MSE}(\ghat_H^\star)$ & "
            r"detection & unit & adaptive & oracle\\"
        ),
        colspec="cccccccc",
        caption=(
            r"Post-specified second-order stress test at $r=5$, sampling a Burr "
            r"law with second-order parameter $\rho$ rather than exact Pareto, so "
            r"the normalized spacings are only approximately exponential and "
            r"ordinary Hill carries threshold bias.  The detection transition "
            r"survives: detection stays near its clean level at $0.5D_n$ and is "
            r"essentially complete at $1.5D_n$ for every $\rho$, and above $D_n$ "
            r"adaptive trimming approaches the trim-count oracle as recovery becomes "
            r"nearly complete.  What changes "
            r"is the relative importance of the contamination: at $\rho=-0.25$ "
            r"the bias term dominates $\operatorname{MSE}(\ghat_H^\star)$, so the "
            r"contamination question is secondary to threshold choice."
        ),
        label="tab:stress-second-order",
        size="footnotesize",
        colsep_pt=4,
        # Pinned: floating it split the Burr model specification.
        placement="H",
    )


def main() -> None:
    """Render the stress-test fragments and digest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "stress_summary.csv")
    parser.add_argument("--outdir", type=Path, default=HERE / "paper" / "generated")
    parser.add_argument(
        "--digest-dir",
        type=Path,
        default=HERE,
        help="where to write the JSON digest of the stress-test numbers",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary)
    digest: dict[str, Any] = {"source": args.summary.name, "post_specified": True}
    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, text in {
        "tab_stress_detection.tex": detection_table(df, digest),
        "tab_stress_profile.tex": profile_table(df, digest),
        "tab_stress_second_order.tex": second_order_table(df, digest),
    }.items():
        (args.outdir / name).write_text(text, encoding="utf-8")
        print(f"wrote {args.outdir / name}")
    args.digest_dir.mkdir(parents=True, exist_ok=True)
    (args.digest_dir / "stress_analysis.json").write_text(
        json.dumps(digest, indent=2), encoding="utf-8"
    )
    ps = digest["profile_summary"]
    print(
        f"\nprofile study: matched-cost cells at >=1.5D differ in Hill cost by at "
        f"most {ps['max_cost_discrepancy']:.1%}; "
        f"minimum spread detection at 1.5D = {ps['min_spread_detection_at_1.5D']:.3f}"
    )


if __name__ == "__main__":
    main()
