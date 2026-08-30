"""Paired beta-versus-adaptive contrasts at every contamination count.

Tables 4 and 5 of the manuscript display one contamination count, and Table 7
gives only the sign pattern across the rest.  This module renders the
magnitudes for every cell, so the family-level statements can be checked one by
one.  It reads the frozen artifacts only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_frozen_run import (
    ADAPTIVE,
    CROSSOVER_SIGNALS,
    HARMONICS,
    N_TEX,
    SIGNAL_TEX,
    cell,
    load_summary,
    paired_contrasts,
    table,
)

HERE = Path(__file__).resolve().parent


def build(df, contrasts: dict) -> tuple[str, list[dict]]:
    """Render the full contrast table and its digest records."""
    rows: list[str] = []
    records: list[dict] = []
    for n in (10000, 50000):
        if rows:
            rows.append(r"\midrule")
        for label in CROSSOVER_SIGNALS:
            sub = df[(df["n"] == n) & (df["signal_label"] == label)]
            if sub.empty:
                continue
            for r_val in sorted(sub["nominal_r"].unique()):
                found = [
                    contrasts.get((n, label, int(r_val), ADAPTIVE, est))
                    for est, _ in HARMONICS
                ]
                adaptive = cell(df, n, label, int(r_val), ADAPTIVE)
                cells = [f"${c['t']:.1f}$" if c else "---" for c in found]
                rows.append(
                    " & ".join(
                        [
                            N_TEX[n],
                            SIGNAL_TEX[label],
                            f"${int(r_val)}$",
                            f"${adaptive['signal_over_D']:.3f}$",
                            *cells,
                        ]
                    )
                    + r"\\"
                )
                records.append(
                    {
                        "n": int(n),
                        "signal": label,
                        "r": int(r_val),
                        **{
                            est: (c["t"] if c else None)
                            for (est, _), c in zip(HARMONICS, found, strict=True)
                        },
                    }
                )
    caption = (
        r"Paired contrasts of every frozen bounded-influence member against "
        r"adaptive trimming, at every contamination count of the grid.  Entries "
        r"are paired $t$ statistics for the replicate loss differences "
        r"$d_i=(\ghat_{A,i}^\star-\gamma)^2-(\ghat_{\beta,i}^\star-\gamma)^2$, "
        r"so $t>0$ favours bounded influence.  Table \ref{tab:beta-contrasts} "
        r"is the $r=5$ block of this table and Table \ref{tab:beta-r-sweep} its "
        r"sign pattern; the magnitudes are given here so that the family-level "
        r"statements of Section \ref{sec:first-run} can be checked cell by cell."
    )
    header = (
        r"$n$ & $S$ & $r$ & $S/D_n$ & "
        + " & ".join(tex for _, tex in HARMONICS)
        + r"\\"
    )
    return (
        table(
            body="\n".join(rows),
            header=header,
            colspec="ccccccc",
            caption=caption,
            label="tab:beta-full-contrasts",
            size="footnotesize",
        ),
        records,
    )


def main() -> None:
    """Render the full contrast fragment and its digest."""
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
        help="where to write the JSON digest of the per-cell contrasts",
    )
    args = parser.parse_args()

    df = load_summary(args.summary)
    contrasts = paired_contrasts(
        args.replicates, [(ADAPTIVE, est) for est, _ in HARMONICS]
    )
    fragment, records = build(df, contrasts)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "tab_beta_full_contrasts.tex").write_text(fragment, encoding="utf-8")
    args.digest_dir.mkdir(parents=True, exist_ok=True)
    (args.digest_dir / "beta_grid_analysis.json").write_text(
        json.dumps({"cells": records}, indent=2), encoding="utf-8"
    )
    print(f"wrote {args.outdir / 'tab_beta_full_contrasts.tex'} ({len(records)} cells)")


if __name__ == "__main__":
    main()
