"""Render the three-scale results figure from the frozen exact-Pareto run.

The frozen simulation plan asks for the results to be shown against each of the
three normalized scales,
    S / I^T_{r,k},   S / D_n,   S / R_n,
so this figure devotes one panel to each.  Every panel puts its own boundary at
x = 1, and two of the three carry an exact prediction rather than only a
qualitative transition:

* Intervention.  Writing B_T for the oracle-trimming benefit against
  contaminated Hill, Proposition 3 gives
      B_T * k (k - r) / (gamma^2 r) = (S / I^T_{r,k})^2 - 1,
  so the curve crosses zero exactly at the intervention scale.
* Detection.  The scan's recovery probability turns on near S = D_n; this
  transition has no closed form and is shown as measured.
* Risk.  Proposition 2 gives MSE inflation of ordinary Hill, in units of the
  clean Hill MSE gamma^2 / k, equal to (S / R_n)^2.

Like ``analyze_frozen_run.py`` this script only reads the frozen artifacts.

Usage::

    python make_scale_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Categorical slots 1-3 of the validated reference palette.  These three clear
# the all-pairs CVD and normal-vision floors, which the scatter form needs.
# Each series also carries a distinct marker so identity never rests on colour
# alone in print or in grayscale.
SERIES = [
    (1000, "#2a78d6", "o", r"$n=10^3$"),
    (10000, "#eb6834", "s", r"$n=10^4$"),
    (50000, "#1baf7a", "^", r"$n=5\times10^4$"),
]

INK = "#0b0b0b"
INK_MUTED = "#52514e"
REFERENCE = "#8a8984"


def load(path: Path) -> pd.DataFrame:
    """Load the frozen summary and attach the three normalized panel series."""
    df = pd.read_csv(path)
    clean_mse = df["gamma"] ** 2 / df["k"]

    hill = df[df["estimator"] == "hill"].copy()
    hill["risk_y"] = hill["contamination_cost"] / clean_mse
    hill["risk_se"] = hill["contamination_cost_se"] / clean_mse

    oracle = df[df["estimator"] == "oracle_trimmed_hill"].copy()
    scale = (
        oracle["k"]
        * (oracle["k"] - oracle["effective_r"])
        / (
            oracle["gamma"] ** 2
            * oracle["effective_r"].where(oracle["effective_r"] > 0)
        )
    )
    oracle["interv_y"] = oracle["benefit_vs_contaminated_hill"] * scale
    oracle["interv_se"] = oracle["benefit_vs_contaminated_hill_se"] * scale

    adaptive = df[df["estimator"] == "adaptive_trimmed_hill"].copy()

    return hill, oracle, adaptive


def wilson_halfwidths(rates: np.ndarray, trials: int) -> np.ndarray:
    """Asymmetric Wilson 95% half-widths, shaped for ``pyplot.errorbar``."""
    z = 1.959963984540054
    denom = 1.0 + z * z / trials
    centre = (rates + z * z / (2 * trials)) / denom
    half = z * np.sqrt(rates * (1 - rates) / trials + z * z / (4 * trials**2)) / denom
    lo, hi = centre - half, centre + half
    return np.vstack([np.maximum(rates - lo, 0), np.maximum(hi - rates, 0)])


def scatter_panel(ax, frame, ycol, secol, *, trials: int | None = None) -> None:
    """Draw one series per sample size, with Monte Carlo uncertainty."""
    for n, colour, marker, label in SERIES:
        sub = frame[(frame["n"] == n) & (frame["signal"] > 0)]
        if sub.empty:
            continue
        yerr = (
            wilson_halfwidths(sub[ycol].to_numpy(), trials)
            if trials is not None
            else sub[secol].to_numpy()
        )
        ax.errorbar(
            sub["x"],
            sub[ycol],
            yerr=yerr,
            fmt=marker,
            ms=4.5,
            mfc=colour,
            mec="white",
            mew=0.5,
            ecolor=colour,
            elinewidth=0.8,
            capsize=0,
            alpha=0.9,
            linestyle="none",
            label=label,
            zorder=3,
        )


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Apply the recessive grid, boundary marker and axis labels."""
    ax.axvline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.grid(True, which="major", color="#e4e3df", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=3, width=0.7)
    ax.set_title(title, fontsize=9, color=INK, pad=7)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK)


def main() -> None:
    """Render the figure to PDF for the manuscript and PNG for inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=HERE / "primary_summary.csv")
    parser.add_argument("--outdir", type=Path, default=HERE / "paper" / "generated")
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=HERE,
        help="where to write the PNG preview used for visual inspection",
    )
    args = parser.parse_args()

    hill, oracle, adaptive = load(args.summary)
    trials = int(hill["trials"].iloc[0])

    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))

    # Panel 1: intervention scale, against the exact prediction (S/I)^2 - 1.
    ax = axes[0]
    oracle = oracle.assign(x=oracle["signal_over_I"])
    grid = np.logspace(np.log10(0.45), np.log10(25), 200)
    ax.plot(grid, grid**2 - 1, color=REFERENCE, lw=1.3, zorder=2)
    ax.axhline(0.0, color=INK_MUTED, lw=0.8, zorder=1)
    scatter_panel(ax, oracle, "interv_y", "interv_se")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlim(0.4, 28)
    style(
        ax,
        r"Trimmed-Hill intervention scale $I^{T}_{r,k}$",
        r"$S/I^{T}_{r,k}$",
        r"normalized oracle benefit",
    )
    ax.annotate(
        r"$(S/I^{T}_{r,k})^2-1$",
        xy=(1.6, 120),
        fontsize=7.5,
        color=INK_MUTED,
        ha="left",
    )

    # Panel 2: detection scale, measured transition.
    ax = axes[1]
    adaptive = adaptive.assign(x=adaptive["signal_over_D"])
    scatter_panel(ax, adaptive, "detection_rate", None, trials=trials)
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.06)
    ax.set_xlim(0.055, 4.2)
    style(
        ax,
        r"Detection scale $D_n$",
        r"$S/D_n$",
        r"detection rate $\Pr(\widehat{r}>0)$",
    )

    # Panel 3: risk scale, against the exact prediction (S/R)^2.
    ax = axes[2]
    hill = hill.assign(x=hill["signal_over_R"])
    grid = np.logspace(np.log10(0.03), np.log10(2.0), 200)
    ax.plot(grid, grid**2, color=REFERENCE, lw=1.3, zorder=2)
    ax.axhline(1.0, color=INK_MUTED, lw=0.8, zorder=1)
    scatter_panel(ax, hill, "risk_y", "risk_se")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.025, 2.4)
    style(
        ax,
        r"Risk scale $R_n$",
        r"$S/R_n$",
        r"Hill MSE inflation, $\gamma^2/k_n$ units",
    )
    ax.annotate(
        r"$(S/R)^2$",
        xy=(0.20, 0.30),
        fontsize=7.5,
        color=INK_MUTED,
        ha="center",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
        bbox_to_anchor=(0.5, -0.02),
        handletextpad=0.4,
        columnspacing=1.8,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    args.outdir.mkdir(parents=True, exist_ok=True)
    pdf = args.outdir / "fig_three_scales.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    preview = args.preview_dir / "fig_three_scales_preview.png"
    fig.savefig(preview, dpi=170, bbox_inches="tight")
    print(f"wrote {pdf}")
    print(f"wrote {preview}")


if __name__ == "__main__":
    main()
