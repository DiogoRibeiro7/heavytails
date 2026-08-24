"""Two checks that close, or reopen, the power result.

**Matched null size.** Each scenario in the power study carries its own
``rho_used``, so comparing Pareto at rho = -1 against Burr at rho = -1/4
changes the law *and* the tuning the orthogonalized weights are built from.
The compatibility statistic's finite-sample null distribution can move with
rho, so part of the apparent power could be a size effect. Exact Pareto is run
here at each rho in turn: if the null acceptance holds up across them, the
comparison is about the law after all.

**Paired loss differences.** Every cutoff in the power study is evaluated on
the same seeds, so ``MSE(c) - MSE(100)`` is a paired quantity and the standard
error of either RMSE says nothing useful about it. The two estimates are
highly correlated, so the uncertainty of their difference is much smaller.
This computes the difference per replication, with a paired bootstrap interval.
"""

from __future__ import annotations

# ruff: noqa: E402
# The sys.path bootstrap has to run before the imports that depend on it.
import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.adaptive_tail.selector_diagnostics import (
    _admissible_max_trim,
    _provenance,
    _thresholds_for_mode,
)
from research.adaptive_tail.selector_power import _cell

NO_SELECTION = 100.0


def _paired_difference(
    treatment: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any] | None:
    """``MSE(treatment) - MSE(reference)`` on the replications both resolved.

    Paired by seed. A replication where either side failed to produce an
    estimate is dropped from both, so the difference is over a common set.
    """
    by_seed = {r["seed"]: r for r in reference}
    differences = []
    for row in treatment:
        other = by_seed.get(row["seed"])
        if other is None:
            continue
        if row["squared_error"] is None or other["squared_error"] is None:
            continue
        differences.append(row["squared_error"] - other["squared_error"])

    if len(differences) < 2:
        return None

    mean = statistics.fmean(differences)
    spread = statistics.stdev(differences)
    rng = random.Random(seed)
    size = len(differences)
    resampled = sorted(
        statistics.fmean([differences[rng.randrange(size)] for _ in range(size)])
        for _ in range(draws)
    )
    lower = resampled[int(0.025 * draws)]
    upper = resampled[min(draws - 1, int(0.975 * draws))]

    return {
        "paired_replications": size,
        "mean_mse_difference": mean,
        "standard_error": spread / (size**0.5),
        "bootstrap_lower": lower,
        "bootstrap_upper": upper,
        # The question the number is being asked: is the selector's loss
        # distinguishable from taking every threshold?
        "interval_contains_zero": lower <= 0.0 <= upper,
        "favours_selection": upper < 0.0,
    }


def build_report(
    *,
    n: int,
    rho_grid: list[float],
    critical_grid: list[float],
    scenarios: list[str],
    trials: int,
    seed_start: int,
    max_trim: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    k_grid = _thresholds_for_mode(
        n,
        k_grid_mode="intermediate",
        k_fractions=[0.02, 0.05, 0.10],
        intermediate_grid_size=10,
        intermediate_min_power=1.0 / 3.0,
        intermediate_max_power=2.0 / 3.0,
    )
    admissible = _admissible_max_trim(n, k_grid[0], max_trim)
    shared = {
        "n": n,
        "k_grid": k_grid,
        "max_trim": admissible,
        "contamination": 0,
        "delta": 0.0,
        "trials": trials,
        "seed_start": seed_start,
    }

    # Exact Pareto at each rho, at every cutoff, so the null size is matched
    # on the tuning as well as on the law.
    null_size = [
        {
            k: v
            for k, v in _cell(
                scenario_key="pareto", critical=critical, rho_used=rho, **shared
            ).items()
            if k != "replications"
        }
        for rho in rho_grid
        for critical in critical_grid
    ]

    # Paired losses against taking every threshold, at each scenario's own
    # tuning, which is what production would use.
    paired = []
    for scenario_key in scenarios:
        reference = _cell(scenario_key=scenario_key, critical=NO_SELECTION, **shared)
        for critical in critical_grid:
            treatment = (
                reference
                if critical == NO_SELECTION
                else _cell(scenario_key=scenario_key, critical=critical, **shared)
            )
            difference = _paired_difference(
                treatment["replications"],
                reference["replications"],
                draws=bootstrap_draws,
                seed=17,
            )
            paired.append(
                {
                    "scenario": scenario_key,
                    "rho_used": treatment["rho_used"],
                    "critical": critical,
                    "rmse": treatment["rmse"],
                    "reference_rmse": reference["rmse"],
                    "run_stable_fraction_p10": (
                        treatment["run_stable_fraction"]["p10"]
                        if treatment["run_stable_fraction"]
                        else None
                    ),
                    "paired_vs_no_selection": difference,
                }
            )

    return {
        "provenance": _provenance(),
        "purpose": (
            "Whether the power result survives matching the null size on rho, "
            "and whether any cutoff beats taking every threshold once the "
            "comparison is made paired rather than marginal."
        ),
        "configuration": {
            "n": n,
            "k_grid": k_grid,
            "admissible_max_trim": admissible,
            "rho_grid": rho_grid,
            "critical_grid": critical_grid,
            "scenarios": scenarios,
            "trials": trials,
            "seed_start": seed_start,
            "bootstrap_draws": bootstrap_draws,
            "no_selection_critical": NO_SELECTION,
            "crossfit_seed": "None, matching the production estimator default",
        },
        "null_size_by_rho": null_size,
        "paired_against_no_selection": paired,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--rho-grid", default="-1,-0.5,-0.25")
    parser.add_argument("--critical-grid", default="4,5")
    parser.add_argument(
        "--scenarios", default="pareto,hall_rho_half,burr_rho_half,burr_rho_quarter"
    )
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--seed-start", type=int, default=90_000)
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = build_report(
        n=args.n,
        rho_grid=[float(v) for v in args.rho_grid.split(",")],
        critical_grid=[float(v) for v in args.critical_grid.split(",")],
        scenarios=args.scenarios.split(","),
        trials=args.trials,
        seed_start=args.seed_start,
        max_trim=args.max_trim,
        bootstrap_draws=args.bootstrap_draws,
    )

    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    print("Exact Pareto null size, by the rho the weights are tuned with:")
    print(f"  {'rho':>6} {'c':>5} {'full-grid':>10}")
    for row in report["null_size_by_rho"]:
        print(
            f"  {row['rho_used']:>6.2f} {row['critical']:>5.1f} "
            f"{row['joint_full_acceptance_rate']:>10.3f}"
        )

    print("\nPaired MSE difference against taking every threshold:")
    print(f"  {'scenario':<18} {'c':>5} {'dMSE':>10} {'95% CI':>22} {'verdict':>12}")
    for row in report["paired_against_no_selection"]:
        d = row["paired_vs_no_selection"]
        if d is None:
            continue
        interval = f"[{d['bootstrap_lower']:+.5f}, {d['bootstrap_upper']:+.5f}]"
        verdict = (
            "helps"
            if d["favours_selection"]
            else ("indistinct" if d["interval_contains_zero"] else "hurts")
        )
        print(
            f"  {row['scenario']:<18} {row['critical']:>5.1f} "
            f"{d['mean_mse_difference']:>+10.5f} {interval:>22} {verdict:>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
