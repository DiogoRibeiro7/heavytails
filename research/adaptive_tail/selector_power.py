"""Power of the calibrated compatibility rule against second-order bias.

The calibration in ``selector_diagnostics.py`` establishes the null curve:
which cutoff makes the production cross-fit selector reach its top threshold
on 95% of clean Pareto samples. It does not establish that such a cutoff is
*useful*, and one reading of that result was wrong about this.

Under exact Pareto the correct threshold **is** the largest one, so a
correctly sized test should accept essentially the whole grid. High null
acceptance is type-I calibration, not a loss of selectivity. What decides
whether the rule still discriminates is the other curve:

    c -> P(the rule cuts the grid short when large k is biased)

This script measures it. For each scenario and cutoff it records the
distribution of

    K_final / k_max,   K_final = max(stable set)

which is the clean diagnostic: if Pareto concentrates at 1 and Burr shifts
down, the rule is working once calibrated. If both sit at 1, it has lost its
power and the constant cannot be the fix.

Alongside it, the estimator's own error, so that a rule which discriminates
but does not help is visible as such.
"""

from __future__ import annotations

# ruff: noqa: E402
# The sys.path bootstrap has to run before the imports that depend on it.
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.adaptive_tail.oracle_experiment import (
    SCENARIOS,
)
from research.adaptive_tail.selector_diagnostics import (
    _admissible_max_trim,
    _provenance,
    _thresholds_for_mode,
    _trace_crossfit,
)

from heavytails.tail_index import (
    threshold_averaged_orthogonalized_hill_estimator,
)


def _contaminate(sample: list[float], count: int, delta: float) -> list[float]:
    """Multiply the ``count`` largest observations by ``delta``.

    Planted before the split, as in the oracle experiment, so the two folds
    receive dependent amounts of it -- which is one reason the clean-data
    identity ``joint = marginal**2`` should not be assumed to carry over.
    """
    if count <= 0:
        return list(sample)
    values = sorted(sample, reverse=True)
    return [v * delta if i < count else v for i, v in enumerate(values)]


def _cell(
    *,
    scenario_key: str,
    n: int,
    k_grid: list[int],
    max_trim: int,
    critical: float,
    contamination: int,
    delta: float,
    trials: int,
    seed_start: int,
    rho_used: float | None = None,
) -> dict[str, Any]:
    scenario = SCENARIOS[scenario_key]
    max_k = k_grid[-1]
    # Each scenario carries its own rho_used, so comparing Pareto against Burr
    # changes the law *and* the tuning the orthogonalized weights are built
    # from. Overriding it is how the null size gets matched on the geometry.
    rho = scenario.rho_used if rho_used is None else rho_used

    fractions: list[float] = []
    run_fractions: list[float] = []
    accepted_full = 0
    both_succeeded = 0
    errors: list[float] = []
    failures = 0
    # Per replication, so cutoffs evaluated on the same seeds can be compared
    # as paired losses. They are highly correlated, and the uncertainty of the
    # difference is much smaller than the uncertainty of either RMSE.
    replications: list[dict[str, Any]] = []

    for offset in range(trials):
        seed = seed_start + offset
        sample = _contaminate(scenario.sample(n, seed), contamination, delta)

        trace = _trace_crossfit(
            sample,
            k=max_k,
            min_k=k_grid[0],
            grid_size=len(k_grid),
            rho=rho,
            max_trim=max_trim,
            critical=critical,
            seed=None,
        )
        succeeded = [f for f in trace["folds"] if f["stage"] == "success"]
        if len(succeeded) < 2:
            failures += 1
            continue
        both_succeeded += 1

        # How far up its own grid each fold's stable set reached. Pooled
        # across folds for the fold-level view, and reduced to the shallower
        # of the two for a statement about the replication itself.
        per_fold = [
            (
                fold["training_stable_thresholds"][-1] / fold["split_k"]
                if fold["training_stable_thresholds"]
                else 0.0
            )
            for fold in succeeded
        ]
        fractions.extend(per_fold)
        run_fraction = min(per_fold)
        run_fractions.append(run_fraction)
        full = int(
            all(
                fold["training_stable_thresholds"]
                and fold["training_stable_thresholds"][-1] == fold["split_k"]
                for fold in succeeded
            )
        )
        accepted_full += full

        try:
            gamma = threshold_averaged_orthogonalized_hill_estimator(
                sample,
                max_k,
                min_k=k_grid[0],
                grid_size=len(k_grid),
                rho=rho,
                adaptive_trim=True,
                max_trim=max_trim,
                critical=critical,
                crossfit=True,
                seed=None,
            )
        except ValueError:
            failures += 1
            replications.append(
                {"seed": seed, "squared_error": None, "run_fraction": run_fraction}
            )
            continue
        error = gamma - scenario.gamma
        errors.append(error)
        replications.append(
            {
                "seed": seed,
                "squared_error": error * error,
                "run_fraction": run_fraction,
                "accepted_full": bool(full),
            }
        )

    def summary(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(ordered),
            "median": statistics.median(ordered),
            "p10": ordered[max(0, int(0.10 * len(ordered)) - 1)],
            "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
            "fraction_at_full_grid": sum(1 for v in ordered if v >= 1.0) / len(ordered),
        }

    return {
        "scenario": scenario_key,
        "label": scenario.label,
        "gamma": scenario.gamma,
        "rho_true": scenario.rho_true,
        "rho_used": rho,
        "rho_used_is_scenario_default": rho_used is None,
        "critical": critical,
        "contamination_count": contamination,
        "delta": delta if contamination else None,
        "trials": trials,
        "failure_rate": failures / trials,
        "joint_full_acceptance_rate": accepted_full / trials,
        "both_folds_succeeded_rate": both_succeeded / trials,
        # Where the stable set stopped, as a fraction of the fold's own top
        # threshold. `stable_fraction` pools the folds; `run_stable_fraction`
        # takes the shallower of the two, so a statement about it is a
        # statement about the replication rather than about a fold.
        "stable_fraction": summary(fractions),
        "run_stable_fraction": summary(run_fractions),
        "replications": replications,
        "bias": statistics.fmean(errors) if errors else None,
        "rmse": math.sqrt(statistics.fmean([e * e for e in errors]))
        if errors
        else None,
    }


def build_report(
    *,
    n: int,
    critical_grid: list[float],
    scenarios: list[str],
    contaminations: list[int],
    deltas: list[float],
    trials: int,
    seed_start: int,
    max_trim: int,
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

    combinations = [
        (critical, scenario_key, contamination, delta)
        for critical in critical_grid
        for scenario_key in scenarios
        for contamination in contaminations
        for delta in (deltas if contamination else [0.0])
    ]
    cells = [
        _cell(
            scenario_key=scenario_key,
            n=n,
            k_grid=k_grid,
            max_trim=admissible,
            critical=critical,
            contamination=contamination,
            delta=delta,
            trials=trials,
            seed_start=seed_start,
        )
        for critical, scenario_key, contamination, delta in combinations
    ]

    return {
        "provenance": _provenance(),
        "purpose": (
            "Power of the calibrated compatibility rule. High acceptance under "
            "exact Pareto is correct type-I behaviour, since the best threshold "
            "there is the largest one; what decides whether the rule is useful "
            "is whether it still cuts the grid short under second-order bias."
        ),
        "configuration": {
            "n": n,
            "k_grid": k_grid,
            "requested_max_trim": max_trim,
            "admissible_max_trim": admissible,
            "critical_grid": critical_grid,
            "scenarios": scenarios,
            "contamination_counts": contaminations,
            "deltas": deltas,
            "trials_per_cell": trials,
            "seed_start": seed_start,
            "crossfit_seed": "None, matching the production estimator default",
        },
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument("--critical-grid", default="4,4.5,5")
    parser.add_argument(
        "--scenarios", default="pareto,hall_rho_half,burr_rho_half,burr_rho_quarter"
    )
    parser.add_argument("--contaminations", default="0,1,3")
    parser.add_argument("--deltas", default="2,10")
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=70_000)
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = build_report(
        n=args.n,
        critical_grid=[float(v) for v in args.critical_grid.split(",")],
        scenarios=args.scenarios.split(","),
        contaminations=[int(v) for v in args.contaminations.split(",")],
        deltas=[float(v) for v in args.deltas.split(",")],
        trials=args.trials,
        seed_start=args.seed_start,
        max_trim=args.max_trim,
    )

    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    print(
        f"{'scenario':<20} {'c':>4} {'r':>2} {'delta':>6} {'full':>6} {'medfrac':>8} {'rmse':>7}"
    )
    for cell in report["cells"]:
        fraction = cell["stable_fraction"]
        print(
            f"{cell['scenario']:<20} {cell['critical']:>4.1f} "
            f"{cell['contamination_count']:>2} "
            f"{(cell['delta'] or 0):>6.1f} "
            f"{cell['joint_full_acceptance_rate']:>6.3f} "
            f"{(fraction['median'] if fraction else float('nan')):>8.3f} "
            f"{(cell['rmse'] if cell['rmse'] is not None else float('nan')):>7.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
