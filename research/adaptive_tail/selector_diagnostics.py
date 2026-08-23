"""Research diagnostics for threshold-compatibility selection.

This script stays outside the public estimator API. It traces the cross-fit
fold path and calibrates the compatibility cutoff under exact Pareto on
calibration seeds, then evaluates the chosen cutoff on held-out seeds.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heavytails import Pareto
from heavytails.tail_index import (
    _crossfit_split,
    _minimum_variance_weights,
    _normalised_log_spacings,
    _orthogonalized_spacing_weights,
    _scaled_order_count,
    adaptive_trim_selection,
    threshold_averaged_orthogonalized_hill_selection,
)
from research.adaptive_tail.oracle_experiment import (
    _admissible_max_trim,
    _crossfit_min_threshold,
    _parse_floats,
    _provenance,
    _thresholds_for_mode,
)


def _trace_apply_threshold_average(
    data: list[float], selection: dict[str, Any]
) -> dict[str, Any]:
    thresholds = selection["stable_thresholds"]
    rho = selection["rho"]
    adaptive_trim = bool(selection["adaptive_trim"])
    max_trim = selection["max_trim"]
    level = selection["level"]

    x = sorted(data, reverse=True)
    if not thresholds:
        raise ValueError("selection contains no stable thresholds")
    max_k = thresholds[-1]
    if not (1 < max_k < len(x)):
        raise ValueError("selected thresholds do not fit the evaluation sample")
    if x[max_k] <= 0.0:
        raise ValueError("threshold averaging requires positive data")

    spacings = _normalised_log_spacings(x, max_k)
    embedded_weights: list[list[float]] = []
    local_estimates: list[float] = []
    trims: list[int] = []

    for threshold in thresholds:
        trim = 0
        if adaptive_trim:
            trim_selection = adaptive_trim_selection(
                x, threshold, max_trim=max_trim, level=level
            )
            if trim_selection["saturated"]:
                raise ValueError(
                    "Contamination reaches deeper than max_trim in the evaluation "
                    f"fold at threshold {threshold}; raise max_trim above "
                    f"{trim_selection['deepest_anomaly']}."
                )
            trim = int(trim_selection["trim"])
        trims.append(trim)

        weights = _orthogonalized_spacing_weights(threshold, trim, rho)
        embedded = [0.0] * max_k
        for offset, weight in enumerate(weights, start=trim):
            embedded[offset] = weight
        local_estimates.append(
            float(
                sum(
                    weight * spacing
                    for weight, spacing in zip(
                        weights, spacings[trim:threshold], strict=True
                    )
                )
            )
        )
        embedded_weights.append(embedded)

    covariance = [
        [
            sum(w_i * w_j for w_i, w_j in zip(first, second, strict=True))
            for second in embedded_weights
        ]
        for first in embedded_weights
    ]
    averaging_weights = _minimum_variance_weights(
        covariance, nonnegative=bool(selection["convex_weights"])
    )
    gamma = float(
        sum(
            weight * estimate
            for weight, estimate in zip(averaging_weights, local_estimates, strict=True)
        )
    )

    return {
        "gamma": gamma,
        "thresholds": thresholds,
        "trims": trims,
        "candidate_pairs": list(zip(trims, thresholds, strict=True)),
        "local_estimates": local_estimates,
        "weights": averaging_weights,
    }


def _trace_crossfit(
    data: list[float],
    *,
    k: int,
    min_k: int,
    grid_size: int,
    rho: float,
    max_trim: int,
    critical: float | None,
    seed: int | None,
) -> dict[str, Any]:
    first, second = _crossfit_split(data, seed)
    full_n = len(data)
    fold_specs = [
        ("first_to_second", first, second),
        ("second_to_first", second, first),
    ]
    folds = []
    fold_estimates = []

    for label, train, target in fold_specs:
        split_k = _scaled_order_count(k, len(train), full_n)
        split_min_k = min(_scaled_order_count(min_k, len(train), full_n), split_k)
        try:
            selection = threshold_averaged_orthogonalized_hill_selection(
                train,
                split_k,
                min_k=split_min_k,
                grid_size=grid_size,
                rho=rho,
                adaptive_trim=True,
                max_trim=max_trim,
                critical=critical,
            )
        except ValueError as exc:
            folds.append(
                {
                    "direction": label,
                    "stage": "selection",
                    "failure_reason": str(exc),
                    "split_k": split_k,
                    "split_min_k": split_min_k,
                }
            )
            continue

        try:
            evaluation = _trace_apply_threshold_average(target, selection)
        except ValueError as exc:
            folds.append(
                {
                    "direction": label,
                    "stage": "evaluation",
                    "failure_reason": str(exc),
                    "split_k": split_k,
                    "split_min_k": split_min_k,
                    "training_thresholds": selection["thresholds"],
                    "training_trims": selection["trims"],
                    "training_stable_thresholds": selection["stable_thresholds"],
                    "training_stable_candidate_pairs": selection[
                        "stable_candidate_pairs"
                    ],
                    "training_weights": selection["weights"],
                }
            )
            continue

        fold_estimates.append(evaluation["gamma"])
        folds.append(
            {
                "direction": label,
                "stage": "success",
                "split_k": split_k,
                "split_min_k": split_min_k,
                "training_thresholds": selection["thresholds"],
                "training_trims": selection["trims"],
                "training_stable_thresholds": selection["stable_thresholds"],
                "training_stable_candidate_pairs": selection["stable_candidate_pairs"],
                "training_weights": selection["weights"],
                "evaluation_thresholds": evaluation["thresholds"],
                "evaluation_trims": evaluation["trims"],
                "evaluation_candidate_pairs": evaluation["candidate_pairs"],
                "evaluation_weights": evaluation["weights"],
                "fold_gamma": evaluation["gamma"],
            }
        )

    return {
        "gamma": (
            sum(fold_estimates) / len(fold_estimates)
            if len(fold_estimates) == 2
            else None
        ),
        "failure_rate": 1.0 - len(fold_estimates) / 2.0,
        "folds": folds,
    }


def _selection_rate(
    *,
    n: int,
    k_grid: list[int],
    max_trim: int,
    rho: float,
    critical: float,
    trials: int,
    seed_start: int,
) -> dict[str, Any]:
    hits = 0
    failures = 0
    stable_sizes: list[int] = []
    max_k = k_grid[-1]
    for offset in range(trials):
        data = Pareto(alpha=2.0, xm=1.0).rvs(n, seed=seed_start + offset)
        try:
            selection = threshold_averaged_orthogonalized_hill_selection(
                data,
                max_k,
                min_k=k_grid[0],
                grid_size=len(k_grid),
                rho=rho,
                adaptive_trim=True,
                max_trim=max_trim,
                critical=critical,
            )
        except ValueError:
            failures += 1
            continue
        hits += int(selection["stable_thresholds"][-1] == max_k)
        stable_sizes.append(len(selection["stable_thresholds"]))

    successes = trials - failures
    return {
        "critical": critical,
        "trials": trials,
        "successes": successes,
        "failure_rate": failures / trials,
        "k_max_acceptance_rate": hits / successes if successes else None,
        "mean_stable_set_size": (
            sum(stable_sizes) / len(stable_sizes) if stable_sizes else None
        ),
    }


def build_report(
    *,
    n: int,
    k_grid_mode: str,
    k_fractions: list[float],
    intermediate_grid_size: int,
    intermediate_min_power: float,
    intermediate_max_power: float,
    max_trim: int,
    rho: float,
    target_acceptance: float,
    calibration_trials: int,
    holdout_trials: int,
    calibration_seed_start: int,
    holdout_seed_start: int,
    critical_grid: list[float],
    trace_count: int,
) -> dict[str, Any]:
    k_grid = _thresholds_for_mode(
        n,
        k_grid_mode=k_grid_mode,  # type: ignore[arg-type]
        k_fractions=k_fractions,
        intermediate_grid_size=intermediate_grid_size,
        intermediate_min_power=intermediate_min_power,
        intermediate_max_power=intermediate_max_power,
    )
    max_k = k_grid[-1]
    min_k = k_grid[0]
    crossfit_min_k = _crossfit_min_threshold(n, min_k)
    admissible_max_trim = _admissible_max_trim(n, min_k, max_trim)

    calibration = [
        _selection_rate(
            n=n,
            k_grid=k_grid,
            max_trim=admissible_max_trim,
            rho=rho,
            critical=critical,
            trials=calibration_trials,
            seed_start=calibration_seed_start,
        )
        for critical in critical_grid
    ]
    selected = next(
        (
            row
            for row in calibration
            if row["k_max_acceptance_rate"] is not None
            and row["k_max_acceptance_rate"] >= target_acceptance
        ),
        calibration[-1],
    )
    holdout = _selection_rate(
        n=n,
        k_grid=k_grid,
        max_trim=admissible_max_trim,
        rho=rho,
        critical=selected["critical"],
        trials=holdout_trials,
        seed_start=holdout_seed_start,
    )
    traces = []
    for seed in range(holdout_seed_start, holdout_seed_start + trace_count):
        data = Pareto(alpha=2.0, xm=1.0).rvs(n, seed=seed)
        traces.append(
            {
                "data_seed": seed,
                "default_critical": _trace_crossfit(
                    data,
                    k=max_k,
                    min_k=min_k,
                    grid_size=len(k_grid),
                    rho=rho,
                    max_trim=admissible_max_trim,
                    critical=None,
                    seed=None,
                ),
                "calibrated_critical": _trace_crossfit(
                    data,
                    k=max_k,
                    min_k=min_k,
                    grid_size=len(k_grid),
                    rho=rho,
                    max_trim=admissible_max_trim,
                    critical=selected["critical"],
                    seed=None,
                ),
            }
        )

    return {
        "provenance": _provenance(),
        "configuration": {
            "n": n,
            "k_grid_mode": k_grid_mode,
            "k_fractions": k_fractions,
            "intermediate_grid_size": intermediate_grid_size,
            "intermediate_min_power": intermediate_min_power,
            "intermediate_max_power": intermediate_max_power,
            "k_grid": k_grid,
            "crossfit_min_k": crossfit_min_k,
            "requested_max_trim": max_trim,
            "admissible_max_trim": admissible_max_trim,
            "rho": rho,
            "target_acceptance": target_acceptance,
            "calibration_trials": calibration_trials,
            "holdout_trials": holdout_trials,
            "calibration_seed_start": calibration_seed_start,
            "holdout_seed_start": holdout_seed_start,
            "trace_count": trace_count,
            "crossfit_seed": "None, matching the production estimator default",
        },
        "calibration": calibration,
        "selected_critical": selected,
        "holdout": holdout,
        "traces": traces,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=10_000)
    parser.add_argument(
        "--k-grid-mode", choices=("fractions", "intermediate"), default="intermediate"
    )
    parser.add_argument("--k-fractions", default="0.02,0.05,0.10")
    parser.add_argument("--intermediate-grid-size", type=int, default=10)
    parser.add_argument("--intermediate-min-power", type=float, default=1.0 / 3.0)
    parser.add_argument("--intermediate-max-power", type=float, default=2.0 / 3.0)
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--rho", type=float, default=-1.0)
    parser.add_argument("--target-acceptance", type=float, default=0.95)
    parser.add_argument("--calibration-trials", type=int, default=200)
    parser.add_argument("--holdout-trials", type=int, default=200)
    parser.add_argument("--calibration-seed-start", type=int, default=10_000)
    parser.add_argument("--holdout-seed-start", type=int, default=20_000)
    parser.add_argument("--critical-grid", default="1.0,1.25,1.5,1.75,2,2.25,2.5,3")
    parser.add_argument("--trace-count", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = build_report(
        n=args.n,
        k_grid_mode=args.k_grid_mode,
        k_fractions=_parse_floats(args.k_fractions),
        intermediate_grid_size=args.intermediate_grid_size,
        intermediate_min_power=args.intermediate_min_power,
        intermediate_max_power=args.intermediate_max_power,
        max_trim=args.max_trim,
        rho=args.rho,
        target_acceptance=args.target_acceptance,
        calibration_trials=args.calibration_trials,
        holdout_trials=args.holdout_trials,
        calibration_seed_start=args.calibration_seed_start,
        holdout_seed_start=args.holdout_seed_start,
        critical_grid=_parse_floats(args.critical_grid),
        trace_count=args.trace_count,
    )
    print(
        "selected critical "
        f"{report['selected_critical']['critical']:.3f}; "
        "holdout k_max acceptance "
        f"{report['holdout']['k_max_acceptance_rate']:.3f}"
    )
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
