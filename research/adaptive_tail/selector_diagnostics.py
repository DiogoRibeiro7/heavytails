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
    """Rate at which the *production* cross-fit selector reaches its top threshold.

    Two things about this are deliberate, and an earlier version of it got both
    wrong.

    **It calibrates the cross-fit path, not the full-sample selector.** The
    production estimator does not select on the whole sample: it splits, then
    selects independently on each training half at a scaled threshold
    ``split_k = scale(k, n_f, n)`` with its own scaled ``min_k`` and its own
    fold-level vanishing trim. Those selectors see half the data, a different
    grid and a different finite-sample covariance geometry, so the full-sample
    acceptance rate is a different quantity. Calibrating it would tune a
    constant against a procedure nobody runs. This goes through
    :func:`_trace_crossfit`, which walks exactly the production path.

    **The denominator is every trial.** A cutoff that fails a tenth of the time
    and accepts on the rest is not a cutoff that works 100% of the time; it is
    usable nine times in ten. Dividing hits by successes reported the former.
    The study treats an estimator failure as invalidating the risk ratio
    everywhere else, and calibration follows the same rule: the indicator is
    joint over both folds succeeding *and* both accepting, averaged over all
    trials.

    Conditional fold-level acceptance is reported alongside as a diagnostic,
    because it says something different and is worth seeing -- but it is not
    the number the cutoff is chosen on.
    """
    joint_accepted = 0
    both_succeeded = 0
    fold_successes = 0
    fold_acceptances = 0
    fold_total = 0
    stable_sizes: list[int] = []
    outcomes: list[dict[str, Any]] = []

    for offset in range(trials):
        data = Pareto(alpha=2.0, xm=1.0).rvs(n, seed=seed_start + offset)
        trace = _trace_crossfit(
            data,
            k=k_grid[-1],
            min_k=k_grid[0],
            grid_size=len(k_grid),
            rho=rho,
            max_trim=max_trim,
            critical=critical,
            # None, as production defaults and oracle_experiment.py uses.
            seed=None,
        )
        fold_total += 2
        succeeded = [fold for fold in trace["folds"] if fold["stage"] == "success"]
        fold_successes += len(succeeded)

        # A fold accepts when its stable set reaches that fold's own top
        # threshold, which is the scaled split_k rather than the full-sample k.
        hits = 0
        for fold in succeeded:
            stable = fold["training_stable_thresholds"]
            stable_sizes.append(len(stable))
            if stable and stable[-1] == fold["split_k"]:
                hits += 1
        fold_acceptances += hits

        if len(succeeded) == 2:
            both_succeeded += 1
            if hits == 2:
                joint_accepted += 1

        # How far the shallower of the two folds got, as a fraction of its own
        # grid. 1.0 is full acceptance; smaller is an earlier stop.
        depths = [
            len(fold["training_stable_thresholds"]) / len(fold["training_thresholds"])
            for fold in succeeded
            if fold["training_thresholds"]
        ]
        outcomes.append(
            {
                "seed": seed_start + offset,
                "category": (
                    "failure"
                    if len(succeeded) < 2
                    else "accepted"
                    if hits == 2
                    else "premature_stop"
                ),
                "folds_succeeded": len(succeeded),
                "folds_accepted": hits,
                "shallowest_stable_fraction": min(depths) if depths else None,
            }
        )

    return {
        "critical": critical,
        "trials": trials,
        # The calibration target. Every trial counts, including the failures.
        "joint_acceptance_rate": joint_accepted / trials,
        "both_folds_succeeded_rate": both_succeeded / trials,
        "fold_failure_rate": 1.0 - fold_successes / fold_total,
        # Diagnostics: what an earlier version of this reported as the target.
        "fold_acceptance_rate_given_success": (
            fold_acceptances / fold_successes if fold_successes else None
        ),
        "mean_stable_set_size": (
            sum(stable_sizes) / len(stable_sizes) if stable_sizes else None
        ),
        "outcomes": outcomes,
    }


def _representative_seeds(outcomes: list[dict[str, Any]], count: int) -> list[int]:
    """Choose seeds worth looking at, rather than the first few.

    Tracing seeds 20000..20004 traces whatever those happened to be, and on a
    selector that mostly works they are mostly ordinary. What is worth reading
    is one of each outcome, and above all the *earliest* stop -- the run that
    cut its stable set shortest is the one that shows the mechanism.

    Falls back to filling from the front, so a run where everything succeeded
    still produces traces.
    """
    chosen: list[int] = []

    def take(seed: int) -> None:
        if seed not in chosen and len(chosen) < count:
            chosen.append(seed)

    failures = [o for o in outcomes if o["category"] == "failure"]
    premature = [
        o
        for o in outcomes
        if o["category"] == "premature_stop"
        and o["shallowest_stable_fraction"] is not None
    ]
    accepted = [o for o in outcomes if o["category"] == "accepted"]

    if premature:
        take(min(premature, key=lambda o: o["shallowest_stable_fraction"])["seed"])
    if failures:
        take(failures[0]["seed"])
    if accepted:
        take(accepted[0]["seed"])
    # Then the next-earliest stops, which are the informative tail of the run.
    for outcome in sorted(premature, key=lambda o: o["shallowest_stable_fraction"]):
        take(outcome["seed"])
    for outcome in outcomes:
        take(outcome["seed"])
    return chosen


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
    # Say so when nothing on the grid reaches the target, rather than handing
    # back the largest cutoff searched as though it had been calibrated. The
    # best available is reported instead of the last, because monotonicity in
    # `critical` is an expectation about the selector, not something this
    # script is entitled to assume.
    qualifying = [
        row for row in calibration if row["joint_acceptance_rate"] >= target_acceptance
    ]
    target_met = bool(qualifying)
    selected = (
        qualifying[0]
        if target_met
        else max(calibration, key=lambda row: row["joint_acceptance_rate"])
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
    for seed in _representative_seeds(holdout["outcomes"], trace_count):
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
        "target_met": target_met,
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
    # Write before printing. The summary reads keys out of the report, and a
    # rename here should not be able to throw away a run that has already
    # finished -- which is exactly what it did once.
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    holdout = report["holdout"]
    selected = report["selected_critical"]
    note = "" if report["target_met"] else "  (TARGET NOT MET on the calibration grid)"
    print(f"selected critical {selected['critical']:.3f}{note}")
    print(
        "holdout joint acceptance "
        f"{holdout['joint_acceptance_rate']:.3f} "
        f"(target {report['configuration']['target_acceptance']:.3f}, "
        f"{holdout['trials']} trials, every trial counted)"
    )
    print(f"  both folds succeeded      {holdout['both_folds_succeeded_rate']:.3f}")
    print(f"  fold failure rate         {holdout['fold_failure_rate']:.3f}")
    print(
        f"  fold acceptance | success {holdout['fold_acceptance_rate_given_success']}"
    )
    print(f"  mean stable set size      {holdout['mean_stable_set_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
