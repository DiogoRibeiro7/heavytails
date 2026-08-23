"""Clean-Pareto decomposition for the adaptive tail estimator.

This script isolates where the exact-Pareto pilot penalty comes from. It uses
the same simulated samples for four estimators:

* the best fixed local ``(r, k)`` estimator on the research grid;
* the full-sample adaptive selected local estimator;
* the full-sample adaptive threshold aggregation;
* the production cross-fitted adaptive estimator.

The first target is an in-sample empirical decomposition baseline, not the
out-of-sample oracle risk used by ``oracle_experiment.py``.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heavytails.tail_index import (
    threshold_averaged_orthogonalized_hill_estimator,
    threshold_averaged_orthogonalized_hill_selection,
)
from research.adaptive_tail.oracle_experiment import (
    SCENARIOS,
    Candidate,
    KGridMode,
    _candidate_estimate,
    _format_optional,
    _mse,
    _parse_floats,
    _parse_ints,
    _provenance,
    _rmse_from_squared,
    _standard_error,
    _thresholds_for_mode,
)


def _jsonable_pair(pair: Candidate | None) -> list[int] | None:
    return None if pair is None else [pair[0], pair[1]]


def _pair_key(pair: Candidate | None) -> str:
    return "failure" if pair is None else f"r={pair[0]},k={pair[1]}"


def _rate_counts(values: Sequence[int]) -> dict[str, float]:
    if not values:
        return {}
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {str(key): counts[key] / len(values) for key in sorted(counts)}


def _inclusion_rates(values: Sequence[int], denominator: int) -> dict[str, float]:
    if denominator == 0:
        return {}
    counts: dict[int, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return {str(key): counts[key] / denominator for key in sorted(counts)}


def _pair_counts(pairs: Sequence[Candidate | None]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        key = _pair_key(pair)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _method_summary(
    estimates: Sequence[float | None],
    *,
    truth: float,
    baseline_mse: float | None,
) -> dict[str, float | None]:
    success_squared = [
        (estimate - truth) ** 2 for estimate in estimates if estimate is not None
    ]
    failure_count = len(estimates) - len(success_squared)
    mse = _mse(success_squared) if failure_count == 0 and success_squared else None
    success_mse = _mse(success_squared) if success_squared else None
    return {
        "failure_rate": failure_count / len(estimates) if estimates else None,
        "mse": mse,
        "rmse": math.sqrt(mse) if mse is not None else None,
        "mse_se": _standard_error(success_squared) if mse is not None else None,
        "success_mse": success_mse,
        "success_rmse": (
            _rmse_from_squared(success_squared) if success_squared else None
        ),
        "ratio_to_best_local": (
            mse / baseline_mse
            if mse is not None and baseline_mse is not None and baseline_mse > 0.0
            else None
        ),
    }


def _select_best_local(
    estimates: dict[Candidate, list[float | None]], truth: float
) -> tuple[Candidate | None, float | None]:
    scored: list[tuple[float, Candidate]] = []
    for candidate, values in estimates.items():
        if any(value is None for value in values):
            continue
        squared = [(value - truth) ** 2 for value in values]  # type: ignore[operator]
        scored.append((_mse(squared), candidate))
    if not scored:
        return None, None
    mse, candidate = min(scored, key=lambda item: item[0])
    return candidate, mse


def _evaluate_clean_pareto_cell(
    *,
    n: int,
    trials: int,
    k_fractions: list[float],
    max_trim: int,
    k_grid_mode: KGridMode = "fractions",
    intermediate_grid_size: int = 10,
    intermediate_min_power: float = 1.0 / 3.0,
    intermediate_max_power: float = 2.0 / 3.0,
) -> dict[str, Any]:
    if trials < 1:
        raise ValueError("at least one trial is required")

    scenario = SCENARIOS["pareto"]
    k_grid = _thresholds_for_mode(
        n,
        k_grid_mode=k_grid_mode,
        k_fractions=k_fractions,
        intermediate_grid_size=intermediate_grid_size,
        intermediate_min_power=intermediate_min_power,
        intermediate_max_power=intermediate_max_power,
    )
    min_k = k_grid[0]
    max_k = k_grid[-1]
    r_grid = list(range(max_trim + 1))
    adaptive_max_trim = min(max_trim, max(1, min_k // 2 - 1))
    candidates = [(r, k) for r in r_grid for k in k_grid if r < k - 1]

    candidate_estimates: dict[Candidate, list[float | None]] = {
        candidate: [] for candidate in candidates
    }
    selected_local_estimates: list[float | None] = []
    full_sample_aggregate_estimates: list[float | None] = []
    crossfit_estimates: list[float | None] = []
    selected_pairs: list[Candidate | None] = []
    stable_set_sizes: list[int] = []
    stable_thresholds: list[int] = []
    stable_trims: list[int] = []

    for seed in range(trials):
        data = scenario.sample(n, seed)

        for candidate in candidates:
            candidate_estimates[candidate].append(
                _candidate_estimate(data, candidate, scenario.rho_used)
            )

        try:
            selection = threshold_averaged_orthogonalized_hill_selection(
                data,
                max_k,
                min_k=min_k,
                grid_size=len(k_grid),
                rho=scenario.rho_used,
                adaptive_trim=True,
                max_trim=adaptive_max_trim,
            )
            selected_index = len(selection["stable_candidate_pairs"]) - 1
            selected_pair = selection["stable_candidate_pairs"][selected_index]
            selected_pairs.append(selected_pair)
            selected_local_estimates.append(
                selection["local_estimates"][selected_index]
            )
            stable_set_sizes.append(len(selection["stable_candidate_pairs"]))
            stable_thresholds.extend(selection["stable_thresholds"])
            stable_trims.extend(pair[0] for pair in selection["stable_candidate_pairs"])
        except ValueError:
            selected_pairs.append(None)
            selected_local_estimates.append(None)

        try:
            full_sample_aggregate_estimates.append(
                threshold_averaged_orthogonalized_hill_estimator(
                    data,
                    max_k,
                    min_k=min_k,
                    grid_size=len(k_grid),
                    rho=scenario.rho_used,
                    adaptive_trim=True,
                    max_trim=adaptive_max_trim,
                    crossfit=False,
                )
            )
        except ValueError:
            full_sample_aggregate_estimates.append(None)

        try:
            crossfit_estimates.append(
                threshold_averaged_orthogonalized_hill_estimator(
                    data,
                    max_k,
                    min_k=min_k,
                    grid_size=len(k_grid),
                    rho=scenario.rho_used,
                    adaptive_trim=True,
                    max_trim=adaptive_max_trim,
                    seed=seed,
                )
            )
        except ValueError:
            crossfit_estimates.append(None)

    best_pair, best_mse = _select_best_local(candidate_estimates, scenario.gamma)
    best_local_estimates = (
        candidate_estimates[best_pair] if best_pair is not None else [None] * trials
    )
    methods = {
        "best_local_oracle": _method_summary(
            best_local_estimates, truth=scenario.gamma, baseline_mse=best_mse
        ),
        "full_sample_selected_local": _method_summary(
            selected_local_estimates, truth=scenario.gamma, baseline_mse=best_mse
        ),
        "full_sample_adaptive_aggregation": _method_summary(
            full_sample_aggregate_estimates,
            truth=scenario.gamma,
            baseline_mse=best_mse,
        ),
        "cross_fitted_adaptive": _method_summary(
            crossfit_estimates, truth=scenario.gamma, baseline_mse=best_mse
        ),
    }
    methods["best_local_oracle"]["selected_pair"] = _jsonable_pair(best_pair)

    return {
        "scenario": scenario.key,
        "label": scenario.label,
        "gamma": scenario.gamma,
        "rho_used": scenario.rho_used,
        "n": n,
        "trials": trials,
        "k_grid_mode": k_grid_mode,
        "k_grid": k_grid,
        "r_grid": r_grid,
        "adaptive_max_trim": adaptive_max_trim,
        "methods": methods,
        "full_sample_selected_local_pair_counts": _pair_counts(selected_pairs),
        "stable_set_size_mean": (
            statistics.fmean(stable_set_sizes) if stable_set_sizes else None
        ),
        "stable_set_size_median": (
            statistics.median(stable_set_sizes) if stable_set_sizes else None
        ),
        "stable_threshold_inclusion": _inclusion_rates(
            stable_thresholds, len(stable_set_sizes)
        ),
        "stable_trim_frequency": _rate_counts(stable_trims),
    }


def run_decomposition(
    *,
    trials: int,
    sample_sizes: list[int],
    k_fractions: list[float],
    max_trim: int,
    k_grid_mode: KGridMode = "fractions",
    intermediate_grid_size: int = 10,
    intermediate_min_power: float = 1.0 / 3.0,
    intermediate_max_power: float = 2.0 / 3.0,
) -> list[dict[str, Any]]:
    """Run the clean-Pareto decomposition grid."""
    return [
        _evaluate_clean_pareto_cell(
            n=n,
            trials=trials,
            k_fractions=k_fractions,
            k_grid_mode=k_grid_mode,
            intermediate_grid_size=intermediate_grid_size,
            intermediate_min_power=intermediate_min_power,
            intermediate_max_power=intermediate_max_power,
            max_trim=max_trim,
        )
        for n in sample_sizes
    ]


def build_report(
    *,
    trials: int,
    sample_sizes: list[int],
    k_fractions: list[float],
    max_trim: int,
    k_grid_mode: KGridMode = "fractions",
    intermediate_grid_size: int = 10,
    intermediate_min_power: float = 1.0 / 3.0,
    intermediate_max_power: float = 2.0 / 3.0,
) -> dict[str, Any]:
    """Build the JSON-serializable clean-Pareto decomposition report."""
    rows = run_decomposition(
        trials=trials,
        sample_sizes=sample_sizes,
        k_fractions=k_fractions,
        k_grid_mode=k_grid_mode,
        intermediate_grid_size=intermediate_grid_size,
        intermediate_min_power=intermediate_min_power,
        intermediate_max_power=intermediate_max_power,
        max_trim=max_trim,
    )
    return {
        "provenance": _provenance(),
        "configuration": {
            "trials": trials,
            "sample_sizes": sample_sizes,
            "scenario": "pareto",
            "k_grid_mode": k_grid_mode,
            "k_fractions": k_fractions,
            "intermediate_grid_size": intermediate_grid_size,
            "intermediate_min_power": intermediate_min_power,
            "intermediate_max_power": intermediate_max_power,
            "max_trim": max_trim,
            "seeds": f"0..{trials - 1} per sample size",
            "target": (
                "clean-Pareto decomposition of local oracle, selection, "
                "aggregation and cross-fitting"
            ),
            "best_local_oracle": (
                "in-sample empirical MSE minimum over fixed local (r, k) candidates"
            ),
        },
        "results": rows,
    }


def _print_rows(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'n':>7}{'grid':>14}{'best':>10}{'selected':>10}"
        f"{'aggregate':>10}{'crossfit':>10}"
    )
    print(header)
    for row in rows:
        methods = row["methods"]
        print(
            f"{row['n']:>7}{row['k_grid_mode']:>14}"
            f"{_format_optional(methods['best_local_oracle']['rmse'], 10)}"
            f"{_format_optional(methods['full_sample_selected_local']['rmse'], 10)}"
            f"{_format_optional(methods['full_sample_adaptive_aggregation']['rmse'], 10)}"
            f"{_format_optional(methods['cross_fitted_adaptive']['rmse'], 10)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--sample-sizes", default="1000,5000,10000")
    parser.add_argument(
        "--k-fractions",
        default="0.02,0.05,0.10",
        help="Envelope fractions used when --k-grid-mode=fractions.",
    )
    parser.add_argument(
        "--k-grid-mode",
        choices=("fractions", "intermediate"),
        default="fractions",
        help="Use fixed fractions or an intermediate n^a..n^b threshold envelope.",
    )
    parser.add_argument("--intermediate-grid-size", type=int, default=10)
    parser.add_argument("--intermediate-min-power", type=float, default=1.0 / 3.0)
    parser.add_argument("--intermediate-max-power", type=float, default=2.0 / 3.0)
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = build_report(
        trials=args.trials,
        sample_sizes=_parse_ints(args.sample_sizes),
        k_fractions=_parse_floats(args.k_fractions),
        k_grid_mode=args.k_grid_mode,
        intermediate_grid_size=args.intermediate_grid_size,
        intermediate_min_power=args.intermediate_min_power,
        intermediate_max_power=args.intermediate_max_power,
        max_trim=args.max_trim,
    )
    _print_rows(report["results"])
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
