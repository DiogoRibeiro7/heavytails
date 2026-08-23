"""Oracle-risk experiment for the adaptive tail estimator.

This is deliberately separate from ``scripts/tail_index_study.py``. The generic
benchmark asks how named estimators behave at a fixed externally chosen
threshold. This experiment asks the research question instead: how far is the
adaptive estimator from an empirical oracle over contaminated, second-order
``(r, k)`` candidates?

The oracle is selected out of sample. One half of the Monte Carlo replications
chooses the best ``(r, k)`` pair, and the other half evaluates that pair; then
the roles are swapped. This avoids reporting the same minimum-selected RMSE
that chose the candidate.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib.metadata
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heavytails import BurrXII, Frechet, Pareto
from heavytails.tail_index import (
    _threshold_grid,
    _vanishing_level,
    adaptive_trim_selection,
    orthogonalized_bias_reduced_hill_estimator,
    threshold_averaged_orthogonalized_hill_estimator,
)


Sampler = Callable[[int, int], list[float]]
Candidate = tuple[int, int]


@dataclass(frozen=True)
class Scenario:
    """Sampling law with known EVI and second-order tuning information."""

    key: str
    label: str
    gamma: float
    rho_true: float | None
    rho_used: float
    sample: Sampler


@dataclass(frozen=True)
class FoldEvaluation:
    """Oracle selection on one Monte Carlo fold and evaluation on another."""

    selected_pair: Candidate | None
    selection_mse: float | None
    evaluation_indices: list[int]
    evaluation_squared_errors: list[tuple[int, float]]


def _hall_sample(
    n: int,
    seed: int,
    *,
    gamma: float,
    rho: float,
    beta: float,
) -> list[float]:
    """Sample from a simple Hall quantile model.

    The tail quantile is ``U(t) = t**gamma * (1 + beta * t**rho)``. Since
    ``rho < 0``, the perturbation vanishes in the tail and the EVI is
    ``gamma``.
    """
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        tail_rank = 1.0 / (1.0 - rng.random())
        out.append(tail_rank**gamma * (1.0 + beta * tail_rank**rho))
    return out


SCENARIOS: dict[str, Scenario] = {
    "pareto": Scenario(
        "pareto",
        "Exact Pareto(alpha=2), rho_used=-1",
        0.5,
        None,
        -1.0,
        lambda n, seed: Pareto(alpha=2.0, xm=1.0).rvs(n, seed=seed),
    ),
    "frechet": Scenario(
        "frechet",
        "Frechet(alpha=2)",
        0.5,
        -1.0,
        -1.0,
        lambda n, seed: Frechet(alpha=2.0, s=1.0, m=0.0).rvs(n, seed=seed),
    ),
    "burr_rho_half": Scenario(
        "burr_rho_half",
        "BurrXII(gamma=0.5,rho=-0.5)",
        0.5,
        -0.5,
        -0.5,
        lambda n, seed: BurrXII(c=1.0, k=2.0, s=1.0).rvs(n, seed=seed),
    ),
    "burr_rho_quarter": Scenario(
        "burr_rho_quarter",
        "BurrXII(gamma=0.5,rho=-0.25)",
        0.5,
        -0.25,
        -0.25,
        lambda n, seed: BurrXII(c=0.5, k=4.0, s=1.0).rvs(n, seed=seed),
    ),
    "hall_rho_half": Scenario(
        "hall_rho_half",
        "Hall(gamma=0.5,rho=-0.5)",
        0.5,
        -0.5,
        -0.5,
        lambda n, seed: _hall_sample(n, seed, gamma=0.5, rho=-0.5, beta=1.0),
    ),
}


def _contaminate(sample: list[float], count: int, delta: float) -> list[float]:
    """Replace the largest observations by separated top outliers."""
    if count == 0:
        return list(sample)
    out = sorted(sample, reverse=True)
    anchor = out[0]
    for index in range(count):
        out[index] = anchor * delta * (1.0 + 0.01 * (count - index))
    return out


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _mse(squared_errors: Sequence[float]) -> float:
    return _mean(squared_errors)


def _rmse_from_squared(squared_errors: Sequence[float]) -> float:
    return math.sqrt(_mse(squared_errors))


def _standard_error(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values) / math.sqrt(len(values))


def _bootstrap_summary(ratios: Sequence[float]) -> dict[str, float | None]:
    """Summarize bootstrap replicates of a statistic."""
    if len(ratios) < 2:
        return {"se": None, "lower": None, "upper": None}
    sorted_ratios = sorted(ratios)
    lower_index = math.floor(0.025 * (len(sorted_ratios) - 1))
    upper_index = math.ceil(0.975 * (len(sorted_ratios) - 1))
    return {
        "se": statistics.stdev(sorted_ratios),
        "lower": sorted_ratios[lower_index],
        "upper": sorted_ratios[upper_index],
    }


def _wilson_interval(
    successes: int, total: int, z: float = 1.96
) -> dict[str, float | None]:
    """Wilson interval for a binomial proportion."""
    if total == 0:
        return {"estimate": None, "lower": None, "upper": None}
    p_hat = successes / total
    denominator = 1.0 + z * z / total
    centre = (p_hat + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * total)) / total)
        / denominator
    )
    return {
        "estimate": p_hat,
        "lower": max(0.0, centre - radius),
        "upper": min(1.0, centre + radius),
    }


def _thresholds_from_fractions(n: int, k_fractions: list[float]) -> list[int]:
    """Build the exact logarithmic threshold grid used by the adaptive estimator."""
    if not k_fractions:
        raise ValueError("at least one k fraction is required")
    raw = sorted({max(2, min(n - 1, round(n * fraction))) for fraction in k_fractions})
    return _threshold_grid(raw[-1], raw[0], len(raw), n)


def _candidate_estimate(
    data: list[float], candidate: Candidate, rho: float
) -> float | None:
    r, k = candidate
    try:
        return orthogonalized_bias_reduced_hill_estimator(data, k, r=r, rho=rho)
    except ValueError:
        return None


def _select_oracle_candidate(
    estimates: dict[Candidate, list[float | None]],
    indices: Sequence[int],
    truth: float,
) -> tuple[Candidate | None, float | None]:
    """Choose the lowest-MSE candidate on a Monte Carlo selection fold."""
    scored: list[tuple[float, Candidate]] = []
    for candidate, values in estimates.items():
        if any(values[index] is None for index in indices):
            continue
        squared = [(values[index] - truth) ** 2 for index in indices]  # type: ignore[operator]
        scored.append((_mse(squared), candidate))
    if not scored:
        return None, None
    selection_mse, candidate = min(scored, key=lambda item: item[0])
    return candidate, selection_mse


def _evaluate_oracle_fold(
    estimates: dict[Candidate, list[float | None]],
    *,
    select_indices: Sequence[int],
    evaluate_indices: Sequence[int],
    truth: float,
) -> FoldEvaluation:
    """Select on one fold and evaluate on the other."""
    candidate, selection_mse = _select_oracle_candidate(
        estimates, select_indices, truth
    )
    if candidate is None:
        return FoldEvaluation(None, None, list(evaluate_indices), [])
    values = estimates[candidate]
    squared = []
    for index in evaluate_indices:
        value = values[index]
        if value is not None:
            squared.append((index, (value - truth) ** 2))
    return FoldEvaluation(candidate, selection_mse, list(evaluate_indices), squared)


def _oracle_squared_by_index(
    folds: Sequence[FoldEvaluation], trials: int
) -> list[float] | None:
    """Return oracle errors in replication order, or None if any are missing."""
    by_index: list[float | None] = [None] * trials
    for fold in folds:
        for index, squared in fold.evaluation_squared_errors:
            by_index[index] = squared
    if any(value is None for value in by_index):
        return None
    return [value for value in by_index if value is not None]


def _bootstrap_select_evaluate_ratio(
    adaptive_estimates: Sequence[float | None],
    candidate_estimates: dict[Candidate, list[float | None]],
    *,
    truth: float,
    draws: int,
    seed: int = 0,
) -> dict[str, float | None]:
    """Bootstrap risk-ratio uncertainty, redoing oracle selection in each draw."""
    if draws <= 0 or len(adaptive_estimates) < 2:
        return {"se": None, "lower": None, "upper": None}
    if any(estimate is None for estimate in adaptive_estimates):
        return {"se": None, "lower": None, "upper": None}
    adaptive_values = [
        estimate for estimate in adaptive_estimates if estimate is not None
    ]
    rng = random.Random(seed)
    ratios = []
    n = len(adaptive_values)
    for _ in range(draws):
        bootstrap_indices = [rng.randrange(n) for _ in range(n)]
        midpoint = n // 2
        first = bootstrap_indices[:midpoint]
        second = bootstrap_indices[midpoint:]
        folds = [
            _evaluate_oracle_fold(
                candidate_estimates,
                select_indices=first,
                evaluate_indices=second,
                truth=truth,
            ),
            _evaluate_oracle_fold(
                candidate_estimates,
                select_indices=second,
                evaluate_indices=first,
                truth=truth,
            ),
        ]
        oracle_squared = [
            squared for fold in folds for _, squared in fold.evaluation_squared_errors
        ]
        if len(oracle_squared) != n:
            continue
        numerator = _mean(
            [(adaptive_values[index] - truth) ** 2 for index in bootstrap_indices]
        )
        denominator = _mean(oracle_squared)
        if denominator > 0.0:
            ratios.append(numerator / denominator)
    return _bootstrap_summary(ratios)


def _jsonable_pair(pair: Candidate | None) -> list[int] | None:
    return None if pair is None else [pair[0], pair[1]]


def _evaluate_cell(
    scenario: Scenario,
    *,
    n: int,
    contamination_count: int,
    delta: float,
    trials: int,
    k_fractions: list[float],
    max_trim: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    """Evaluate one scenario/contamination/sample-size cell."""
    if trials < 2:
        raise ValueError("at least two trials are needed for split oracle evaluation")

    k_grid = _thresholds_from_fractions(n, k_fractions)
    min_k = k_grid[0]
    max_k = k_grid[-1]
    r_grid = list(range(max_trim + 1))
    adaptive_max_trim = min(max_trim, max(1, min_k // 2 - 1))
    candidates = [(r, k) for r in r_grid for k in k_grid if r < k - 1]

    adaptive_estimates: list[float | None] = []
    candidate_estimates: dict[Candidate, list[float | None]] = {
        candidate: [] for candidate in candidates
    }
    trim_hits_vanishing = 0
    trim_hits_fixed = 0
    trim_trials_vanishing = 0
    trim_trials_fixed = 0
    failures = {"adaptive": 0, "trim_selection_vanishing": 0, "trim_selection_fixed": 0}

    for seed in range(trials):
        clean = scenario.sample(n, seed)
        data = _contaminate(clean, contamination_count, delta)

        try:
            adaptive_estimates.append(
                threshold_averaged_orthogonalized_hill_estimator(
                    data,
                    max_k,
                    min_k=min_k,
                    grid_size=len(k_grid),
                    rho=scenario.rho_used,
                    adaptive_trim=True,
                    max_trim=adaptive_max_trim,
                )
            )
        except ValueError:
            failures["adaptive"] += 1
            adaptive_estimates.append(None)

        try:
            selection = adaptive_trim_selection(
                data,
                max_k,
                max_trim=adaptive_max_trim,
                level=_vanishing_level(n),
            )
            trim_hits_vanishing += int(selection["trim"] == contamination_count)
            trim_trials_vanishing += 1
        except ValueError:
            failures["trim_selection_vanishing"] += 1

        try:
            selection = adaptive_trim_selection(data, max_k, max_trim=adaptive_max_trim)
            trim_hits_fixed += int(selection["trim"] == contamination_count)
            trim_trials_fixed += 1
        except ValueError:
            failures["trim_selection_fixed"] += 1

        for candidate in candidates:
            candidate_estimates[candidate].append(
                _candidate_estimate(data, candidate, scenario.rho_used)
            )

    midpoint = trials // 2
    first = list(range(midpoint))
    second = list(range(midpoint, trials))
    folds = [
        _evaluate_oracle_fold(
            candidate_estimates,
            select_indices=first,
            evaluate_indices=second,
            truth=scenario.gamma,
        ),
        _evaluate_oracle_fold(
            candidate_estimates,
            select_indices=second,
            evaluate_indices=first,
            truth=scenario.gamma,
        ),
    ]
    oracle_squared = _oracle_squared_by_index(folds, trials)

    adaptive_success_squared = [
        (estimate - scenario.gamma) ** 2
        for estimate in adaptive_estimates
        if estimate is not None
    ]
    adaptive_unconditional_squared = (
        adaptive_success_squared if failures["adaptive"] == 0 else None
    )

    oracle_mse = _mse(oracle_squared) if oracle_squared is not None else None
    adaptive_mse = (
        _mse(adaptive_unconditional_squared)
        if adaptive_unconditional_squared is not None
        else None
    )
    risk_ratio = (
        adaptive_mse / oracle_mse
        if adaptive_mse is not None and oracle_mse is not None and oracle_mse > 0.0
        else None
    )
    bootstrap = (
        _bootstrap_select_evaluate_ratio(
            adaptive_estimates,
            candidate_estimates,
            truth=scenario.gamma,
            draws=bootstrap_draws,
        )
        if adaptive_unconditional_squared is not None and oracle_squared is not None
        else {"se": None, "lower": None, "upper": None}
    )

    return {
        "scenario": scenario.key,
        "label": scenario.label,
        "gamma": scenario.gamma,
        "rho_true": scenario.rho_true,
        "rho_used": scenario.rho_used,
        "n": n,
        "contamination_count": contamination_count,
        "delta": delta,
        "trials": trials,
        "k_grid": k_grid,
        "r_grid": r_grid,
        "adaptive_max_trim": adaptive_max_trim,
        "adaptive_failure_rate": failures["adaptive"] / trials,
        "adaptive_rmse_success": (
            _rmse_from_squared(adaptive_success_squared)
            if adaptive_success_squared
            else None
        ),
        "adaptive_mse_success_se": _standard_error(adaptive_success_squared),
        "adaptive_mse": adaptive_mse,
        "adaptive_rmse": math.sqrt(adaptive_mse) if adaptive_mse is not None else None,
        "adaptive_mse_se": (
            _standard_error(adaptive_unconditional_squared)
            if adaptive_unconditional_squared is not None
            else None
        ),
        "oracle_pairs": [_jsonable_pair(fold.selected_pair) for fold in folds],
        "oracle_selection_mse": [fold.selection_mse for fold in folds],
        "oracle_mse": oracle_mse,
        "oracle_rmse": math.sqrt(oracle_mse) if oracle_mse is not None else None,
        "oracle_mse_se": (
            _standard_error(oracle_squared) if oracle_squared is not None else None
        ),
        "risk_ratio": risk_ratio,
        "risk_ratio_bootstrap": bootstrap,
        "trim_recovery_vanishing": _wilson_interval(
            trim_hits_vanishing, trim_trials_vanishing
        ),
        "trim_recovery_fixed_005": _wilson_interval(trim_hits_fixed, trim_trials_fixed),
        "failures": failures,
    }


def run_experiment(
    *,
    trials: int,
    sample_sizes: list[int],
    scenarios: list[str],
    contamination_counts: list[int],
    deltas: list[float],
    k_fractions: list[float],
    max_trim: int,
    bootstrap_draws: int,
) -> list[dict[str, Any]]:
    """Run the oracle-risk grid."""
    rows = []
    for scenario_key in scenarios:
        scenario = SCENARIOS[scenario_key]
        for n in sample_sizes:
            for contamination_count in contamination_counts:
                rows.extend(
                    [
                        _evaluate_cell(
                            scenario,
                            n=n,
                            contamination_count=contamination_count,
                            delta=delta,
                            trials=trials,
                            k_fractions=k_fractions,
                            max_trim=max(max_trim, contamination_count),
                            bootstrap_draws=bootstrap_draws,
                        )
                        for delta in deltas
                    ]
                )
    return rows


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _format_optional(value: float | None, width: int, digits: int = 4) -> str:
    if value is None:
        return f"{'invalid':>{width}}"
    return f"{value:>{width}.{digits}f}"


def _print_rows(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'scenario':<18}{'n':>7}{'r':>4}{'delta':>8}{'adapt':>10}"
        f"{'fail%':>8}{'oracle':>10}{'ratio':>10}{'trim_an':>9}"
        f"{'oracle folds':>18}"
    )
    print(header)
    for row in rows:
        pair_text = ",".join(
            "-" if pair is None else f"({pair[0]},{pair[1]})"
            for pair in row["oracle_pairs"]
        )
        trim = row["trim_recovery_vanishing"]["estimate"]
        print(
            f"{row['scenario']:<18}{row['n']:>7}{row['contamination_count']:>4}"
            f"{row['delta']:>8.2f}"
            f"{_format_optional(row['adaptive_rmse'], 10)}"
            f"{100.0 * row['adaptive_failure_rate']:>8.1f}"
            f"{_format_optional(row['oracle_rmse'], 10)}"
            f"{_format_optional(row['risk_ratio'], 10, digits=3)}"
            f"{_format_optional(trim, 9, digits=3)}{pair_text:>18}"
        )


def _git_commit() -> str | None:
    """Return the checked-out commit without invoking git."""
    git_entry = ROOT / ".git"
    try:
        if git_entry.is_file():
            content = git_entry.read_text(encoding="utf-8").strip()
            if content.startswith("gitdir:"):
                git_dir = (ROOT / content.removeprefix("gitdir:").strip()).resolve()
            else:
                return None
        else:
            git_dir = git_entry

        head = git_dir / "HEAD"
        content = head.read_text(encoding="utf-8").strip()
        if not content.startswith("ref:"):
            return content or None
        ref = content.removeprefix("ref:").strip()
        ref_file = git_dir / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {ref}"):
                    return line.split()[0]
    except OSError:
        return None
    return None


def _provenance() -> dict[str, Any]:
    try:
        version = importlib.metadata.version("heavytails")
    except importlib.metadata.PackageNotFoundError:
        version = "0.0.0.dev0"
    return {
        "heavytails_version": version,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
    }


def build_report(
    *,
    trials: int,
    sample_sizes: list[int],
    scenario_keys: list[str],
    contamination_counts: list[int],
    deltas: list[float],
    k_fractions: list[float],
    max_trim: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    """Build the JSON-serializable experiment report."""
    rows = run_experiment(
        trials=trials,
        sample_sizes=sample_sizes,
        scenarios=scenario_keys,
        contamination_counts=contamination_counts,
        deltas=deltas,
        k_fractions=k_fractions,
        max_trim=max_trim,
        bootstrap_draws=bootstrap_draws,
    )
    return {
        "provenance": _provenance(),
        "configuration": {
            "trials": trials,
            "sample_sizes": sample_sizes,
            "scenarios": scenario_keys,
            "contamination_counts": contamination_counts,
            "deltas": deltas,
            "k_fractions": k_fractions,
            "max_trim": max_trim,
            "bootstrap_draws": bootstrap_draws,
            "seeds": f"0..{trials - 1} per cell",
            "oracle": "two-fold Monte Carlo select/evaluate rotation",
            "adaptive_failures": (
                "primary risk ratio is null when any adaptive replication fails"
            ),
        },
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--sample-sizes", default="1000,5000,10000")
    parser.add_argument(
        "--scenarios",
        default="pareto,frechet,burr_rho_half,burr_rho_quarter,hall_rho_half",
        help=f"Comma-separated keys from: {', '.join(SCENARIOS)}",
    )
    parser.add_argument("--contamination-counts", default="0,1,3,5")
    parser.add_argument("--deltas", default="1.5,2,3,5,10")
    parser.add_argument(
        "--k-fractions",
        default="0.02,0.05,0.10",
        help="Envelope fractions; the exact logarithmic adaptive grid is used.",
    )
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if args.trials < 2:
        raise SystemExit("at least two trials are needed for split oracle evaluation")

    scenario_keys = [part for part in args.scenarios.split(",") if part]
    unknown = sorted(set(scenario_keys) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")

    report = build_report(
        trials=args.trials,
        sample_sizes=_parse_ints(args.sample_sizes),
        scenario_keys=scenario_keys,
        contamination_counts=_parse_ints(args.contamination_counts),
        deltas=_parse_floats(args.deltas),
        k_fractions=_parse_floats(args.k_fractions),
        max_trim=args.max_trim,
        bootstrap_draws=args.bootstrap_draws,
    )
    _print_rows(report["results"])
    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
