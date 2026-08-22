"""Oracle-risk experiment for the adaptive tail estimator.

This is deliberately separate from ``scripts/tail_index_study.py``. The generic
benchmark asks how named estimators behave at a fixed externally chosen
threshold. This experiment asks the research question instead: how far is the
adaptive estimator from the empirical oracle over contaminated, second-order
``(r, k)`` candidates?
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from heavytails import BurrXII, Frechet, Pareto
from heavytails.tail_index import (
    adaptive_trim_selection,
    orthogonalized_bias_reduced_hill_estimator,
    threshold_averaged_orthogonalized_hill_estimator,
)


Sampler = Callable[[int, int], list[float]]


@dataclass(frozen=True)
class Scenario:
    """Sampling law with known EVI and second-order shape."""

    key: str
    label: str
    gamma: float
    rho: float
    sample: Sampler


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
        "Pareto(alpha=2)",
        0.5,
        -1.0,
        lambda n, seed: Pareto(alpha=2.0, xm=1.0).rvs(n, seed=seed),
    ),
    "frechet": Scenario(
        "frechet",
        "Frechet(alpha=2)",
        0.5,
        -1.0,
        lambda n, seed: Frechet(alpha=2.0, s=1.0, m=0.0).rvs(n, seed=seed),
    ),
    "burr_rho_half": Scenario(
        "burr_rho_half",
        "BurrXII(gamma=0.5,rho=-0.5)",
        0.5,
        -0.5,
        lambda n, seed: BurrXII(c=1.0, k=2.0, s=1.0).rvs(n, seed=seed),
    ),
    "burr_rho_quarter": Scenario(
        "burr_rho_quarter",
        "BurrXII(gamma=0.5,rho=-0.25)",
        0.5,
        -0.25,
        lambda n, seed: BurrXII(c=0.5, k=4.0, s=1.0).rvs(n, seed=seed),
    ),
    "hall_rho_half": Scenario(
        "hall_rho_half",
        "Hall(gamma=0.5,rho=-0.5)",
        0.5,
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


def _rmse(estimates: list[float], truth: float) -> float:
    return math.sqrt(sum((value - truth) ** 2 for value in estimates) / len(estimates))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _evaluate_cell(
    scenario: Scenario,
    *,
    n: int,
    contamination_count: int,
    delta: float,
    trials: int,
    k_fractions: list[float],
    max_trim: int,
) -> dict[str, Any]:
    """Evaluate one scenario/contamination/sample-size cell."""
    k_grid = sorted(
        {max(10, min(n - 2, round(n * fraction))) for fraction in k_fractions}
    )
    min_k = k_grid[0]
    max_k = k_grid[-1]
    r_grid = list(range(max_trim + 1))
    adaptive_max_trim = min(max_trim, max(1, min_k // 2 - 1))

    adaptive: list[float] = []
    trim_hits = 0
    trim_trials = 0
    oracle_estimates: dict[tuple[int, int], list[float]] = {
        (r, k): [] for r in r_grid for k in k_grid if r < k - 1
    }
    failures = {"adaptive": 0, "trim_selection": 0, "oracle": 0}

    for seed in range(trials):
        clean = scenario.sample(n, seed)
        data = _contaminate(clean, contamination_count, delta)

        try:
            adaptive.append(
                threshold_averaged_orthogonalized_hill_estimator(
                    data,
                    max_k,
                    min_k=min_k,
                    grid_size=len(k_grid),
                    rho=scenario.rho,
                    adaptive_trim=True,
                    max_trim=adaptive_max_trim,
                )
            )
        except ValueError:
            failures["adaptive"] += 1

        try:
            selection = adaptive_trim_selection(data, max_k, max_trim=adaptive_max_trim)
            trim_hits += int(selection["trim"] == contamination_count)
            trim_trials += 1
        except ValueError:
            failures["trim_selection"] += 1

        for candidate, estimates in oracle_estimates.items():
            r, k = candidate
            try:
                estimates.append(
                    orthogonalized_bias_reduced_hill_estimator(
                        data, k, r=r, rho=scenario.rho
                    )
                )
            except ValueError:
                failures["oracle"] += 1

    usable_oracle = {
        candidate: estimates
        for candidate, estimates in oracle_estimates.items()
        if len(estimates) == trials
    }
    if not adaptive:
        adaptive_rmse = math.inf
        adaptive_mean = math.nan
    else:
        adaptive_rmse = _rmse(adaptive, scenario.gamma)
        adaptive_mean = _mean(adaptive)

    if usable_oracle:
        oracle_pair, oracle_values = min(
            usable_oracle.items(), key=lambda item: _rmse(item[1], scenario.gamma)
        )
        oracle_rmse = _rmse(oracle_values, scenario.gamma)
        oracle_mean = _mean(oracle_values)
    else:
        oracle_pair = None
        oracle_rmse = math.inf
        oracle_mean = math.nan

    return {
        "scenario": scenario.key,
        "label": scenario.label,
        "gamma": scenario.gamma,
        "rho": scenario.rho,
        "n": n,
        "contamination_count": contamination_count,
        "delta": delta,
        "trials": trials,
        "k_grid": k_grid,
        "r_grid": r_grid,
        "adaptive_mean": adaptive_mean,
        "adaptive_rmse": adaptive_rmse,
        "oracle_pair": oracle_pair,
        "oracle_mean": oracle_mean,
        "oracle_rmse": oracle_rmse,
        "risk_ratio": (
            math.inf
            if not math.isfinite(adaptive_rmse) or oracle_rmse == 0.0
            else (adaptive_rmse * adaptive_rmse) / (oracle_rmse * oracle_rmse)
        ),
        "trim_recovery": math.nan if trim_trials == 0 else trim_hits / trim_trials,
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
                        )
                        for delta in deltas
                    ]
                )
    return rows


def _parse_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_floats(value: str) -> list[float]:
    return [float(part) for part in value.split(",") if part]


def _print_rows(rows: list[dict[str, Any]]) -> None:
    header = (
        f"{'scenario':<18}{'n':>7}{'r':>4}{'delta':>8}{'adapt':>10}"
        f"{'oracle':>10}{'ratio':>10}{'trim':>8}{'oracle (r,k)':>15}"
    )
    print(header)
    for row in rows:
        oracle_pair = row["oracle_pair"]
        pair_text = "-" if oracle_pair is None else f"{oracle_pair}"
        print(
            f"{row['scenario']:<18}{row['n']:>7}{row['contamination_count']:>4}"
            f"{row['delta']:>8.2f}{row['adaptive_rmse']:>10.4f}"
            f"{row['oracle_rmse']:>10.4f}{row['risk_ratio']:>10.3f}"
            f"{row['trim_recovery']:>8.3f}{pair_text:>15}"
        )


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
    parser.add_argument("--k-fractions", default="0.02,0.05,0.10")
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    scenario_keys = [part for part in args.scenarios.split(",") if part]
    unknown = sorted(set(scenario_keys) - set(SCENARIOS))
    if unknown:
        raise SystemExit(f"unknown scenarios: {', '.join(unknown)}")

    rows = run_experiment(
        trials=args.trials,
        sample_sizes=_parse_ints(args.sample_sizes),
        scenarios=scenario_keys,
        contamination_counts=_parse_ints(args.contamination_counts),
        deltas=_parse_floats(args.deltas),
        k_fractions=_parse_floats(args.k_fractions),
        max_trim=args.max_trim,
    )
    _print_rows(rows)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
