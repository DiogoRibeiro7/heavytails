"""Compare the tail index estimators across known indices and sample sizes.

An estimator is a statement about a sampling distribution, so a single
evaluation says nothing useful about it. This script draws many samples from
distributions with a known extreme-value index and reports the bias, standard
deviation and root mean squared error of each estimator, so the trade-offs
between them can be read off rather than asserted.

    poetry run python scripts/tail_index_study.py
    poetry run python scripts/tail_index_study.py --trials 500 --json study.json

The numbers quoted in the validation studies page come from the default run.
"""

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

from heavytails import Frechet, Pareto
from heavytails.tail_index import (
    generalized_hill_estimator,
    hill_estimator,
    moment_estimator,
    pickands_estimator,
    smoothed_hill_estimator,
)

Estimator = Callable[[list[float], int], float]

ESTIMATORS: dict[str, Estimator] = {
    "hill": hill_estimator,
    "generalized_hill": generalized_hill_estimator,
    "smoothed_hill_u2": lambda d, k: smoothed_hill_estimator(d, k, u=2.0),
    "smoothed_hill_u3": lambda d, k: smoothed_hill_estimator(d, k, u=3.0),
    "moment": lambda d, k: moment_estimator(d, k)[0],
    "pickands": pickands_estimator,
}


@dataclass(frozen=True)
class Scenario:
    """A distribution with a known extreme-value index."""

    name: str
    gamma: float
    sample: Callable[[int, int], list[float]]


def _uniform_sample(n: int, seed: int) -> list[float]:
    """Uniform(0,1): a finite upper endpoint, so gamma = -1."""
    rnd = random.Random(seed)
    return [rnd.random() for _ in range(n)]


SCENARIOS: list[Scenario] = [
    Scenario(
        "Pareto(alpha=1.5)",
        1.0 / 1.5,
        lambda n, s: Pareto(alpha=1.5, xm=1.0).rvs(n, seed=s),
    ),
    Scenario(
        "Pareto(alpha=2)",
        0.5,
        lambda n, s: Pareto(alpha=2.0, xm=1.0).rvs(n, seed=s),
    ),
    Scenario(
        "Pareto(alpha=4)",
        0.25,
        lambda n, s: Pareto(alpha=4.0, xm=1.0).rvs(n, seed=s),
    ),
    Scenario(
        "Frechet(alpha=2)",
        0.5,
        lambda n, s: Frechet(alpha=2.0, s=1.0, m=0.0).rvs(n, seed=s),
    ),
    Scenario("Uniform(0,1)", -1.0, _uniform_sample),
]

SAMPLE_SIZES = [1000, 10000]


def _summarise(values: list[float], truth: float) -> dict[str, float]:
    """Bias, standard deviation and RMSE of a set of estimates."""
    n = len(values)
    mean = sum(values) / n
    bias = mean - truth
    variance = sum((v - mean) ** 2 for v in values) / max(n - 1, 1)
    mse = sum((v - truth) ** 2 for v in values) / n
    return {
        "trials": n,
        "mean": mean,
        "bias": bias,
        "std": math.sqrt(variance),
        "rmse": math.sqrt(mse),
    }


def run_study(trials: int) -> list[dict[str, Any]]:
    """Run every estimator over every scenario and sample size."""
    rows: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for n in SAMPLE_SIZES:
            # A fixed fraction of the sample, which is the usual rule of thumb
            # and keeps the comparison fair across sizes.
            k = max(10, n // 20)
            samples = [scenario.sample(n, seed) for seed in range(trials)]
            for name, estimate in ESTIMATORS.items():
                values = []
                failures = 0
                for data in samples:
                    try:
                        values.append(estimate(data, k))
                    except (ValueError, ZeroDivisionError, OverflowError):
                        failures += 1
                if not values:
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "gamma": scenario.gamma,
                            "n": n,
                            "k": k,
                            "estimator": name,
                            "failures": failures,
                            "error": "no usable estimates",
                        }
                    )
                    continue
                rows.append(
                    {
                        "scenario": scenario.name,
                        "gamma": scenario.gamma,
                        "n": n,
                        "k": k,
                        "estimator": name,
                        "failures": failures,
                        **_summarise(values, scenario.gamma),
                    }
                )
    return rows


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Print the study as a readable table, grouped by scenario."""
    header = (
        f"{'estimator':<20}{'mean':>10}{'bias':>10}{'std':>10}{'rmse':>10}{'fails':>7}"
    )
    current = None
    for row in rows:
        key = (row["scenario"], row["n"])
        if key != current:
            current = key
            print(
                f"\n{row['scenario']}   gamma = {row['gamma']:+.4f}   "
                f"n = {row['n']}   k = {row['k']}"
            )
            print(header)
        if "error" in row:
            print(
                f"{row['estimator']:<20}{'-':>10}{'-':>10}{'-':>10}{'-':>10}"
                f"{row['failures']:>7}"
            )
            continue
        print(
            f"{row['estimator']:<20}{row['mean']:>10.4f}{row['bias']:>+10.4f}"
            f"{row['std']:>10.4f}{row['rmse']:>10.4f}{row['failures']:>7}"
        )


def main() -> int:
    """Run the study from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=200, help="Samples per scenario (default: 200)"
    )
    parser.add_argument("--json", type=Path, default=None, help="Write results as JSON")
    args = parser.parse_args()

    rows = run_study(args.trials)
    _print_table(rows)

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
