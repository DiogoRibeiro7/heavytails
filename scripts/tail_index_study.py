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
from typing import Any, Literal

from _provenance import base_provenance
from heavytails import Frechet, Pareto
from heavytails.tail_index import (
    adaptive_trimmed_hill_estimator,
    bias_reduced_hill_estimator,
    generalized_hill_estimator,
    gpd_mle_estimator,
    harmonic_moment_estimator,
    hill_estimator,
    moment_estimator,
    orthogonalized_bias_reduced_hill_estimator,
    pickands_estimator,
    smoothed_hill_estimator,
    t_hill_estimator,
    threshold_averaged_orthogonalized_hill_estimator,
    trimmed_hill_estimator,
)

Estimator = Callable[[list[float], int], float]
GammaDomain = Literal["positive", "any"]


@dataclass(frozen=True)
class EstimatorSpec:
    """Estimator plus the extreme-value-index range it is meant to estimate."""

    estimate: Estimator
    gamma_domain: GammaDomain


ESTIMATORS: dict[str, EstimatorSpec] = {
    "hill": EstimatorSpec(hill_estimator, "positive"),
    "generalized_hill": EstimatorSpec(generalized_hill_estimator, "any"),
    "smoothed_hill_u2": EstimatorSpec(
        lambda d, k: smoothed_hill_estimator(d, k, u=2.0), "positive"
    ),
    "smoothed_hill_u3": EstimatorSpec(
        lambda d, k: smoothed_hill_estimator(d, k, u=3.0), "positive"
    ),
    "trimmed_hill_r5": EstimatorSpec(
        lambda d, k: trimmed_hill_estimator(d, k, r=5), "positive"
    ),
    "adaptive_trimmed_hill": EstimatorSpec(adaptive_trimmed_hill_estimator, "positive"),
    "t_hill": EstimatorSpec(t_hill_estimator, "positive"),
    "harmonic_beta2": EstimatorSpec(
        lambda d, k: harmonic_moment_estimator(d, k, beta=2.0), "positive"
    ),
    "moment": EstimatorSpec(lambda d, k: moment_estimator(d, k)[0], "any"),
    "pickands": EstimatorSpec(pickands_estimator, "any"),
    "gpd_mle": EstimatorSpec(gpd_mle_estimator, "any"),
    # rho = -1 is the canonical choice; estimating it per sample is both
    # slow and unstable, and the study is about the tail index itself.
    "bias_reduced_hill": EstimatorSpec(
        lambda d, k: bias_reduced_hill_estimator(d, k, rho=-1.0), "positive"
    ),
    "orthogonalized_br_hill": EstimatorSpec(
        lambda d, k: orthogonalized_bias_reduced_hill_estimator(d, k, rho=-1.0),
        "positive",
    ),
    "threshold_avg_orthogonalized": EstimatorSpec(
        lambda d, k: threshold_averaged_orthogonalized_hill_estimator(
            d, k, min_k=max(10, k // 4), grid_size=6, rho=-1.0, adaptive_trim=True
        ),
        "positive",
    ),
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


def _contaminate(sample: list[float], count: int, magnitude: float) -> list[float]:
    """Replace the ``count`` largest values with outliers.

    Robustness is invisible on clean data: trimming five observations from a
    clean Pareto sample moves the standard deviation from 0.0296 to 0.0302.
    It only shows up once something has gone wrong with the data.
    """
    ordered = sorted(sample, reverse=True)
    for i in range(count):
        ordered[i] = magnitude / (i + 1)
    return ordered


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
    Scenario(
        "Pareto(alpha=2) + 3 outliers",
        0.5,
        lambda n, s: _contaminate(
            Pareto(alpha=2.0, xm=1.0).rvs(n, seed=s), count=3, magnitude=1e9
        ),
    ),
    Scenario(
        "Pareto(alpha=2) + 10 outliers",
        0.5,
        lambda n, s: _contaminate(
            Pareto(alpha=2.0, xm=1.0).rvs(n, seed=s), count=10, magnitude=1e9
        ),
    ),
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


def _supports_scenario(estimator: EstimatorSpec, scenario: Scenario) -> bool:
    """Return whether an estimator is meaningful for a scenario's tail index."""
    return estimator.gamma_domain == "any" or scenario.gamma > 0.0


def _provenance(trials: int) -> dict[str, Any]:
    """Describe the run, so a results file can be traced back to its code.

    A table of numbers with no record of the version that produced it
    cannot be reproduced or cited, which matters as soon as the numbers
    leave this repository.

    The version, where it came from, and the git commit are all recorded. The
    commit is the unambiguous one; the version is what a reader will quote.
    """
    return {
        **base_provenance(Path(__file__).resolve().parent.parent),
        "trials": trials,
        "sample_sizes": list(SAMPLE_SIZES),
        "estimators": sorted(ESTIMATORS),
        "scenarios": [sc.name for sc in SCENARIOS],
        "seeds": f"0..{trials - 1} per scenario and sample size",
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
            for name, estimator in ESTIMATORS.items():
                if not _supports_scenario(estimator, scenario):
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "gamma": scenario.gamma,
                            "n": n,
                            "k": k,
                            "estimator": name,
                            "failures": 0,
                            "skipped": True,
                            "reason": "requires gamma > 0",
                        }
                    )
                    continue
                values = []
                failures = 0
                for data in samples:
                    try:
                        values.append(estimator.estimate(data, k))
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
    name_width = max(20, *(len(row["estimator"]) for row in rows))
    header = (
        f"{'estimator':<{name_width}}{'mean':>10}{'bias':>10}{'std':>10}"
        f"{'rmse':>10}{'fails':>7}"
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
        if row.get("skipped"):
            print(
                f"{row['estimator']:<{name_width}}{'N/A':>10}{'N/A':>10}"
                f"{'N/A':>10}{'N/A':>10}{'N/A':>7}"
            )
            continue
        if "error" in row:
            print(
                f"{row['estimator']:<{name_width}}{'-':>10}{'-':>10}{'-':>10}"
                f"{'-':>10}{row['failures']:>7}"
            )
            continue
        print(
            f"{row['estimator']:<{name_width}}{row['mean']:>10.4f}"
            f"{row['bias']:>+10.4f}{row['std']:>10.4f}"
            f"{row['rmse']:>10.4f}{row['failures']:>7}"
        )


def main() -> int:
    """Run the study from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trials", type=int, default=200, help="Samples per scenario (default: 200)"
    )
    parser.add_argument("--json", type=Path, default=None, help="Write results as JSON")
    args = parser.parse_args()

    prov = _provenance(args.trials)
    print(
        f"heavytails {prov['heavytails_version']} on Python "
        f"{prov['python_version']}, {args.trials} trials per scenario"
    )

    rows = run_study(args.trials)
    _print_table(rows)

    if args.json:
        report = {"provenance": _provenance(args.trials), "results": rows}
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
