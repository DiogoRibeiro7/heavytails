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

import heavytails
from heavytails import Frechet, Pareto
from heavytails.tail_index import (
    adaptive_trimmed_hill_estimator,
    bias_reduced_hill_estimator,
    generalized_hill_estimator,
    gpd_mle_estimator,
    harmonic_moment_estimator,
    hill_estimator,
    moment_estimator,
    pickands_estimator,
    smoothed_hill_estimator,
    t_hill_estimator,
    trimmed_hill_estimator,
)

Estimator = Callable[[list[float], int], float]

ESTIMATORS: dict[str, Estimator] = {
    "hill": hill_estimator,
    "generalized_hill": generalized_hill_estimator,
    "smoothed_hill_u2": lambda d, k: smoothed_hill_estimator(d, k, u=2.0),
    "smoothed_hill_u3": lambda d, k: smoothed_hill_estimator(d, k, u=3.0),
    "trimmed_hill_r5": lambda d, k: trimmed_hill_estimator(d, k, r=5),
    "adaptive_trimmed_hill": adaptive_trimmed_hill_estimator,
    "t_hill": t_hill_estimator,
    "harmonic_beta2": lambda d, k: harmonic_moment_estimator(d, k, beta=2.0),
    "moment": lambda d, k: moment_estimator(d, k)[0],
    "pickands": pickands_estimator,
    "gpd_mle": gpd_mle_estimator,
    # rho = -1 is the canonical choice; estimating it per sample is both
    # slow and unstable, and the study is about the tail index itself.
    "bias_reduced_hill": lambda d, k: bias_reduced_hill_estimator(d, k, rho=-1.0),
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


def _git_commit() -> str | None:
    """Return the checked-out commit, or None if this is not a git checkout.

    The package version alone is not reliable provenance for a run made from a
    working tree: ``importlib.metadata`` reports what is *installed*, which
    silently lags behind after a version bump in an editable install. The
    commit is unambiguous.

    Read from the .git directory directly rather than by invoking git, so this
    needs no subprocess and no git on PATH.
    """
    git_dir = Path(__file__).resolve().parent.parent / ".git"
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    try:
        content = head.read_text(encoding="utf-8").strip()
        if not content.startswith("ref:"):
            return content or None  # detached HEAD
        ref = content.removeprefix("ref:").strip()
        ref_file = git_dir / ref
        if ref_file.is_file():
            return ref_file.read_text(encoding="utf-8").strip() or None
        # Packed refs, which is how a freshly cloned repository stores them.
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.endswith(f" {ref}"):
                    return line.split()[0]
    except OSError:
        return None
    return None


def _provenance(trials: int) -> dict[str, Any]:
    """Describe the run, so a results file can be traced back to its code.

    A table of numbers with no record of the version that produced it
    cannot be reproduced or cited, which matters as soon as the numbers
    leave this repository.

    Both the package version and the git commit are recorded. The version
    comes from the installed distribution metadata and can lag a working tree
    after a version bump; the commit cannot.
    """
    return {
        "heavytails_version": heavytails.__version__,
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
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
