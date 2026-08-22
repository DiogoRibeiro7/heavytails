"""Measure what the NumPy fast path is worth, per family and per method.

The complexity of a second evaluation path has to be paid for by a number, and
the number has to be the one a user would see rather than a microbenchmark of
the inner expression. So this times the public calls: a plain Python loop over
the scalar method against :mod:`heavytails.vectorized` on the same points.

It also reports the families that have no kernel, where the honest answer is
that there is no speedup and cannot be one. NumPy has neither the error
function nor the incomplete beta and gamma, so LogNormal, StudentT,
InverseGamma and BetaPrime fall back to the loop. Reporting them as 1.0x rather
than omitting them is the point: a caller sizing a job needs to know which
half they are in.

Usage::

    poetry run python scripts/vectorization_benchmark.py
    poetry run python scripts/vectorization_benchmark.py --n 1000000 --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from heavytails import (
    Cauchy,
    Frechet,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)
from heavytails.extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
)
from heavytails.vectorized import accelerated, cdf, pdf, ppf, sf

FAMILIES: list[Any] = [
    Pareto(alpha=2.5, xm=1.0),
    Cauchy(x0=0.0, gamma=1.0),
    Weibull(k=0.7, lam=2.0),
    Frechet(alpha=2.0, s=1.0, m=0.0),
    GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0),
    GeneralizedPareto(xi=0.4, sigma=1.0, mu=0.0),
    BurrXII(c=2.0, k=1.5, s=1.0),
    LogLogistic(kappa=2.0, lam=1.0),
    LogNormal(mu=0.0, sigma=1.0),
    StudentT(nu=3.0),
    InverseGamma(alpha=2.0, beta=1.0),
    BetaPrime(a=2.0, b=3.0, s=1.0),
]

VECTORIZED = {"pdf": pdf, "cdf": cdf, "sf": sf, "ppf": ppf}


def _best_of(call: Any, repeats: int = 3) -> float:
    """Fastest of several runs.

    The minimum rather than the mean: the slow runs are the machine doing
    something else, and including them measures the machine.
    """
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def _measure(dist: Any, method: str, points: list[float]) -> dict[str, Any]:
    scalar = getattr(dist, method)
    loop = _best_of(lambda: [scalar(value) for value in points])
    fast = _best_of(lambda: VECTORIZED[method](dist, points))
    return {
        "family": type(dist).__name__,
        "method": method,
        "accelerated": accelerated(dist, method),
        "loop_ms": loop * 1e3,
        "fast_ms": fast * 1e3,
        "speedup": loop / fast,
    }


def _run(n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dist in FAMILIES:
        sample = dist.rvs(n, seed=1)
        probabilities = [(i + 0.5) / n for i in range(n)]
        rows.extend(_measure(dist, method, sample) for method in ("pdf", "cdf", "sf"))
        rows.append(_measure(dist, "ppf", probabilities))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100_000, help="Points per call.")
    parser.add_argument("--json", type=Path, help="Write the results as JSON.")
    args = parser.parse_args()

    try:
        import numpy  # noqa: PLC0415

        numpy_version = numpy.__version__
    except ModuleNotFoundError:
        print("NumPy is not installed, so there is nothing to measure.")
        return

    print(
        f"heavytails vectorisation, {args.n:,} points per call\n"
        f"NumPy {numpy_version} on {platform.python_version()} / {platform.platform()}\n"
    )
    rows = _run(args.n)

    header = f"{'family':<20}{'method':<7}{'loop ms':>10}{'fast ms':>10}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    for row in rows:
        marker = "" if row["accelerated"] else "  (no kernel)"
        print(
            f"{row['family']:<20}{row['method']:<7}"
            f"{row['loop_ms']:>10.1f}{row['fast_ms']:>10.2f}"
            f"{row['speedup']:>9.1f}x{marker}"
        )

    fast_rows = [r for r in rows if r["accelerated"]]
    if fast_rows:
        speeds = sorted(r["speedup"] for r in fast_rows)
        print(
            f"\nAccelerated calls: {len(fast_rows)} of {len(rows)}. "
            f"Speedup {speeds[0]:.1f}x to {speeds[-1]:.1f}x, "
            f"median {speeds[len(speeds) // 2]:.1f}x."
        )
    without = sorted({r["family"] for r in rows if not r["accelerated"]})
    if without:
        print(
            f"No kernel: {', '.join(without)} -- NumPy has no erf, betainc or gammainc."
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "n": args.n,
                    "numpy": numpy_version,
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
