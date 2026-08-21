"""Measure the accuracy of the special functions against arbitrary precision.

The library implements the regularized incomplete beta and the regularized
lower incomplete gamma from the standard library alone. This script sweeps the
parameter ranges the distributions actually use and reports the worst relative
error against `mpmath` evaluated at 50 decimal digits, which is far beyond
double precision and so serves as exact.

Run it directly to reproduce the numbers quoted in the validation studies page:

    poetry run python scripts/special_function_accuracy.py

The same bounds are asserted by ``tests/test_special_accuracy.py``, so a
regression fails the build rather than only changing a number in the docs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

from mpmath import mp, mpf

from heavytails._special import _betainc_reg, _gammainc_lower_reg

# Well beyond double precision, so the reference is exact for our purposes.
mp.dps = 50

# Shape parameters spanning what the distributions use. The beta function is
# reached through Student-t (a = nu/2, b = 1/2) and Beta-Prime; the gamma
# function through Inverse-Gamma.
BETA_SHAPES: list[tuple[float, float]] = [
    (0.5, 0.5),
    (0.5, 1.5),
    (1.0, 1.0),
    (1.5, 0.5),
    (2.0, 3.0),
    (5.0, 0.5),
    (0.05, 0.05),
    (15.0, 0.5),
    (50.0, 0.5),
    (100.0, 0.5),
    (0.001, 2.0),
    (1000.0, 0.5),
]

GAMMA_SHAPES: list[float] = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0]

# Arguments concentrated near the endpoints, where cancellation bites.
BETA_ARGS: list[float] = [
    1e-12,
    1e-8,
    1e-4,
    0.01,
    0.1,
    0.25,
    0.4,
    0.5,
    0.6,
    0.75,
    0.9,
    0.99,
    1 - 1e-4,
    1 - 1e-8,
    1 - 1e-12,
]


@dataclass
class Worst:
    """Worst observed relative error and where it occurred."""

    error: float = 0.0
    where: str = ""

    def update(self, error: float, where: str) -> None:
        if error > self.error:
            self.error = error
            self.where = where


def _relative_error(got: float, exact: Any) -> float:
    """Relative error, falling back to absolute error near zero."""
    exact_f = float(exact)
    if exact_f == 0.0:
        return abs(got)
    return abs(got - exact_f) / abs(exact_f)


def audit_incomplete_beta() -> dict[str, Any]:
    """Sweep the regularized incomplete beta against mpmath."""
    worst = Worst()
    worst_by_region = {"series": Worst(), "continued_fraction": Worst()}
    samples = 0

    for a, b in BETA_SHAPES:
        for x in BETA_ARGS:
            got = _betainc_reg(a, b, x)
            exact = mp.betainc(mpf(a), mpf(b), 0, mpf(x), regularized=True)
            err = _relative_error(got, exact)
            where = f"a={a}, b={b}, x={x}"
            worst.update(err, where)
            samples += 1

            # The implementation flips to the mirrored problem above this point.
            region = "continued_fraction" if x > (a + 1.0) / (a + b + 2.0) else "series"
            worst_by_region[region].update(err, where)

    return {
        "function": "regularized incomplete beta",
        "samples": samples,
        "worst_relative_error": worst.error,
        "worst_at": worst.where,
        "by_region": {
            name: {"worst_relative_error": w.error, "worst_at": w.where}
            for name, w in worst_by_region.items()
        },
    }


def audit_incomplete_gamma() -> dict[str, Any]:
    """Sweep the regularized lower incomplete gamma against mpmath."""
    worst = Worst()
    worst_by_region = {"series": Worst(), "continued_fraction": Worst()}
    samples = 0

    for a in GAMMA_SHAPES:
        # Cover both sides of the x < a + 1 switch, plus the far tail.
        xs = [
            1e-6 * a,
            0.1 * a,
            0.5 * a,
            a,
            a + 0.5,
            a + 1.0,
            a + 2.0,
            2.0 * a,
            5.0 * a,
            20.0 * a,
        ]
        for x in xs:
            if x <= 0.0:
                continue
            got = _gammainc_lower_reg(a, x)
            exact = mp.gammainc(mpf(a), 0, mpf(x), regularized=True)
            err = _relative_error(got, exact)
            where = f"a={a}, x={x:.6g}"
            worst.update(err, where)
            samples += 1

            region = "series" if x < a + 1.0 else "continued_fraction"
            worst_by_region[region].update(err, where)

    return {
        "function": "regularized lower incomplete gamma",
        "samples": samples,
        "worst_relative_error": worst.error,
        "worst_at": worst.where,
        "by_region": {
            name: {"worst_relative_error": w.error, "worst_at": w.where}
            for name, w in worst_by_region.items()
        },
    }


def audit_switch_point() -> dict[str, Any]:
    """Check that accuracy does not degrade where the method changes.

    A badly placed switch shows up as a spike in error on one side of the
    boundary. This compares the error just below and just above it.
    """
    beta_below = Worst()
    beta_above = Worst()
    for a, b in BETA_SHAPES:
        boundary = (a + 1.0) / (a + b + 2.0)
        for delta, bucket in ((-1e-6, beta_below), (1e-6, beta_above)):
            x = boundary + delta
            if not (0.0 < x < 1.0):
                continue
            got = _betainc_reg(a, b, x)
            exact = mp.betainc(mpf(a), mpf(b), 0, mpf(x), regularized=True)
            bucket.update(_relative_error(got, exact), f"a={a}, b={b}, x={x:.9g}")

    gamma_below = Worst()
    gamma_above = Worst()
    for a in GAMMA_SHAPES:
        for delta, bucket in ((-1e-6, gamma_below), (1e-6, gamma_above)):
            x = a + 1.0 + delta
            got = _gammainc_lower_reg(a, x)
            exact = mp.gammainc(mpf(a), 0, mpf(x), regularized=True)
            bucket.update(_relative_error(got, exact), f"a={a}, x={x:.9g}")

    return {
        "incomplete_beta": {
            "just_below_switch": beta_below.error,
            "just_above_switch": beta_above.error,
        },
        "incomplete_gamma": {
            "just_below_switch": gamma_below.error,
            "just_above_switch": gamma_above.error,
        },
    }


def run_audit() -> dict[str, Any]:
    """Run every sweep and return the combined report."""
    return {
        "mpmath_decimal_digits": mp.dps,
        "incomplete_beta": audit_incomplete_beta(),
        "incomplete_gamma": audit_incomplete_gamma(),
        "switch_point": audit_switch_point(),
    }


def main() -> int:
    """Run the audit from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path, default=None, help="Write the full report as JSON"
    )
    args = parser.parse_args()

    report = run_audit()

    for key in ("incomplete_beta", "incomplete_gamma"):
        section = report[key]
        print(f"\n{section['function']} ({section['samples']} points)")
        print(f"  worst relative error: {section['worst_relative_error']:.3e}")
        print(f"  at:                   {section['worst_at']}")
        for name, stats in section["by_region"].items():
            print(f"  {name:<20} {stats['worst_relative_error']:.3e}")

    print("\nswitch point")
    for key, stats in report["switch_point"].items():
        print(
            f"  {key:<18} below {stats['just_below_switch']:.3e}"
            f"   above {stats['just_above_switch']:.3e}"
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
