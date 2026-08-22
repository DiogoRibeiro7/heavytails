"""Build the reference value database from an independent high-precision source.

The library's tests mostly assert that it is *self-consistent*: that the
quantile inverts the distribution function, that the survival function
complements it, that a sample has the moments it should. Those are worth
having, and they cannot catch a formula that is simply wrong, because a
consistently wrong implementation is still self-consistent.

Two bugs in this library's history make the point. ``InverseGamma.cdf`` was
wrong by factors of two to seventeen across the lower tail from 0.1.0 until it
was found, and every property test passed the whole time: the values were
monotone, in [0, 1], and complementary with the survival function. They were
just not the right values. More recently ``InverseGamma.cdf`` returned exactly
zero throughout the lower tail, and the accuracy test that should have caught
it was comparing against a reference that computed ``1 - P`` in double
precision and had itself quantised to multiples of 2**-48.

So this writes a table of values computed by mpmath at 50 decimal digits, from
the mathematical definitions rather than from this library's formulas. The test
that reads it needs no mpmath and does no arithmetic beyond a comparison, which
keeps it fast enough to run everywhere.

Usage::

    poetry run python scripts/generate_reference_values.py
    poetry run python scripts/generate_reference_values.py --out other.json

Regenerate only when adding a family or a grid point. **Regenerating to make a
failing test pass would defeat the entire purpose**: the table is the
independent opinion, and a disagreement means the library changed, not that the
table is stale.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import mpmath as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mp.mp.dps = 50

# Where each family's probability is evaluated. Deliberately weighted towards
# the tails, since that is what this library is for and where a wrong formula
# is least likely to be noticed by eye.
QUANTILE_GRID = [1e-9, 1e-6, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1 - 1e-6]


def _bisect(f: Any, lo: mp.mpf, hi: mp.mpf, iterations: int = 300) -> mp.mpf:
    """Bisect a monotone increasing ``f`` for its root.

    Newton and secant both fail here at the extreme grid points -- mpmath's
    findroot gives up on the Student-t quantile at u = 1e-9 -- and bisection
    cannot. Its cost is irrelevant: this runs once when the table is built,
    never when it is read.
    """
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _bisect_log(f: Any, lo: mp.mpf, hi: mp.mpf, iterations: int = 300) -> mp.mpf:
    """Bisect in log space, for quantiles spanning many orders of magnitude."""
    t = _bisect(lambda t: f(mp.e**t), mp.log(lo), mp.log(hi), iterations)
    return mp.e**t


def _pareto(alpha: float, xm: float) -> dict[str, Any]:
    a, m = mp.mpf(alpha), mp.mpf(xm)
    return {
        "sf": lambda x: (m / x) ** a if x >= m else mp.mpf(1),
        "cdf": lambda x: 1 - (m / x) ** a if x >= m else mp.mpf(0),
        "pdf": lambda x: a * m**a / x ** (a + 1) if x >= m else mp.mpf(0),
        "ppf": lambda u: m / (1 - mp.mpf(u)) ** (1 / a),
    }


def _cauchy(x0: float, gamma: float) -> dict[str, Any]:
    c, g = mp.mpf(x0), mp.mpf(gamma)
    return {
        "cdf": lambda x: mp.atan((x - c) / g) / mp.pi + mp.mpf(1) / 2,
        "sf": lambda x: mp.mpf(1) / 2 - mp.atan((x - c) / g) / mp.pi,
        "pdf": lambda x: 1 / (mp.pi * g * (1 + ((x - c) / g) ** 2)),
        "ppf": lambda u: c + g * mp.tan(mp.pi * (mp.mpf(u) - mp.mpf(1) / 2)),
    }


def _student_t(nu: float) -> dict[str, Any]:
    n = mp.mpf(nu)

    def cdf(x: mp.mpf) -> mp.mpf:
        # I_{n/(n+x^2)}(n/2, 1/2) halved, mirrored about zero.
        z = n / (n + x * x)
        half = mp.betainc(n / 2, mp.mpf(1) / 2, 0, z, regularized=True) / 2
        return half if x <= 0 else 1 - half

    def pdf(x: mp.mpf) -> mp.mpf:
        return (
            mp.gamma((n + 1) / 2)
            / (mp.sqrt(n * mp.pi) * mp.gamma(n / 2))
            * (1 + x * x / n) ** (-(n + 1) / 2)
        )

    return {
        "cdf": cdf,
        "sf": lambda x: 1 - cdf(x),
        "pdf": pdf,
        # -(10**40), not (-10)**40, which is positive and collapses the
        # bracket to a single point.
        "ppf": lambda u: _bisect(
            lambda x: cdf(x) - mp.mpf(u), -(mp.mpf(10) ** 40), mp.mpf(10) ** 40
        ),
    }


def _lognormal(mu: float, sigma: float) -> dict[str, Any]:
    m, s = mp.mpf(mu), mp.mpf(sigma)
    root2 = mp.sqrt(2)
    return {
        "cdf": lambda x: (1 + mp.erf((mp.log(x) - m) / (s * root2))) / 2,
        "sf": lambda x: mp.erfc((mp.log(x) - m) / (s * root2)) / 2,
        "pdf": lambda x: mp.exp(-((mp.log(x) - m) ** 2) / (2 * s * s))
        / (x * s * mp.sqrt(2 * mp.pi)),
        "ppf": lambda u: mp.exp(m + s * root2 * mp.erfinv(2 * mp.mpf(u) - 1)),
    }


def _weibull(k: float, lam: float) -> dict[str, Any]:
    shape, scale = mp.mpf(k), mp.mpf(lam)
    return {
        "cdf": lambda x: 1 - mp.exp(-((x / scale) ** shape)),
        "sf": lambda x: mp.exp(-((x / scale) ** shape)),
        "pdf": lambda x: (shape / scale)
        * (x / scale) ** (shape - 1)
        * mp.exp(-((x / scale) ** shape)),
        "ppf": lambda u: scale * (-mp.log(1 - mp.mpf(u))) ** (1 / shape),
    }


def _frechet(alpha: float, s: float, m: float) -> dict[str, Any]:
    a, scale, loc = mp.mpf(alpha), mp.mpf(s), mp.mpf(m)
    return {
        "cdf": lambda x: mp.exp(-(((x - loc) / scale) ** -a)),
        "sf": lambda x: 1 - mp.exp(-(((x - loc) / scale) ** -a)),
        "pdf": lambda x: (a / scale)
        * ((x - loc) / scale) ** (-1 - a)
        * mp.exp(-(((x - loc) / scale) ** -a)),
        "ppf": lambda u: loc + scale * (-mp.log(mp.mpf(u))) ** (-1 / a),
    }


def _gev_frechet(xi: float, mu: float, sigma: float) -> dict[str, Any]:
    x_i, loc, scale = mp.mpf(xi), mp.mpf(mu), mp.mpf(sigma)

    def t(x: mp.mpf) -> mp.mpf:
        return (1 + x_i * (x - loc) / scale) ** (-1 / x_i)

    return {
        "cdf": lambda x: mp.exp(-t(x)),
        "sf": lambda x: 1 - mp.exp(-t(x)),
        "pdf": lambda x: t(x) ** (x_i + 1) * mp.exp(-t(x)) / scale,
        "ppf": lambda u: loc + scale * ((-mp.log(mp.mpf(u))) ** -x_i - 1) / x_i,
    }


def _gpd(xi: float, sigma: float, mu: float) -> dict[str, Any]:
    x_i, scale, loc = mp.mpf(xi), mp.mpf(sigma), mp.mpf(mu)
    return {
        "cdf": lambda x: 1 - (1 + x_i * (x - loc) / scale) ** (-1 / x_i),
        "sf": lambda x: (1 + x_i * (x - loc) / scale) ** (-1 / x_i),
        "pdf": lambda x: (1 + x_i * (x - loc) / scale) ** (-1 / x_i - 1) / scale,
        "ppf": lambda u: loc + scale * ((1 - mp.mpf(u)) ** -x_i - 1) / x_i,
    }


def _burr(c: float, k: float, s: float) -> dict[str, Any]:
    cc, kk, ss = mp.mpf(c), mp.mpf(k), mp.mpf(s)
    return {
        "cdf": lambda x: 1 - (1 + (x / ss) ** cc) ** -kk,
        "sf": lambda x: (1 + (x / ss) ** cc) ** -kk,
        "pdf": lambda x: (cc * kk / ss)
        * (x / ss) ** (cc - 1)
        * (1 + (x / ss) ** cc) ** (-kk - 1),
        "ppf": lambda u: ss * ((1 - mp.mpf(u)) ** (-1 / kk) - 1) ** (1 / cc),
    }


def _loglogistic(kappa: float, lam: float) -> dict[str, Any]:
    b, a = mp.mpf(kappa), mp.mpf(lam)
    return {
        "cdf": lambda x: 1 / (1 + (x / a) ** -b),
        "sf": lambda x: 1 / (1 + (x / a) ** b),
        "pdf": lambda x: (b / a) * (x / a) ** (b - 1) / (1 + (x / a) ** b) ** 2,
        "ppf": lambda u: a * (mp.mpf(u) / (1 - mp.mpf(u))) ** (1 / b),
    }


def _inverse_gamma(alpha: float, beta: float) -> dict[str, Any]:
    a, b = mp.mpf(alpha), mp.mpf(beta)

    def cdf(x: mp.mpf) -> mp.mpf:
        # Q(a, b/x), taken as the upper integral. Computing it as 1 - P is how
        # both the implementation and its test went wrong before.
        return mp.gammainc(a, b / x, regularized=True)

    return {
        "cdf": cdf,
        "sf": lambda x: mp.gammainc(a, 0, b / x, regularized=True),
        "pdf": lambda x: b**a / mp.gamma(a) * x ** (-a - 1) * mp.exp(-b / x),
        "ppf": lambda u: _bisect_log(
            lambda x: cdf(x) - mp.mpf(u), mp.mpf(10) ** -40, mp.mpf(10) ** 40
        ),
    }


def _beta_prime(a: float, b: float, s: float) -> dict[str, Any]:
    aa, bb, ss = mp.mpf(a), mp.mpf(b), mp.mpf(s)

    def cdf(x: mp.mpf) -> mp.mpf:
        return mp.betainc(aa, bb, 0, x / (x + ss), regularized=True)

    return {
        "cdf": cdf,
        # I_{1-z}(b,a), whose argument s/(x+s) is computed rather than
        # subtracted -- the same reason the implementation does it this way.
        "sf": lambda x: mp.betainc(bb, aa, 0, ss / (x + ss), regularized=True),
        "pdf": lambda x: (x / ss) ** (aa - 1)
        * (1 + x / ss) ** (-aa - bb)
        / (ss * mp.beta(aa, bb)),
        "ppf": lambda u: ss
        * _beta_quantile(aa, bb, u)
        / (1 - _beta_quantile(aa, bb, u)),
    }


def _beta_quantile(a: mp.mpf, b: mp.mpf, u: float) -> mp.mpf:
    """The Beta(a,b) quantile. mpmath has no inverse incomplete beta."""
    return _bisect(
        lambda z: mp.betainc(a, b, 0, z, regularized=True) - mp.mpf(u),
        mp.mpf(0),
        mp.mpf(1),
    )


# Each entry: the class path, its keyword arguments, and the mpmath definitions.
FAMILIES: list[dict[str, Any]] = [
    {"cls": "heavy_tails.Pareto", "kwargs": {"alpha": 2.5, "xm": 1.0}, "ref": _pareto},
    {"cls": "heavy_tails.Pareto", "kwargs": {"alpha": 0.7, "xm": 3.0}, "ref": _pareto},
    {"cls": "heavy_tails.Cauchy", "kwargs": {"x0": 0.0, "gamma": 1.0}, "ref": _cauchy},
    {"cls": "heavy_tails.Cauchy", "kwargs": {"x0": 2.0, "gamma": 0.5}, "ref": _cauchy},
    {"cls": "heavy_tails.StudentT", "kwargs": {"nu": 3.0}, "ref": _student_t},
    {"cls": "heavy_tails.StudentT", "kwargs": {"nu": 1.5}, "ref": _student_t},
    {
        "cls": "heavy_tails.LogNormal",
        "kwargs": {"mu": 0.0, "sigma": 1.0},
        "ref": _lognormal,
    },
    {
        "cls": "heavy_tails.LogNormal",
        "kwargs": {"mu": -0.5, "sigma": 1.8},
        "ref": _lognormal,
    },
    {"cls": "heavy_tails.Weibull", "kwargs": {"k": 0.6, "lam": 2.0}, "ref": _weibull},
    {"cls": "heavy_tails.Weibull", "kwargs": {"k": 1.5, "lam": 1.0}, "ref": _weibull},
    {
        "cls": "heavy_tails.Frechet",
        "kwargs": {"alpha": 2.0, "s": 1.0, "m": 0.0},
        "ref": _frechet,
    },
    {
        "cls": "heavy_tails.GEV_Frechet",
        "kwargs": {"xi": 0.5, "mu": 0.0, "sigma": 1.0},
        "ref": _gev_frechet,
    },
    {
        "cls": "extra_distributions.GeneralizedPareto",
        "kwargs": {"xi": 0.4, "sigma": 1.0, "mu": 0.0},
        "ref": _gpd,
    },
    {
        "cls": "extra_distributions.GeneralizedPareto",
        "kwargs": {"xi": 0.9, "sigma": 2.5, "mu": 1.0},
        "ref": _gpd,
    },
    {
        "cls": "extra_distributions.BurrXII",
        "kwargs": {"c": 2.0, "k": 1.5, "s": 1.0},
        "ref": _burr,
    },
    {
        "cls": "extra_distributions.LogLogistic",
        "kwargs": {"kappa": 2.0, "lam": 1.0},
        "ref": _loglogistic,
    },
    {
        "cls": "extra_distributions.InverseGamma",
        "kwargs": {"alpha": 2.0, "beta": 1.0},
        "ref": _inverse_gamma,
    },
    {
        "cls": "extra_distributions.InverseGamma",
        "kwargs": {"alpha": 0.5, "beta": 3.0},
        "ref": _inverse_gamma,
    },
    {
        "cls": "extra_distributions.BetaPrime",
        "kwargs": {"a": 2.0, "b": 3.0, "s": 1.0},
        "ref": _beta_prime,
    },
]


def _build() -> dict[str, Any]:
    """Evaluate every family on the grid and collect the results."""
    entries = []
    for spec in FAMILIES:
        ref = spec["ref"](**spec["kwargs"])
        points = []
        for u in QUANTILE_GRID:
            exact_x = ref["ppf"](u)

            # Evaluate the probabilities at the *double-rounded* x, not at the
            # 50-digit one. The library never sees the exact quantile -- it
            # gets whatever survives the round to a float -- so comparing its
            # cdf at the rounded point against a reference cdf at the exact
            # point measures that rounding rather than the library. Near the
            # lower support of a Pareto that rounding is a relative 1e-6 on the
            # probability, which would read as a library defect and is not one.
            x = mp.mpf(float(exact_x))
            cdf, sf, pdf = ref["cdf"](x), ref["sf"](x), ref["pdf"](x)
            points.append(
                {
                    "u": u,
                    "x": mp.nstr(exact_x, 20),
                    "pdf": mp.nstr(pdf, 20),
                    "cdf": mp.nstr(cdf, 20),
                    "sf": mp.nstr(sf, 20),
                    # How much a relative error in the input becomes a relative
                    # error in the output. Near a support boundary this is
                    # enormous: the double x carries only about seven relative
                    # digits of its distance from the boundary, so cdf(x) there
                    # cannot be recovered to more than that by any formula. A
                    # flat tolerance would read the arithmetic's own limit as a
                    # library defect, and would hide real defects elsewhere by
                    # having to be loose enough to accommodate it.
                    "cond_cdf": mp.nstr(abs(x * pdf / cdf), 8) if cdf else "1",
                    "cond_sf": mp.nstr(abs(x * pdf / sf), 8) if sf else "1",
                    "cond_pdf": mp.nstr(abs(x * mp.diff(ref["pdf"], x) / pdf), 8)
                    if pdf
                    else "1",
                    "cond_ppf": mp.nstr(abs(mp.mpf(u) / (x * pdf)), 8)
                    if x and pdf
                    else "1",
                }
            )
        entries.append({"cls": spec["cls"], "kwargs": spec["kwargs"], "points": points})
        print(f"  {spec['cls']}({spec['kwargs']}) -> {len(points)} points")
    return {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": "mpmath",
        "mpmath_version": mp.__version__,
        "decimal_digits": mp.mp.dps,
        "note": (
            "Computed from the mathematical definitions, independently of the "
            "heavytails implementations. Regenerating to make a failing test "
            "pass defeats the purpose: a disagreement means the library "
            "changed, not that this file is stale."
        ),
        "quantile_grid": QUANTILE_GRID,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "reference_values.json",
        help="Where to write the database.",
    )
    args = parser.parse_args()

    print(f"Evaluating {len(FAMILIES)} parameterisations at {mp.mp.dps} digits:")
    database = _build()
    args.out.write_text(json.dumps(database, indent=2) + "\n", encoding="utf-8")
    total = sum(len(e["points"]) for e in database["entries"])
    print(f"\nWrote {total} reference points to {args.out}")


if __name__ == "__main__":
    main()
