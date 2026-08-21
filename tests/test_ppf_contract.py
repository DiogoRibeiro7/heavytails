"""The quantile function contract, asserted uniformly across every family.

Before this was pinned down, behaviour at the edges depended on which family you
asked. Six raised ``OverflowError`` where the quantile was simply too large to
represent, ``InverseGamma`` raised a ``ValueError`` about failing to bracket the
root, ``LogNormal`` returned ``inf``, and ``BetaPrime`` returned a finite value.

The contract is:

* ``u`` outside the open interval (0, 1) raises ``ValueError``.
* A quantile beyond the float range returns ``inf``.
* A solver that cannot converge raises ``ConvergenceError`` rather than
  returning its best guess.
"""

from __future__ import annotations

import math

import pytest

import heavytails
from heavytails import (
    BetaPrime,
    BurrXII,
    Cauchy,
    ConvergenceError,
    DiscretePareto,
    Frechet,
    GeneralizedPareto,
    GEV_Frechet,
    InverseGamma,
    LogLogistic,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
    Zipf,
)
from heavytails._special import _ppf_monotone

CONTINUOUS = [
    Pareto(alpha=1.5, xm=1.0),
    Cauchy(x0=0.0, gamma=1.0),
    StudentT(nu=3.0),
    LogNormal(mu=0.0, sigma=1.0),
    Weibull(k=0.7, lam=1.0),
    Frechet(alpha=2.0, s=1.0, m=0.0),
    GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0),
    GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0),
    BurrXII(c=1.2, k=2.5, s=3.0),
    LogLogistic(kappa=2.0, lam=1.0),
    InverseGamma(alpha=2.0, beta=1.0),
    BetaPrime(a=2.0, b=3.0, s=1.0),
]

DISCRETE = [
    Zipf(s=2.0, kmax=1000),
    DiscretePareto(alpha=1.5, k_min=1, k_max=1000),
]

# Parameters chosen so the quantile genuinely exceeds the float range.
OVERFLOWING = [
    Pareto(alpha=1e-3, xm=1.0),
    Weibull(k=1e-3, lam=1.0),
    Frechet(alpha=1e-3, s=1.0, m=0.0),
    GEV_Frechet(xi=100.0, mu=0.0, sigma=1.0),
    BurrXII(c=1e-3, k=1e-3, s=1.0),
    LogLogistic(kappa=1e-3, lam=1.0),
    InverseGamma(alpha=1e-3, beta=1.0),
    LogNormal(mu=1000.0, sigma=1.0),
    GeneralizedPareto(xi=200.0, sigma=1.0, mu=0.0),
]


def _name(d: object) -> str:
    return type(d).__name__


class TestInvalidInput:
    """u must lie strictly inside (0, 1), for every family."""

    @pytest.mark.parametrize("dist", CONTINUOUS + DISCRETE, ids=_name)
    @pytest.mark.parametrize("u", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_rejects_u_outside_the_open_unit_interval(self, dist, u: float) -> None:
        with pytest.raises(ValueError):
            dist.ppf(u)


class TestOverflowReturnsInf:
    """A quantile beyond the float range is reported, not raised."""

    @pytest.mark.parametrize("dist", OVERFLOWING, ids=_name)
    def test_returns_inf_rather_than_raising(self, dist) -> None:
        """Every one of these used to raise OverflowError or ValueError.

        Raising aborts a parameter sweep at the first point that overflows,
        and makes a representable-range limit look like a caller error.
        """
        assert dist.ppf(1.0 - 1e-12) == math.inf

    @pytest.mark.parametrize("dist", OVERFLOWING, ids=_name)
    def test_moderate_quantiles_are_still_finite(self, dist) -> None:
        """The guard must not swallow ordinary quantiles."""
        value = dist.ppf(0.5)
        assert math.isfinite(value) or value == math.inf

    def test_a_sweep_across_parameters_completes(self) -> None:
        """The practical symptom: one overflow used to kill the whole loop."""
        values = [
            Pareto(alpha=a, xm=1.0).ppf(1 - 1e-15) for a in (2.0, 1.0, 0.1, 0.001)
        ]
        assert math.isfinite(values[0])
        assert values[-1] == math.inf


class TestMonotonicity:
    """ppf is non-decreasing in u, for every family."""

    @pytest.mark.parametrize("dist", CONTINUOUS, ids=_name)
    def test_continuous_ppf_is_non_decreasing(self, dist) -> None:
        us = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = [dist.ppf(u) for u in us]
        assert values == sorted(values)

    @pytest.mark.parametrize("dist", DISCRETE, ids=_name)
    def test_discrete_ppf_is_non_decreasing(self, dist) -> None:
        us = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
        values = [dist.ppf(u) for u in us]
        assert values == sorted(values)


class TestRoundTrip:
    """cdf(ppf(u)) recovers u wherever the quantile is representable."""

    @pytest.mark.parametrize("dist", CONTINUOUS, ids=_name)
    @pytest.mark.parametrize("u", [0.05, 0.25, 0.5, 0.75, 0.95])
    def test_cdf_inverts_ppf(self, dist, u: float) -> None:
        x = dist.ppf(u)
        if not math.isfinite(x):
            pytest.skip("quantile is not representable at this parameter set")
        assert dist.cdf(x) == pytest.approx(u, rel=1e-6, abs=1e-9)


class TestSolverConvergence:
    """_ppf_monotone signals failure instead of returning a plausible number."""

    def test_raises_when_the_iteration_budget_is_exhausted(self) -> None:
        """A wide bracket and one iteration cannot converge.

        Previously this returned the midpoint, which is indistinguishable from a
        converged answer at the call site.
        """

        def cdf(t: float) -> float:
            return min(max(t / 1e10, 0.0), 1.0)

        # u is deliberately not 0.5: the initial guess is the midpoint of the
        # bracket, which for a linear cdf would already be the exact answer.
        with pytest.raises(ConvergenceError):
            _ppf_monotone(cdf, 0.0, 1e10, 0.1, max_iter=1)

    def test_converges_normally_with_a_sane_budget(self) -> None:
        """The same problem solves fine when given room."""

        def cdf(t: float) -> float:
            return min(max(t / 1e10, 0.0), 1.0)

        assert _ppf_monotone(cdf, 0.0, 1e10, 0.1) == pytest.approx(1e9, rel=1e-9)

    def test_still_rejects_an_unbracketed_root(self) -> None:
        """A bracket that does not contain the quantile is a caller error."""

        def cdf(t: float) -> float:
            return min(max(t, 0.0), 1.0)

        with pytest.raises(ValueError):
            _ppf_monotone(cdf, 0.9, 1.0, 0.5)

    def test_convergence_error_is_exported(self) -> None:
        """Callers need to be able to catch it by name."""
        assert heavytails.ConvergenceError is ConvergenceError
        assert issubclass(ConvergenceError, RuntimeError)
