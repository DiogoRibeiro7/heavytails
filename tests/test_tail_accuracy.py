"""Tests for the distribution methods added to complete the common interface.

These cover the Student-t CDF/SF/PPF, the survival functions added to Cauchy,
Frechet and GEV_Frechet, and the Yule-Simon overflow fix. The recurring theme is
that ``1 - cdf(x)`` is not an acceptable survival function in the far tail: once
``cdf(x)`` rounds to 1.0 the complement is exactly zero.
"""

from __future__ import annotations

import math

import pytest

from heavytails import (
    Cauchy,
    DiscretePareto,
    Frechet,
    GeneralizedPareto,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
    YuleSimon,
    Zipf,
)

CONTINUOUS = [
    Pareto(alpha=1.5, xm=1.0),
    Cauchy(x0=0.0, gamma=1.0),
    StudentT(nu=3.0),
    LogNormal(mu=0.0, sigma=1.0),
    Weibull(k=0.7, lam=1.0),
    Frechet(alpha=2.0, s=1.0, m=0.0),
    GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0),
    GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0),
]

DISCRETE = [
    Zipf(s=2.0, kmax=1000),
    YuleSimon(rho=1.5),
    DiscretePareto(alpha=1.5, k_min=1, k_max=1000),
]


class TestCommonInterface:
    """Every family exposes the interface the documentation advertises."""

    @pytest.mark.parametrize("dist", CONTINUOUS, ids=lambda d: type(d).__name__)
    def test_continuous_families_are_complete(self, dist) -> None:
        """pdf, cdf, sf, ppf and rvs are all present."""
        for method in ("pdf", "cdf", "sf", "ppf", "rvs"):
            assert callable(getattr(dist, method)), f"{dist} is missing {method}"

    @pytest.mark.parametrize("dist", DISCRETE, ids=lambda d: type(d).__name__)
    def test_discrete_families_are_complete(self, dist) -> None:
        """pmf, cdf, ppf and rvs are all present."""
        for method in ("pmf", "cdf", "ppf", "rvs"):
            assert callable(getattr(dist, method)), f"{dist} is missing {method}"

    @pytest.mark.parametrize("dist", CONTINUOUS, ids=lambda d: type(d).__name__)
    def test_sf_complements_cdf(self, dist) -> None:
        """sf(x) + cdf(x) == 1 wherever both are comfortably representable."""
        for u in (0.1, 0.25, 0.5, 0.75, 0.9):
            x = dist.ppf(u)
            assert dist.cdf(x) + dist.sf(x) == pytest.approx(1.0, abs=1e-12)


class TestSurvivalFunctionTailAccuracy:
    """The survival functions must not collapse to zero in the far tail."""

    @pytest.mark.parametrize(
        ("dist", "x"),
        [
            (Cauchy(x0=0.0, gamma=1.0), 1e15),
            (Frechet(alpha=2.0, s=1.0, m=0.0), 1e8),
            (GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0), 1e8),
            (StudentT(nu=4.0), 1e8),
        ],
        ids=["cauchy", "frechet", "gev", "studentt"],
    )
    def test_sf_beats_one_minus_cdf(self, dist, x: float) -> None:
        """sf stays positive and accurate where 1 - cdf has lost every digit."""
        sf = dist.sf(x)
        assert sf > 0.0, "survival function underflowed to zero"

        naive = 1.0 - dist.cdf(x)
        # The naive form is quantised to multiples of the float spacing at 1.0,
        # so it cannot resolve a value this small.
        assert abs(naive - sf) > 0.05 * sf, (
            "expected 1 - cdf to have lost precision here; if this fails the "
            "test no longer exercises what it claims to"
        )

    def test_cauchy_sf_matches_asymptotic(self) -> None:
        """For large x the Cauchy survival function tends to 1/(pi x)."""
        c = Cauchy(x0=0.0, gamma=1.0)
        for x in (1e6, 1e10, 1e15):
            assert c.sf(x) == pytest.approx(1.0 / (math.pi * x), rel=1e-9)


class TestStudentT:
    """The Student-t CDF, survival function and quantile function."""

    def test_symmetry(self) -> None:
        """The Student-t is symmetric about zero."""
        t = StudentT(nu=3.0)
        assert t.cdf(0.0) == pytest.approx(0.5)
        for x in (0.5, 1.0, 4.0, 20.0):
            assert t.cdf(-x) == pytest.approx(t.sf(x), rel=1e-12)
            assert t.ppf(0.5) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("nu", [1.0, 2.0, 4.0, 30.0])
    def test_ppf_inverts_cdf(self, nu: float) -> None:
        """ppf(cdf(x)) recovers x wherever cdf(x) is not saturated.

        Round-tripping through the CDF is only meaningful while cdf(x) stays
        strictly below 1.0. For light-tailed parameters such as nu = 30 the CDF
        reaches 1.0 exactly by x = 20, and no quantile function can invert that.
        """
        t = StudentT(nu=nu)
        for x in (-20.0, -3.0, -0.5, 0.5, 3.0, 20.0):
            u = t.cdf(x)
            if not (0.0 < u < 1.0):
                continue
            assert t.ppf(u) == pytest.approx(x, rel=1e-8, abs=1e-8)

    def test_cdf_saturates_but_sf_stays_informative(self) -> None:
        """Where the CDF rounds to 1.0, the survival function still carries the answer.

        This is the whole reason sf is a separate method rather than a helper
        computing 1 - cdf(x).
        """
        t = StudentT(nu=30.0)
        assert t.cdf(20.0) == 1.0
        assert 0.0 < t.sf(20.0) < 1e-15

    @pytest.mark.parametrize("u", [1e-9, 1e-4, 0.1, 0.9, 1 - 1e-4, 1 - 1e-9])
    def test_cdf_inverts_ppf(self, u: float) -> None:
        """cdf(ppf(u)) recovers u, including at extreme quantiles."""
        t = StudentT(nu=2.5)
        assert t.cdf(t.ppf(u)) == pytest.approx(u, rel=1e-9)

    def test_cauchy_is_studentt_with_one_degree_of_freedom(self) -> None:
        """Student-t with nu = 1 is exactly the standard Cauchy."""
        t = StudentT(nu=1.0)
        c = Cauchy(x0=0.0, gamma=1.0)
        for x in (-10.0, -1.0, 0.0, 1.0, 10.0):
            assert t.cdf(x) == pytest.approx(c.cdf(x), rel=1e-12)
        for u in (0.05, 0.3, 0.7, 0.95):
            assert t.ppf(u) == pytest.approx(c.ppf(u), rel=1e-8)

    def test_ppf_rejects_out_of_range(self) -> None:
        """u must lie strictly inside (0, 1)."""
        t = StudentT(nu=3.0)
        for u in (0.0, 1.0, -0.5, 2.0):
            with pytest.raises(ValueError):
                t.ppf(u)

    def test_cdf_is_monotone(self) -> None:
        """The CDF never decreases."""
        t = StudentT(nu=3.0)
        xs = [-50.0, -5.0, -1.0, 0.0, 1.0, 5.0, 50.0]
        values = [t.cdf(x) for x in xs]
        assert values == sorted(values)


class TestYuleSimon:
    """The Yule-Simon overflow fix and its new closed-form methods."""

    @pytest.mark.parametrize("k", [170, 171, 500, 1000, 10_000])
    def test_pmf_does_not_overflow(self, k: int) -> None:
        """math.gamma(k) overflows past k = 170; lgamma does not.

        Values this large are ordinary for a heavy tail, so the old
        implementation raised OverflowError on entirely reasonable input.
        """
        y = YuleSimon(rho=1.5)
        value = y.pmf(k)
        assert 0.0 < value < 1.0
        assert math.isfinite(value)

    def test_sampling_reaches_the_tail(self) -> None:
        """Sampling used to call the overflowing pmf in a loop."""
        y = YuleSimon(rho=0.6)
        samples = y.rvs(200, seed=42)
        assert len(samples) == 200
        assert all(k >= 1 for k in samples)

    def test_pmf_sums_to_one(self) -> None:
        """The mass function is normalised."""
        y = YuleSimon(rho=2.0)
        total = sum(y.pmf(k) for k in range(1, 50_000))
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_sf_matches_the_summed_tail(self) -> None:
        """The closed-form survival function agrees with direct summation."""
        y = YuleSimon(rho=2.0)
        for k in (1, 3, 10, 40):
            summed = sum(y.pmf(j) for j in range(k + 1, 200_000))
            assert y.sf(k) == pytest.approx(summed, rel=1e-6)

    def test_cdf_is_monotone(self) -> None:
        """The CDF never decreases."""
        y = YuleSimon(rho=1.5)
        values = [y.cdf(k) for k in (1, 2, 5, 20, 100, 1000)]
        assert values == sorted(values)

    def test_ppf_is_the_smallest_k_reaching_u(self) -> None:
        """ppf(u) returns the smallest k with cdf(k) >= u."""
        y = YuleSimon(rho=1.5)
        for u in (0.1, 0.5, 0.9, 0.99, 0.999):
            k = y.ppf(u)
            assert y.cdf(k) >= u
            if k > 1:
                assert y.cdf(k - 1) < u

    def test_ppf_rejects_out_of_range(self) -> None:
        """u must lie strictly inside (0, 1)."""
        y = YuleSimon(rho=1.5)
        for u in (0.0, 1.0, -1.0):
            with pytest.raises(ValueError):
                y.ppf(u)
