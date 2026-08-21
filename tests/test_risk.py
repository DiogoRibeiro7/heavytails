"""Tail-risk metrics.

Every closed-form expected shortfall is checked three ways: against the
quadrature fallback, against Monte Carlo, and against its defining identity
where one is available. Agreement of the analytic and numeric paths is the
strongest of the three, since they share no code.
"""

from __future__ import annotations

import math
import statistics

import pytest

import heavytails
from heavytails import (
    BurrXII,
    Cauchy,
    Frechet,
    GeneralizedPareto,
    LogLogistic,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)
from heavytails.risk import (
    expected_shortfall,
    mean_exists,
    monte_carlo_tail_risk,
    tail_conditional_expectation,
    value_at_risk,
)

CLOSED_FORM = [
    Pareto(alpha=2.0, xm=1.0),
    Pareto(alpha=3.0, xm=2.0),
    LogNormal(mu=0.0, sigma=1.0),
    LogNormal(mu=1.0, sigma=0.5),
    GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0),
    GeneralizedPareto(xi=0.5, sigma=2.0, mu=1.0),
    Weibull(k=0.7, lam=1.0),
    Weibull(k=1.5, lam=2.0),
]

NUMERIC_ONLY = [
    Frechet(alpha=2.0, s=1.0, m=0.0),
    BurrXII(c=2.0, k=2.0, s=1.0),
    LogLogistic(kappa=3.0, lam=1.0),
    StudentT(nu=3.0),
]


def _name(d: object) -> str:
    return type(d).__name__


class TestValueAtRisk:
    def test_is_the_quantile(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        for level in (0.9, 0.95, 0.99, 0.999):
            assert value_at_risk(dist, level) == dist.ppf(level)

    def test_increases_with_the_level(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        values = [value_at_risk(dist, p) for p in (0.9, 0.95, 0.99, 0.999)]
        assert values == sorted(values)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_rejects_a_level_outside_the_unit_interval(self, level: float) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            value_at_risk(Pareto(alpha=2.0, xm=1.0), level)


class TestExpectedShortfallClosedForms:
    """The analytic path, against two independent references."""

    @pytest.mark.parametrize("dist", CLOSED_FORM, ids=_name)
    @pytest.mark.parametrize("level", [0.95, 0.99])
    def test_analytic_matches_quadrature(self, dist: object, level: float) -> None:
        """These share no code, so agreement is real evidence.

        The closed forms come from integrating the density; the quadrature
        integrates the quantile function. Both being wrong the same way is
        implausible.
        """
        analytic = expected_shortfall(dist, level, method="analytic")
        numeric = expected_shortfall(dist, level, method="numeric")
        assert analytic == pytest.approx(numeric, rel=1e-4)

    @pytest.mark.slow
    @pytest.mark.parametrize("dist", CLOSED_FORM, ids=_name)
    def test_analytic_matches_monte_carlo(self, dist: object) -> None:
        analytic = expected_shortfall(dist, 0.95, method="analytic")
        sample = sorted(dist.rvs(300000, seed=1))  # type: ignore[attr-defined]
        empirical = statistics.mean(sample[int(0.95 * 300000) :])
        assert analytic == pytest.approx(empirical, rel=0.03)

    def test_pareto_matches_its_defining_identity(self) -> None:
        """For a Pareto tail, ES is exactly VaR * alpha / (alpha - 1)."""
        for alpha in (1.5, 2.0, 4.0):
            dist = Pareto(alpha=alpha, xm=1.0)
            for level in (0.9, 0.99, 0.999):
                expected = value_at_risk(dist, level) * alpha / (alpha - 1.0)
                assert expected_shortfall(dist, level) == pytest.approx(
                    expected, rel=1e-12
                )

    @pytest.mark.parametrize("dist", CLOSED_FORM, ids=_name)
    def test_exceeds_value_at_risk(self, dist: object) -> None:
        """A conditional mean above a threshold must exceed the threshold."""
        for level in (0.9, 0.99):
            assert expected_shortfall(dist, level) > value_at_risk(dist, level)


class TestExpectedShortfallNumeric:
    @pytest.mark.parametrize("dist", NUMERIC_ONLY, ids=_name)
    @pytest.mark.slow
    def test_quadrature_matches_monte_carlo(self, dist: object) -> None:
        numeric = expected_shortfall(dist, 0.95)
        sample = sorted(dist.rvs(300000, seed=1))  # type: ignore[attr-defined]
        empirical = statistics.mean(sample[int(0.95 * 300000) :])
        assert numeric == pytest.approx(empirical, rel=0.05)

    @pytest.mark.parametrize("dist", NUMERIC_ONLY, ids=_name)
    def test_analytic_is_refused_where_there_is_no_closed_form(
        self, dist: object
    ) -> None:
        """Better to refuse than to silently fall back and imply exactness."""
        with pytest.raises(ValueError, match="No closed-form"):
            expected_shortfall(dist, 0.99, method="analytic")

    def test_more_nodes_do_not_change_the_answer_much(self) -> None:
        """A stable quadrature means the substitution is doing its job."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        coarse = expected_shortfall(dist, 0.99, method="numeric", nodes=5000)
        fine = expected_shortfall(dist, 0.99, method="numeric", nodes=40000)
        assert coarse == pytest.approx(fine, rel=1e-3)


class TestInfiniteMean:
    """The case that must not quietly return a number."""

    @pytest.mark.parametrize(
        "dist",
        [
            Pareto(alpha=0.5, xm=1.0),
            Pareto(alpha=1.0, xm=1.0),
            Cauchy(x0=0.0, gamma=1.0),
            StudentT(nu=1.0),
            GeneralizedPareto(xi=1.5, sigma=1.0, mu=0.0),
            BurrXII(c=0.5, k=1.0, s=1.0),
            LogLogistic(kappa=0.8, lam=1.0),
        ],
        ids=_name,
    )
    def test_expected_shortfall_is_infinite(self, dist: object) -> None:
        """Returning a large finite number would look like a result."""
        assert not mean_exists(dist)
        assert expected_shortfall(dist, 0.99) == math.inf

    @pytest.mark.parametrize(
        "dist",
        [
            Pareto(alpha=2.0, xm=1.0),
            LogNormal(mu=0.0, sigma=1.0),
            Weibull(k=0.7, lam=1.0),
            StudentT(nu=3.0),
            GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0),
        ],
        ids=_name,
    )
    def test_finite_mean_gives_a_finite_answer(self, dist: object) -> None:
        assert mean_exists(dist)
        assert math.isfinite(expected_shortfall(dist, 0.99))

    def test_the_boundary_is_at_alpha_one(self) -> None:
        """alpha = 1 has no finite mean; just above it does."""
        assert not mean_exists(Pareto(alpha=1.0, xm=1.0))
        assert mean_exists(Pareto(alpha=1.0001, xm=1.0))

    def test_monte_carlo_reports_infinity_rather_than_a_sample_mean(self) -> None:
        """The sample mean of the exceedances exists but does not converge.

        Reporting it with a standard error would dress up a meaningless number.
        """
        result = monte_carlo_tail_risk(
            Pareto(alpha=0.5, xm=1.0), 0.99, n_samples=20000, seed=1
        )
        assert result["expected_shortfall"] == math.inf
        assert result["expected_shortfall_std_error"] is None
        # Value at risk is still perfectly well defined.
        assert math.isfinite(result["value_at_risk"])


class TestMonteCarlo:
    def test_reports_standard_errors(self) -> None:
        result = monte_carlo_tail_risk(
            Pareto(alpha=2.0, xm=1.0), 0.99, n_samples=100000, seed=3
        )
        assert result["expected_shortfall_std_error"] > 0.0
        assert result["value_at_risk_std_error"] > 0.0
        assert result["n_exceedances"] == 1000

    def test_estimates_are_close_to_the_analytic_values(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        result = monte_carlo_tail_risk(dist, 0.99, n_samples=200000, seed=5)
        assert result["value_at_risk"] == pytest.approx(
            value_at_risk(dist, 0.99), rel=0.05
        )
        assert result["expected_shortfall"] == pytest.approx(
            expected_shortfall(dist, 0.99), rel=0.10
        )

    def test_the_analytic_value_lies_within_a_few_standard_errors(self) -> None:
        """The standard error has to mean something."""
        dist = Pareto(alpha=2.0, xm=1.0)
        truth = expected_shortfall(dist, 0.99)
        result = monte_carlo_tail_risk(dist, 0.99, n_samples=200000, seed=5)
        z = (
            abs(result["expected_shortfall"] - truth)
            / result["expected_shortfall_std_error"]
        )
        assert z < 4.0

    def test_standard_error_shrinks_with_more_samples(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        small = monte_carlo_tail_risk(dist, 0.99, n_samples=20000, seed=1)
        large = monte_carlo_tail_risk(dist, 0.99, n_samples=200000, seed=1)
        assert (
            large["expected_shortfall_std_error"]
            < small["expected_shortfall_std_error"]
        )

    def test_is_reproducible_from_a_seed(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        first = monte_carlo_tail_risk(dist, 0.99, n_samples=20000, seed=42)
        second = monte_carlo_tail_risk(dist, 0.99, n_samples=20000, seed=42)
        assert first == second

    def test_rejects_a_level_leaving_too_few_exceedances(self) -> None:
        with pytest.raises(ValueError, match="exceedances"):
            monte_carlo_tail_risk(
                Pareto(alpha=2.0, xm=1.0), 0.9999, n_samples=1000, seed=1
            )

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5])
    def test_rejects_an_invalid_level(self, level: float) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            monte_carlo_tail_risk(Pareto(alpha=2.0, xm=1.0), level)


class TestTailConditionalExpectation:
    def test_equals_expected_shortfall_for_continuous_families(self) -> None:
        """They differ only for distributions with an atom at the quantile."""
        for dist in CLOSED_FORM:
            for level in (0.95, 0.99):
                assert tail_conditional_expectation(dist, level) == (
                    expected_shortfall(dist, level)
                )

    def test_passes_keyword_arguments_through(self) -> None:
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        assert tail_conditional_expectation(
            dist, 0.99, method="numeric", nodes=5000
        ) == expected_shortfall(dist, 0.99, method="numeric", nodes=5000)


class TestArgumentValidation:
    def test_rejects_an_unknown_method(self) -> None:
        with pytest.raises(ValueError, match="Available"):
            expected_shortfall(Pareto(alpha=2.0, xm=1.0), 0.99, method="nonsense")

    @pytest.mark.parametrize("level", [0.0, 1.0, 2.0])
    def test_rejects_an_invalid_level(self, level: float) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            expected_shortfall(Pareto(alpha=2.0, xm=1.0), level)


class TestExports:
    def test_everything_is_exported(self) -> None:
        for name in (
            "value_at_risk",
            "expected_shortfall",
            "tail_conditional_expectation",
            "monte_carlo_tail_risk",
            "mean_exists",
        ):
            assert hasattr(heavytails, name)
            assert name in heavytails.__all__
