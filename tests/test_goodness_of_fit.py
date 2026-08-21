"""Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit tests.

The statistics are checked against their definitions and against SciPy; the
p-values are checked against published critical values and, more importantly,
by Monte Carlo: a correctly specified null must be rejected at about the
nominal rate, and a wrong family must be rejected decisively.
"""

from __future__ import annotations

import math

import pytest

from heavytails import LogNormal, Pareto
from heavytails.utilities import AutoFit
from heavytails.validation import (
    GoodnessOfFitTests,
    _anderson_darling_cdf,
    _anderson_darling_p_value,
    _kolmogorov_p_value,
    _resolve_distribution,
)


@pytest.fixture
def tests() -> GoodnessOfFitTests:
    return GoodnessOfFitTests()


class TestKolmogorovSmirnovStatistic:
    """The statistic itself, against its definition."""

    def test_matches_the_definition(self, tests: GoodnessOfFitTests) -> None:
        """D = max_i max(i/n - F(x_i), F(x_i) - (i-1)/n)."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(200, seed=42)

        values = sorted(data)
        n = len(values)
        expected = max(
            max(i / n - dist.cdf(x), dist.cdf(x) - (i - 1) / n)
            for i, x in enumerate(values, start=1)
        )

        result = tests.kolmogorov_smirnov_test(data, "pareto", alpha=2.5, xm=1.0)
        assert result["statistic"] == pytest.approx(expected, rel=1e-12)

    def test_statistic_is_zero_for_a_perfect_fit(
        self, tests: GoodnessOfFitTests
    ) -> None:
        """Data placed exactly on the quantiles gives a near-zero statistic."""
        dist = Pareto(alpha=2.0, xm=1.0)
        n = 500
        data = [dist.ppf((i - 0.5) / n) for i in range(1, n + 1)]
        result = tests.kolmogorov_smirnov_test(data, "pareto", alpha=2.0, xm=1.0)
        assert result["statistic"] < 1.0 / n

    def test_rejects_an_empty_sample(self, tests: GoodnessOfFitTests) -> None:
        with pytest.raises(ValueError, match="empty"):
            tests.kolmogorov_smirnov_test([], "pareto", alpha=2.0, xm=1.0)


class TestAndersonDarlingStatistic:
    """The statistic itself, against its definition."""

    def test_matches_the_definition(self, tests: GoodnessOfFitTests) -> None:
        """A^2 = -n - (1/n) sum (2i-1)[ln F(x_i) + ln(1 - F(x_{n+1-i}))]."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(200, seed=7)

        values = sorted(data)
        n = len(values)
        cdfs = [dist.cdf(x) for x in values]
        total = sum(
            (2 * i - 1) * (math.log(cdfs[i - 1]) + math.log(1.0 - cdfs[n - i]))
            for i in range(1, n + 1)
        )
        expected = -n - total / n

        result = tests.anderson_darling_test(data, "pareto", alpha=2.5, xm=1.0)
        assert result["statistic"] == pytest.approx(expected, rel=1e-10)

    def test_saturated_cdf_values_do_not_produce_infinity(
        self, tests: GoodnessOfFitTests
    ) -> None:
        """A single observation with cdf rounding to 1.0 would send A^2 to inf.

        The implementation clamps away from the endpoints for exactly this
        reason.
        """
        data = [1.0, 2.0, 3.0, 1e300]
        result = tests.anderson_darling_test(data, "pareto", alpha=2.0, xm=1.0)
        assert math.isfinite(result["statistic"])


class TestPValueCalibration:
    """The p-values are the part that is easy to get plausibly wrong."""

    @pytest.mark.parametrize(
        ("statistic", "level"),
        [(1.933, 0.10), (2.492, 0.05), (3.070, 0.025), (3.857, 0.01)],
    )
    def test_anderson_darling_reproduces_published_critical_values(
        self, statistic: float, level: float
    ) -> None:
        """Asymptotic critical values for the fully specified case.

        These are the values that distinguish the correct null distribution
        from the D'Agostino-Stephens formulas for the estimated-parameter
        normality test, whose 5% critical value is 0.787. Using those here
        would reject a correctly specified distribution about half the time.
        """
        assert _anderson_darling_p_value(statistic) == pytest.approx(level, abs=1e-3)

    def test_anderson_darling_cdf_is_monotone(self) -> None:
        values = [_anderson_darling_cdf(z) for z in (0.1, 0.5, 1.0, 2.0, 3.0, 10.0)]
        assert values == sorted(values)
        assert all(0.0 <= v <= 1.0 for v in values)

    def test_kolmogorov_p_value_is_one_at_zero_distance(self) -> None:
        assert _kolmogorov_p_value(0.0, 100) == 1.0

    def test_kolmogorov_p_value_decreases_with_the_statistic(self) -> None:
        values = [_kolmogorov_p_value(d, 100) for d in (0.02, 0.05, 0.1, 0.2, 0.4)]
        assert values == sorted(values, reverse=True)

    @pytest.mark.slow
    @pytest.mark.parametrize("test_name", ["kolmogorov_smirnov", "anderson_darling"])
    def test_rejection_rate_under_the_null_is_near_the_nominal_level(
        self, tests: GoodnessOfFitTests, test_name: str
    ) -> None:
        """A correctly specified null must be rejected at roughly alpha.

        This is the check that caught the wrong p-value approximation: it was
        rejecting the true distribution far more often than 5%.
        """
        method = getattr(tests, f"{test_name}_test")
        trials = 300
        rejections = sum(
            method(
                Pareto(alpha=2.0, xm=1.0).rvs(200, seed=seed),
                "pareto",
                alpha=2.0,
                xm=1.0,
            )["reject"]
            for seed in range(trials)
        )
        rate = rejections / trials
        # Binomial standard error at p=0.05, n=300 is about 0.013.
        assert 0.01 < rate < 0.11, f"rejection rate {rate:.3f} is not near 0.05"


class TestPower:
    """A wrong family must be rejected."""

    @pytest.mark.parametrize("test_name", ["kolmogorov_smirnov", "anderson_darling"])
    def test_does_not_reject_the_correct_family(
        self, tests: GoodnessOfFitTests, test_name: str
    ) -> None:
        method = getattr(tests, f"{test_name}_test")
        data = Pareto(alpha=2.0, xm=1.0).rvs(500, seed=1)
        assert not method(data, "pareto", alpha=2.0, xm=1.0)["reject"]

    @pytest.mark.parametrize("test_name", ["kolmogorov_smirnov", "anderson_darling"])
    def test_rejects_the_wrong_family(
        self, tests: GoodnessOfFitTests, test_name: str
    ) -> None:
        method = getattr(tests, f"{test_name}_test")
        data = Pareto(alpha=2.0, xm=1.0).rvs(500, seed=1)
        result = method(data, "lognormal", mu=0.0, sigma=1.0)
        assert result["reject"]
        assert result["p_value"] < 1e-6

    def test_anderson_darling_is_more_sensitive_in_the_tail(
        self, tests: GoodnessOfFitTests
    ) -> None:
        """The reason to prefer A^2 for this library.

        Two Pareto tails that agree in the body but differ in the tail: the
        Anderson-Darling statistic weights the tail, the KS statistic does not.
        """
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=11)
        ad = tests.anderson_darling_test(data, "pareto", alpha=2.4, xm=1.0)
        ks = tests.kolmogorov_smirnov_test(data, "pareto", alpha=2.4, xm=1.0)
        assert ad["p_value"] < ks["p_value"]


class TestReportShape:
    """The returned mapping is what callers rely on."""

    @pytest.mark.parametrize("test_name", ["kolmogorov_smirnov", "anderson_darling"])
    def test_report_fields(self, tests: GoodnessOfFitTests, test_name: str) -> None:
        method = getattr(tests, f"{test_name}_test")
        data = Pareto(alpha=2.0, xm=1.0).rvs(100, seed=3)
        result = method(data, "pareto", alpha=2.0, xm=1.0)
        for key in (
            "test",
            "statistic",
            "p_value",
            "reject",
            "alpha_level",
            "n",
            "distribution",
            "parameters",
            "method",
        ):
            assert key in result
        assert result["n"] == 100
        assert result["parameters"] == {"alpha": 2.0, "xm": 1.0}
        assert 0.0 <= result["p_value"] <= 1.0

    @pytest.mark.parametrize("test_name", ["kolmogorov_smirnov", "anderson_darling"])
    def test_estimated_parameters_carries_a_caveat(
        self, tests: GoodnessOfFitTests, test_name: str
    ) -> None:
        """The asymptotic null does not hold when parameters came from the data.

        Saying so is the difference between a usable number and a misleading
        one, so it is part of the result rather than only the documentation.
        """
        method = getattr(tests, f"{test_name}_test")
        data = Pareto(alpha=2.0, xm=1.0).rvs(100, seed=3)

        plain = method(data, "pareto", alpha=2.0, xm=1.0)
        assert "caveat" not in plain

        estimated = method(data, "pareto", parameters_estimated=True, alpha=2.0, xm=1.0)
        assert "conservative" in estimated["caveat"]

    def test_alpha_level_is_configurable(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(200, seed=2)
        strict = GoodnessOfFitTests(alpha_level=0.5)
        assert (
            strict.kolmogorov_smirnov_test(data, "pareto", alpha=2.0, xm=1.0)[
                "alpha_level"
            ]
            == 0.5
        )

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 2.0])
    def test_rejects_an_invalid_alpha_level(self, level: float) -> None:
        with pytest.raises(ValueError, match="alpha_level"):
            GoodnessOfFitTests(alpha_level=level)


class TestDistributionResolver:
    """Name resolution shared by both tests."""

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            ("pareto", {"alpha": 2.0, "xm": 1.0}),
            ("Pareto", {"alpha": 2.0, "xm": 1.0}),
            ("student-t", {"nu": 3.0}),
            ("studentt", {"nu": 3.0}),
            ("gpd", {"xi": 0.5, "sigma": 1.0, "mu": 0.0}),
            ("burrxii", {"c": 1.2, "k": 2.5, "s": 3.0}),
            ("betaprime", {"a": 2.0, "b": 3.0, "s": 1.0}),
        ],
    )
    def test_resolves_known_names(self, name: str, params: dict) -> None:
        assert _resolve_distribution(name, params) is not None

    def test_unknown_name_lists_the_alternatives(self) -> None:
        with pytest.raises(ValueError, match="Available"):
            _resolve_distribution("not-a-distribution", {})

    def test_bad_parameters_are_reported_as_a_value_error(self) -> None:
        with pytest.raises(ValueError, match="Invalid parameters"):
            _resolve_distribution("pareto", {"nonsense": 1.0})


class TestComparisonIntegration:
    """AutoFit.compare_distributions reports fit as well as rank."""

    def test_comparison_includes_goodness_of_fit(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(400, seed=5)
        results = AutoFit().compare_distributions(data, ["pareto", "lognormal"])

        for entry in results.values():
            assert "anderson_darling" in entry
            assert "kolmogorov_smirnov" in entry

        # The right family ranks first and is not rejected; the wrong one is.
        assert results["pareto"]["rank_AIC"] == 1
        assert not results["pareto"]["anderson_darling"]["reject"]
        assert results["lognormal"]["anderson_darling"]["reject"]

    def test_comparison_marks_parameters_as_estimated(self) -> None:
        """compare_distributions fits from the same sample it then tests."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(200, seed=6)
        results = AutoFit().compare_distributions(data, ["pareto"])
        assert "caveat" in results["pareto"]["anderson_darling"]

    def test_a_failing_test_does_not_abort_the_comparison(self) -> None:
        """A family the resolver cannot build must not lose the whole table."""
        data = LogNormal(mu=0.0, sigma=1.0).rvs(200, seed=8)
        results = AutoFit._attach_goodness_of_fit(
            data, {"nonexistent": {"params": {"a": 1.0}}}
        )
        assert "error" in results["nonexistent"]["anderson_darling"]
