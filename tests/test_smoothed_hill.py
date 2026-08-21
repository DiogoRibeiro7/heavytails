"""The Resnick-Starica smoothed Hill estimator.

The claim being tested is not just that the estimator is consistent, but that
it does the thing it exists to do: reduce the variance of the ordinary Hill
estimator by the factor the theory predicts.
"""

from __future__ import annotations

import random
import statistics

import pytest

import heavytails
from heavytails import Frechet, Pareto
from heavytails.tail_index import (
    hill_estimator,
    smoothed_hill_estimator,
    smoothed_hill_variance_ratio,
    tail_index_confidence_interval,
)


class TestConsistency:
    """It must estimate the right thing before anything else matters."""

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
    @pytest.mark.parametrize("u", [2.0, 3.0])
    def test_recovers_the_tail_index_of_a_pareto(self, alpha: float, u: float) -> None:
        data = Pareto(alpha=alpha, xm=1.0).rvs(20000, seed=11)
        assert smoothed_hill_estimator(data, k=1000, u=u) == pytest.approx(
            1.0 / alpha, abs=0.03
        )

    def test_recovers_the_tail_index_of_a_frechet(self) -> None:
        """A second family, so the test is not tuned to Pareto."""
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(20000, seed=4)
        assert smoothed_hill_estimator(data, k=1000) == pytest.approx(0.5, abs=0.04)

    def test_is_scale_invariant(self) -> None:
        """gamma is a shape property; rescaling must not change it."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=6)
        base = smoothed_hill_estimator(data, k=200)
        scaled = smoothed_hill_estimator([1000.0 * x for x in data], k=200)
        assert scaled == pytest.approx(base, rel=1e-9)

    def test_equals_the_average_of_the_hill_estimates_it_smooths(self) -> None:
        """The definition, evaluated directly.

        This is the check that the prefix-sum formulation is equivalent to the
        O(k^2) reading of the definition.
        """
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=3)
        k, u = 100, 2.0
        upper = int(u * k)
        expected = sum(hill_estimator(data, j) for j in range(k + 1, upper + 1)) / (
            upper - k
        )
        assert smoothed_hill_estimator(data, k, u) == pytest.approx(expected, rel=1e-12)


class TestVarianceReduction:
    """The property the estimator exists for."""

    def test_variance_ratio_formula(self) -> None:
        """2*(u - 1 - ln u) / (u - 1)^2, at the values usually quoted."""
        assert smoothed_hill_variance_ratio(2.0) == pytest.approx(0.6137, abs=1e-4)
        assert smoothed_hill_variance_ratio(3.0) == pytest.approx(0.4507, abs=1e-4)

    def test_variance_ratio_tends_to_one_as_u_approaches_one(self) -> None:
        """With no smoothing there is no reduction."""
        assert smoothed_hill_variance_ratio(1.0001) == pytest.approx(1.0, abs=1e-3)

    def test_variance_ratio_decreases_with_u(self) -> None:
        ratios = [smoothed_hill_variance_ratio(u) for u in (1.5, 2.0, 3.0, 5.0, 10.0)]
        assert ratios == sorted(ratios, reverse=True)
        assert all(0.0 < r < 1.0 for r in ratios)

    @pytest.mark.parametrize("u", [1.0, 0.5, 0.0, -1.0])
    def test_variance_ratio_rejects_u_not_above_one(self, u: float) -> None:
        with pytest.raises(ValueError, match="strictly greater than 1"):
            smoothed_hill_variance_ratio(u)

    @pytest.mark.slow
    @pytest.mark.parametrize("u", [2.0, 3.0])
    def test_measured_variance_reduction_matches_the_theory(self, u: float) -> None:
        """Simulation, against the asymptotic formula.

        This is the test that would fail if the averaging range were off by
        one, or if the estimator were subtly not the one it claims to be: the
        estimate would still look consistent, but the variance would not fall
        by the predicted factor.
        """
        trials = 250
        hill_values = []
        smoothed_values = []
        for seed in range(trials):
            data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=seed)
            hill_values.append(hill_estimator(data, 500))
            smoothed_values.append(smoothed_hill_estimator(data, 500, u=u))

        measured = statistics.variance(smoothed_values) / statistics.variance(
            hill_values
        )
        predicted = smoothed_hill_variance_ratio(u)
        # The formula is asymptotic and this is a finite Monte Carlo estimate of
        # a ratio of variances, so the tolerance is deliberately loose.
        assert measured == pytest.approx(predicted, abs=0.10)

    @pytest.mark.slow
    def test_is_less_variable_than_plain_hill(self) -> None:
        """The headline claim, stated without reference to any formula."""
        trials = 150
        hill_values = []
        smoothed_values = []
        for seed in range(trials):
            data = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=seed)
            hill_values.append(hill_estimator(data, 300))
            smoothed_values.append(smoothed_hill_estimator(data, 300, u=3.0))
        assert statistics.stdev(smoothed_values) < statistics.stdev(hill_values)


class TestLimitations:
    """What it does not fix, asserted so nobody assumes otherwise."""

    def test_inherits_the_positive_only_limitation_of_hill(self) -> None:
        """It averages Hill estimates, so it cannot represent gamma < 0 either.

        On a Uniform(0,1) sample, whose index is -1, it returns a positive
        number just as Hill does. Smoothing addresses variance in k, not the
        range of gamma; generalized_hill_estimator is the one for that.
        """
        rnd = random.Random(3)
        data = [rnd.random() for _ in range(20000)]
        assert smoothed_hill_estimator(data, k=500) > 0.0


class TestArgumentValidation:
    @pytest.mark.parametrize("u", [1.0, 0.5, 0.0, -2.0])
    def test_rejects_u_not_above_one(self, u: float) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="strictly greater than 1"):
            smoothed_hill_estimator(data, k=100, u=u)

    @pytest.mark.parametrize("k", [0, 1, -5])
    def test_rejects_k_below_two(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="k must be greater than 1"):
            smoothed_hill_estimator(data, k=k)

    def test_rejects_a_range_running_past_the_sample(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(100, seed=1)
        with pytest.raises(ValueError, match="less than the sample size"):
            smoothed_hill_estimator(data, k=60, u=2.0)

    def test_rejects_an_empty_averaging_range(self) -> None:
        """u only slightly above 1 can leave floor(u*k) == k."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="empty"):
            smoothed_hill_estimator(data, k=10, u=1.01)

    def test_rejects_non_positive_data(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            smoothed_hill_estimator([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0], k=2)


class TestIntegration:
    """It is a first-class estimator, not a bolt-on."""

    def test_is_available_to_the_confidence_interval_helper(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=7)
        result = tail_index_confidence_interval(
            data,
            k=100,
            estimator="smoothed_hill",
            method="bootstrap",
            n_bootstrap=60,
            seed=1,
        )
        assert result["estimator"] == "smoothed_hill"
        assert result["lower"] <= result["gamma"] <= result["upper"]

    def test_asymptotic_interval_is_refused(self) -> None:
        """Only Hill has the closed form implemented; do not invent one."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=7)
        with pytest.raises(ValueError, match="only established for the Hill"):
            tail_index_confidence_interval(
                data, k=100, estimator="smoothed_hill", method="asymptotic"
            )

    def test_is_exported_from_the_package(self) -> None:
        assert heavytails.smoothed_hill_estimator is smoothed_hill_estimator
        assert "smoothed_hill_estimator" in heavytails.__all__
        assert "smoothed_hill_variance_ratio" in heavytails.__all__
