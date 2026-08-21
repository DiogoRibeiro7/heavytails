"""GPD maximum likelihood, and peaks-over-threshold tail index estimation.

The fit is checked against ``scipy.stats.genpareto.fit`` where SciPy is
available, and against known parameters by simulation where it is not. SciPy is
a benchmark-group dependency, so these tests skip rather than fail without it.
"""

from __future__ import annotations

import math
import random

import pytest

import heavytails
from heavytails import GeneralizedPareto, Pareto
from heavytails.tail_index import (
    _gpd_profile_log_likelihood,
    fit_generalized_pareto,
    gpd_mle_estimator,
    tail_index_confidence_interval,
)

scipy_stats = pytest.importorskip(
    "scipy.stats", reason="scipy is a benchmark-group dependency"
)


class TestAgainstScipy:
    """The fit must match an independent implementation."""

    @pytest.mark.parametrize(
        ("xi", "sigma"),
        [(0.5, 1.0), (0.25, 2.0), (0.8, 1.0), (1.5, 0.5), (-0.2, 1.0), (-0.5, 1.0)],
    )
    def test_matches_scipy_across_the_sign_of_xi(self, xi: float, sigma: float) -> None:
        """Positive, near-zero and negative shape all agree to four decimals."""
        sample = (
            scipy_stats.genpareto(c=xi, scale=sigma)
            .rvs(size=5000, random_state=7)
            .tolist()
        )
        ours = fit_generalized_pareto(sample)
        their_xi, _, their_sigma = scipy_stats.genpareto.fit(sample, floc=0)

        assert ours["xi"] == pytest.approx(their_xi, abs=1e-3)
        assert ours["sigma"] == pytest.approx(their_sigma, abs=1e-3)

    def test_matches_scipy_near_the_exponential_case(self) -> None:
        """xi = 0 is the boundary the theta substitution has to handle."""
        sample = (
            scipy_stats.genpareto(c=0.0, scale=1.0)
            .rvs(size=20000, random_state=1)
            .tolist()
        )
        ours = fit_generalized_pareto(sample)
        their_xi, _, their_sigma = scipy_stats.genpareto.fit(sample, floc=0)
        assert ours["xi"] == pytest.approx(their_xi, abs=1e-3)
        assert ours["sigma"] == pytest.approx(their_sigma, abs=1e-3)


class TestRecoversKnownParameters:
    @pytest.mark.parametrize(
        ("xi", "sigma"), [(0.5, 1.0), (0.25, 2.0), (0.75, 1.5), (-0.3, 1.0)]
    )
    def test_recovers_the_generating_parameters(self, xi: float, sigma: float) -> None:
        sample = (
            scipy_stats.genpareto(c=xi, scale=sigma)
            .rvs(size=20000, random_state=3)
            .tolist()
        )
        fitted = fit_generalized_pareto(sample)
        assert fitted["xi"] == pytest.approx(xi, abs=0.03)
        assert fitted["sigma"] == pytest.approx(sigma, rel=0.05)

    def test_reports_the_sample_size_and_likelihood(self) -> None:
        sample = GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0).rvs(1000, seed=1)
        fitted = fit_generalized_pareto(sample)
        assert fitted["n"] == 1000
        assert math.isfinite(fitted["log_likelihood"])

    def test_the_reported_likelihood_is_the_maximum(self) -> None:
        """Perturbing theta either way must not improve the objective."""
        sample = GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0).rvs(2000, seed=5)
        fitted = fit_generalized_pareto(sample)
        theta = fitted["xi"] / fitted["sigma"]
        best = fitted["log_likelihood"]
        for factor in (0.9, 0.99, 1.01, 1.1):
            assert _gpd_profile_log_likelihood(theta * factor, sample) <= best + 1e-9


class TestAsATailIndexEstimator:
    @pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
    def test_recovers_the_tail_index_on_average(self, alpha: float) -> None:
        """An estimator is a statement about a sampling distribution.

        A single evaluation of this estimator can sit several percent from the
        truth, so the check averages. Testing one seed would either pass or
        fail for reasons unrelated to correctness.
        """
        estimates = [
            gpd_mle_estimator(Pareto(alpha=alpha, xm=1.0).rvs(20000, seed=seed), 1000)
            for seed in range(30)
        ]
        mean = sum(estimates) / len(estimates)
        assert mean == pytest.approx(1.0 / alpha, abs=0.02)

    def test_handles_a_negative_index(self) -> None:
        """Unlike the whole Hill family, which cannot represent gamma < 0."""
        rnd = random.Random(3)
        data = [rnd.random() for _ in range(20000)]
        assert gpd_mle_estimator(data, 1000) == pytest.approx(-1.0, abs=0.2)

    def test_is_scale_invariant(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=6)
        base = gpd_mle_estimator(data, 500)
        scaled = gpd_mle_estimator([1000.0 * x for x in data], 500)
        assert scaled == pytest.approx(base, rel=1e-6)


class TestArgumentValidation:
    def test_rejects_too_few_exceedances(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            fit_generalized_pareto([1.0])

    @pytest.mark.parametrize("bad", [[1.0, 0.0, 2.0], [1.0, -1.0, 2.0]])
    def test_rejects_non_positive_exceedances(self, bad: list[float]) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            fit_generalized_pareto(bad)

    @pytest.mark.parametrize("k", [0, 1, 1000])
    def test_rejects_k_out_of_range(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="k must be between"):
            gpd_mle_estimator(data, k=k)

    def test_rejects_a_sample_tied_at_the_top(self) -> None:
        """No observation strictly exceeds the threshold, so nothing to fit."""
        with pytest.raises(ValueError, match="not identifiable"):
            gpd_mle_estimator([5.0] * 20 + [1.0] * 20, k=10)


class TestIntegration:
    def test_available_to_the_interval_helper(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator="gpd_mle",
            method="bootstrap",
            # Deliberately small: this estimator optimises, so it is far slower
            # than the closed-form ones.
            n_bootstrap=20,
            seed=1,
        )
        assert result["estimator"] == "gpd_mle"
        assert result["lower"] <= result["gamma"] <= result["upper"]

    def test_is_exported(self) -> None:
        assert heavytails.gpd_mle_estimator is gpd_mle_estimator
        assert heavytails.fit_generalized_pareto is fit_generalized_pareto
