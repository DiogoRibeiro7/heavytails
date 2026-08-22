"""GARCH, and telling volatility clustering apart from heavy shocks.

The claim this module exists to support is that raw returns overstate how heavy
the innovations are, and the tests check it in the direction it is actually
true rather than in the stronger form that would be convenient:

**Standardising helps, and does not fully undo the effect.** On 60,000
simulated returns from a GARCH with ``nu = 4`` innovations, the raw series
gives a tail index of 2.76 and the residuals give 3.33 against a truth of 4.
The direction is unambiguous and the magnitude is not complete, because the
Hill estimator has its own finite-sample bias and the variance recursion has to
be started somewhere. A test asserting the residuals *recover* the innovation
index would be asserting something false.

**Gaussian innovations still produce a heavy-tailed series.** That is the
sharpest form of the point: a tail index of 4.51 out of shocks that have no
power-law tail at all.
"""

from __future__ import annotations

from itertools import pairwise
import statistics

import pytest

from heavytails import Pareto
from heavytails.heavy_tails import ParameterError
from heavytails.tail_index import hill_estimator
from heavytails.timeseries import GARCH11, decluster, extremal_index, fit_garch11


def _tail_index(values: list[float], k: int = 1500) -> float:
    """Hill estimate of the tail index of the absolute values."""
    ordered = sorted((abs(v) for v in values), reverse=True)
    return 1.0 / hill_estimator(ordered, k=k)


class TestTheModel:
    def test_the_persistence_and_long_run_variance(self) -> None:
        model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85)
        assert model.persistence == pytest.approx(0.95)
        assert model.unconditional_variance == pytest.approx(1e-6 / 0.05)

    def test_a_simulated_series_has_the_variance_it_should(self) -> None:
        """Loosely, because a persistent process makes the sample variance
        itself a high-variance statistic."""
        model = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
        sample = model.simulate(200_000, seed=1)
        assert statistics.pvariance(sample) == pytest.approx(
            model.unconditional_variance, rel=0.25
        )

    def test_the_series_is_uncorrelated_but_its_squares_are_not(self) -> None:
        """The signature of volatility clustering, and the reason a static
        distribution cannot describe it.

        Returns themselves show no autocorrelation -- which is why they look
        independent to a correlation test -- while their squares clearly do.
        """
        model = GARCH11(omega=1e-6, alpha=0.12, beta=0.85, nu=6.0)
        sample = model.simulate(50_000, seed=2)
        squares = [v * v for v in sample]

        def autocorrelation(series: list[float]) -> float:
            mean = statistics.fmean(series)
            centred = [v - mean for v in series]
            numerator = sum(a * b for a, b in pairwise(centred))
            return numerator / sum(v * v for v in centred)

        assert abs(autocorrelation(sample)) < 0.03
        assert autocorrelation(squares) > 0.10

    def test_the_conditional_variances_follow_the_recursion(self) -> None:
        model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85)
        returns = [0.01, -0.02, 0.005]
        variances = model.conditional_variances(returns)
        assert variances[0] == pytest.approx(model.unconditional_variance)
        for i in range(1, len(returns)):
            expected = (
                model.omega
                + model.alpha * returns[i - 1] ** 2
                + model.beta * variances[i - 1]
            )
            assert variances[i] == pytest.approx(expected, rel=1e-12)

    def test_the_residuals_have_unit_variance(self) -> None:
        """They are the innovations, which are standardised by construction."""
        model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85, nu=8.0)
        residuals = model.standardized_residuals(model.simulate(50_000, seed=3))
        assert statistics.pvariance(residuals) == pytest.approx(1.0, rel=0.15)

    def test_the_seed_makes_simulation_reproducible(self) -> None:
        model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85, nu=5.0)
        assert model.simulate(200, seed=7) == model.simulate(200, seed=7)

    def test_the_likelihood_prefers_the_model_that_generated_the_data(self) -> None:
        truth = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
        sample = truth.simulate(6_000, seed=4)
        wrong = GARCH11(omega=2e-6, alpha=0.4, beta=0.4, nu=6.0)
        assert truth.log_likelihood(sample) > wrong.log_likelihood(sample)


class TestClusteringManufacturesHeavyTails:
    """The claim the module is built on."""

    def test_gaussian_innovations_give_a_heavy_tailed_series(self) -> None:
        """The sharpest form: a power-law tail out of shocks that have none.

        A Gaussian has no tail index at all -- its tail falls faster than any
        power. The returns it drives through a GARCH recursion have a tail
        index around four and a half, and a Hill estimate of the raw series
        would report that as though it were a property of the shocks.
        """
        model = GARCH11(omega=1e-6, alpha=0.10, beta=0.88)
        index = _tail_index(model.simulate(60_000, seed=1))
        assert 3.0 < index < 7.0

    @pytest.mark.parametrize("nu", [4.0, 6.0])
    def test_raw_returns_look_heavier_than_their_innovations(self, nu: float) -> None:
        model = GARCH11(omega=1e-6, alpha=0.10, beta=0.88, nu=nu)
        assert _tail_index(model.simulate(60_000, seed=1)) < nu

    @pytest.mark.parametrize("nu", [4.0, 6.0, None])
    def test_standardising_moves_the_estimate_towards_the_innovations(
        self, nu: float | None
    ) -> None:
        """Towards, not onto.

        The residual estimate is still below the innovation index -- 3.33
        against 4, and 4.04 against 6 -- because the Hill estimator carries its
        own finite-sample bias and the variance recursion has to start
        somewhere. Asserting recovery would be asserting something false, so
        this asserts the direction, which is unambiguous.
        """
        model = GARCH11(omega=1e-6, alpha=0.10, beta=0.88, nu=nu)
        returns = model.simulate(60_000, seed=1)
        raw = _tail_index(returns)
        residual = _tail_index(model.standardized_residuals(returns))
        assert residual > raw + 0.3


class TestFitting:
    def test_it_recovers_the_persistence(self) -> None:
        """The parameter that matters, and the one identified best.

        omega, alpha and beta trade off against each other in the likelihood;
        their sum is what the data pins down.
        """
        truth = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
        fit = fit_garch11(truth.simulate(8_000, seed=1))
        assert fit["model"].persistence == pytest.approx(truth.persistence, abs=0.03)

    def test_it_recovers_the_degrees_of_freedom_roughly(self) -> None:
        truth = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
        fit = fit_garch11(truth.simulate(8_000, seed=1))
        assert 4.0 < fit["model"].nu < 10.0

    def test_the_fit_beats_the_truth_on_its_own_sample(self) -> None:
        """Maximum likelihood does what it says: nothing scores higher on the
        sample than the maximiser, including the model that generated it."""
        truth = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
        sample = truth.simulate(4_000, seed=2)
        fit = fit_garch11(sample)
        assert fit["log_likelihood"] >= truth.log_likelihood(sample) - 1e-6

    def test_the_student_t_fit_beats_the_normal_on_heavy_data(self) -> None:
        truth = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=4.0)
        sample = truth.simulate(6_000, seed=3)
        assert (
            fit_garch11(sample, innovations="t")["aic"]
            < fit_garch11(sample, innovations="normal")["aic"]
        )

    def test_the_reported_likelihood_is_the_fitted_one(self) -> None:
        sample = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0).simulate(
            2_000, seed=4
        )
        fit = fit_garch11(sample)
        assert fit["log_likelihood"] == pytest.approx(
            fit["model"].log_likelihood(sample), rel=1e-12
        )

    def test_the_fitted_model_is_always_stationary(self) -> None:
        """The parameterisation guarantees it, so no fit can wander out.

        A penalty-based constraint would let the optimiser sit just outside the
        region and come back with a number, which is worse than refusing.
        """
        for seed in range(4):
            sample = GARCH11(omega=1e-6, alpha=0.15, beta=0.80, nu=5.0).simulate(
                2_000, seed=seed
            )
            model = fit_garch11(sample)["model"]
            assert model.persistence < 1.0
            assert model.alpha >= 0.0
            assert model.beta >= 0.0
            assert model.omega > 0.0

    def test_it_refuses_a_series_too_short_to_fit(self) -> None:
        with pytest.raises(ValueError, match="at least 50"):
            fit_garch11([0.01] * 10)

    def test_it_refuses_a_constant_series(self) -> None:
        with pytest.raises(ValueError, match="no variation"):
            fit_garch11([0.0] * 100)

    def test_it_refuses_an_unknown_innovation_family(self) -> None:
        sample = GARCH11(omega=1e-6, alpha=0.1, beta=0.85).simulate(200, seed=1)
        with pytest.raises(ValueError, match="Available"):
            fit_garch11(sample, innovations="cauchy")


class TestExtremalIndex:
    def test_independent_data_gives_one(self) -> None:
        """What classical extreme value theory assumes, and the case where the
        assumption happens to be true."""
        sample = Pareto(alpha=2.0, xm=1.0).rvs(40_000, seed=1)
        threshold = sorted(sample)[int(0.99 * len(sample))]
        assert extremal_index(sample, threshold)["extremal_index"] > 0.85

    def test_clustered_data_gives_much_less_than_one(self) -> None:
        """And the reciprocal is the mean cluster size, which is the number to
        divide a return period by."""
        model = GARCH11(omega=1e-6, alpha=0.12, beta=0.86, nu=5.0)
        sample = [abs(v) for v in model.simulate(40_000, seed=2)]
        threshold = sorted(sample)[int(0.99 * len(sample))]
        result = extremal_index(sample, threshold)
        assert result["extremal_index"] < 0.6
        assert result["mean_cluster_size"] > 1.6

    def test_it_stays_a_proportion(self) -> None:
        for seed in range(3):
            sample = Pareto(alpha=1.5, xm=1.0).rvs(10_000, seed=seed)
            threshold = sorted(sample)[int(0.98 * len(sample))]
            assert 0.0 < extremal_index(sample, threshold)["extremal_index"] <= 1.0

    def test_stronger_persistence_means_more_clustering(self) -> None:
        estimates = []
        for beta in (0.5, 0.75, 0.88):
            model = GARCH11(omega=1e-6, alpha=0.10, beta=beta, nu=5.0)
            sample = [abs(v) for v in model.simulate(40_000, seed=3)]
            threshold = sorted(sample)[int(0.99 * len(sample))]
            estimates.append(extremal_index(sample, threshold)["extremal_index"])
        assert estimates[0] > estimates[-1]

    def test_too_few_exceedances_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least three"):
            extremal_index([1.0, 2.0, 10.0, 1.0], threshold=5.0)

    def test_it_reports_which_branch_applied(self) -> None:
        """The bias correction is only valid where the gaps can carry it."""
        sample = Pareto(alpha=2.0, xm=1.0).rvs(5_000, seed=4)
        threshold = sorted(sample)[int(0.98 * len(sample))]
        assert extremal_index(sample, threshold)["branch"] in {
            "bias-corrected",
            "uncorrected",
        }


class TestDeclustering:
    def test_it_keeps_the_largest_of_each_cluster(self) -> None:
        series = [0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0]
        result = decluster(series, threshold=1.0, run_length=2)
        assert result["cluster_maxima"] == [6.0, 7.0]
        assert result["cluster_sizes"] == [2, 1]

    def test_the_run_length_changes_the_answer(self) -> None:
        """Which is the known weakness of the runs method, and the reason the
        extremal index here uses the intervals estimator instead."""
        series = [0.0, 5.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 7.0]
        tight = decluster(series, threshold=1.0, run_length=1)
        loose = decluster(series, threshold=1.0, run_length=5)
        assert tight["n_clusters"] > loose["n_clusters"]

    def test_it_reduces_a_clustered_series(self) -> None:
        model = GARCH11(omega=1e-6, alpha=0.12, beta=0.86, nu=5.0)
        sample = [abs(v) for v in model.simulate(20_000, seed=5)]
        threshold = sorted(sample)[int(0.99 * len(sample))]
        result = decluster(sample, threshold, run_length=5)
        assert result["n_clusters"] < result["n_exceedances"]

    def test_nothing_above_the_threshold_gives_nothing(self) -> None:
        result = decluster([1.0, 2.0, 3.0], threshold=10.0)
        assert result["n_clusters"] == 0
        assert result["cluster_maxima"] == []

    def test_a_bad_run_length_is_refused(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            decluster([1.0, 5.0, 1.0], threshold=2.0, run_length=0)


class TestValidation:
    @pytest.mark.parametrize(
        ("omega", "alpha", "beta"),
        [(0.0, 0.1, 0.8), (-1e-6, 0.1, 0.8), (1e-6, -0.1, 0.8), (1e-6, 0.5, 0.6)],
    )
    def test_bad_parameters_are_refused(
        self, omega: float, alpha: float, beta: float
    ) -> None:
        with pytest.raises(ParameterError):
            GARCH11(omega=omega, alpha=alpha, beta=beta)

    def test_a_non_stationary_process_is_refused_with_the_sum(self) -> None:
        with pytest.raises(ParameterError, match=r"1\.05"):
            GARCH11(omega=1e-6, alpha=0.15, beta=0.90)

    @pytest.mark.parametrize("nu", [2.0, 1.0, 0.0, -1.0])
    def test_degrees_of_freedom_at_or_below_two_are_refused(self, nu: float) -> None:
        """The innovations are standardised to unit variance, and below two
        there is no variance to standardise."""
        with pytest.raises(ParameterError, match="nu must exceed two"):
            GARCH11(omega=1e-6, alpha=0.1, beta=0.8, nu=nu)

    def test_simulation_arguments_are_checked(self) -> None:
        model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85)
        with pytest.raises(ValueError, match="positive integer"):
            model.simulate(0)
        with pytest.raises(ValueError, match="burn_in"):
            model.simulate(10, burn_in=-1)

    def test_an_empty_series_is_refused(self) -> None:
        model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85)
        with pytest.raises(ValueError, match="must not be empty"):
            model.conditional_variances([])

    def test_the_burn_in_matters_at_high_persistence(self) -> None:
        """A shock takes hundreds of steps to decay at 0.98, so too short a
        burn-in leaves the series still remembering where it started."""
        model = GARCH11(omega=1e-6, alpha=0.10, beta=0.88, nu=6.0)
        assert model.simulate(500, seed=1, burn_in=0) != model.simulate(
            500, seed=1, burn_in=1000
        )
