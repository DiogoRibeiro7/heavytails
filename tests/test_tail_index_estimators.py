"""The generalized Hill estimator, the Hill plot, and confidence intervals.

Consistency is checked by simulation against distributions with a known tail
index, and interval coverage by repeated sampling. Neither can be checked by
a single point evaluation, because an estimator is a statement about a
sampling distribution rather than about one number.
"""

from __future__ import annotations

import math
import random

import pytest

from heavytails import Frechet, Pareto
from heavytails.tail_index import (
    generalized_hill_estimator,
    hill_estimator,
    hill_plot,
    moment_estimator,
    tail_index_confidence_interval,
)


class TestGeneralizedHill:
    """Beirlant, Vynckier and Teugels (1996)."""

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
    def test_recovers_the_tail_index_of_a_pareto(self, alpha: float) -> None:
        """gamma = 1/alpha for a Pareto tail."""
        data = Pareto(alpha=alpha, xm=1.0).rvs(20000, seed=11)
        estimate = generalized_hill_estimator(data, k=1000)
        assert estimate == pytest.approx(1.0 / alpha, abs=0.05)

    def test_recovers_the_tail_index_of_a_frechet(self) -> None:
        """A second family, so the test is not tuned to Pareto."""
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(20000, seed=4)
        assert generalized_hill_estimator(data, k=1000) == pytest.approx(0.5, abs=0.06)

    def test_handles_a_negative_index_where_hill_cannot(self) -> None:
        """This is the whole point of the generalized estimator.

        A Uniform(0,1) sample has a finite upper endpoint, so gamma = -1. The
        Hill estimator is only defined for gamma > 0 and cannot represent this
        at all; the generalized Hill estimator recovers it.
        """
        rnd = random.Random(3)
        data = [rnd.random() for _ in range(20000)]
        assert generalized_hill_estimator(data, k=1000) == pytest.approx(-1.0, abs=0.15)
        # Hill, by construction, cannot return a negative value.
        assert hill_estimator(data, k=1000) > 0.0

    def test_agrees_with_hill_where_both_apply(self) -> None:
        """For a heavy tail and a well-chosen k the two should be close."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(50000, seed=8)
        gh = generalized_hill_estimator(data, k=2500)
        h = hill_estimator(data, k=2500)
        assert abs(gh - h) < 0.05

    @pytest.mark.parametrize("k", [0, 1, -5])
    def test_rejects_k_below_the_valid_range(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(100, seed=1)
        with pytest.raises(ValueError, match="k must satisfy"):
            generalized_hill_estimator(data, k)

    def test_rejects_k_at_or_above_the_sample_size(self) -> None:
        """It needs one more order statistic than Hill, for the UH_{k+1} term."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(50, seed=1)
        with pytest.raises(ValueError, match="k must satisfy"):
            generalized_hill_estimator(data, 49)

    def test_rejects_non_positive_data(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            generalized_hill_estimator([1.0, 2.0, 3.0, 0.0, -1.0], k=2)

    def test_is_scale_invariant(self) -> None:
        """gamma is a shape property; rescaling the data must not change it."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=6)
        base = generalized_hill_estimator(data, k=250)
        scaled = generalized_hill_estimator([1000.0 * x for x in data], k=250)
        assert scaled == pytest.approx(base, rel=1e-9)


class TestHillPlot:
    """The sweep across k that the documentation tells people to look at."""

    def test_returns_ordered_pairs(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=2)
        points = hill_plot(data)
        assert len(points) > 10
        ks = [k for k, _ in points]
        assert ks == sorted(ks)
        assert len(set(ks)) == len(ks), "k values must not repeat"

    def test_plateau_sits_near_the_true_index(self) -> None:
        """The reason the plot is the recommended tool."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=2)
        plateau = [g for k, g in hill_plot(data) if 100 <= k <= 800]
        assert sum(plateau) / len(plateau) == pytest.approx(0.5, abs=0.05)

    def test_accepts_explicit_k_values(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=2)
        points = hill_plot(data, ks=[10, 50, 100])
        assert [k for k, _ in points] == [10, 50, 100]

    def test_silently_drops_k_outside_the_valid_range(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(100, seed=2)
        points = hill_plot(data, ks=[1, 50, 100, 500])
        assert [k for k, _ in points] == [50]

    def test_rejects_a_sample_too_small_to_plot(self) -> None:
        with pytest.raises(ValueError, match="at least 8"):
            hill_plot([1.0, 2.0, 3.0])

    def test_each_point_matches_the_hill_estimator(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=9)
        for k, gamma in hill_plot(data, ks=[20, 100, 400]):
            assert gamma == pytest.approx(hill_estimator(data, k), rel=1e-12)


class TestConfidenceIntervals:
    """An estimate without an interval is not usable."""

    def test_asymptotic_interval_brackets_the_estimate(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)
        r = tail_index_confidence_interval(data, k=250)
        assert r["lower"] < r["gamma"] < r["upper"]
        assert r["method"] == "asymptotic"
        assert r["alpha"] == pytest.approx(1.0 / r["gamma"])

    def test_asymptotic_interval_matches_the_closed_form(self) -> None:
        """gamma_hat * (1 +/- z / sqrt(k)) with z the normal quantile."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)
        k = 250
        r = tail_index_confidence_interval(data, k=k, level=0.95)
        z = 1.959963984540054
        half = z * r["gamma"] / math.sqrt(k)
        assert r["lower"] == pytest.approx(r["gamma"] - half, rel=1e-6)
        assert r["upper"] == pytest.approx(r["gamma"] + half, rel=1e-6)

    def test_a_higher_level_gives_a_wider_interval(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)
        narrow = tail_index_confidence_interval(data, k=250, level=0.80)
        wide = tail_index_confidence_interval(data, k=250, level=0.99)
        assert (wide["upper"] - wide["lower"]) > (narrow["upper"] - narrow["lower"])

    def test_the_interval_narrows_as_k_grows(self) -> None:
        """Standard error is gamma / sqrt(k)."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=7)
        small = tail_index_confidence_interval(data, k=100)
        large = tail_index_confidence_interval(data, k=2000)
        assert (large["upper"] - large["lower"]) < (small["upper"] - small["lower"])

    @pytest.mark.parametrize(
        "estimator", ["hill", "generalized_hill", "moment", "pickands"]
    )
    def test_bootstrap_works_for_every_estimator(self, estimator: str) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=7)
        r = tail_index_confidence_interval(
            data,
            k=100,
            estimator=estimator,
            method="bootstrap",
            n_bootstrap=60,
            seed=1,
        )
        assert r["lower"] <= r["upper"]
        assert r["estimator"] == estimator

    def test_bootstrap_is_reproducible_from_a_seed(self) -> None:
        """Every random operation in the library is seedable; this is no exception."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        kwargs = {"k": 100, "method": "bootstrap", "n_bootstrap": 50, "seed": 42}
        first = tail_index_confidence_interval(data, **kwargs)
        second = tail_index_confidence_interval(data, **kwargs)
        assert first["lower"] == second["lower"]
        assert first["upper"] == second["upper"]

    def test_different_seeds_give_different_intervals(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        a = tail_index_confidence_interval(
            data, k=100, method="bootstrap", n_bootstrap=50, seed=1
        )
        b = tail_index_confidence_interval(
            data, k=100, method="bootstrap", n_bootstrap=50, seed=2
        )
        assert (a["lower"], a["upper"]) != (b["lower"], b["upper"])

    def test_asymptotic_is_refused_for_the_other_estimators(self) -> None:
        """Only the Hill estimator has an established closed-form interval here.

        Reporting one for the others would be inventing a result.
        """
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        with pytest.raises(ValueError, match="only established for the Hill"):
            tail_index_confidence_interval(
                data, k=100, estimator="moment", method="asymptotic"
            )

    def test_rejects_an_unknown_estimator(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        with pytest.raises(ValueError, match="Available"):
            tail_index_confidence_interval(data, k=100, estimator="nonsense")

    def test_rejects_an_unknown_method(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        with pytest.raises(ValueError, match="Available"):
            tail_index_confidence_interval(data, k=100, method="nonsense")

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 2.0])
    def test_rejects_an_invalid_level(self, level: float) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        with pytest.raises(ValueError, match="level"):
            tail_index_confidence_interval(data, k=100, level=level)

    def test_rejects_too_few_bootstrap_resamples(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=7)
        with pytest.raises(ValueError, match="n_bootstrap"):
            tail_index_confidence_interval(
                data, k=100, method="bootstrap", n_bootstrap=1
            )

    @pytest.mark.slow
    def test_coverage_is_close_to_the_nominal_level(self) -> None:
        """The property an interval actually has to have.

        Coverage runs slightly below nominal because the interval captures
        sampling variance but not the bias introduced by the choice of k. That
        is a real limitation and is documented rather than tuned away.
        """
        trials = 100
        covered = sum(
            tail_index_confidence_interval(
                Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=seed), k=1000
            )["lower"]
            <= 0.5
            <= tail_index_confidence_interval(
                Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=seed), k=1000
            )["upper"]
            for seed in range(trials)
        )
        assert 0.85 <= covered / trials <= 1.0


class TestExistingEstimatorsStillAgree:
    """The refactor moved _phi_inverse; make sure nothing else shifted."""

    def test_hill_is_unchanged(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
        x = sorted(data, reverse=True)
        k = 250
        expected = sum(math.log(x[i] / x[k]) for i in range(k)) / k
        assert hill_estimator(data, k) == pytest.approx(expected, rel=1e-15)

    def test_moment_still_returns_gamma_and_alpha(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
        gamma, alpha = moment_estimator(data, 250)
        assert alpha == pytest.approx(1.0 / gamma, rel=1e-12)
