"""The trimmed Hill estimator and its robustness to contamination.

Robustness is invisible on clean data, so most of these tests deliberately
damage the sample first. The Hill estimator gives enormous leverage to the
largest order statistics, because they enter through unbounded logarithms of
ratios, and a handful of bad observations is enough to destroy it.
"""

from __future__ import annotations

import statistics

import pytest

from heavytails import Pareto
from heavytails.tail_index import (
    hill_estimator,
    tail_index_confidence_interval,
    trimmed_hill_estimator,
    trimmed_hill_plot,
)


def contaminate(sample: list[float], count: int, magnitude: float = 1e9) -> list[float]:
    """Replace the ``count`` largest observations with outliers."""
    ordered = sorted(sample, reverse=True)
    for i in range(count):
        ordered[i] = magnitude / (i + 1)
    return ordered


class TestAgreementWithHill:
    """With no trimming it must be the Hill estimator."""

    def test_r_zero_reproduces_hill(self) -> None:
        """The two formulations are algebraically identical.

        Hill sums ``log(X_(i)/X_(k+1))``; this sums the normalised spacings
        ``i*(log X_(i) - log X_(i+1))``, which telescope to the same thing. They
        differ only in floating-point summation order, by about one unit in the
        last place.
        """
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
        assert trimmed_hill_estimator(data, 250, r=0) == pytest.approx(
            hill_estimator(data, 250), rel=1e-12
        )

    def test_is_scale_invariant(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=6)
        base = trimmed_hill_estimator(data, k=250, r=5)
        scaled = trimmed_hill_estimator([1000.0 * x for x in data], k=250, r=5)
        assert scaled == pytest.approx(base, rel=1e-9)


class TestConsistencyOnCleanData:
    """Trimming must not break the estimator when nothing is wrong."""

    @pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
    @pytest.mark.parametrize("r", [0, 5, 20])
    def test_recovers_the_tail_index(self, alpha: float, r: int) -> None:
        data = Pareto(alpha=alpha, xm=1.0).rvs(20000, seed=11)
        assert trimmed_hill_estimator(data, k=1000, r=r) == pytest.approx(
            1.0 / alpha, abs=0.04
        )

    @pytest.mark.slow
    def test_trimming_costs_little_efficiency(self) -> None:
        """The trade-off that makes trimming worth doing by default.

        Discarding ten observations raises the standard deviation by only a few
        percent, which is a small price for surviving contamination.
        """
        untrimmed = []
        trimmed = []
        for seed in range(150):
            data = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=seed)
            untrimmed.append(trimmed_hill_estimator(data, 300, r=0))
            trimmed.append(trimmed_hill_estimator(data, 300, r=10))
        inflation = statistics.stdev(trimmed) / statistics.stdev(untrimmed)
        assert 1.0 <= inflation < 1.15


class TestRobustness:
    """The property the estimator exists for."""

    def test_hill_is_destroyed_by_three_outliers(self) -> None:
        """Pins the premise. Without this the tests below prove nothing."""
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=0), count=3)
        assert hill_estimator(data, 500) > 0.55  # true value is 0.5

    def test_trimming_more_than_the_contamination_recovers_the_estimate(self) -> None:
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=0), count=3)
        assert trimmed_hill_estimator(data, 500, r=5) == pytest.approx(0.5, abs=0.02)

    def test_trimming_less_than_the_contamination_does_not_help(self) -> None:
        """The rule that has to be stated: r must exceed the contamination.

        Trimming two when three are contaminated leaves the third entering
        through a spacing, and the estimate is essentially as bad as untrimmed.
        """
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=0), count=10)
        assert trimmed_hill_estimator(data, 500, r=5) > 0.55

    @pytest.mark.parametrize("count", [1, 3, 8])
    def test_recovers_across_contamination_levels(self, count: int) -> None:
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=3), count=count)
        assert trimmed_hill_estimator(data, 500, r=count + 3) == pytest.approx(
            0.5, abs=0.03
        )

    def test_robust_to_the_magnitude_of_the_outliers(self) -> None:
        """Once trimmed, how extreme the outliers were stops mattering."""
        base = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=4)
        estimates = [
            trimmed_hill_estimator(contaminate(base, 3, magnitude=m), 500, r=5)
            for m in (1e3, 1e9, 1e30)
        ]
        assert max(estimates) - min(estimates) < 1e-9


class TestTrimmedHillPlot:
    """The diagnostic for choosing r."""

    def test_elbow_falls_at_the_contamination_count(self) -> None:
        """The plot is readable in the way the documentation claims.

        The estimate moves while r is below the number of contaminated
        observations and flattens once they are all discarded, so the elbow
        says how much contamination is present.
        """
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1), count=3)
        points = dict(trimmed_hill_plot(data, k=300, max_trim=8))

        # Still contaminated below r = 3.
        assert points[0] > 0.6
        assert points[2] > 0.6
        # Clean from r = 3 onwards, and stable thereafter.
        assert points[3] == pytest.approx(0.5, abs=0.03)
        assert abs(points[8] - points[3]) < 0.01

    def test_is_flat_on_clean_data(self) -> None:
        """No elbow means no contamination."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1)
        values = [g for _, g in trimmed_hill_plot(data, k=300, max_trim=10)]
        assert max(values) - min(values) < 0.02

    def test_matches_direct_evaluation(self) -> None:
        """The incremental computation must equal the direct one."""
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1), count=3)
        for r, gamma in trimmed_hill_plot(data, k=250, max_trim=6):
            assert gamma == pytest.approx(
                trimmed_hill_estimator(data, 250, r), rel=1e-12
            )

    def test_starts_at_no_trimming(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=1)
        points = trimmed_hill_plot(data, k=100)
        assert points[0][0] == 0
        assert [r for r, _ in points] == sorted(r for r, _ in points)

    def test_rejects_an_invalid_max_trim(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=1)
        with pytest.raises(ValueError, match="max_trim"):
            trimmed_hill_plot(data, k=100, max_trim=100)


class TestArgumentValidation:
    @pytest.mark.parametrize("r", [-1, -10])
    def test_rejects_negative_trimming(self, r: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="0 <= r < k"):
            trimmed_hill_estimator(data, k=100, r=r)

    @pytest.mark.parametrize("r", [100, 200])
    def test_rejects_trimming_everything(self, r: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="0 <= r < k"):
            trimmed_hill_estimator(data, k=100, r=r)

    @pytest.mark.parametrize("k", [0, 1, 1000])
    def test_rejects_k_out_of_range(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="k must be between"):
            trimmed_hill_estimator(data, k=k)

    def test_rejects_non_positive_data(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            trimmed_hill_estimator([5.0, 4.0, 3.0, 2.0, 1.0, 0.0, -1.0], k=5)


class TestIntervalIntegration:
    """estimator_kwargs is what makes the estimator usable from the helper."""

    def test_kwargs_reach_the_estimator(self) -> None:
        """Without them, trimmed_hill would silently run at r = 0.

        That is not a cosmetic problem: r = 0 is the ordinary Hill estimator,
        so the caller would get no robustness at all while believing otherwise.
        """
        data = contaminate(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1), count=3)
        default = tail_index_confidence_interval(
            data,
            k=300,
            estimator="trimmed_hill",
            method="bootstrap",
            n_bootstrap=40,
            seed=1,
        )
        trimmed = tail_index_confidence_interval(
            data,
            k=300,
            estimator="trimmed_hill",
            method="bootstrap",
            n_bootstrap=40,
            seed=1,
            estimator_kwargs={"r": 5},
        )
        assert default["gamma"] > 0.6
        assert trimmed["gamma"] == pytest.approx(0.5, abs=0.03)
        assert trimmed["estimator_kwargs"] == {"r": 5}

    def test_kwargs_work_for_the_smoothed_estimator_too(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator="smoothed_hill",
            method="bootstrap",
            n_bootstrap=40,
            seed=1,
            estimator_kwargs={"u": 3.0},
        )
        assert result["estimator_kwargs"] == {"u": 3.0}

    def test_omitting_kwargs_is_still_valid(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
        result = tail_index_confidence_interval(data, k=250)
        assert result["estimator_kwargs"] == {}


class TestNotation:
    """Every estimator returns gamma, not alpha. See issue #322."""

    def test_estimators_return_gamma_not_alpha(self) -> None:
        """gamma = 1/alpha, so a Pareto(2) sample gives about 0.5, not 2."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        for value in (
            hill_estimator(data, 1000),
            trimmed_hill_estimator(data, 1000, r=5),
        ):
            assert value == pytest.approx(0.5, abs=0.05)

    def test_interval_reports_both_conventions(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        result = tail_index_confidence_interval(data, k=1000)
        assert result["gamma"] == pytest.approx(0.5, abs=0.05)
        assert result["alpha"] == pytest.approx(1.0 / result["gamma"], rel=1e-12)
