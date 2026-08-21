"""Threshold selection diagnostics for peaks-over-threshold analysis.

Most of these tests exploit a convenient fact: for a Pareto tail the
generalized Pareto model holds *exactly* above every threshold, with the same
shape and a scale proportional to the threshold. So the diagnostics have known
answers rather than merely plausible ones.

For ``Pareto(alpha)`` above ``u``:

* the excesses are ``GPD(xi = 1/alpha, sigma = u/alpha)``
* the mean excess is ``u / (alpha - 1)``, linear in ``u``
* the modified scale ``sigma_u - xi * u`` is exactly zero
"""

from __future__ import annotations

import pytest

import heavytails
from heavytails import GeneralizedPareto, LogNormal, Pareto
from heavytails.threshold import (
    mean_residual_life,
    parameter_stability,
    return_level,
    select_threshold,
)

N = 20000

# A coarse sweep: the diagnostics are about shape across thresholds, and
# fitting forty generalized Paretos per call makes the suite needlessly slow.
SWEEP = [1.9, 2.1, 2.4, 2.8, 3.3, 4.0, 5.0, 6.5, 9.0, 13.0]


@pytest.fixture(scope="module")
def pareto_sample() -> list[float]:
    return Pareto(alpha=2.0, xm=1.0).rvs(N, seed=1)


class TestMeanResidualLife:
    def test_is_linear_for_a_pareto_tail(self, pareto_sample: list[float]) -> None:
        """e(u) = u / (alpha - 1), which is u for alpha = 2.

        Linearity is the whole point of the diagnostic, so it is worth
        asserting against the known slope rather than just checking the plot
        is monotone.
        """
        points = [
            p for p in mean_residual_life(pareto_sample) if p["n_exceedances"] > 500
        ]
        assert len(points) > 5
        for point in points:
            assert point["mean_excess"] == pytest.approx(point["threshold"], rel=0.15)

    def test_intervals_bracket_the_estimate(self, pareto_sample: list[float]) -> None:
        for point in mean_residual_life(pareto_sample):
            assert point["lower"] < point["mean_excess"] < point["upper"]

    def test_intervals_widen_as_exceedances_run_out(
        self, pareto_sample: list[float]
    ) -> None:
        """The right-hand end of the plot is always noisy, and must look it."""
        points = mean_residual_life(pareto_sample, thresholds=SWEEP)
        widths = [(p["upper"] - p["lower"]) / p["mean_excess"] for p in points]
        assert widths[-1] > widths[0]

    def test_thresholds_are_increasing(self, pareto_sample: list[float]) -> None:
        thresholds = [p["threshold"] for p in mean_residual_life(pareto_sample)]
        assert thresholds == sorted(thresholds)

    def test_accepts_explicit_thresholds(self, pareto_sample: list[float]) -> None:
        points = mean_residual_life(pareto_sample, thresholds=[2.0, 5.0, 10.0])
        assert [p["threshold"] for p in points] == [2.0, 5.0, 10.0]

    def test_drops_thresholds_with_too_few_exceedances(
        self, pareto_sample: list[float]
    ) -> None:
        points = mean_residual_life(pareto_sample, thresholds=[2.0, 1e12])
        assert [p["threshold"] for p in points] == [2.0]

    def test_rejects_a_tiny_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            mean_residual_life([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5])
    def test_rejects_an_invalid_level(
        self, pareto_sample: list[float], level: float
    ) -> None:
        with pytest.raises(ValueError, match="level must be in"):
            mean_residual_life(pareto_sample, level=level)


class TestParameterStability:
    def test_shape_is_stable_for_a_pareto_tail(
        self, pareto_sample: list[float]
    ) -> None:
        """xi = 1/alpha at every threshold, so the plot must be flat."""
        points = [
            p
            for p in parameter_stability(pareto_sample, thresholds=SWEEP)
            if p["n_exceedances"] > 500
        ]
        assert len(points) > 5
        for point in points:
            assert point["xi"] == pytest.approx(0.5, abs=0.08)

    def test_modified_scale_is_near_zero_for_a_pareto_tail(
        self, pareto_sample: list[float]
    ) -> None:
        """sigma_u = u/alpha and xi = 1/alpha, so sigma_u - xi*u is exactly 0.

        This is the sharper of the two checks: the raw scale grows with the
        threshold and only the modified one is constant, so a sign error in
        the modification would show up here and nowhere else.
        """
        points = [
            p
            for p in parameter_stability(pareto_sample, thresholds=SWEEP)
            if p["n_exceedances"] > 1000
        ]
        for point in points:
            assert abs(point["modified_scale"]) < 0.15 * point["threshold"]

    def test_raw_scale_grows_with_the_threshold(
        self, pareto_sample: list[float]
    ) -> None:
        """The contrast that makes the modified scale worth reporting."""
        points = [
            p
            for p in parameter_stability(pareto_sample, thresholds=SWEEP)
            if p["n_exceedances"] > 500
        ]
        assert points[-1]["sigma"] > points[0]["sigma"] * 1.5

    def test_skips_thresholds_below_the_exceedance_floor(
        self, pareto_sample: list[float]
    ) -> None:
        points = parameter_stability(
            pareto_sample, thresholds=[2.0, 1e12], min_exceedances=30
        )
        assert [p["threshold"] for p in points] == [2.0]

    def test_rejects_a_tiny_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 10"):
            parameter_stability([1.0, 2.0, 3.0])


class TestSelectThreshold:
    def test_finds_a_threshold_for_a_pareto_tail(
        self, pareto_sample: list[float]
    ) -> None:
        """The model holds at every threshold, so one must be found."""
        result = select_threshold(pareto_sample)
        assert result["threshold"] is not None
        assert result["xi"] == pytest.approx(0.5, abs=0.1)
        assert result["n_exceedances"] >= 50

    def test_selects_the_lowest_passing_candidate(
        self, pareto_sample: list[float]
    ) -> None:
        """Lower keeps more data, so the rule stops at the first pass.

        Because the p-values are conservative, that first pass tends to be the
        very lowest candidate; the docstring warns about exactly this.
        """
        result = select_threshold(pareto_sample)
        rejected = [c for c in result["candidates_tested"] if c.get("rejected")]
        assert rejected == []

    def test_reports_every_candidate_it_tried(self, pareto_sample: list[float]) -> None:
        """A failure has to be inspectable rather than opaque."""
        result = select_threshold(pareto_sample, thresholds=[1.5, 2.0, 5.0])
        assert len(result["candidates_tested"]) >= 1

    def test_reports_no_threshold_rather_than_guessing(self) -> None:
        """A sample with no exceedances anywhere yields None, not a fabrication."""
        result = select_threshold([1.0] * 100, thresholds=[10.0, 20.0])
        assert result["threshold"] is None
        assert result["xi"] is None

    def test_exceedance_rate_is_consistent(self, pareto_sample: list[float]) -> None:
        result = select_threshold(pareto_sample)
        assert result["exceedance_rate"] == pytest.approx(
            result["n_exceedances"] / len(pareto_sample), rel=1e-12
        )

    def test_rejects_a_tiny_sample(self) -> None:
        with pytest.raises(ValueError, match="at least 20"):
            select_threshold([1.0, 2.0, 3.0])

    @pytest.mark.parametrize("alpha_level", [0.0, 1.0, 2.0])
    def test_rejects_an_invalid_alpha_level(
        self, pareto_sample: list[float], alpha_level: float
    ) -> None:
        with pytest.raises(ValueError, match="alpha_level"):
            select_threshold(pareto_sample, alpha_level=alpha_level)


class TestReturnLevel:
    @pytest.mark.parametrize("period", [200, 1000, 10000])
    def test_recovers_the_true_quantile(
        self, pareto_sample: list[float], period: int
    ) -> None:
        """The return level for period T is the 1 - 1/T quantile."""
        truth = Pareto(alpha=2.0, xm=1.0).ppf(1.0 - 1.0 / period)
        result = return_level(
            pareto_sample, threshold=5.0, period=period, n_bootstrap=30, seed=1
        )
        assert result["return_level"] == pytest.approx(truth, rel=0.25)

    def test_interval_brackets_the_estimate(self, pareto_sample: list[float]) -> None:
        result = return_level(
            pareto_sample, threshold=5.0, period=1000, n_bootstrap=40, seed=1
        )
        assert result["lower"] <= result["return_level"] <= result["upper"]

    def test_interval_widens_with_the_period(self, pareto_sample: list[float]) -> None:
        """Extrapolating further should be reported as less certain."""
        near = return_level(
            pareto_sample, threshold=5.0, period=200, n_bootstrap=40, seed=1
        )
        far = return_level(
            pareto_sample, threshold=5.0, period=10000, n_bootstrap=40, seed=1
        )
        near_width = (near["upper"] - near["lower"]) / near["return_level"]
        far_width = (far["upper"] - far["lower"]) / far["return_level"]
        assert far_width > near_width

    def test_increases_with_the_period(self, pareto_sample: list[float]) -> None:
        levels = [
            return_level(
                pareto_sample, threshold=5.0, period=T, n_bootstrap=20, seed=1
            )["return_level"]
            for T in (200, 1000, 10000)
        ]
        assert levels == sorted(levels)

    def test_is_reproducible_from_a_seed(self, pareto_sample: list[float]) -> None:
        kwargs = {"threshold": 5.0, "period": 1000, "n_bootstrap": 25, "seed": 7}
        first = return_level(pareto_sample, **kwargs)
        second = return_level(pareto_sample, **kwargs)
        assert first == second

    def test_refuses_a_period_inside_the_body(self, pareto_sample: list[float]) -> None:
        """A return level at or below the threshold is not something the model
        can speak to, so it raises rather than extrapolating backwards."""
        with pytest.raises(ValueError, match="return level at or below"):
            # threshold=2 leaves roughly a quarter of the sample, so the
            # exceedance-count guard does not fire first and the period guard
            # is what is being tested.
            return_level(pareto_sample, threshold=2.0, period=2, n_bootstrap=10)

    def test_refuses_a_threshold_with_too_few_exceedances(
        self, pareto_sample: list[float]
    ) -> None:
        with pytest.raises(ValueError, match="exceed"):
            return_level(pareto_sample, threshold=1e9, period=1000, n_bootstrap=10)

    @pytest.mark.parametrize("period", [0.5, 1.0, -10])
    def test_rejects_an_invalid_period(
        self, pareto_sample: list[float], period: float
    ) -> None:
        with pytest.raises(ValueError, match="period"):
            return_level(pareto_sample, threshold=5.0, period=period)

    def test_rejects_too_few_bootstrap_resamples(
        self, pareto_sample: list[float]
    ) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            return_level(pareto_sample, threshold=5.0, period=1000, n_bootstrap=1)

    @pytest.mark.slow
    def test_interval_coverage_is_near_the_nominal_level(self) -> None:
        """The property an interval has to have.

        Coverage runs slightly under nominal because the bootstrap captures
        sampling variability but not the error from choosing the threshold.
        That is documented rather than tuned away.
        """
        truth = Pareto(alpha=2.0, xm=1.0).ppf(1.0 - 1.0 / 1000)
        covered = 0
        trials = 25
        for seed in range(trials):
            sample = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=seed)
            result = return_level(
                sample, threshold=5.0, period=1000, n_bootstrap=20, seed=seed
            )
            covered += result["lower"] <= truth <= result["upper"]
        # Measured around 0.88 at this sample size. The bound is loose because
        # a proportion from 25 trials carries a standard error of about 0.065,
        # and because coverage degrades further on smaller samples: at
        # n = 8000 it falls to roughly 0.76, which the docstring records.
        assert 0.70 <= covered / trials <= 1.0


class TestOnOtherDistributions:
    """The diagnostics must not be tuned to Pareto."""

    def test_works_on_a_generalized_pareto_sample(self) -> None:
        data = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0).rvs(N, seed=2)
        points = [
            p
            for p in parameter_stability(data, thresholds=SWEEP)
            if p["n_exceedances"] > 500
        ]
        for point in points:
            assert point["xi"] == pytest.approx(0.3, abs=0.12)

    def test_works_on_a_lognormal_sample(self) -> None:
        """Lognormal is not Pareto-tailed, so the shape should sit near zero."""
        data = LogNormal(mu=0.0, sigma=1.0).rvs(N, seed=2)
        points = [
            p
            for p in parameter_stability(data, thresholds=SWEEP)
            if p["n_exceedances"] > 1000
        ]
        assert all(abs(p["xi"]) < 0.4 for p in points)


class TestExports:
    def test_everything_is_exported(self) -> None:
        for name in (
            "mean_residual_life",
            "parameter_stability",
            "select_threshold",
            "return_level",
        ):
            assert hasattr(heavytails, name)
            assert name in heavytails.__all__
