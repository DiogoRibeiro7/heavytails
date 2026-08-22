"""Data-driven choice of the trimming parameter for the trimmed Hill estimator.

Two claims have to hold for this to be worth having, and they pull against each
other:

1. it must find the contamination -- and the *right amount* of it, not merely
   some;
2. it must cost nothing on clean data, or it is just fixed trimming with extra
   steps.

Both are measured here rather than asserted loosely. The first is checked by
planting a known number of outliers and asserting the median choice equals that
number; the second by comparing the standard deviation against the plain Hill
estimator on the same clean samples.

The p-value has an exact null distribution, so its calibration is checked
directly rather than through the estimator that uses it.
"""

from __future__ import annotations

import math
import statistics

import pytest

from heavytails import Pareto
from heavytails.tail_index import (
    _normalised_log_spacings,
    _spacing_p_value,
    adaptive_trim_selection,
    adaptive_trimmed_hill_estimator,
    hill_estimator,
    tail_index_confidence_interval,
    trimmed_hill_estimator,
)

K = 300
N = 10_000


def contaminate(seed: int, count: int, magnitude: float = 1e6) -> list[float]:
    """A Pareto(2) sample with ``count`` of its largest values replaced."""
    x = sorted(Pareto(alpha=2.0, xm=1.0).rvs(N, seed=seed), reverse=True)
    for j in range(count):
        x[j] = magnitude * (j + 1)
    return sorted(x, reverse=True)


class TestSpacingPValue:
    """The null distribution is exact, so it can be checked as one."""

    def test_it_is_uniform_under_the_null(self) -> None:
        """``P(R > t) = (m/(m+t))^m`` exactly, by a Beta(1, m) argument.

        No asymptotics and no tabulated critical values, so the p-value must be
        uniform on (0,1) for any ``m``. If it were not, every level below would
        mean something other than what it says.
        """
        values = []
        for seed in range(2000):
            x = sorted(Pareto(alpha=2.0, xm=1.0).rvs(400, seed=seed), reverse=True)
            values.append(_spacing_p_value(_normalised_log_spacings(x, K), 0))
        for nominal in [0.01, 0.05, 0.10, 0.25, 0.50]:
            observed = sum(1 for p in values if p < nominal) / len(values)
            assert observed == pytest.approx(
                nominal, abs=3.5 * math.sqrt(nominal * (1 - nominal) / len(values))
            )

    def test_it_falls_when_a_spacing_is_inflated(self) -> None:
        x = contaminate(seed=1, count=1)
        spacings = _normalised_log_spacings(x, K)
        assert _spacing_p_value(spacings, 0) < 1e-6
        assert _spacing_p_value(spacings, 5) > 0.01

    def test_it_lies_in_the_unit_interval(self) -> None:
        x = sorted(Pareto(alpha=2.0, xm=1.0).rvs(N, seed=2), reverse=True)
        spacings = _normalised_log_spacings(x, K)
        assert all(0.0 <= _spacing_p_value(spacings, j) <= 1.0 for j in range(50))


class TestChoosingTheTrimming:
    @pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 8])
    def test_the_median_choice_equals_the_number_planted(self, count: int) -> None:
        """The estimator recovers how much contamination there is.

        Getting *some* trimming would be easy and nearly useless: too little
        leaves the outliers in, too much throws away good observations. The
        median choice matching the planted count is the claim worth making.
        """
        picks = [
            adaptive_trim_selection(contaminate(seed, count), K)["trim"]
            for seed in range(120)
        ]
        assert statistics.median(picks) == count

    def test_several_outliers_are_found_even_though_the_gaps_between_them_are_small(
        self,
    ) -> None:
        """The reason the scan runs from the deepest spacing upwards.

        Outliers of similar size sit close together, so only the gap *below* the
        last one is large. A rule that stopped at the first ordinary-looking
        spacing would report a badly contaminated sample as clean, which is
        exactly what the first version of this did.
        """
        x = contaminate(seed=3, count=5)
        spacings = _normalised_log_spacings(x, K)
        assert _spacing_p_value(spacings, 0) > 0.05  # the gap between outliers
        assert _spacing_p_value(spacings, 4) < 1e-6  # the gap below them
        assert adaptive_trim_selection(x, K)["trim"] == 5

    @pytest.mark.parametrize(
        ("multiple", "floor"), [(3.0, 0.85), (5.0, 0.97), (1e4, 1.0)]
    )
    def test_detection_improves_with_the_size_of_the_outlier(
        self, multiple: float, floor: float
    ) -> None:
        """Detection is a rate, not a certainty, and the docstring says which.

        Asserting a floor that rises with the outlier size states what is
        actually true instead of pretending detection is guaranteed.
        """
        found = 0
        for seed in range(150):
            x = sorted(Pareto(alpha=2.0, xm=1.0).rvs(N, seed=seed), reverse=True)
            top = x[0]
            for j in range(3):
                x[j] = top * multiple * (1.0 + 0.1 * j)
            found += adaptive_trim_selection(sorted(x, reverse=True), K)["trim"] >= 3
        assert found / 150 >= floor

    def test_a_barely_extreme_outlier_is_admitted_to_be_hard(self) -> None:
        """Half again the sample maximum: found about half the time.

        Nothing could do better -- an outlier that close to the largest genuine
        observation is not reliably distinguishable from the tail itself. The
        bound is two-sided so the claim cannot quietly drift in either
        direction.
        """
        found = 0
        for seed in range(150):
            x = sorted(Pareto(alpha=2.0, xm=1.0).rvs(N, seed=seed), reverse=True)
            top = x[0]
            for j in range(3):
                x[j] = top * 1.5 * (1.0 + 0.1 * j)
            found += adaptive_trim_selection(sorted(x, reverse=True), K)["trim"] >= 3
        assert 0.30 < found / 150 < 0.65

    @pytest.mark.parametrize("level", [0.01, 0.05, 0.10])
    def test_clean_data_is_over_trimmed_at_about_the_stated_level(
        self, level: float
    ) -> None:
        """The Bonferroni correction is what makes the level mean something.

        Without it, scanning 75 spacings at 0.05 each would over-trim almost
        every clean sample.
        """
        samples = 600
        over = sum(
            1
            for seed in range(samples)
            if adaptive_trim_selection(contaminate(seed, 0), K, level=level)["trim"] > 0
        )
        assert over / samples == pytest.approx(level, abs=0.025)

    def test_it_reports_the_tests_behind_the_choice(self) -> None:
        result = adaptive_trim_selection(contaminate(seed=1, count=2), K, max_trim=20)
        assert len(result["p_values"]) == 20
        assert result["p_values"][1] < result["p_values"][10]
        assert result["gamma"] == pytest.approx(
            trimmed_hill_estimator(contaminate(1, 2), K, r=result["trim"])
        )


class TestTheEstimator:
    @pytest.mark.parametrize("count", [0, 1, 3, 8])
    def test_it_recovers_the_clean_sample_estimate(self, count: int) -> None:
        """The right target is the clean-data answer, not the true gamma.

        Comparing against 0.5 would conflate robustness with the estimator's own
        finite-sample error, which is the same for both. What robustness means
        is that contamination does not move the answer.
        """
        for seed in range(25):
            clean = Pareto(alpha=2.0, xm=1.0).rvs(N, seed=seed)
            reference = hill_estimator(clean, k=K)
            adaptive = adaptive_trimmed_hill_estimator(contaminate(seed, count), k=K)
            assert adaptive == pytest.approx(reference, abs=0.03)

    def test_it_costs_nothing_measurable_on_clean_data(self) -> None:
        """Robustness usually costs variance. Here it does not.

        Trimming is applied only when the data asks for it, so on clean samples
        the estimator is the Hill estimator almost every time.
        """
        adaptive, plain = [], []
        for seed in range(400):
            clean = Pareto(alpha=2.0, xm=1.0).rvs(N, seed=seed)
            adaptive.append(adaptive_trimmed_hill_estimator(clean, k=K))
            plain.append(hill_estimator(clean, k=K))
        assert statistics.stdev(adaptive) == pytest.approx(
            statistics.stdev(plain), rel=0.10
        )
        assert statistics.fmean(adaptive) == pytest.approx(0.5, abs=0.01)

    def test_it_beats_the_plain_hill_estimator_under_contamination(self) -> None:
        adaptive, plain = [], []
        for seed in range(60):
            x = contaminate(seed, 5)
            adaptive.append(adaptive_trimmed_hill_estimator(x, k=K))
            plain.append(hill_estimator(x, k=K))
        assert abs(statistics.fmean(adaptive) - 0.5) < 0.02
        assert statistics.fmean(plain) > 0.6

    @pytest.mark.parametrize("alpha", [0.7, 1.0, 2.0, 4.0])
    def test_it_works_across_tail_indices(self, alpha: float) -> None:
        """Nothing in the rule depends on the scale of gamma.

        The spacings are exponential with mean gamma whatever gamma is, and the
        test compares one against the mean of the others, so it is scale free.
        """
        errors = []
        for seed in range(30):
            x = sorted(Pareto(alpha=alpha, xm=1.0).rvs(N, seed=seed), reverse=True)
            reference = hill_estimator(x, k=K)
            for j in range(4):
                x[j] = x[0] * 500.0 * (j + 1)
            estimate = adaptive_trimmed_hill_estimator(sorted(x, reverse=True), k=K)
            errors.append(abs(estimate - reference))
        assert statistics.fmean(errors) < 0.05 * (1.0 / alpha)

    def test_no_contamination_leaves_the_hill_estimator_untouched(self) -> None:
        """When nothing is trimmed the two must agree exactly, not nearly."""
        clean = Pareto(alpha=2.0, xm=1.0).rvs(N, seed=11)
        if adaptive_trim_selection(clean, K)["trim"] == 0:
            assert adaptive_trimmed_hill_estimator(clean, k=K) == pytest.approx(
                hill_estimator(clean, k=K), rel=1e-15
            )


class TestTheFailureModeIsReported:
    def test_contamination_deeper_than_the_scan_raises(self) -> None:
        """The one outcome worse than no estimate: a wrong one that looks clean.

        With 30 outliers and ``max_trim = 20`` every scanned spacing is a gap
        *between* outliers, so nothing looks anomalous and the estimator would
        report 1.79 for a true 0.5 while claiming no contamination. It is an
        error rather than a number.
        """
        x = contaminate(seed=1, count=30)
        with pytest.raises(ValueError, match="deeper than max_trim"):
            adaptive_trimmed_hill_estimator(x, k=K, max_trim=20)

    def test_the_message_says_what_to_raise_the_limit_to(self) -> None:
        x = contaminate(seed=1, count=30)
        with pytest.raises(ValueError, match="30"):
            adaptive_trimmed_hill_estimator(x, k=K, max_trim=20)

    def test_the_selection_reports_it_rather_than_raising(self) -> None:
        """The diagnostic function returns the finding; the estimator refuses."""
        result = adaptive_trim_selection(contaminate(1, 30), K, max_trim=20)
        assert result["saturated"] is True
        assert result["deepest_anomaly"] >= 30
        assert result["trim"] == 0
        assert result["gamma"] > 1.5  # what the estimator refuses to return

    def test_a_large_enough_limit_finds_it(self) -> None:
        result = adaptive_trim_selection(contaminate(1, 30), K, max_trim=60)
        assert result["saturated"] is False
        assert result["trim"] == 30
        assert result["gamma"] == pytest.approx(0.5, abs=0.05)

    def test_the_default_limit_is_generous_enough_for_this(self) -> None:
        """``k // 4`` rather than ``k // 10``, because too low is how it fails."""
        assert adaptive_trimmed_hill_estimator(
            contaminate(1, 30), k=K
        ) == pytest.approx(0.5, abs=0.05)

    def test_clean_data_is_essentially_never_reported_as_saturated(self) -> None:
        """The interlock is far stricter than the trimming test, deliberately.

        A false alarm here turns a perfectly good estimate into an exception,
        while real contamination gives p-values around exp(-200). At the same
        level as the trimming test it fired on one clean sample in fifty; at
        1e-4 family-wise it fired on none of two thousand.
        """
        flagged = sum(
            adaptive_trim_selection(contaminate(seed, 0), K)["saturated"]
            for seed in range(500)
        )
        assert flagged == 0


class TestValidation:
    @pytest.mark.parametrize("k", [0, 1, N, N + 5])
    def test_a_bad_k_is_rejected(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            adaptive_trim_selection(Pareto(alpha=2.0, xm=1.0).rvs(N, seed=1), k)

    @pytest.mark.parametrize("max_trim", [0, -1, K, K + 1])
    def test_a_bad_max_trim_is_rejected(self, max_trim: int) -> None:
        with pytest.raises(ValueError, match="max_trim"):
            adaptive_trim_selection(
                Pareto(alpha=2.0, xm=1.0).rvs(N, seed=1), K, max_trim=max_trim
            )

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.1, 1.5])
    def test_a_bad_level_is_rejected(self, level: float) -> None:
        with pytest.raises(ValueError, match="level must be"):
            adaptive_trim_selection(
                Pareto(alpha=2.0, xm=1.0).rvs(N, seed=1), K, level=level
            )

    def test_non_positive_data_is_rejected(self) -> None:
        data = [-1.0] * 50 + list(Pareto(alpha=2.0, xm=1.0).rvs(100, seed=1))
        with pytest.raises(ValueError, match="positive"):
            adaptive_trim_selection(data, k=120)


class TestIntegration:
    def test_it_is_available_to_the_confidence_interval_machinery(self) -> None:
        """Registered like the others, so the benchmark harness picks it up."""
        clean = Pareto(alpha=2.0, xm=1.0).rvs(4000, seed=1)
        interval = tail_index_confidence_interval(
            clean,
            k=200,
            estimator="adaptive_trimmed_hill",
            method="bootstrap",
            n_bootstrap=60,
        )
        assert interval["lower"] < interval["gamma"] < interval["upper"]
        assert interval["estimator"] == "adaptive_trimmed_hill"

    def test_it_matches_fixed_trimming_at_the_r_it_picked(self) -> None:
        x = contaminate(seed=7, count=4)
        result = adaptive_trim_selection(x, K)
        assert adaptive_trimmed_hill_estimator(x, k=K) == pytest.approx(
            trimmed_hill_estimator(x, k=K, r=result["trim"]), rel=1e-15
        )
