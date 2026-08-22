"""Tail index estimation over a stream.

The claim that matters is an identity, not an approximation: the streaming
estimate is the *same number* the batch estimator produces from the whole
sample, to the last bit. That holds because both depend on the sample only
through its top ``k + 1`` values, and the streaming version keeps exactly
those. Asserting equality rather than closeness is what pins it down — a
tolerance would pass on an implementation that quietly dropped or duplicated an
order statistic.

The memory claim is checked the same way: by counting what is retained, not by
timing or by measuring the process.
"""

from __future__ import annotations

import math
import random

import pytest

from heavytails import Pareto
from heavytails.streaming import StreamingTailIndex, TopK, WindowedTailIndex
from heavytails.tail_index import hill_estimator, moment_estimator

SAMPLE = Pareto(alpha=2.0, xm=1.0).rvs(20_000, seed=1)


class TestTopK:
    def test_it_keeps_the_largest_values(self) -> None:
        top = TopK(3)
        top.extend([5.0, 1.0, 9.0, 3.0, 7.0])
        assert top.descending() == [9.0, 7.0, 5.0]

    def test_it_agrees_with_sorting_the_whole_sample(self) -> None:
        top = TopK(50)
        top.extend(SAMPLE)
        assert top.descending() == sorted(SAMPLE, reverse=True)[:50]

    @pytest.mark.parametrize("n", [1_000, 50_000, 500_000])
    def test_memory_is_bounded_by_k_whatever_the_stream_length(self, n: int) -> None:
        """The whole point: what is retained does not grow with the stream.

        Counted rather than measured. Inspecting the process would report the
        interpreter's allocator as much as this structure, and would flake.
        """
        top = TopK(100)
        top.extend(float(i % 997) + 1.0 for i in range(n))
        assert len(top) == 100
        assert top.n_seen == n

    def test_it_holds_everything_until_it_is_full(self) -> None:
        top = TopK(10)
        top.extend([1.0, 2.0, 3.0])
        assert len(top) == 3
        assert top.descending() == [3.0, 2.0, 1.0]

    def test_the_order_of_arrival_does_not_matter(self) -> None:
        values = list(SAMPLE[:5_000])
        first, second = TopK(40), TopK(40)
        first.extend(values)
        random.Random(0).shuffle(values)
        second.extend(values)
        assert first.descending() == second.descending()

    def test_duplicates_are_kept_as_separate_observations(self) -> None:
        """A tail with ties still has ``k`` order statistics, not ``k`` distinct
        values."""
        top = TopK(3)
        top.extend([4.0, 4.0, 4.0, 1.0])
        assert top.descending() == [4.0, 4.0, 4.0]

    def test_negative_values_are_kept_when_nothing_beats_them(self) -> None:
        """``TopK`` is about order, not about sign; the estimators check sign."""
        top = TopK(2)
        top.extend([-5.0, -1.0, -9.0])
        assert top.descending() == [-1.0, -5.0]

    @pytest.mark.parametrize("k", [0, -1, 2.5, "3"])
    def test_a_bad_capacity_is_rejected(self, k: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            TopK(k)  # type: ignore[arg-type]


class TestStreamingMatchesBatch:
    """The identity the whole module rests on."""

    @pytest.mark.parametrize("k", [10, 100, 1_000, 5_000])
    def test_the_hill_estimate_is_bit_for_bit_the_batch_one(self, k: int) -> None:
        stream = StreamingTailIndex(k=k)
        stream.extend(SAMPLE)
        assert stream.hill() == hill_estimator(SAMPLE, k=k)

    @pytest.mark.parametrize("k", [10, 100, 1_000])
    def test_the_moment_estimate_is_bit_for_bit_the_batch_one(self, k: int) -> None:
        stream = StreamingTailIndex(k=k)
        stream.extend(SAMPLE)
        assert stream.moment() == moment_estimator(SAMPLE, k=k)

    @pytest.mark.parametrize("alpha", [0.7, 1.0, 2.0, 5.0])
    def test_it_holds_across_tail_indices(self, alpha: float) -> None:
        sample = Pareto(alpha=alpha, xm=1.0).rvs(10_000, seed=3)
        stream = StreamingTailIndex(k=300)
        stream.extend(sample)
        assert stream.hill() == hill_estimator(sample, k=300)

    def test_it_holds_when_the_stream_arrives_in_a_different_order(self) -> None:
        """Order statistics do not depend on arrival order, and neither may this."""
        shuffled = list(SAMPLE)
        random.Random(7).shuffle(shuffled)
        ordered, jumbled = StreamingTailIndex(k=200), StreamingTailIndex(k=200)
        ordered.extend(SAMPLE)
        jumbled.extend(shuffled)
        assert ordered.hill() == jumbled.hill()

    def test_it_holds_when_fed_one_observation_at_a_time(self) -> None:
        stream = StreamingTailIndex(k=100)
        for value in SAMPLE:
            stream.update(value)
        assert stream.hill() == hill_estimator(SAMPLE, k=100)

    def test_the_threshold_is_the_batch_order_statistic(self) -> None:
        stream = StreamingTailIndex(k=200)
        stream.extend(SAMPLE)
        assert stream.threshold == sorted(SAMPLE, reverse=True)[200]


class TestStreamingBehaviour:
    def test_it_reports_how_much_it_has_seen(self) -> None:
        stream = StreamingTailIndex(k=10)
        stream.extend(SAMPLE[:500])
        assert stream.n_seen == 500

    def test_it_refuses_to_estimate_before_it_can(self) -> None:
        """Returning something from ten observations at ``k = 100`` would be a
        number with no meaning attached."""
        stream = StreamingTailIndex(k=100)
        stream.extend(SAMPLE[:10])
        assert not stream.ready
        with pytest.raises(ValueError, match="need at least 101"):
            stream.hill()

    def test_it_becomes_ready_at_exactly_k_plus_one(self) -> None:
        stream = StreamingTailIndex(k=5)
        stream.extend(SAMPLE[:5])
        assert not stream.ready
        stream.update(SAMPLE[5])
        assert stream.ready
        assert math.isfinite(stream.hill())

    def test_non_positive_data_in_the_tail_is_rejected(self) -> None:
        """The Hill estimator takes logarithms of ratios of order statistics."""
        stream = StreamingTailIndex(k=3)
        stream.extend([-1.0, -2.0, -3.0, -4.0, -5.0])
        with pytest.raises(ValueError, match="positive"):
            stream.hill()

    def test_values_below_the_threshold_are_discarded_without_affecting_it(
        self,
    ) -> None:
        """Only the top ``k + 1`` matter, so the rest may be anything at all.

        Including values the estimator could not take a logarithm of: they
        never reach it.
        """
        stream = StreamingTailIndex(k=3)
        stream.extend([100.0, 90.0, 80.0, 70.0])
        before = stream.hill()
        stream.extend([-5.0, 0.0, 1e-300])
        assert stream.hill() == before

    def test_the_estimate_converges_on_the_true_index(self) -> None:
        stream = StreamingTailIndex(k=2_000)
        stream.extend(Pareto(alpha=2.0, xm=1.0).rvs(100_000, seed=11))
        assert stream.hill() == pytest.approx(0.5, abs=0.03)

    @pytest.mark.parametrize("k", [0, 1, -3, 2.5])
    def test_a_bad_k_is_rejected(self, k: object) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            StreamingTailIndex(k=k)  # type: ignore[arg-type]


class TestWindowedMatchesBatchOnItsWindow:
    def test_it_equals_the_batch_estimate_over_the_retained_observations(
        self,
    ) -> None:
        """The same identity, applied to what the window actually holds."""
        monitor = WindowedTailIndex(window=2_000, k=100)
        monitor.extend(SAMPLE)
        assert monitor.hill() == hill_estimator(monitor.values(), k=100)

    def test_the_window_holds_the_most_recent_observations(self) -> None:
        monitor = WindowedTailIndex(window=500, k=50)
        monitor.extend(SAMPLE)
        assert monitor.values() == list(SAMPLE[-500:])

    def test_it_matches_the_whole_stream_before_anything_is_evicted(self) -> None:
        monitor = WindowedTailIndex(window=10_000, k=200)
        monitor.extend(SAMPLE[:5_000])
        assert monitor.hill() == hill_estimator(SAMPLE[:5_000], k=200)

    def test_eviction_removes_one_copy_of_a_repeated_value(self) -> None:
        """A window of ties must not lose more than it evicted."""
        monitor = WindowedTailIndex(window=4, k=2)
        monitor.extend([3.0, 3.0, 3.0, 3.0, 1.0])
        assert sorted(monitor.values()) == [1.0, 3.0, 3.0, 3.0]

    def test_the_moment_estimate_matches_too(self) -> None:
        monitor = WindowedTailIndex(window=2_000, k=100)
        monitor.extend(SAMPLE)
        assert monitor.moment() == moment_estimator(monitor.values(), k=100)


class TestWindowedForgets:
    """The reason the windowed version exists at all."""

    def test_it_follows_a_change_in_the_tail_where_the_stream_does_not(self) -> None:
        """A tail index that moves from 3 to 1.5 is the case to get right.

        The whole-stream estimator averages the two regimes and reports
        something in between that describes neither. The window reports the
        regime the data is actually in, which is what monitoring is for.
        """
        light = Pareto(alpha=3.0, xm=1.0).rvs(5_000, seed=1)
        heavy = Pareto(alpha=1.5, xm=1.0).rvs(5_000, seed=2)

        monitor = WindowedTailIndex(window=5_000, k=400)
        stream = StreamingTailIndex(k=400)
        for source in (light, heavy):
            monitor.extend(source)
            stream.extend(source)

        assert monitor.hill() == pytest.approx(2.0 / 3.0, abs=0.05)
        assert abs(stream.hill() - 2.0 / 3.0) > abs(monitor.hill() - 2.0 / 3.0)

    def test_the_window_tracks_the_first_regime_before_the_change(self) -> None:
        monitor = WindowedTailIndex(window=5_000, k=400)
        monitor.extend(Pareto(alpha=3.0, xm=1.0).rvs(5_000, seed=1))
        assert monitor.hill() == pytest.approx(1.0 / 3.0, abs=0.05)

    def test_an_extreme_outlier_leaves_when_the_window_moves_past_it(self) -> None:
        """Contamination is temporary here, which fixed trimming cannot manage.

        A single bad observation corrupts every whole-stream estimate from the
        moment it arrives. In a window it corrupts ``window`` of them and then
        it is gone.
        """
        monitor = WindowedTailIndex(window=1_000, k=100)
        monitor.extend(SAMPLE[:1_000])
        clean = monitor.hill()

        monitor.update(1e12)
        assert monitor.hill() > clean * 1.1

        monitor.extend(SAMPLE[1_000:2_100])
        assert monitor.hill() == pytest.approx(clean, abs=0.15)


class TestWindowedValidation:
    def test_it_refuses_to_estimate_before_the_window_fills_enough(self) -> None:
        monitor = WindowedTailIndex(window=100, k=50)
        monitor.extend(SAMPLE[:20])
        assert not monitor.ready
        with pytest.raises(ValueError, match="need at least 51"):
            monitor.hill()

    def test_non_positive_data_in_the_tail_is_rejected(self) -> None:
        monitor = WindowedTailIndex(window=10, k=3)
        monitor.extend([-1.0, -2.0, -3.0, -4.0, -5.0])
        with pytest.raises(ValueError, match="positive"):
            monitor.hill()

    @pytest.mark.parametrize(
        ("window", "k"), [(0, 2), (1, 2), (100, 1), (100, 0), (10, 10), (10, 20)]
    )
    def test_bad_sizes_are_rejected(self, window: int, k: int) -> None:
        with pytest.raises(ValueError):
            WindowedTailIndex(window=window, k=k)

    def test_k_must_be_below_the_window(self) -> None:
        """Otherwise there is no ``k + 1``-th order statistic to divide by."""
        with pytest.raises(ValueError, match="below window"):
            WindowedTailIndex(window=50, k=50)

    def test_it_reports_the_whole_stream_length_not_the_window(self) -> None:
        monitor = WindowedTailIndex(window=100, k=10)
        monitor.extend(SAMPLE[:5_000])
        assert monitor.n_seen == 5_000
        assert len(monitor.values()) == 100
