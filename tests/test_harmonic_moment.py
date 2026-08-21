"""The harmonic moment and t-Hill estimators.

Hill's contributions ``log(X_(i)/u)`` are unbounded above, so one sufficiently
extreme observation moves the estimate arbitrarily far. These estimators use
the reciprocal ratios ``u/X_(i)``, which lie in (0, 1], so the influence of a
single contaminated observation is bounded however extreme it is.

That is a different robustness mechanism from trimming: trimming needs to know
roughly how many observations are bad, bounded influence does not.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

import heavytails
from heavytails import Frechet, Pareto
from heavytails.tail_index import (
    harmonic_moment_estimator,
    hill_estimator,
    t_hill_estimator,
    tail_index_confidence_interval,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tail_index_study import _provenance as _study_provenance


def contaminate(sample: list[float], count: int, magnitude: float) -> list[float]:
    ordered = sorted(sample, reverse=True)
    for i in range(count):
        ordered[i] = magnitude / (i + 1)
    return ordered


class TestConsistency:
    @pytest.mark.parametrize("alpha", [1.5, 2.0, 4.0])
    @pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
    def test_recovers_the_tail_index(self, alpha: float, beta: float) -> None:
        data = Pareto(alpha=alpha, xm=1.0).rvs(50000, seed=11)
        assert harmonic_moment_estimator(data, k=2500, beta=beta) == pytest.approx(
            1.0 / alpha, abs=0.03
        )

    def test_recovers_the_tail_index_of_a_frechet(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(50000, seed=4)
        assert t_hill_estimator(data, k=2500) == pytest.approx(0.5, abs=0.04)

    def test_is_scale_invariant(self) -> None:
        """The ratios u/X_(i) are already scale free, so this should be exact."""
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=6)
        base = harmonic_moment_estimator(data, k=250, beta=1.5)
        scaled = harmonic_moment_estimator([1e6 * x for x in data], k=250, beta=1.5)
        assert scaled == pytest.approx(base, rel=1e-12)

    def test_t_hill_is_the_beta_one_case(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
        assert t_hill_estimator(data, 250) == harmonic_moment_estimator(
            data, 250, beta=1.0
        )


class TestHillLimit:
    """beta -> 0 recovers the Hill estimator.

    This is the strongest available check that the family is the right one: an
    estimator can look consistent while being a different estimator, but it
    cannot converge to Hill by accident.
    """

    @pytest.mark.parametrize(
        ("beta", "tolerance"), [(0.1, 2e-3), (0.01, 2e-4), (0.001, 2e-5)]
    )
    def test_converges_to_hill_as_beta_shrinks(
        self, beta: float, tolerance: float
    ) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=3)
        assert harmonic_moment_estimator(data, 1000, beta=beta) == pytest.approx(
            hill_estimator(data, 1000), abs=tolerance
        )

    def test_the_gap_shrinks_monotonically(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=3)
        target = hill_estimator(data, 1000)
        gaps = [
            abs(harmonic_moment_estimator(data, 1000, beta=b) - target)
            for b in (1.0, 0.1, 0.01, 0.001)
        ]
        assert gaps == sorted(gaps, reverse=True)


class TestBoundedInfluence:
    """The property these estimators exist for."""

    @pytest.mark.parametrize("magnitude", [1e6, 1e12, 1e30, 1e100])
    def test_insensitive_to_the_magnitude_of_one_outlier(
        self, magnitude: float
    ) -> None:
        """A single contaminated value cannot move the estimate far, however
        extreme it is, because its contribution u/X tends to zero rather than
        to infinity."""
        base = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=0)
        clean = t_hill_estimator(base, 500)
        dirty = t_hill_estimator(contaminate(base, 1, magnitude), 500)
        assert abs(dirty - clean) < 0.01

    def test_hill_is_not_insensitive(self) -> None:
        """Pins the premise: without this the test above proves nothing."""
        base = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=0)
        clean = hill_estimator(base, 500)
        dirty = hill_estimator(contaminate(base, 1, 1e30), 500)
        assert dirty - clean > 0.1

    def test_influence_saturates_rather_than_growing(self) -> None:
        """Pushing the outlier further has almost no additional effect."""
        base = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=0)
        estimates = [
            t_hill_estimator(contaminate(base, 1, m), 500)
            for m in (1e6, 1e12, 1e30, 1e100)
        ]
        assert max(estimates) - min(estimates) < 1e-6

    def test_larger_beta_is_more_robust(self) -> None:
        """beta is the robustness dial, so it should behave like one.

        The comparison is against the estimate on the *clean* sample at the
        same beta, not against the true value. Every estimator carries some
        finite-sample bias of its own, and measuring against the truth mixes
        that in with the contamination effect being tested.
        """
        base = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=2)
        dirty = contaminate(base, 20, 1e12)
        shifts = [
            abs(
                harmonic_moment_estimator(dirty, 500, beta=b)
                - harmonic_moment_estimator(base, 500, beta=b)
            )
            for b in (0.25, 0.5, 1.0, 3.0)
        ]
        assert shifts == sorted(shifts, reverse=True)


class TestArgumentValidation:
    @pytest.mark.parametrize("beta", [0.0, -1.0, -0.5])
    def test_rejects_non_positive_beta(self, beta: float) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="beta must be strictly positive"):
            harmonic_moment_estimator(data, k=100, beta=beta)

    @pytest.mark.parametrize("k", [0, 1, 1000])
    def test_rejects_k_out_of_range(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="k must be between"):
            harmonic_moment_estimator(data, k=k)

    def test_rejects_a_non_positive_threshold(self) -> None:
        """Only the top k+1 observations are read, so that is what must be positive.

        Values below the threshold are never touched and may legitimately be
        zero or negative, which is common when only positive exceedances are
        modelled.
        """
        with pytest.raises(ValueError, match="positive"):
            harmonic_moment_estimator([5.0, 4.0, 0.0, -1.0, -2.0], k=2)

    def test_accepts_non_positive_values_below_the_threshold(self) -> None:
        """They are outside the region the estimator looks at."""
        data = [100.0, 50.0, 25.0, 12.0, 6.0, 3.0, 0.0, -5.0]
        assert harmonic_moment_estimator(data, k=4) > 0.0

    def test_rejects_a_degenerate_sample(self) -> None:
        """All ties gives H = 1 and a division by zero.

        Reporting that the index is not identifiable is more useful than
        returning inf or raising ZeroDivisionError from inside the formula.
        """
        with pytest.raises(ValueError, match="not identifiable"):
            harmonic_moment_estimator([2.0] * 50, k=10)


class TestIntegration:
    @pytest.mark.parametrize("estimator", ["harmonic_moment", "t_hill"])
    def test_available_to_the_interval_helper(self, estimator: str) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator=estimator,
            method="bootstrap",
            n_bootstrap=50,
            seed=1,
        )
        assert result["estimator"] == estimator
        assert result["lower"] <= result["gamma"] <= result["upper"]

    def test_beta_reaches_the_estimator_through_kwargs(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator="harmonic_moment",
            method="bootstrap",
            n_bootstrap=50,
            seed=1,
            estimator_kwargs={"beta": 3.0},
        )
        assert result["estimator_kwargs"] == {"beta": 3.0}
        assert result["gamma"] == pytest.approx(
            harmonic_moment_estimator(data, 250, beta=3.0), rel=1e-12
        )

    def test_both_are_exported(self) -> None:
        assert heavytails.harmonic_moment_estimator is harmonic_moment_estimator
        assert heavytails.t_hill_estimator is t_hill_estimator


class TestStudyProvenance:
    """A results file has to say what produced it."""

    def test_provenance_records_version_and_commit(self) -> None:
        prov = _study_provenance(trials=7)
        assert prov["trials"] == 7
        assert prov["heavytails_version"]
        assert prov["python_version"]
        assert prov["estimators"]
        assert prov["scenarios"]
        # git_commit is None outside a checkout, which is acceptable; when
        # present it must look like a SHA.
        commit = prov["git_commit"]
        assert commit is None or (
            len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)
        )
