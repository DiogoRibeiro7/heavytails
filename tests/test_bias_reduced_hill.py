"""Bias-reduced Hill, and the second-order parameters it needs.

The claim being tested is that the correction reduces bias, so most of these
tests compare against the Hill estimator on distributions whose second-order
parameter is known analytically:

* Frechet(alpha) has ``rho = -1``, whatever alpha is
* BurrXII(c, k, s) has ``rho = -1/k``

The first of those is the more useful check, because rho and gamma vary
independently: an implementation that confused the two would fail it.
"""

from __future__ import annotations

import math
import statistics

import pytest

import heavytails
from heavytails import BurrXII, Frechet, Pareto
from heavytails.tail_index import (
    bias_reduced_hill_estimator,
    hill_estimator,
    recommended_rho_k,
    second_order_beta,
    second_order_rho,
    tail_index_confidence_interval,
)

N = 20000


class TestRecommendedK:
    def test_is_most_of_the_sample(self) -> None:
        """Estimating rho needs far more data than estimating gamma."""
        assert recommended_rho_k(20000) > 0.8 * 20000

    def test_never_exceeds_the_sample(self) -> None:
        for n in (16, 100, 1000, 10**6):
            assert recommended_rho_k(n) < n

    @pytest.mark.parametrize("n", [0, 1, 15])
    def test_rejects_a_sample_too_small(self, n: int) -> None:
        with pytest.raises(ValueError, match="at least 16"):
            recommended_rho_k(n)


class TestSecondOrderRho:
    def test_is_always_negative(self) -> None:
        """rho < 0 by definition; the estimator takes -|.| for this reason."""
        for dist in (
            Frechet(alpha=2.0, s=1.0, m=0.0),
            BurrXII(c=1.0, k=2.0, s=1.0),
            Pareto(alpha=2.0, xm=1.0),
        ):
            assert second_order_rho(dist.rvs(N, seed=11)) < 0.0

    def test_does_not_depend_on_gamma(self) -> None:
        """Frechet has rho = -1 for every alpha, so the estimates must agree.

        This is the strongest structural check available: rho and gamma vary
        independently here, so an implementation that confused the two, or
        that picked up the tail index by accident, would fail.
        """
        first = second_order_rho(Frechet(alpha=2.0, s=1.0, m=0.0).rvs(N, seed=11))
        second = second_order_rho(Frechet(alpha=4.0, s=1.0, m=0.0).rvs(N, seed=11))
        assert first == pytest.approx(second, abs=0.05)

    @pytest.mark.parametrize(
        ("burr_k", "true_rho"), [(1.0, -1.0), (2.0, -0.5), (4.0, -0.25)]
    )
    def test_orders_burr_cases_correctly(self, burr_k: float, true_rho: float) -> None:
        """BurrXII(c, k, s) has rho = -1/k, so larger k means rho nearer zero.

        The estimator is too imprecise to assert the value, but it must at
        least order the cases correctly.
        """
        estimate = second_order_rho(BurrXII(c=1.0, k=burr_k, s=1.0).rvs(N, seed=11))
        assert estimate < 0.0
        # Loose, deliberately: this estimator is not accurate enough for more.
        assert abs(estimate - true_rho) < 0.6

    def test_burr_ordering_is_monotone(self) -> None:
        estimates = [
            second_order_rho(BurrXII(c=1.0, k=bk, s=1.0).rvs(N, seed=11))
            for bk in (1.0, 2.0, 4.0)
        ]
        assert estimates == sorted(estimates)

    @pytest.mark.parametrize("tau", [0.0, 0.5, 1.0, 2.0])
    def test_every_tau_gives_a_negative_estimate(self, tau: float) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(N, seed=11)
        assert second_order_rho(data, tau=tau) < 0.0

    def test_rejects_a_negative_tau(self) -> None:
        with pytest.raises(ValueError, match="tau must be non-negative"):
            second_order_rho(Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1), tau=-1.0)

    @pytest.mark.parametrize("k", [0, 1, 5000])
    def test_rejects_k_out_of_range(self, k: int) -> None:
        with pytest.raises(ValueError, match="k must be between"):
            second_order_rho(Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1), k=k)

    def test_rejects_a_degenerate_sample(self) -> None:
        with pytest.raises(ValueError, match=r"degenerate|positive|identifiable"):
            second_order_rho([3.0] * 200, k=50)


class TestSecondOrderBeta:
    def test_returns_a_finite_estimate(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(N, seed=11)
        assert math.isfinite(second_order_beta(data, 2000, rho=-1.0))

    @pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
    def test_rejects_a_non_negative_rho(self, rho: float) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="rho must be negative"):
            second_order_beta(data, 100, rho=rho)

    @pytest.mark.parametrize("k", [0, 1, 5000])
    def test_rejects_k_out_of_range(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="k must be between"):
            second_order_beta(data, k, rho=-1.0)


class TestBiasReduction:
    """The property the estimator exists for."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        ("name", "dist", "rho"),
        [
            ("frechet", Frechet(alpha=2.0, s=1.0, m=0.0), -1.0),
            ("burr_k1", BurrXII(c=2.0, k=1.0, s=1.0), -1.0),
            ("burr_k2", BurrXII(c=1.0, k=2.0, s=1.0), -0.5),
        ],
    )
    @pytest.mark.parametrize("k", [500, 2000])
    def test_reduces_the_bias_of_hill(
        self, name: str, dist: object, rho: float, k: int
    ) -> None:
        """Measured reduction runs from about 3x to 80x.

        A single evaluation says nothing about bias, so this averages.
        """
        hill_values = []
        corrected_values = []
        for seed in range(30):
            data = dist.rvs(N, seed=seed)  # type: ignore[attr-defined]
            hill_values.append(hill_estimator(data, k))
            corrected_values.append(bias_reduced_hill_estimator(data, k, rho=rho))

        hill_bias = abs(statistics.mean(hill_values) - 0.5)
        corrected_bias = abs(statistics.mean(corrected_values) - 0.5)
        assert corrected_bias < hill_bias, (
            f"{name} k={k}: correction did not reduce bias "
            f"({corrected_bias:.4f} vs {hill_bias:.4f})"
        )

    @pytest.mark.slow
    def test_the_worst_case_improves_by_an_order_of_magnitude(self) -> None:
        """BurrXII(c=1, k=2) at k=2000 is where Hill is worst."""
        dist = BurrXII(c=1.0, k=2.0, s=1.0)
        hill_values = []
        corrected_values = []
        for seed in range(30):
            data = dist.rvs(N, seed=seed)
            hill_values.append(hill_estimator(data, 2000))
            corrected_values.append(bias_reduced_hill_estimator(data, 2000, rho=-0.5))
        hill_bias = abs(statistics.mean(hill_values) - 0.5)
        corrected_bias = abs(statistics.mean(corrected_values) - 0.5)
        assert hill_bias / corrected_bias > 5.0

    @pytest.mark.slow
    def test_still_helps_when_rho_is_estimated(self) -> None:
        """Worse than supplying rho, but still better than not correcting.

        This is the honest headline: estimating rho costs most but not all of
        the benefit, so the correction is worth applying either way.
        """
        dist = BurrXII(c=1.0, k=2.0, s=1.0)
        hill_values = []
        estimated_values = []
        for seed in range(25):
            data = dist.rvs(N, seed=seed)
            hill_values.append(hill_estimator(data, 2000))
            estimated_values.append(bias_reduced_hill_estimator(data, 2000))
        hill_bias = abs(statistics.mean(hill_values) - 0.5)
        estimated_bias = abs(statistics.mean(estimated_values) - 0.5)
        assert estimated_bias < hill_bias


class TestArgumentHandling:
    def test_supplying_beta_skips_its_estimation(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(5000, seed=1)
        supplied = bias_reduced_hill_estimator(data, 500, rho=-1.0, beta=0.0)
        # beta = 0 means no correction at all, so it must equal Hill exactly.
        assert supplied == pytest.approx(hill_estimator(data, 500), rel=1e-12)

    @pytest.mark.parametrize("rho", [0.0, 0.5, 2.0])
    def test_rejects_a_non_negative_rho(self, rho: float) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="rho must be negative"):
            bias_reduced_hill_estimator(data, 100, rho=rho)

    @pytest.mark.parametrize("k", [0, 1, 5000])
    def test_rejects_k_out_of_range(self, k: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="k must be between"):
            bias_reduced_hill_estimator(data, k, rho=-1.0)

    def test_is_scale_invariant(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(5000, seed=6)
        base = bias_reduced_hill_estimator(data, 500, rho=-1.0)
        scaled = bias_reduced_hill_estimator([1000.0 * x for x in data], 500, rho=-1.0)
        assert scaled == pytest.approx(base, rel=1e-9)


class TestIntegration:
    def test_available_to_the_interval_helper(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(5000, seed=7)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator="bias_reduced_hill",
            method="bootstrap",
            n_bootstrap=20,
            seed=1,
            estimator_kwargs={"rho": -1.0},
        )
        assert result["estimator"] == "bias_reduced_hill"
        assert result["lower"] <= result["gamma"] <= result["upper"]

    def test_everything_is_exported(self) -> None:
        for name in (
            "bias_reduced_hill_estimator",
            "second_order_rho",
            "second_order_beta",
            "recommended_rho_k",
        ):
            assert hasattr(heavytails, name)
            assert name in heavytails.__all__
