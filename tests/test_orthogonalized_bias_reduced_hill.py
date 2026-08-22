"""Orthogonalized bias-reduced Hill estimator.

The estimator is useful only if the weights do exactly what the derivation says:
keep the intercept and remove the first second-order direction. These tests
therefore check the algebra directly, then check the finite-sample behaviour on
distributions already used by the reduced-bias tests.
"""

from __future__ import annotations

import statistics

import pytest

import heavytails
from heavytails import BurrXII, Frechet, Pareto
from heavytails.tail_index import (
    _apply_threshold_average,
    _normalised_log_spacings,
    _orthogonalized_spacing_weights,
    hill_estimator,
    orthogonalized_bias_reduced_hill_estimator,
    tail_index_confidence_interval,
    threshold_averaged_orthogonalized_hill_estimator,
    threshold_averaged_orthogonalized_hill_selection,
)

N = 20_000
K = 2_000


def contaminate(seed: int, count: int) -> list[float]:
    """A Pareto(2) sample with ``count`` of its largest values replaced."""
    x = sorted(Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=seed), reverse=True)
    for j in range(count):
        x[j] = 1e6 * (j + 1)
    return sorted(x, reverse=True)


class TestOrthogonalizedWeights:
    def test_the_weights_keep_the_intercept_and_kill_the_bias_direction(self) -> None:
        k, r, rho = 200, 7, -0.5
        weights = _orthogonalized_spacing_weights(k, r, rho)
        covariates = [((j + 1) / (k + 1)) ** (-rho) for j in range(r, k)]

        assert sum(weights) == pytest.approx(1.0, abs=1e-12)
        assert sum(w * x for w, x in zip(weights, covariates, strict=True)) == (
            pytest.approx(0.0, abs=1e-12)
        )

    def test_the_weighted_form_equals_the_regression_intercept(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(N, seed=1)
        x = sorted(data, reverse=True)
        k, r, rho = 500, 5, -1.0
        spacings = _normalised_log_spacings(x, k)[r:]
        covariates = [((j + 1) / (k + 1)) ** (-rho) for j in range(r, k)]

        mean_z = statistics.fmean(spacings)
        mean_x = statistics.fmean(covariates)
        numerator = sum(
            (x_j - mean_x) * z_j for x_j, z_j in zip(covariates, spacings, strict=True)
        )
        denominator = sum((x_j - mean_x) ** 2 for x_j in covariates)
        intercept = mean_z - mean_x * numerator / denominator

        assert orthogonalized_bias_reduced_hill_estimator(
            data, k, r=r, rho=rho
        ) == pytest.approx(intercept, rel=1e-12)


class TestEstimatorBehaviour:
    @pytest.mark.parametrize(
        ("dist", "rho"),
        [
            (Frechet(alpha=2.0, s=1.0, m=0.0), -1.0),
            (BurrXII(c=1.0, k=2.0, s=1.0), -0.5),
        ],
    )
    def test_it_reduces_second_order_bias(self, dist: object, rho: float) -> None:
        hill_values = []
        corrected_values = []
        for seed in range(25):
            data = dist.rvs(N, seed=seed)  # type: ignore[attr-defined]
            hill_values.append(hill_estimator(data, K))
            corrected_values.append(
                orthogonalized_bias_reduced_hill_estimator(data, K, rho=rho)
            )

        hill_bias = abs(statistics.fmean(hill_values) - 0.5)
        corrected_bias = abs(statistics.fmean(corrected_values) - 0.5)
        assert corrected_bias < hill_bias

    @pytest.mark.parametrize("count", [0, 3, 8])
    def test_adaptive_trimming_protects_against_top_outliers(self, count: int) -> None:
        errors = []
        for seed in range(20):
            clean = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=seed)
            reference = orthogonalized_bias_reduced_hill_estimator(
                clean, k=300, rho=-1.0
            )
            estimate = orthogonalized_bias_reduced_hill_estimator(
                contaminate(seed, count),
                k=300,
                rho=-1.0,
                adaptive_trim=True,
            )
            errors.append(abs(estimate - reference))

        assert statistics.fmean(errors) < 0.02


class TestThresholdAggregation:
    def test_reports_the_stable_threshold_set_and_weights(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=1)
        result = threshold_averaged_orthogonalized_hill_selection(
            data,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=False,
        )

        assert result["thresholds"][0] == 150
        assert result["thresholds"][-1] == 500
        assert result["candidate_pairs"] == list(
            zip(result["trims"], result["thresholds"], strict=True)
        )
        assert (
            result["stable_candidate_pairs"]
            == result["candidate_pairs"][: len(result["stable_thresholds"])]
        )
        assert len(result["variance_proxy"]) == len(result["thresholds"])
        assert all(value > 0.0 for value in result["variance_proxy"])
        assert 1 <= len(result["stable_thresholds"]) <= len(result["thresholds"])
        assert len(result["weights"]) == len(result["stable_thresholds"])
        assert sum(result["weights"]) == pytest.approx(1.0, abs=1e-12)
        assert result["gamma"] == pytest.approx(
            sum(
                weight * estimate
                for weight, estimate in zip(
                    result["weights"],
                    result["local_estimates"][: len(result["weights"])],
                    strict=True,
                )
            )
        )

    def test_threshold_averaging_has_the_point_estimator_wrapper(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=2)
        details = threshold_averaged_orthogonalized_hill_selection(
            data,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=False,
        )
        point = threshold_averaged_orthogonalized_hill_estimator(
            data,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=False,
            crossfit=False,
        )

        assert point == pytest.approx(details["gamma"], rel=1e-15)
        assert point == pytest.approx(0.5, abs=0.12)

    def test_threshold_averaging_is_cross_fitted_by_default(self) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=3)
        full_sample = threshold_averaged_orthogonalized_hill_estimator(
            data,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=False,
            crossfit=False,
        )
        crossfit = threshold_averaged_orthogonalized_hill_estimator(
            data,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=False,
        )

        assert crossfit != pytest.approx(full_sample, rel=1e-12)
        assert crossfit == pytest.approx(0.5, abs=0.12)

    def test_threshold_averaging_can_use_adaptive_trimming(self) -> None:
        clean = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=4)
        contaminated = sorted(clean, reverse=True)
        for j in range(4):
            contaminated[j] = 1e6 * (j + 1)

        reference = threshold_averaged_orthogonalized_hill_estimator(
            clean,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=False,
        )
        robust = threshold_averaged_orthogonalized_hill_estimator(
            sorted(contaminated, reverse=True),
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=True,
        )

        assert robust == pytest.approx(reference, abs=0.04)

    @pytest.mark.parametrize("count", [1, 3, 5])
    def test_cross_fit_reestimates_trimming_on_the_evaluation_fold(
        self, count: int
    ) -> None:
        """Odd contamination counts split unevenly, so trims cannot be transferred.

        A cross-fit direction that learns zero contaminants on the clean split
        and applies that zero to the contaminated split leaves an arbitrary
        outlier in the final estimate. The estimator must transfer threshold
        decisions, then re-estimate the trimming count on the evaluation fold.
        """
        errors = []
        for seed in range(10):
            clean = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=seed)
            contaminated = sorted(clean, reverse=True)
            for j in range(count):
                contaminated[j] = 1e6 * (j + 1)

            reference = threshold_averaged_orthogonalized_hill_estimator(
                clean,
                k=500,
                min_k=150,
                grid_size=6,
                rho=-1.0,
                adaptive_trim=True,
            )
            robust = threshold_averaged_orthogonalized_hill_estimator(
                sorted(contaminated, reverse=True),
                k=500,
                min_k=150,
                grid_size=6,
                rho=-1.0,
                adaptive_trim=True,
            )
            errors.append(abs(robust - reference))

        assert statistics.fmean(errors) < 0.025

    def test_cross_fit_recomputes_weights_after_evaluation_trimming(self) -> None:
        """Training-fold weights correspond to training-fold trims, not target trims."""
        train = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=21)
        target = Pareto(alpha=2.0, xm=1.0).rvs(10_000, seed=22)
        selection = threshold_averaged_orthogonalized_hill_selection(
            train,
            k=500,
            min_k=150,
            grid_size=6,
            rho=-1.0,
            adaptive_trim=True,
            critical=100.0,
        )
        assert len(selection["stable_thresholds"]) > 1

        baseline = _apply_threshold_average(target, selection)
        tampered = dict(selection)
        tampered["weights"] = [1.0] + [0.0] * (len(selection["weights"]) - 1)

        assert _apply_threshold_average(target, tampered) == pytest.approx(
            baseline, rel=1e-15
        )


class TestValidationAndIntegration:
    @pytest.mark.parametrize("rho", [0.0, 0.5, 1.0])
    def test_rejects_a_non_negative_rho(self, rho: float) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match="rho must be negative"):
            orthogonalized_bias_reduced_hill_estimator(data, 100, rho=rho)

    @pytest.mark.parametrize("r", [-1, 99, 100])
    def test_rejects_bad_trim_counts(self, r: int) -> None:
        data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        with pytest.raises(ValueError, match=r"r must|at least two"):
            orthogonalized_bias_reduced_hill_estimator(data, 100, r=r)

    def test_available_to_the_interval_helper(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(5000, seed=7)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator="orthogonalized_bias_reduced_hill",
            method="bootstrap",
            n_bootstrap=20,
            seed=1,
            estimator_kwargs={"rho": -1.0},
        )
        assert result["estimator"] == "orthogonalized_bias_reduced_hill"
        assert result["lower"] <= result["gamma"] <= result["upper"]

    def test_threshold_averaging_is_available_to_the_interval_helper(self) -> None:
        data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(5000, seed=8)
        result = tail_index_confidence_interval(
            data,
            k=250,
            estimator="threshold_averaged_orthogonalized_hill",
            method="bootstrap",
            n_bootstrap=10,
            seed=1,
            estimator_kwargs={
                "min_k": 100,
                "grid_size": 4,
                "rho": -1.0,
                "adaptive_trim": False,
            },
        )
        assert result["estimator"] == "threshold_averaged_orthogonalized_hill"
        assert result["lower"] <= result["gamma"] <= result["upper"]

    def test_everything_is_exported(self) -> None:
        assert hasattr(heavytails, "orthogonalized_bias_reduced_hill_estimator")
        assert hasattr(heavytails, "threshold_averaged_orthogonalized_hill_estimator")
        assert hasattr(heavytails, "threshold_averaged_orthogonalized_hill_selection")
        assert "orthogonalized_bias_reduced_hill_estimator" in heavytails.__all__
        assert "threshold_averaged_orthogonalized_hill_estimator" in heavytails.__all__
        assert "threshold_averaged_orthogonalized_hill_selection" in heavytails.__all__
