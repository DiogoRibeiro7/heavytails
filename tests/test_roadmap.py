"""Tests for roadmap.py statistical features."""

import math
import random

import pytest

from heavytails import Cauchy, LogNormal, Pareto
from heavytails.roadmap import (
    bootstrap_confidence_intervals,
    fit_mle,
    improved_incomplete_beta,
    model_comparison,
    robust_hill_estimator,
    safe_lognormal_ppf,
)


class TestMLEFitting:
    """Test Maximum Likelihood Estimation implementations."""

    def test_pareto_mle_basic(self):
        """Test Pareto MLE with known parameters."""
        # Generate sample from known distribution
        true_alpha = 2.5
        true_xm = 1.0
        dist = Pareto(alpha=true_alpha, xm=true_xm)
        data = dist.rvs(1000, seed=42)

        # Fit using MLE
        params = fit_mle(data, "pareto")

        # Check parameters are close to true values
        assert "alpha" in params
        assert "xm" in params
        assert abs(params["alpha"] - true_alpha) < 0.3  # Within reasonable tolerance
        assert abs(params["xm"] - true_xm) < 0.1

    def test_lognormal_mle(self):
        """Test LogNormal MLE fitting."""
        true_mu = 0.5
        true_sigma = 1.0
        dist = LogNormal(mu=true_mu, sigma=true_sigma)
        data = dist.rvs(1000, seed=42)

        params = fit_mle(data, "lognormal")

        assert "mu" in params
        assert "sigma" in params
        assert abs(params["mu"] - true_mu) < 0.1
        assert abs(params["sigma"] - true_sigma) < 0.1

    def test_exponential_mle(self):
        """Test Exponential MLE fitting."""
        # Generate exponential data manually (no Exponential class in library)
        random.seed(42)
        true_lambda = 2.0
        # Exponential data: -ln(U)/lambda where U ~ Uniform(0,1)
        data = [-math.log(random.random()) / true_lambda for _ in range(1000)]

        params = fit_mle(data, "exponential")

        assert "lambda" in params
        assert abs(params["lambda"] - true_lambda) < 0.2

    def test_cauchy_mle(self):
        """Test Cauchy MLE fitting."""
        true_x0 = 0.0
        true_gamma = 1.0
        dist = Cauchy(x0=true_x0, gamma=true_gamma)
        data = dist.rvs(1000, seed=42)

        params = fit_mle(data, "cauchy")

        assert "x0" in params
        assert "gamma" in params
        # Cauchy has heavy tails, so fitting can be less precise
        assert abs(params["x0"] - true_x0) < 0.5
        assert abs(params["gamma"] - true_gamma) < 0.5

    def test_invalid_distribution(self):
        """Test that invalid distribution name raises error."""
        data = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="MLE not implemented"):
            fit_mle(data, "invalid_distribution")

    def test_empty_data(self):
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            fit_mle([], "pareto")

    def test_non_finite_data(self):
        """Test that non-finite data raises error."""
        data = [1.0, 2.0, float("nan"), 3.0]
        with pytest.raises(ValueError, match="non-finite values"):
            fit_mle(data, "pareto")

    def test_lognormal_negative_data(self):
        """Test that LogNormal rejects negative data."""
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="LogNormal requires all data > 0"):
            fit_mle(data, "lognormal")

    def test_exponential_negative_data(self):
        """Test that Exponential rejects negative data."""
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="Exponential requires all data >= 0"):
            fit_mle(data, "exponential")

    def test_weibull_mle(self):
        """Test Weibull MLE fitting."""
        try:
            from heavytails import Weibull
            # Generate Weibull data
            dist = Weibull(k=2.0, lam=1.5)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "weibull")

            assert "k" in params
            assert "lam" in params
            assert params["k"] > 0
            assert params["lam"] > 0
        except ImportError:
            pytest.skip("Weibull distribution not available")

    def test_weibull_negative_data(self):
        """Test that Weibull rejects non-positive data."""
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="Weibull requires all data > 0"):
            fit_mle(data, "weibull")

    def test_studentt_mle(self):
        """Test Student-t MLE fitting."""
        try:
            from heavytails import StudentT
            # Generate Student-t data
            dist = StudentT(nu=5.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "studentt")

            assert "nu" in params
            assert params["nu"] > 0
        except ImportError:
            pytest.skip("StudentT distribution not available")

    def test_frechet_mle(self):
        """Test Frechet MLE fitting."""
        pytest.importorskip("scipy")
        try:
            from heavytails import Frechet
            # Generate Frechet data
            dist = Frechet(alpha=2.0, s=1.0, m=0.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "frechet")

            assert "alpha" in params
            assert "s" in params
            assert "m" in params
        except ImportError:
            pytest.skip("Frechet distribution not available")

    def test_frechet_negative_data(self):
        """Test that Frechet rejects non-positive data."""
        pytest.importorskip("scipy")
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="Frechet requires all data > 0"):
            fit_mle(data, "frechet")

    def test_gpd_mle(self):
        """Test GPD MLE fitting."""
        pytest.importorskip("scipy")
        try:
            from heavytails import GeneralizedPareto
            # Generate GPD data
            dist = GeneralizedPareto(xi=0.1, sigma=1.0, mu=0.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "generalizedpareto")

            assert "xi" in params
            assert "sigma" in params
            assert "mu" in params
        except ImportError:
            pytest.skip("GeneralizedPareto distribution not available")

    def test_burrxii_mle(self):
        """Test BurrXII MLE fitting."""
        pytest.importorskip("scipy")
        try:
            from heavytails import BurrXII
            # Generate BurrXII data
            dist = BurrXII(c=2.0, k=2.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "burrxii")

            assert "c" in params
            assert "k" in params
        except ImportError:
            pytest.skip("BurrXII distribution not available")

    def test_burrxii_negative_data(self):
        """Test that BurrXII rejects non-positive data."""
        pytest.importorskip("scipy")
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="BurrXII requires all data > 0"):
            fit_mle(data, "burrxii")

    def test_loglogistic_mle(self):
        """Test LogLogistic MLE fitting."""
        pytest.importorskip("scipy")
        try:
            from heavytails import LogLogistic
            # Generate LogLogistic data
            dist = LogLogistic(kappa=2.0, lam=1.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "loglogistic")

            # roadmap.py uses kappa and lam for MLE parameters
            assert "kappa" in params
            assert "lam" in params
            assert params["kappa"] > 0
            assert params["lam"] > 0
        except ImportError:
            pytest.skip("LogLogistic distribution not available")

    def test_loglogistic_negative_data(self):
        """Test that LogLogistic rejects non-positive data."""
        pytest.importorskip("scipy")
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="LogLogistic requires all data > 0"):
            fit_mle(data, "loglogistic")

    def test_inversegamma_mle(self):
        """Test InverseGamma MLE fitting."""
        pytest.importorskip("scipy")
        try:
            from heavytails import InverseGamma
            # Generate InverseGamma data
            dist = InverseGamma(alpha=3.0, beta=2.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "inversegamma")

            assert "alpha" in params
            assert "beta" in params
        except ImportError:
            pytest.skip("InverseGamma distribution not available")

    def test_inversegamma_negative_data(self):
        """Test that InverseGamma rejects non-positive data."""
        pytest.importorskip("scipy")
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="InverseGamma requires all data > 0"):
            fit_mle(data, "inversegamma")

    def test_betaprime_mle(self):
        """Test BetaPrime MLE fitting."""
        pytest.importorskip("scipy")
        try:
            from heavytails import BetaPrime
            # Generate BetaPrime data
            dist = BetaPrime(a=2.0, b=2.0, s=1.0)
            data = dist.rvs(500, seed=42)

            params = fit_mle(data, "betaprime")

            # roadmap.py uses a and b for MLE parameters
            assert "a" in params
            assert "b" in params
            assert params["a"] > 0
            assert params["b"] > 0
        except ImportError:
            pytest.skip("BetaPrime distribution not available")

    def test_betaprime_negative_data(self):
        """Test that BetaPrime rejects non-positive data."""
        pytest.importorskip("scipy")
        data = [-1.0, 1.0, 2.0]
        with pytest.raises(ValueError, match="BetaPrime requires all data > 0"):
            fit_mle(data, "betaprime")


class TestModelComparison:
    """Test model comparison utilities."""

    def test_model_comparison_pareto_data(self):
        """Test model comparison on Pareto-generated data."""
        # Generate Pareto data
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(1000, seed=42)

        # Compare Pareto vs LogNormal (Exponential class not implemented)
        results = model_comparison(data, ["pareto", "lognormal"])

        # All distributions should have results
        assert "pareto" in results
        assert "lognormal" in results

        # Check structure of results for valid distributions
        for dist_name in results:
            assert "params" in results[dist_name]
            assert "log_likelihood" in results[dist_name]
            assert "AIC" in results[dist_name]
            assert "BIC" in results[dist_name]
            # Only check for rank if log_likelihood is not -inf
            if results[dist_name]["log_likelihood"] != float("-inf"):
                assert "rank_AIC" in results[dist_name]
                assert "rank_BIC" in results[dist_name]

        # Pareto should rank best (rank 1) for Pareto data
        assert results["pareto"]["rank_AIC"] == 1
        assert results["pareto"]["rank_BIC"] == 1

    def test_model_comparison_aic_bic_ordering(self):
        """Test that AIC and BIC rankings are consistent."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        results = model_comparison(data, ["pareto", "lognormal"])

        # Rankings should be 1 and 2
        ranks_aic = sorted([results[d]["rank_AIC"] for d in results])
        ranks_bic = sorted([results[d]["rank_BIC"] for d in results])

        assert ranks_aic == [1, 2]
        assert ranks_bic == [1, 2]

    def test_model_comparison_aic_formula(self):
        """Test that AIC calculation follows correct formula: AIC = 2k - 2*log(L)."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        results = model_comparison(data, ["pareto"])

        # Verify AIC calculation
        k = results["pareto"]["n_params"]
        log_lik = results["pareto"]["log_likelihood"]
        aic_expected = 2 * k - 2 * log_lik
        aic_actual = results["pareto"]["AIC"]

        assert abs(aic_actual - aic_expected) < 1e-6

    def test_model_comparison_bic_formula(self):
        """Test that BIC calculation follows correct formula: BIC = k*ln(n) - 2*log(L)."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        results = model_comparison(data, ["pareto"])

        # Verify BIC calculation
        k = results["pareto"]["n_params"]
        n = len(data)
        log_lik = results["pareto"]["log_likelihood"]
        bic_expected = k * math.log(n) - 2 * log_lik
        bic_actual = results["pareto"]["BIC"]

        assert abs(bic_actual - bic_expected) < 1e-6

    def test_model_comparison_empty_data(self):
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            model_comparison([], ["pareto"])

    def test_model_comparison_with_failed_distribution(self):
        """Test model comparison when some distributions fail to fit."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        # Include an invalid distribution
        results = model_comparison(data, ["pareto", "invalid_dist"])

        # Pareto should fit successfully
        assert results["pareto"]["log_likelihood"] != float("-inf")
        assert "rank_AIC" in results["pareto"]

        # Invalid dist should have error
        assert results["invalid_dist"]["log_likelihood"] == float("-inf")
        assert "error" in results["invalid_dist"]


class TestBootstrapConfidenceIntervals:
    """Test bootstrap confidence interval estimation."""

    def test_bootstrap_ci_pareto(self):
        """Test bootstrap CIs for Pareto distribution."""
        true_alpha = 2.5
        true_xm = 1.0
        dist = Pareto(alpha=true_alpha, xm=true_xm)
        data = dist.rvs(500, seed=42)

        # Calculate 95% CI
        ci = bootstrap_confidence_intervals(
            data, "pareto", n_bootstrap=200, confidence_level=0.95, seed=42
        )

        # Check structure
        assert "alpha" in ci
        assert "xm" in ci

        # Each CI should be a tuple (lower, upper)
        assert len(ci["alpha"]) == 2
        assert len(ci["xm"]) == 2

        # Lower bound should be less than upper bound
        assert ci["alpha"][0] < ci["alpha"][1]
        assert ci["xm"][0] < ci["xm"][1]

        # True value should be within CI (most of the time)
        # This is a statistical test, so it could fail occasionally
        assert ci["alpha"][0] <= true_alpha <= ci["alpha"][1]
        # For xm, allow small tolerance since it's a boundary parameter
        assert ci["xm"][0] - 0.01 <= true_xm <= ci["xm"][1] + 0.01

    def test_bootstrap_ci_different_confidence_levels(self):
        """Test that wider confidence levels give wider intervals."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        ci_95 = bootstrap_confidence_intervals(
            data, "pareto", n_bootstrap=200, confidence_level=0.95, seed=42
        )
        ci_90 = bootstrap_confidence_intervals(
            data, "pareto", n_bootstrap=200, confidence_level=0.90, seed=42
        )

        # 95% CI should be wider than 90% CI
        width_95 = ci_95["alpha"][1] - ci_95["alpha"][0]
        width_90 = ci_90["alpha"][1] - ci_90["alpha"][0]

        assert width_95 >= width_90

    def test_bootstrap_ci_invalid_confidence_level(self):
        """Test that invalid confidence level raises error."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            bootstrap_confidence_intervals(data, "pareto", confidence_level=1.5)

        with pytest.raises(
            ValueError, match="Confidence level must be between 0 and 1"
        ):
            bootstrap_confidence_intervals(data, "pareto", confidence_level=0.0)

    def test_bootstrap_ci_reproducibility(self):
        """Test that same seed gives reproducible results."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        ci1 = bootstrap_confidence_intervals(data, "pareto", n_bootstrap=100, seed=42)
        ci2 = bootstrap_confidence_intervals(data, "pareto", n_bootstrap=100, seed=42)

        # Should get identical results with same seed
        assert ci1["alpha"] == ci2["alpha"]
        assert ci1["xm"] == ci2["xm"]

    def test_bootstrap_ci_empty_data(self):
        """Test that bootstrap with empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            bootstrap_confidence_intervals([], "pareto")

    def test_bootstrap_ci_low_n_warning(self):
        """Test that low n_bootstrap triggers warning."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(100, seed=42)

        with pytest.warns(UserWarning, match="n_bootstrap < 100"):
            bootstrap_confidence_intervals(data, "pareto", n_bootstrap=50, seed=42)


class TestRobustHillEstimator:
    """Test robust Hill estimator implementation."""

    def test_robust_hill_pareto_data(self):
        """Test Hill estimator on Pareto-generated data."""
        true_alpha = 2.5
        dist = Pareto(alpha=true_alpha, xm=1.0)
        data = dist.rvs(1000, seed=42)

        result = robust_hill_estimator(data)

        # Check result structure
        assert "gamma" in result
        assert "alpha" in result
        assert "k_used" in result
        assert "bias_corrected" in result
        assert "n" in result
        assert "reliability" in result

        # Gamma should be 1/alpha
        expected_gamma = 1.0 / true_alpha
        assert abs(result["gamma"] - expected_gamma) < 0.1

        # Alpha should be close to true value
        assert abs(result["alpha"] - true_alpha) < 0.5

        # Should use bias correction by default
        assert result["bias_corrected"] is True

        # Sample size should match
        assert result["n"] == 1000

        # Reliability should be valid
        assert result["reliability"] in ["good", "fair", "poor"]

    def test_robust_hill_without_bias_correction(self):
        """Test Hill estimator without bias correction."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(1000, seed=42)

        result = robust_hill_estimator(data, bias_correction=False)

        assert result["bias_corrected"] is False

    def test_robust_hill_manual_k(self):
        """Test Hill estimator with manually specified k."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(1000, seed=42)

        k_manual = 100
        result = robust_hill_estimator(data, k=k_manual)

        # Should use the specified k
        assert result["k_used"] == k_manual

    def test_robust_hill_automatic_k(self):
        """Test automatic k selection."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(1000, seed=42)

        result = robust_hill_estimator(data, k=None)

        # k should be selected automatically
        assert result["k_used"] > 0
        assert result["k_used"] < len(data) // 2

    def test_robust_hill_small_sample_warning(self):
        """Test that small sample sizes trigger warning."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(100, seed=42)

        with pytest.warns(UserWarning, match="Hill estimator may be unreliable"):
            robust_hill_estimator(data)

    def test_robust_hill_too_small_sample_error(self):
        """Test that very small samples raise error."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]

        with pytest.raises(ValueError, match="Sample size too small"):
            robust_hill_estimator(data)

    def test_robust_hill_reliability_assessment(self):
        """Test reliability assessment for different sample sizes."""
        dist = Pareto(alpha=2.5, xm=1.0)

        # Large sample should be 'good'
        data_large = dist.rvs(1000, seed=42)
        result_large = robust_hill_estimator(data_large)
        assert result_large["reliability"] in ["good", "fair"]

        # Small sample should be 'poor'
        data_small = dist.rvs(100, seed=42)
        result_small = robust_hill_estimator(data_small)
        assert result_small["reliability"] == "poor"

    def test_robust_hill_gamma_alpha_relationship(self):
        """Test that gamma and alpha are inverse of each other."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(1000, seed=42)

        result = robust_hill_estimator(data)

        # gamma * alpha should equal 1.0
        product = result["gamma"] * result["alpha"]
        assert abs(product - 1.0) < 1e-6

    def test_robust_hill_k_adjustment(self):
        """Test that invalid k values are adjusted."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(100, seed=42)

        # Try with k that's too large
        with pytest.warns(UserWarning):
            result = robust_hill_estimator(data, k=90)
            assert 5 < result["k_used"] < len(data) // 2


class TestIntegration:
    """Integration tests for combined functionality."""

    def test_full_workflow_pareto(self):
        """Test complete workflow: generate data, fit, compare, bootstrap, Hill."""
        # 1. Generate data
        true_alpha = 2.5
        true_xm = 1.0
        dist = Pareto(alpha=true_alpha, xm=true_xm)
        data = dist.rvs(1000, seed=42)

        # 2. Fit MLE
        params = fit_mle(data, "pareto")
        assert abs(params["alpha"] - true_alpha) < 0.5

        # 3. Model comparison
        comparison = model_comparison(data, ["pareto", "lognormal"])
        assert comparison["pareto"]["rank_AIC"] == 1

        # 4. Bootstrap CI
        ci = bootstrap_confidence_intervals(data, "pareto", n_bootstrap=100, seed=42)
        assert ci["alpha"][0] < true_alpha < ci["alpha"][1]

        # 5. Hill estimator
        hill_result = robust_hill_estimator(data)
        assert abs(hill_result["alpha"] - true_alpha) < 0.5

    def test_compare_mle_and_hill_estimates(self):
        """Test that MLE and Hill estimates are consistent."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(1000, seed=42)

        # MLE estimate
        mle_params = fit_mle(data, "pareto")
        mle_alpha = mle_params["alpha"]

        # Hill estimate
        hill_result = robust_hill_estimator(data)
        hill_alpha = hill_result["alpha"]

        # They should be reasonably close (both estimate the same parameter)
        assert abs(mle_alpha - hill_alpha) < 1.0


class TestHelperFunctions:
    """Test helper functions in roadmap module."""

    def test_improved_incomplete_beta(self):
        """Test improved incomplete beta function."""
        # Test basic functionality
        result = improved_incomplete_beta(2.0, 3.0, 0.5)
        assert 0 <= result <= 1
        assert math.isfinite(result)

    def test_safe_lognormal_ppf(self):
        """Test safe lognormal PPF function."""
        # Test normal case
        result = safe_lognormal_ppf(0.0, 1.0, 0.5)
        assert result > 0
        assert math.isfinite(result)

    def test_safe_lognormal_ppf_overflow(self):
        """Test safe lognormal PPF with extreme parameters."""
        # Test extreme parameters that might cause overflow
        result = safe_lognormal_ppf(100.0, 10.0, 0.9999)
        # Should return inf or a very large number without crashing
        assert result > 0  # Either finite or inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
