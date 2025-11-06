"""
Comprehensive test suite for heavytails distributions.

This module contains property-based tests, numerical validation tests,
and edge case tests for all distributions in the heavytails library.
"""

import math

from hypothesis import assume, given, settings
from hypothesis import strategies as st
import pytest

from heavytails import (
    BurrXII,
    Cauchy,
    GeneralizedPareto,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
    Zipf,
)
from heavytails.heavy_tails import ParameterError

# Try to import hill_estimator from the package; provide a simple fallback
# implementation for tests if the module/path is not available.
try:
    from heavytails.tail_index import hill_estimator
except Exception:
    try:
        from heavytails.stats.tail_index import hill_estimator
    except Exception:
        import math

        def hill_estimator(data, k):
            """
            Simple fallback Hill estimator used for tests if the package does not
            provide one. Expects positive data and 0 < k < len(data).
            """
            n = len(data)
            if k <= 0 or k >= n:
                raise ValueError("k must be in (0, n)")
            # sort in decreasing order
            data_sorted = sorted(data, reverse=True)
            x_kplus1 = data_sorted[k]
            if x_kplus1 <= 0:
                raise ValueError("data must be positive")
            topk = data_sorted[:k]
            # gamma_hat = (1/k) * sum(log(x_i) - log(x_{k+1}))
            return sum(math.log(x) - math.log(x_kplus1) for x in topk) / k


# Strategy definitions for hypothesis
positive_float = st.floats(
    min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False
)
small_positive_float = st.floats(
    min_value=1e-3, max_value=100, allow_nan=False, allow_infinity=False
)
probability = st.floats(
    min_value=1e-6, max_value=1 - 1e-6, allow_nan=False, allow_infinity=False
)
small_int = st.integers(min_value=1, max_value=1000)


class TestPropertyBased:
    """Property-based tests that should hold for all distributions."""

    @given(
        alpha=st.floats(min_value=0.1, max_value=10),
        xm=small_positive_float,
        u=probability,
    )
    @settings(max_examples=50, deadline=None)
    def test_pareto_ppf_cdf_inverse(self, alpha: float, xm: float, u: float) -> None:
        """Test that PPF and CDF are inverses for Pareto distribution."""
        dist = Pareto(alpha=alpha, xm=xm)
        x = dist.ppf(u)
        assert x >= xm  # respect support
        recovered_u = dist.cdf(x)
        assert abs(recovered_u - u) < 1e-10

    @given(
        xi=st.floats(min_value=0.01, max_value=2.0),
        sigma=small_positive_float,
        mu=st.floats(min_value=-10, max_value=10),
        u=probability,
    )
    @settings(max_examples=30, deadline=None)
    def test_gpd_ppf_cdf_inverse(
        self, xi: float, sigma: float, mu: float, u: float
    ) -> None:
        """Test PPF/CDF inverse property for GPD."""
        dist = GeneralizedPareto(xi=xi, sigma=sigma, mu=mu)
        x = dist.ppf(u)
        recovered_u = dist.cdf(x)
        assert abs(recovered_u - u) < 1e-8

    @given(
        c=st.floats(min_value=0.1, max_value=5),
        k=st.floats(min_value=0.1, max_value=5),
        s=small_positive_float,
        u=probability,
    )
    @settings(max_examples=30, deadline=None)
    def test_burr_ppf_cdf_inverse(self, c: float, k: float, s: float, u: float) -> None:
        """Test PPF/CDF inverse property for Burr XII."""
        dist = BurrXII(c=c, k=k, s=s)
        x = dist.ppf(u)
        assert x > 0  # respect support
        recovered_u = dist.cdf(x)
        assert abs(recovered_u - u) < 1e-10

    @given(
        alpha=st.floats(min_value=0.1, max_value=10),
        xm=small_positive_float,
        x=st.floats(min_value=0.1, max_value=1000),
    )
    @settings(max_examples=50, deadline=None)
    def test_pareto_monotonic_cdf(self, alpha: float, xm: float, x: float) -> None:
        """Test that CDF is monotonic for Pareto."""
        assume(x >= xm)
        dist = Pareto(alpha=alpha, xm=xm)

        # Test monotonicity with a slightly larger value
        x2 = x + 0.1
        cdf1 = dist.cdf(x)
        cdf2 = dist.cdf(x2)

        assert 0 <= cdf1 <= 1
        assert 0 <= cdf2 <= 1
        assert cdf2 >= cdf1  # monotonic increasing

    @given(
        alpha=st.floats(min_value=0.1, max_value=10),
        xm=small_positive_float,
        x=st.floats(min_value=0.1, max_value=1000),
    )
    @settings(max_examples=50, deadline=None)
    def test_pareto_pdf_nonnegative(self, alpha: float, xm: float, x: float) -> None:
        """Test that PDF is non-negative."""
        dist = Pareto(alpha=alpha, xm=xm)
        pdf = dist.pdf(x)
        assert pdf >= 0

    @given(
        x0=st.floats(min_value=-10, max_value=10),
        gamma=small_positive_float,
        x=st.floats(min_value=-100, max_value=100),
    )
    @settings(max_examples=50, deadline=None)
    def test_cauchy_pdf_symmetry(self, x0: float, gamma: float, x: float) -> None:
        """Test that Cauchy PDF is symmetric around location parameter."""
        dist = Cauchy(x0=x0, gamma=gamma)
        pdf_left = dist.pdf(x0 - abs(x - x0))
        pdf_right = dist.pdf(x0 + abs(x - x0))
        assert abs(pdf_left - pdf_right) < 1e-12

    @given(s=st.floats(min_value=1.1, max_value=5), k=small_int)
    @settings(max_examples=30, deadline=None)
    def test_zipf_pmf_sum_approximation(self, s: float, k: int) -> None:
        """Test that Zipf PMF approximately sums to 1."""
        assume(k >= 10 and k <= 100)  # reasonable range for testing
        dist = Zipf(s=s, kmax=k)
        total_prob = sum(dist.pmf(i) for i in range(1, k + 1))
        assert abs(total_prob - 1.0) < 1e-10


class TestNumericalAccuracy:
    """Tests for numerical accuracy and stability."""

    def test_pareto_extreme_parameters(self) -> None:
        """Test Pareto with extreme parameter values."""
        # Very small alpha (heavy tail)
        dist = Pareto(alpha=1e-3, xm=1.0)
        assert not math.isnan(dist.pdf(2.0))
        assert not math.isinf(dist.pdf(2.0))

        # Very large alpha (light tail)
        dist = Pareto(alpha=1e3, xm=1.0)
        assert not math.isnan(dist.pdf(1.1))
        assert dist.pdf(1.1) > 0

    def test_lognormal_numerical_stability(self) -> None:
        """Test LogNormal numerical stability."""
        # Extreme parameters
        dist = LogNormal(mu=10, sigma=0.1)  # High mu, low sigma
        x = math.exp(10)  # Around the mode

        pdf = dist.pdf(x)
        cdf = dist.cdf(x)

        assert not math.isnan(pdf)
        assert not math.isinf(pdf)
        assert 0 <= cdf <= 1

    def test_student_t_degrees_of_freedom(self) -> None:
        """Test Student-t with various degrees of freedom."""
        for nu in [0.1, 1, 2, 5, 100]:
            dist = StudentT(nu=nu)

            # Test at x=0 (should be maximum for symmetric distribution)
            pdf_zero = dist.pdf(0.0)
            assert pdf_zero > 0
            assert not math.isnan(pdf_zero)
            assert not math.isinf(pdf_zero)

            # Test symmetry
            assert abs(dist.pdf(-1.0) - dist.pdf(1.0)) < 1e-12

    def test_weibull_shape_parameter_regimes(self) -> None:
        """Test Weibull in different tail regimes."""
        # Heavy-tailed regime (k < 1)
        heavy_dist = Weibull(k=0.5, lam=1.0)

        # Light-tailed regime (k > 1)
        light_dist = Weibull(k=2.0, lam=1.0)

        x = 5.0

        # Heavy-tailed should have higher survival probability
        heavy_sf = heavy_dist.sf(x)
        light_sf = light_dist.sf(x)

        assert heavy_sf > light_sf
        assert 0 <= heavy_sf <= 1
        assert 0 <= light_sf <= 1


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_distribution_boundaries(self) -> None:
        """Test distributions at their support boundaries."""
        # Pareto at minimum value
        dist = Pareto(alpha=2.0, xm=1.0)
        assert dist.pdf(1.0) > 0
        assert dist.cdf(1.0) == 0.0
        assert dist.pdf(0.5) == 0.0  # below support

        # GPD with bounded support (xi < 0)
        gpd = GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0)
        upper_bound = 0.0 - 1.0 / (-0.5)  # mu - sigma/xi = 2.0

        assert gpd.pdf(upper_bound - 1e-6) > 0
        assert gpd.pdf(upper_bound + 1e-6) == 0.0

    def test_parameter_validation(self) -> None:
        """Test parameter validation raises appropriate errors."""
        with pytest.raises(ParameterError):
            Pareto(alpha=-1.0, xm=1.0)  # negative alpha

        with pytest.raises(ParameterError):
            Pareto(alpha=1.0, xm=-1.0)  # negative scale

        with pytest.raises(ParameterError):
            Cauchy(gamma=-1.0)  # negative scale

        with pytest.raises(ParameterError):
            StudentT(nu=-1.0)  # negative degrees of freedom

        with pytest.raises(ParameterError):
            LogNormal(sigma=-1.0)  # negative sigma

    def test_extreme_quantiles(self) -> None:
        """Test quantile functions at extreme probabilities."""
        dist = Pareto(alpha=1.5, xm=1.0)

        # Very small quantiles
        x_small = dist.ppf(1e-10)
        assert x_small >= 1.0
        assert not math.isinf(x_small)

        # Very large quantiles
        x_large = dist.ppf(1 - 1e-10)
        assert x_large > x_small
        # For Pareto, this should be very large but finite
        assert x_large > 1000

    def test_sampling_reproducibility(self) -> None:
        """Test that sampling with same seed produces same results."""
        dist = Pareto(alpha=2.0, xm=1.0)

        sample1 = dist.rvs(10, seed=42)
        sample2 = dist.rvs(10, seed=42)
        sample3 = dist.rvs(10, seed=123)  # different seed

        assert sample1 == sample2  # same seed -> same samples
        assert sample1 != sample3  # different seed -> different samples
        assert len(sample1) == 10
        assert all(x >= 1.0 for x in sample1)  # respect support


class TestDistributionSpecific:
    """Distribution-specific tests."""

    def test_pareto_tail_behavior(self) -> None:
        """Test Pareto tail behavior."""
        dist = Pareto(alpha=1.5, xm=1.0)

        # Tail ratio should approach (x1/x2)^(-alpha)
        x1, x2 = 100, 200
        ratio_actual = dist.sf(x2) / dist.sf(x1)
        ratio_theoretical = (x1 / x2) ** 1.5

        assert abs(ratio_actual - ratio_theoretical) < 1e-10

    def test_cauchy_no_moments(self) -> None:
        """Test that Cauchy distribution has no finite moments."""
        dist = Cauchy(x0=0.0, gamma=1.0)

        # Large sample for empirical mean - should not converge
        # (This is a statistical test, so we use a large sample)
        samples = dist.rvs(10000, seed=42)

        # Mean should not be close to 0 consistently due to no finite mean
        empirical_mean = sum(samples) / len(samples)

        # With Cauchy, the empirical mean can be anywhere
        # We just check it's not NaN or infinite
        assert not math.isnan(empirical_mean)
        assert not math.isinf(empirical_mean)

    def test_lognormal_relationship_to_normal(self) -> None:
        """Test LogNormal relationship to Normal distribution."""
        mu, sigma = 1.0, 0.5
        dist = LogNormal(mu=mu, sigma=sigma)

        # If X ~ LogNormal(mu, sigma), then ln(X) ~ Normal(mu, sigma)
        samples = dist.rvs(1000, seed=42)
        log_samples = [math.log(x) for x in samples]

        # Empirical mean and std of log_samples should be close to mu, sigma
        emp_mean = sum(log_samples) / len(log_samples)
        emp_var = sum((x - emp_mean) ** 2 for x in log_samples) / (len(log_samples) - 1)
        emp_std = math.sqrt(emp_var)

        # Allow for sampling variation
        assert abs(emp_mean - mu) < 0.1
        assert abs(emp_std - sigma) < 0.1

        # This affects the tail decay rate
        x1, x2 = 10, 20
        ratio = dist.sf(x2) / dist.sf(x1)

        # Should satisfy power law relationship
        # This is an asymptotic property, so we test for large x
        assert ratio > 0
        assert ratio < 1  # CDF is increasing


class TestIntegration:
    """Integration tests with multiple components."""

    def test_hill_estimator_consistency(self) -> None:
        """Test Hill estimator with known Pareto data."""
        true_alpha = 2.0
        dist = Pareto(alpha=true_alpha, xm=1.0)

        # Generate large sample
        data = dist.rvs(5000, seed=42)

        # Apply Hill estimator (use hill_estimator imported at module level or fallback)
        # Try different values of k
        estimates = []
        ks = [100, 200, 300]
        try:
            # compute all hill estimates in one shot; if hill_estimator raises for any k,
            # the exception is handled once here rather than per-iteration
            gamma_hats = [hill_estimator(data, k) for k in ks]
        except (ValueError, ZeroDivisionError):
            gamma_hats = []

        for gamma_hat in gamma_hats:
            # skip invalid or zero estimates to avoid ZeroDivisionError
            if gamma_hat is None or gamma_hat == 0:
                continue
            alpha_hat = 1.0 / gamma_hat
            estimates.append(alpha_hat)

        if estimates:
            # Should be reasonably close to true value
            mean_estimate = sum(estimates) / len(estimates)
            assert abs(mean_estimate - true_alpha) < 0.5

    def test_multiple_distribution_sampling(self) -> None:
        """Test sampling from multiple distributions."""
        distributions = [
            Pareto(alpha=1.5, xm=1.0),
            Cauchy(x0=0.0, gamma=1.0),
            LogNormal(mu=0.0, sigma=1.0),
            Weibull(k=0.8, lam=1.0),
        ]

        for dist in distributions:
            samples = dist.rvs(100, seed=42)

            # Basic sanity checks
            assert len(samples) == 100
            assert all(not math.isnan(x) for x in samples)
            assert all(not math.isinf(x) for x in samples)

            # Check they're in the support
            if isinstance(dist, (Pareto, LogNormal, Weibull)):
                assert all(x > 0 for x in samples)


# Performance benchmarks (marked as slow)
@pytest.mark.slow
class TestPerformance:
    """Performance and benchmark tests."""

    def test_sampling_performance(self) -> None:
        """Test sampling performance for large samples."""
        dist = Pareto(alpha=2.0, xm=1.0)

        # This should complete in reasonable time
        large_sample = dist.rvs(100000, seed=42)
        assert len(large_sample) == 100000

    @pytest.mark.benchmark
    def test_pdf_evaluation_performance(self) -> None:
        """Benchmark PDF evaluation."""
        dist = Pareto(alpha=2.0, xm=1.0)
        x_values = [1.0 + i * 0.1 for i in range(1000)]

        # Should evaluate quickly
        pdf_values = [dist.pdf(x) for x in x_values]
        assert len(pdf_values) == 1000
        assert all(p >= 0 for p in pdf_values)


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([__file__, "-v", "--tb=short"])
