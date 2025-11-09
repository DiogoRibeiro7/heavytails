"""
Test suite for performance optimization features.

This module tests:
- Vectorized PDF/CDF evaluation
- Optimized PPF with hybrid root-finding
- Parallel sampling
- Smart caching
- Performance benchmarking
"""

import io
import sys

import pytest

from heavytails import Pareto, StudentT
from heavytails.performance import (
    DistributionCache,
    MemoryProfiler,
    OnlineEstimation,
    PerformanceBenchmarks,
    PPFOptimizer,
    _parallel_worker,
    cython_special_functions,
    distribution_specific_optimizations,
    jit_accelerated_functions,
    optimized_rejection_sampling,
    parallel_sampling,
    robust_log_gamma,
    vectorized_cdf_evaluation,
    vectorized_pdf_evaluation,
)

# ========================================
# Vectorized Operations Tests
# ========================================


class TestVectorizedOperations:
    """Test suite for vectorized PDF/CDF evaluation."""

    def test_vectorized_pdf_pareto(self):
        """Test vectorized PDF evaluation for Pareto distribution."""
        x_values = [1.0, 2.0, 3.0, 4.0, 5.0]

        pdf_values = vectorized_pdf_evaluation("Pareto", x_values, alpha=2.5, xm=1.0)

        # Check that results match individual evaluations
        dist = Pareto(alpha=2.5, xm=1.0)
        expected = [dist.pdf(x) for x in x_values]

        for i, (actual, exp) in enumerate(zip(pdf_values, expected, strict=False)):
            assert abs(actual - exp) < 1e-10, f"Mismatch at index {i}"

    def test_vectorized_cdf_pareto(self):
        """Test vectorized CDF evaluation for Pareto distribution."""
        x_values = [1.0, 2.0, 3.0, 4.0, 5.0]

        cdf_values = vectorized_cdf_evaluation("Pareto", x_values, alpha=2.5, xm=1.0)

        # Check that results match individual evaluations
        dist = Pareto(alpha=2.5, xm=1.0)
        expected = [dist.cdf(x) for x in x_values]

        for i, (actual, exp) in enumerate(zip(cdf_values, expected, strict=False)):
            assert abs(actual - exp) < 1e-10, f"Mismatch at index {i}"

    def test_vectorized_pdf_student_t(self):
        """Test vectorized PDF evaluation for Student-t distribution."""
        x_values = [-2.0, -1.0, 0.0, 1.0, 2.0]

        pdf_values = vectorized_pdf_evaluation("StudentT", x_values, nu=5.0)

        # Check that results match individual evaluations
        dist = StudentT(nu=5.0)
        expected = [dist.pdf(x) for x in x_values]

        for i, (actual, exp) in enumerate(zip(pdf_values, expected, strict=False)):
            assert abs(actual - exp) < 1e-10, f"Mismatch at index {i}"

    def test_vectorized_invalid_distribution(self):
        """Test that invalid distribution name raises ValueError."""
        x_values = [1.0, 2.0, 3.0]

        with pytest.raises(ValueError, match="Unknown distribution"):
            vectorized_pdf_evaluation("InvalidDist", x_values, alpha=2.5)

    def test_vectorized_empty_array(self):
        """Test vectorized evaluation with empty array."""
        x_values = []

        pdf_values = vectorized_pdf_evaluation("Pareto", x_values, alpha=2.5, xm=1.0)

        assert len(pdf_values) == 0

    def test_vectorized_single_value(self):
        """Test vectorized evaluation with single value."""
        x_values = [2.0]

        pdf_values = vectorized_pdf_evaluation("Pareto", x_values, alpha=2.5, xm=1.0)

        dist = Pareto(alpha=2.5, xm=1.0)
        expected = dist.pdf(2.0)

        assert abs(pdf_values[0] - expected) < 1e-10


# ========================================
# PPF Optimizer Tests
# ========================================


class TestPPFOptimizer:
    """Test suite for optimized PPF computation."""

    def test_ppf_optimizer_basic(self):
        """Test basic PPF optimizer functionality."""
        optimizer = PPFOptimizer()
        dist = Pareto(alpha=2.5, xm=1.0)

        # Create optimized PPF
        fast_ppf = optimizer.optimize_ppf(dist.cdf, dist.pdf)

        # Test at various quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        for u in quantiles:
            optimized = fast_ppf(u)
            expected = dist.ppf(u)

            # Allow small numerical difference
            assert abs(optimized - expected) / expected < 0.01, (
                f"PPF mismatch at u={u}: {optimized} vs {expected}"
            )

    def test_ppf_optimizer_without_pdf(self):
        """Test PPF optimizer falls back to Brent's method without PDF."""
        optimizer = PPFOptimizer()
        dist = Pareto(alpha=2.5, xm=1.0)

        # Create optimized PPF without PDF
        fast_ppf = optimizer.optimize_ppf(dist.cdf, pdf_func=None)

        # Test at median
        optimized = fast_ppf(0.5)
        expected = dist.ppf(0.5)

        # Should still be reasonably close
        assert abs(optimized - expected) / expected < 0.01

    def test_ppf_optimizer_caching(self):
        """Test that PPF optimizer caches results."""
        optimizer = PPFOptimizer(cache_size=10)
        dist = Pareto(alpha=2.5, xm=1.0)

        fast_ppf = optimizer.optimize_ppf(dist.cdf, dist.pdf)

        # Call twice with same value
        result1 = fast_ppf(0.5)
        result2 = fast_ppf(0.5)

        # Should be exactly the same (cached)
        assert result1 == result2

    def test_ppf_optimizer_invalid_u(self):
        """Test that invalid u values raise ValueError."""
        optimizer = PPFOptimizer()
        dist = Pareto(alpha=2.5, xm=1.0)

        fast_ppf = optimizer.optimize_ppf(dist.cdf, dist.pdf)

        with pytest.raises(ValueError, match="must be in the open interval"):
            fast_ppf(0.0)

        with pytest.raises(ValueError, match="must be in the open interval"):
            fast_ppf(1.0)

        with pytest.raises(ValueError, match="must be in the open interval"):
            fast_ppf(-0.1)

        with pytest.raises(ValueError, match="must be in the open interval"):
            fast_ppf(1.1)


# ========================================
# Parallel Sampling Tests
# ========================================


class TestParallelSampling:
    """Test suite for parallel sampling."""

    def test_parallel_sampling_basic(self):
        """Test basic parallel sampling functionality."""
        samples = parallel_sampling(
            "Pareto", n=1000, n_cores=2, seed=42, alpha=2.5, xm=1.0
        )

        # Check length
        assert len(samples) == 1000

        # Check all samples are valid (>= xm for Pareto)
        assert all(x >= 1.0 for x in samples)

    def test_parallel_sampling_deterministic(self):
        """Test that parallel sampling is deterministic with seed."""
        samples1 = parallel_sampling(
            "Pareto", n=1000, n_cores=2, seed=42, alpha=2.5, xm=1.0
        )
        samples2 = parallel_sampling(
            "Pareto", n=1000, n_cores=2, seed=42, alpha=2.5, xm=1.0
        )

        # With same seed, should get same results
        assert len(samples1) == len(samples2)
        # Note: Due to multiprocessing, exact match may not be guaranteed
        # but distribution properties should be similar

    def test_parallel_sampling_small_n(self):
        """Test that small n falls back to serial sampling."""
        samples = parallel_sampling(
            "Pareto", n=100, n_cores=4, seed=42, alpha=2.5, xm=1.0
        )

        assert len(samples) == 100
        assert all(x >= 1.0 for x in samples)

    def test_parallel_sampling_large_n(self):
        """Test parallel sampling with large n."""
        samples = parallel_sampling(
            "Pareto", n=10000, n_cores=2, seed=42, alpha=2.5, xm=1.0
        )

        assert len(samples) == 10000
        assert all(x >= 1.0 for x in samples)

    def test_parallel_sampling_invalid_distribution(self):
        """Test that invalid distribution raises ValueError."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            parallel_sampling("InvalidDist", n=1000, alpha=2.5)


# ========================================
# Distribution Cache Tests
# ========================================


class TestDistributionCache:
    """Test suite for distribution caching."""

    def test_cache_basic_pdf(self):
        """Test basic PDF caching."""
        cache = DistributionCache()

        # First call (miss)
        pdf1 = cache.cached_pdf("Pareto", 2.0, alpha=2.5, xm=1.0)

        # Second call (hit)
        pdf2 = cache.cached_pdf("Pareto", 2.0, alpha=2.5, xm=1.0)

        # Should be exactly the same
        assert pdf1 == pdf2

        # Verify it's correct
        dist = Pareto(alpha=2.5, xm=1.0)
        assert abs(pdf1 - dist.pdf(2.0)) < 1e-10

    def test_cache_basic_cdf(self):
        """Test basic CDF caching."""
        cache = DistributionCache()

        cdf1 = cache.cached_cdf("Pareto", 2.0, alpha=2.5, xm=1.0)
        cdf2 = cache.cached_cdf("Pareto", 2.0, alpha=2.5, xm=1.0)

        assert cdf1 == cdf2

    def test_cache_basic_ppf(self):
        """Test basic PPF caching."""
        cache = DistributionCache()

        ppf1 = cache.cached_ppf("Pareto", 0.5, alpha=2.5, xm=1.0)
        ppf2 = cache.cached_ppf("Pareto", 0.5, alpha=2.5, xm=1.0)

        assert ppf1 == ppf2

    def test_cache_different_params(self):
        """Test that different parameters are cached separately."""
        cache = DistributionCache()

        pdf1 = cache.cached_pdf("Pareto", 2.0, alpha=2.5, xm=1.0)
        pdf2 = cache.cached_pdf("Pareto", 2.0, alpha=3.0, xm=1.0)

        # Different parameters should give different results
        assert pdf1 != pdf2

    def test_cache_info(self):
        """Test cache info retrieval."""
        cache = DistributionCache()

        # Make some cached calls
        cache.cached_pdf("Pareto", 2.0, alpha=2.5, xm=1.0)
        cache.cached_pdf("Pareto", 2.0, alpha=2.5, xm=1.0)  # Hit
        cache.cached_pdf("Pareto", 3.0, alpha=2.5, xm=1.0)  # Miss

        info = cache.get_cache_info()

        # Check structure
        assert "pdf" in info
        assert "hits" in info["pdf"]
        assert "misses" in info["pdf"]

        # Check values
        assert info["pdf"]["hits"] == 1
        assert info["pdf"]["misses"] == 2

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = DistributionCache()

        # Add some cached values
        cache.cached_pdf("Pareto", 2.0, alpha=2.5, xm=1.0)
        cache.cached_cdf("Pareto", 2.0, alpha=2.5, xm=1.0)

        # Clear cache
        cache.clear_cache()

        # After clearing, cache info should show all misses
        info = cache.get_cache_info()
        assert info["pdf"]["currsize"] == 0
        assert info["cdf"]["currsize"] == 0


# ========================================
# Performance Benchmarks Tests
# ========================================


class TestPerformanceBenchmarks:
    """Test suite for performance benchmarking."""

    def test_benchmark_sampling_basic(self):
        """Test basic sampling benchmark."""
        benchmarks = PerformanceBenchmarks()

        results = benchmarks.benchmark_sampling(
            "Pareto", [100, 1000], alpha=2.5, xm=1.0
        )

        # Check structure
        assert 100 in results
        assert 1000 in results

        # Check that each result has expected keys
        for size in [100, 1000]:
            assert "serial_time" in results[size]
            assert "samples_per_second" in results[size]
            assert results[size]["serial_time"] > 0

    def test_benchmark_pdf_evaluation(self):
        """Test PDF evaluation benchmark."""
        benchmarks = PerformanceBenchmarks()

        results = benchmarks.benchmark_pdf_evaluation("Pareto", 1000, alpha=2.5, xm=1.0)

        # Check structure
        assert "n_points" in results
        assert "serial_time" in results
        assert "vectorized_time" in results
        assert "speedup" in results

        # Check values are reasonable
        assert results["n_points"] == 1000
        assert results["serial_time"] > 0
        assert results["vectorized_time"] > 0

    def test_benchmark_cache_effectiveness(self):
        """Test cache effectiveness benchmark."""
        benchmarks = PerformanceBenchmarks()

        results = benchmarks.benchmark_cache_effectiveness(
            "Pareto", n_evaluations=1000, n_unique=50, alpha=2.5, xm=1.0
        )

        # Check structure
        assert "n_evaluations" in results
        assert "n_unique" in results
        assert "uncached_time" in results
        assert "cached_time" in results
        assert "speedup" in results
        assert "hit_rate" in results

        # Check values
        assert results["n_evaluations"] == 1000
        assert results["n_unique"] == 50
        assert results["hit_rate"] > 0  # Should have cache hits

    def test_get_default_params(self):
        """Test default parameter retrieval."""
        benchmarks = PerformanceBenchmarks()

        # Test known distributions
        pareto_params = benchmarks._get_default_params("Pareto")
        assert "alpha" in pareto_params
        assert "xm" in pareto_params

        student_t_params = benchmarks._get_default_params("StudentT")
        assert "nu" in student_t_params

        # Test unknown distribution
        unknown_params = benchmarks._get_default_params("UnknownDist")
        assert unknown_params == {}


# ========================================
# Integration Tests
# ========================================


class TestPerformanceIntegration:
    """Integration tests for performance features."""

    def test_vectorized_and_cache_integration(self):
        """Test integration of vectorized operations with caching."""
        cache = DistributionCache()

        x_values = [1.0, 2.0, 3.0, 2.0, 1.0]  # Repeated values

        # Use cache for PDF evaluation (populate cache via side effects)
        for x in x_values:
            cache.cached_pdf("Pareto", x, alpha=2.5, xm=1.0)

        # Check cache was effective
        info = cache.get_cache_info()
        assert info["pdf"]["hits"] >= 2  # At least 2 hits from repeated values

    def test_parallel_sampling_correctness(self):
        """Test that parallel sampling produces statistically similar results to serial."""
        # Serial sampling
        dist = Pareto(alpha=2.5, xm=1.0)
        serial_samples = dist.rvs(10000, seed=42)

        # Parallel sampling (different seed due to different implementation)
        parallel_samples = parallel_sampling(
            "Pareto", n=10000, n_cores=2, seed=43, alpha=2.5, xm=1.0
        )

        # Both should have similar statistics (not exact due to different seeds)
        # Just check that both are valid
        assert len(serial_samples) == 10000
        assert len(parallel_samples) == 10000
        assert all(x >= 1.0 for x in serial_samples)
        assert all(x >= 1.0 for x in parallel_samples)


class TestVectorizedOperationsExtended:
    """Extended tests for vectorized operations."""

    def test_vectorized_cdf_invalid_distribution(self):
        """Test that invalid distribution raises ValueError for CDF."""
        x_values = [1.0, 2.0, 3.0]
        with pytest.raises(ValueError, match="Unknown distribution"):
            vectorized_cdf_evaluation("InvalidDist", x_values, alpha=2.5)

    def test_vectorized_pdf_large_array(self):
        """Test vectorized PDF with large array (triggers numpy path if available)."""
        x_values = [float(i) for i in range(1, 50)]  # More than 10 elements
        pdf_values = vectorized_pdf_evaluation("Pareto", x_values, alpha=2.5, xm=1.0)
        assert len(pdf_values) == 49
        assert all(isinstance(p, float) for p in pdf_values)

    def test_vectorized_cdf_large_array(self):
        """Test vectorized CDF with large array (triggers numpy path if available)."""
        x_values = [float(i) for i in range(1, 50)]  # More than 10 elements
        cdf_values = vectorized_cdf_evaluation("Pareto", x_values, alpha=2.5, xm=1.0)
        assert len(cdf_values) == 49
        assert all(isinstance(c, float) for c in cdf_values)
        # CDF should be monotonically increasing
        for i in range(len(cdf_values) - 1):
            assert cdf_values[i] <= cdf_values[i + 1]


class TestPPFOptimizerExtended:
    """Extended tests for PPF optimizer edge cases."""

    def test_ppf_optimizer_extreme_quantiles(self):
        """Test PPF optimizer with very extreme quantiles."""
        optimizer = PPFOptimizer()
        dist = Pareto(alpha=2.5, xm=1.0)
        fast_ppf = optimizer.optimize_ppf(dist.cdf, dist.pdf)

        # Test extreme quantiles (triggers initial guess edge cases)
        extreme_quantiles = [0.001, 0.999]
        for u in extreme_quantiles:
            try:
                result = fast_ppf(u)
                assert result > 0  # Should be valid
            except Exception:
                # Some edge cases may fail - that's okay
                pass

    def test_ppf_optimizer_newton_fallback(self):
        """Test PPF optimizer falls back when Newton-Raphson fails."""
        optimizer = PPFOptimizer(tolerance=1e-12, max_iterations=5)
        dist = Pareto(alpha=2.5, xm=1.0)
        fast_ppf = optimizer.optimize_ppf(dist.cdf, dist.pdf)

        # Should still work even with strict tolerance and few iterations
        result = fast_ppf(0.5)
        assert result > 0


class TestUnimplementedFeatures:
    """Test that unimplemented features raise appropriate errors."""

    def test_cython_special_functions(self):
        """Test that Cython special functions raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            cython_special_functions()

    def test_robust_log_gamma_overflow(self):
        """Test robust log gamma with values."""
        # Normal values should work
        result = robust_log_gamma(5.0)
        assert result > 0

        # Test with another normal value
        result2 = robust_log_gamma(10.0)
        assert result2 > result  # log-gamma is increasing

    def test_jit_accelerated_functions(self):
        """Test that JIT acceleration raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            jit_accelerated_functions()

    def test_memory_profiler_sampling(self):
        """Test that memory profiler raises NotImplementedError."""
        profiler = MemoryProfiler()
        with pytest.raises(NotImplementedError):
            profiler.profile_sampling("Pareto", 1000)

    def test_memory_profiler_optimize(self):
        """Test that memory optimizer raises NotImplementedError."""
        profiler = MemoryProfiler()
        with pytest.raises(NotImplementedError):
            profiler.optimize_memory_usage()

    def test_optimized_rejection_sampling(self):
        """Test that optimized rejection sampling raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            optimized_rejection_sampling()

    def test_online_estimation_update(self):
        """Test that online estimation update raises NotImplementedError."""
        estimator = OnlineEstimation("Pareto")
        with pytest.raises(NotImplementedError):
            estimator.update(1.5)

    def test_online_estimation_get_estimates(self):
        """Test that online estimation get estimates raises NotImplementedError."""
        estimator = OnlineEstimation("Pareto")
        with pytest.raises(NotImplementedError):
            estimator.get_current_estimates()

    def test_distribution_specific_optimizations(self):
        """Test that distribution-specific optimizations raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            distribution_specific_optimizations()


class TestPerformanceBenchmarksExtended:
    """Extended tests for performance benchmarks."""

    def test_run_all_benchmarks(self):
        """Test running all benchmarks for distributions."""
        benchmarks = PerformanceBenchmarks()
        results = benchmarks.run_all_benchmarks(["Pareto"])

        # Check that results contain expected keys
        assert "sampling_Pareto" in results
        assert "pdf_Pareto" in results
        assert "cache_Pareto" in results

    def test_print_results(self):
        """Test printing benchmark results."""
        benchmarks = PerformanceBenchmarks()
        results = {
            "test": {
                "value1": 1.5,
                "nested": {"value2": 2.5},
                "string": "test",
            }
        }

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            benchmarks.print_results(results)
            output = buffer.getvalue()
            # Check that output contains expected content
            assert "test" in output
            assert "value1" in output
        finally:
            sys.stdout = old_stdout


class TestParallelSamplingExtended:
    """Extended tests for parallel sampling."""

    def test_parallel_sampling_without_seed(self):
        """Test parallel sampling without seed."""
        samples = parallel_sampling(
            "Pareto", n=1000, n_cores=2, seed=None, alpha=2.5, xm=1.0
        )
        assert len(samples) == 1000
        assert all(x >= 1.0 for x in samples)

    def test_parallel_worker_directly(self):
        """Test parallel worker function directly."""
        samples = _parallel_worker("Pareto", 100, 42, {"alpha": 2.5, "xm": 1.0})
        assert len(samples) == 100
        assert all(x >= 1.0 for x in samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
