"""
Performance optimization module for HeavyTails library.

This module contains performance-critical functions and optimization
TODO items that will be tracked as GitHub Issues.
"""

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
import math
import multiprocessing as mp
from typing import Any

import numpy as np

# NOTE: Future enhancement - Cython extensions for critical mathematical functions


def vectorized_pdf_evaluation(
    distribution: str, x_array: list[float], **params: Any
) -> list[float]:
    """
    Vectorized PDF evaluation for array inputs.

    .. note::
        This is now a thin wrapper. ``dist.pdf`` takes an array and returns one,
        so looking a distribution up by name and handing it the points is all
        that is left to do -- prefer calling the method directly. It used to go
        through ``np.vectorize``, which NumPy's own documentation describes as
        provided for convenience rather than performance, and which was a
        Python loop either way.

    Args:
        distribution: Name of the distribution class (e.g., 'Pareto', 'StudentT')
        x_array: Array of x values to evaluate
        **params: Distribution parameters (e.g., alpha=2.5, xm=1.0)

    Returns:
        List of PDF values evaluated at each point in x_array

    Examples:
        >>> from heavytails.performance import vectorized_pdf_evaluation
        >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> pdf_values = vectorized_pdf_evaluation('Pareto', x, alpha=2.5, xm=1.0)
    """
    # Import distribution classes (inside function to avoid circular imports)
    import heavytails  # noqa: PLC0415
    from heavytails import registry  # noqa: PLC0415

    # Get distribution class
    try:
        dist_class = registry.resolve(distribution)
    except AttributeError as e:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: {', '.join(heavytails.__all__)}"
        ) from e

    # Create distribution instance
    dist = dist_class(**params)

    # One call. `dist.pdf` takes an array and returns one, so np.vectorize --
    # which is a Python loop wearing an array interface -- has nothing left to
    # do here, and the branch for NumPy being absent has nothing left to catch.
    return [
        float(value) for value in np.asarray(dist.pdf(np.asarray(x_array, dtype=float)))
    ]


def vectorized_cdf_evaluation(
    distribution: str, x_array: list[float], **params: Any
) -> list[float]:
    """
    Vectorized CDF evaluation for array inputs.

    .. note::
        See :func:`vectorized_pdf_evaluation`. Prefer calling ``dist.cdf``
        directly, which takes an array.

    Args:
        distribution: Name of the distribution class (e.g., 'Pareto', 'StudentT')
        x_array: Array of x values to evaluate
        **params: Distribution parameters

    Returns:
        List of CDF values evaluated at each point in x_array

    Examples:
        >>> from heavytails.performance import vectorized_cdf_evaluation
        >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> cdf_values = vectorized_cdf_evaluation('Pareto', x, alpha=2.5, xm=1.0)
    """
    # Import distribution classes (inside function to avoid circular imports)
    import heavytails  # noqa: PLC0415
    from heavytails import registry  # noqa: PLC0415

    # Get distribution class
    try:
        dist_class = registry.resolve(distribution)
    except AttributeError as e:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: {', '.join(heavytails.__all__)}"
        ) from e

    # Create distribution instance
    dist = dist_class(**params)

    # One call. `dist.cdf` takes an array and returns one, so np.vectorize --
    # which is a Python loop wearing an array interface -- has nothing left to
    # do here, and the branch for NumPy being absent has nothing left to catch.
    return [
        float(value) for value in np.asarray(dist.cdf(np.asarray(x_array, dtype=float)))
    ]


@dataclass
class PPFOptimizer:
    """
    Optimized quantile function (PPF) computation using hybrid root-finding.

    Uses Newton-Raphson when PDF is available, with fallback to Brent's method
    for robustness. Includes caching for frequently used quantiles.

    Attributes:
        max_iterations: Maximum iterations for root-finding (default: 100)
        tolerance: Convergence tolerance (default: 1e-9)
        cache_size: Size of LRU cache for quantiles (default: 128)
    """

    max_iterations: int = 100
    tolerance: float = 1e-9
    cache_size: int = 128

    def optimize_ppf(
        self,
        cdf_func: Callable[[float], float],
        pdf_func: Callable[[float], float] | None = None,
    ) -> Callable[[float], float]:
        """
        Create optimized PPF using hybrid root-finding methods.

        Tries Newton-Raphson first (if PDF available), falls back to Brent's
        method for robustness. Caches frequently used quantiles.

        Args:
            cdf_func: Cumulative distribution function F(x)
            pdf_func: Probability density function f(x), optional

        Returns:
            Optimized PPF function that takes u in (0,1) and returns x

        Examples:
            >>> optimizer = PPFOptimizer()
            >>> from heavytails import Pareto
            >>> dist = Pareto(alpha=2.5, xm=1.0)
            >>> fast_ppf = optimizer.optimize_ppf(dist.cdf, dist.pdf)
            >>> x = fast_ppf(0.5)
        """

        @lru_cache(maxsize=self.cache_size)
        def cached_ppf(u: float) -> float:
            """Cached PPF with hybrid root-finding."""
            if not (0.0 < u < 1.0):
                raise ValueError("u must be in the open interval (0, 1)")

            # Try Newton-Raphson if PDF is available
            if pdf_func is not None:
                try:
                    result = self._newton_raphson_ppf(u, cdf_func, pdf_func)
                    if result is not None:
                        return result
                except (ValueError, ZeroDivisionError, OverflowError):
                    pass  # Fall through to Brent's method

            # Fall back to Brent's method
            return self._brent_ppf(u, cdf_func)

        return cached_ppf

    def _newton_raphson_ppf(
        self,
        u: float,
        cdf_func: Callable[[float], float],
        pdf_func: Callable[[float], float],
    ) -> float | None:
        """
        Newton-Raphson method for PPF computation.

        Uses the derivative information from PDF for faster convergence.
        Formula: x_{n+1} = x_n - (F(x_n) - u) / f(x_n)

        Returns None if convergence fails.
        """
        # Initial guess using linear interpolation
        x = self._initial_guess(u, cdf_func)

        for _ in range(self.max_iterations):
            fx = cdf_func(x)
            fpx = pdf_func(x)

            if abs(fpx) < 1e-15:  # Avoid division by zero
                return None

            # Newton-Raphson update
            x_new = x - (fx - u) / fpx

            # Check convergence
            if abs(x_new - x) < self.tolerance:
                return x_new

            x = x_new

            # Bounds checking
            if not (-1e10 < x < 1e10):  # Prevent divergence
                return None

        return None  # Failed to converge

    def _brent_ppf(self, u: float, cdf_func: Callable[[float], float]) -> float:
        """
        Brent's method for robust PPF computation.

        More robust than bisection, with similar convergence guarantees
        but faster in practice. Combines bisection, secant, and inverse
        quadratic interpolation.
        """
        # Find bracketing interval [a, b] such that F(a) < u < F(b)
        a, b = self._find_bracket(u, cdf_func)

        # Brent's method implementation
        fa = cdf_func(a) - u
        fb = cdf_func(b) - u

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        c = a
        fc = fa
        mflag = True

        for _ in range(self.max_iterations):
            if abs(fb) < self.tolerance:
                return b

            if fc not in (fa, fb):
                # Inverse quadratic interpolation
                s = (
                    a * fb * fc / ((fa - fb) * (fa - fc))
                    + b * fa * fc / ((fb - fa) * (fb - fc))
                    + c * fa * fb / ((fc - fa) * (fc - fb))
                )
            else:
                # Secant method
                s = b - fb * (b - a) / (fb - fa)

            # Check if bisection should be used
            tmp2 = (3 * a + b) / 4
            if not (
                (s > tmp2 and s < b)
                or (
                    (s < tmp2 and s > b)
                    and (
                        (mflag and abs(s - b) < abs(b - c) / 2)
                        or (not mflag and abs(s - b) < abs(c - c) / 2)
                    )
                )
            ):
                s = (a + b) / 2
                mflag = True
            else:
                mflag = False

            fs = cdf_func(s) - u
            c = b

            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        return b

    def _initial_guess(self, u: float, cdf_func: Callable[[float], float]) -> float:  # noqa: ARG002
        """Generate initial guess for root-finding based on u."""
        # Simple heuristic: try a few standard points
        # Note: cdf_func parameter reserved for future use with adaptive guessing
        if u < 0.01:
            return -10.0
        elif u > 0.99:
            return 10.0
        else:
            return 0.0

    def _find_bracket(
        self, u: float, cdf_func: Callable[[float], float]
    ) -> tuple[float, float]:
        """
        Find bracketing interval [a, b] such that F(a) < u < F(b).

        Uses exponential search to find suitable bounds.
        """
        # Start with a reasonable initial interval
        a, b = -1.0, 1.0

        # Expand interval until we bracket the quantile
        max_expansions = 50
        for _ in range(max_expansions):
            fa = cdf_func(a)
            fb = cdf_func(b)

            if fa < u < fb:
                return a, b
            elif u <= fa:
                # Need to search lower
                b = a
                a = 2 * a - 1.0 if a < 0 else a / 2.0 - 1.0
            elif u >= fb:
                # Need to search higher
                a = b
                b = 2 * b + 1.0 if b > 0 else b / 2.0 + 1.0

        # Default to wide interval if bracketing fails
        return -100.0, 100.0


def parallel_sampling(
    distribution: str,
    n: int,
    n_cores: int | None = None,
    seed: int | None = None,
    **params: Any,
) -> list[float]:
    """
    Parallel random number generation for large samples using multiprocessing.

    Splits the sampling task across multiple CPU cores for faster generation.
    Particularly beneficial for large samples (n > 10000) and distributions
    with expensive sampling procedures.

    Args:
        distribution: Name of the distribution class (e.g., 'Pareto', 'StudentT')
        n: Number of samples to generate
        n_cores: Number of CPU cores to use (default: all available cores)
        seed: Random seed for reproducibility (default: None)
        **params: Distribution parameters

    Returns:
        List of n random samples from the distribution

    Examples:
        >>> from heavytails.performance import parallel_sampling
        >>> samples = parallel_sampling('Pareto', n=100000, n_cores=4, seed=42, alpha=2.5, xm=1.0)
        >>> len(samples)
        100000

    Note:
        - For small n (< 1000), overhead may outweigh benefits
        - Uses different random seeds for each worker to ensure independence
        - Results are deterministic when seed is provided
    """
    # Import distribution classes (inside function to avoid circular imports)
    import heavytails  # noqa: PLC0415
    from heavytails import registry  # noqa: PLC0415

    # Get distribution class
    try:
        dist_class = registry.resolve(distribution)
    except AttributeError as e:
        raise ValueError(
            f"Unknown distribution '{distribution}'. "
            f"Available: {', '.join(heavytails.__all__)}"
        ) from e

    # Determine number of cores
    n_cores = mp.cpu_count() if n_cores is None else min(n_cores, mp.cpu_count())

    # For small samples, don't use parallelization (overhead too high)
    if n < 1000 or n_cores == 1:
        dist = dist_class(**params)
        return [float(value) for value in dist.rvs(n, seed=seed)]

    # Split work across cores
    chunk_size = n // n_cores
    remainder = n % n_cores

    # Create chunk sizes ensuring all samples are generated
    chunk_sizes = [chunk_size] * n_cores
    for i in range(remainder):
        chunk_sizes[i] += 1

    # Generate different seeds for each worker
    if seed is not None:
        import random  # noqa: PLC0415

        rng = random.Random(seed)
        worker_seeds: list[int | None] = [
            rng.randint(0, 2**31 - 1) for _ in range(n_cores)
        ]
    else:
        worker_seeds = [None] * n_cores

    # Create worker arguments
    worker_args = [
        (distribution, chunk_sizes[i], worker_seeds[i], params) for i in range(n_cores)
    ]

    # Use multiprocessing pool for parallel sampling
    with mp.Pool(processes=n_cores) as pool:
        results = pool.starmap(_parallel_worker, worker_args)

    # Combine results from all workers
    samples = []
    for chunk in results:
        samples.extend(chunk)

    return samples


def _parallel_worker(
    distribution: str, chunk_size: int, seed: int | None, params: dict[str, Any]
) -> list[float]:
    """
    Worker function for parallel sampling.

    This function is executed in each worker process to generate a chunk
    of samples independently.

    Args:
        distribution: Name of the distribution class
        chunk_size: Number of samples to generate in this worker
        seed: Random seed for this worker
        params: Distribution parameters

    Returns:
        List of samples generated by this worker
    """
    from heavytails import registry  # noqa: PLC0415

    # Get distribution class
    dist_class = registry.resolve(distribution)

    # Create distribution instance
    dist = dist_class(**params)

    # Generate samples
    return [float(value) for value in dist.rvs(chunk_size, seed=seed)]


# NOTE: Future enhancement - memory profiling and optimization tools


# NOTE: Current implementation uses Python's math.lgamma
def robust_log_gamma(x: float) -> float:
    """
    Numerically stable log-gamma function.

    Currently uses math.lgamma which works for most cases but can fail for:
    - Very large arguments (> 1e308)
    - Arguments very close to negative integers

    Future enhancements could include:
    - Stirling's approximation for large x
    - Series expansions for problematic regions
    - Complex plane extensions
    - Error bounds and accuracy guarantees
    """
    try:
        return math.lgamma(x)
    except (OverflowError, ValueError) as e:
        raise NotImplementedError(
            "Extended log-gamma for edge cases not yet implemented"
        ) from e


# NOTE: Future enhancement - Just-In-Time (JIT) compilation with Numba


class DistributionCache:
    """
    Intelligent caching system for distribution computations.

    Uses LRU (Least Recently Used) caching to store results of expensive
    computations. Particularly useful for:
    - Repeated PDF/CDF evaluations with same parameters
    - Quantile calculations for common probabilities
    - Parameter fitting results

    Attributes:
        max_size: Maximum number of cached entries (default: 1000)
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize the distribution cache.

        Args:
            max_size: Maximum number of entries to cache
        """
        self.max_size = max_size
        self._pdf_cache: dict[tuple[Any, ...], float] = {}
        self._cdf_cache: dict[tuple[Any, ...], float] = {}
        self._ppf_cache: dict[tuple[Any, ...], float] = {}

        # Create LRU-decorated methods
        self._cached_pdf_impl = lru_cache(maxsize=max_size)(self._compute_pdf)
        self._cached_cdf_impl = lru_cache(maxsize=max_size)(self._compute_cdf)
        self._cached_ppf_impl = lru_cache(maxsize=max_size)(self._compute_ppf)

    def cached_pdf(self, distribution: str, x: float, **params: Any) -> float:
        """
        Cached PDF evaluation.

        Args:
            distribution: Name of the distribution class
            x: Point at which to evaluate PDF
            **params: Distribution parameters

        Returns:
            PDF value at x

        Examples:
            >>> cache = DistributionCache()
            >>> pdf1 = cache.cached_pdf('Pareto', 2.0, alpha=2.5, xm=1.0)
            >>> pdf2 = cache.cached_pdf('Pareto', 2.0, alpha=2.5, xm=1.0)  # Retrieved from cache
        """
        # Create hashable key from parameters
        param_key = self._make_param_key(params)
        return self._cached_pdf_impl(distribution, x, param_key)

    def cached_cdf(self, distribution: str, x: float, **params: Any) -> float:
        """
        Cached CDF evaluation.

        Args:
            distribution: Name of the distribution class
            x: Point at which to evaluate CDF
            **params: Distribution parameters

        Returns:
            CDF value at x

        Examples:
            >>> cache = DistributionCache()
            >>> cdf = cache.cached_cdf('Pareto', 2.0, alpha=2.5, xm=1.0)
        """
        # Create hashable key from parameters
        param_key = self._make_param_key(params)
        return self._cached_cdf_impl(distribution, x, param_key)

    def cached_ppf(self, distribution: str, u: float, **params: Any) -> float:
        """
        Cached PPF (quantile) evaluation.

        Args:
            distribution: Name of the distribution class
            u: Probability at which to evaluate PPF (0 < u < 1)
            **params: Distribution parameters

        Returns:
            Quantile value at probability u

        Examples:
            >>> cache = DistributionCache()
            >>> q = cache.cached_ppf('Pareto', 0.5, alpha=2.5, xm=1.0)
        """
        # Create hashable key from parameters
        param_key = self._make_param_key(params)
        return self._cached_ppf_impl(distribution, u, param_key)

    def _compute_pdf(
        self, distribution: str, x: float, param_key: tuple[Any, ...]
    ) -> float:
        """Internal method to compute PDF (called by LRU cache)."""
        from heavytails import registry  # noqa: PLC0415

        params = dict(param_key)
        dist_class = registry.resolve(distribution)
        dist = dist_class(**params)
        return float(dist.pdf(x))

    def _compute_cdf(
        self, distribution: str, x: float, param_key: tuple[Any, ...]
    ) -> float:
        """Internal method to compute CDF (called by LRU cache)."""
        from heavytails import registry  # noqa: PLC0415

        params = dict(param_key)
        dist_class = registry.resolve(distribution)
        dist = dist_class(**params)
        return float(dist.cdf(x))

    def _compute_ppf(
        self, distribution: str, u: float, param_key: tuple[Any, ...]
    ) -> float:
        """Internal method to compute PPF (called by LRU cache)."""
        from heavytails import registry  # noqa: PLC0415

        params = dict(param_key)
        dist_class = registry.resolve(distribution)
        dist = dist_class(**params)
        return float(dist.ppf(u))

    def _make_param_key(self, params: dict[str, Any]) -> tuple[Any, ...]:
        """
        Create a hashable key from parameters dictionary.

        Args:
            params: Dictionary of distribution parameters

        Returns:
            Tuple representation suitable for hashing
        """
        # Sort items to ensure consistent ordering
        return tuple(sorted(params.items()))

    def clear_cache(self) -> None:
        """Clear all cached values."""
        self._cached_pdf_impl.cache_clear()
        self._cached_cdf_impl.cache_clear()
        self._cached_ppf_impl.cache_clear()

    def get_cache_info(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary containing cache hit/miss statistics for each function
        """
        return {
            "pdf": self._cached_pdf_impl.cache_info()._asdict(),
            "cdf": self._cached_cdf_impl.cache_info()._asdict(),
            "ppf": self._cached_ppf_impl.cache_info()._asdict(),
        }


class PerformanceBenchmarks:
    """
    Comprehensive performance benchmarking suite.

    Tracks performance metrics for distribution operations including:
    - PDF/CDF evaluation speed
    - Sampling performance
    - Parallel vs serial performance
    - Cache effectiveness

    Attributes:
        benchmarks: Dictionary storing benchmark results
    """

    def __init__(self) -> None:
        """Initialize the benchmarking suite."""
        # Heterogeneous: sampling results are keyed by sample size, the
        # others by operation name.
        self.benchmarks: dict[str, Any] = {}

    def benchmark_sampling(
        self, distribution: str, sizes: list[int], **params: Any
    ) -> dict[int, dict[str, Any]]:
        """
        Benchmark sampling performance across different sample sizes.

        Args:
            distribution: Name of the distribution class
            sizes: List of sample sizes to benchmark
            **params: Distribution parameters

        Returns:
            Dictionary containing timing results for each size

        Examples:
            >>> benchmarks = PerformanceBenchmarks()
            >>> results = benchmarks.benchmark_sampling('Pareto', [100, 1000, 10000], alpha=2.5, xm=1.0)
        """
        import time  # noqa: PLC0415

        from heavytails import registry  # noqa: PLC0415

        results: dict[int, dict[str, Any]] = {}
        dist_class = registry.resolve(distribution)
        dist = dist_class(**params)

        for n in sizes:
            # Time serial sampling
            start = time.perf_counter()
            _ = dist.rvs(n, seed=42)
            serial_time = time.perf_counter() - start

            # Time parallel sampling (if n is large enough)
            if n >= 1000:
                start = time.perf_counter()
                _ = parallel_sampling(distribution, n, seed=42, **params)
                parallel_time = time.perf_counter() - start
                speedup = serial_time / parallel_time if parallel_time > 0 else 0
            else:
                parallel_time = None
                speedup = None

            results[n] = {
                "serial_time": serial_time,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "samples_per_second": n / serial_time if serial_time > 0 else 0,
            }

        self.benchmarks[f"sampling_{distribution}"] = results
        return results

    def benchmark_pdf_evaluation(
        self, distribution: str, n_points: int, **params: Any
    ) -> dict[str, Any]:
        """
        Benchmark PDF evaluation performance.

        Args:
            distribution: Name of the distribution class
            n_points: Number of points to evaluate
            **params: Distribution parameters

        Returns:
            Dictionary containing timing results

        Examples:
            >>> benchmarks = PerformanceBenchmarks()
            >>> results = benchmarks.benchmark_pdf_evaluation('Pareto', 10000, alpha=2.5, xm=1.0)
        """
        import time  # noqa: PLC0415

        from heavytails import registry  # noqa: PLC0415

        dist_class = registry.resolve(distribution)
        dist = dist_class(**params)

        # Generate test points
        x_values = [1.0 + i * 0.1 for i in range(n_points)]

        # Benchmark element-by-element evaluation
        start = time.perf_counter()
        _ = [dist.pdf(x) for x in x_values]
        serial_time = time.perf_counter() - start

        # Benchmark vectorized evaluation
        start = time.perf_counter()
        _ = vectorized_pdf_evaluation(distribution, x_values, **params)
        vectorized_time = time.perf_counter() - start

        results = {
            "n_points": n_points,
            "serial_time": serial_time,
            "vectorized_time": vectorized_time,
            "speedup": serial_time / vectorized_time if vectorized_time > 0 else 0,
            "evaluations_per_second": n_points / serial_time if serial_time > 0 else 0,
        }

        self.benchmarks[f"pdf_evaluation_{distribution}"] = results
        return results

    def benchmark_cache_effectiveness(
        self, distribution: str, n_evaluations: int, n_unique: int, **params: Any
    ) -> dict[str, Any]:
        """
        Benchmark cache effectiveness for repeated evaluations.

        Args:
            distribution: Name of the distribution class
            n_evaluations: Total number of evaluations to perform
            n_unique: Number of unique evaluation points
            **params: Distribution parameters

        Returns:
            Dictionary containing cache performance metrics

        Examples:
            >>> benchmarks = PerformanceBenchmarks()
            >>> results = benchmarks.benchmark_cache_effectiveness(
            ...     'Pareto', n_evaluations=10000, n_unique=100, alpha=2.5, xm=1.0
            ... )
        """
        import time  # noqa: PLC0415

        from heavytails import registry  # noqa: PLC0415

        dist_class = registry.resolve(distribution)
        dist = dist_class(**params)

        # Generate evaluation points with repetitions
        x_values = [1.0 + (i % n_unique) * 0.1 for i in range(n_evaluations)]

        # Benchmark without cache
        start = time.perf_counter()
        for x in x_values:
            dist.pdf(x)
        uncached_time = time.perf_counter() - start

        # Benchmark with cache
        cache = DistributionCache(max_size=n_unique)
        start = time.perf_counter()
        for x in x_values:
            cache.cached_pdf(distribution, x, **params)
        cached_time = time.perf_counter() - start

        # Get cache statistics
        cache_info = cache.get_cache_info()

        results = {
            "n_evaluations": n_evaluations,
            "n_unique": n_unique,
            "uncached_time": uncached_time,
            "cached_time": cached_time,
            "speedup": uncached_time / cached_time if cached_time > 0 else 0,
            "hit_rate": cache_info["pdf"]["hits"] / n_evaluations
            if n_evaluations > 0
            else 0,
            "cache_info": cache_info["pdf"],
        }

        self.benchmarks[f"cache_{distribution}"] = results
        return results

    def run_all_benchmarks(
        self, distributions: list[str] | None = None
    ) -> dict[str, Any]:
        """
        Execute comprehensive benchmark suite for specified distributions.

        Args:
            distributions: List of distribution names to benchmark
                          (default: ['Pareto', 'StudentT', 'LogNormal'])

        Returns:
            Dictionary containing all benchmark results

        Examples:
            >>> benchmarks = PerformanceBenchmarks()
            >>> results = benchmarks.run_all_benchmarks(['Pareto'])
        """
        if distributions is None:
            distributions = ["Pareto", "StudentT", "LogNormal"]

        # Heterogeneous, like self.benchmarks: sampling entries are keyed
        # by sample size and the rest by operation name.
        all_results: dict[str, Any] = {}

        for dist_name in distributions:
            # Get default parameters for each distribution
            params = self._get_default_params(dist_name)

            # Run sampling benchmarks
            sampling_results = self.benchmark_sampling(
                dist_name, [100, 1000, 10000], **params
            )
            all_results[f"sampling_{dist_name}"] = sampling_results

            # Run PDF evaluation benchmarks
            pdf_results = self.benchmark_pdf_evaluation(dist_name, 10000, **params)
            all_results[f"pdf_{dist_name}"] = pdf_results

            # Run cache benchmarks
            cache_results = self.benchmark_cache_effectiveness(
                dist_name, n_evaluations=10000, n_unique=100, **params
            )
            all_results[f"cache_{dist_name}"] = cache_results

        self.benchmarks.update(all_results)
        return all_results

    def _get_default_params(self, distribution: str) -> dict[str, float]:
        """Get default parameters for a distribution."""
        defaults = {
            "Pareto": {"alpha": 2.5, "xm": 1.0},
            "StudentT": {"nu": 5.0},
            "LogNormal": {"mu": 0.0, "sigma": 1.0},
            "Cauchy": {"x0": 0.0, "gamma": 1.0},
            "Weibull": {"k": 1.5, "lambda_scale": 1.0},
            "GeneralizedPareto": {"xi": 0.5, "sigma": 1.0, "mu": 0.0},
        }
        return defaults.get(distribution, {})

    def print_results(self, results: dict[str, Any] | None = None) -> None:
        """
        Print benchmark results in a readable format.

        Args:
            results: Results dictionary to print (default: all stored benchmarks)
        """
        if results is None:
            results = self.benchmarks

        for benchmark_name, data in results.items():
            print(f"\n{benchmark_name}:")
            print("-" * 60)
            self._print_dict(data, indent=2)

    def _print_dict(self, d: dict[str, Any], indent: int = 0) -> None:
        """Recursively print dictionary with indentation."""
        for key, value in d.items():
            if isinstance(value, dict):
                print(" " * indent + f"{key}:")
                self._print_dict(value, indent + 2)
            elif isinstance(value, float):
                print(" " * indent + f"{key}: {value:.6f}")
            else:
                print(" " * indent + f"{key}: {value}")


# NOTE: Future enhancement - optimized rejection sampling algorithms


# NOTE: Future enhancement - streaming algorithms for online parameter estimation


# NOTE: Future enhancement - distribution-specific optimizations


if __name__ == "__main__":
    print("Performance optimization module loaded.")
    print("Contains TODO items for future performance improvements.")
