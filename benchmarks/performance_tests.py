"""
Performance benchmarking suite for HeavyTails distributions.

This module provides comprehensive benchmarking tools for all distribution
operations including PDF, CDF, sampling, and parameter estimation.
"""

import argparse
import contextlib
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, ClassVar

import heavytails


@dataclass
class BenchmarkResult:
    """
    Container for benchmark result data.

    Attributes:
        distribution: Name of the distribution
        operation: Type of operation benchmarked (pdf, cdf, rvs, etc.)
        mean_time: Mean execution time in seconds
        std_time: Standard deviation of execution time
        min_time: Minimum execution time
        max_time: Maximum execution time
        iterations: Number of iterations performed
        operations_per_second: Throughput in operations per second
        parameters: Distribution parameters used
    """

    distribution: str
    operation: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    operations_per_second: float
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return asdict(self)


class DistributionBenchmark:
    """
    Benchmark suite for heavy-tailed distributions.

    Provides methods to benchmark various operations on all distributions
    in the HeavyTails library.
    """

    # Distribution test configurations
    DISTRIBUTIONS: ClassVar[dict[str, dict[str, float]]] = {
        "Pareto": {"alpha": 2.5, "xm": 1.0},
        "StudentT": {"nu": 5.0},
        "LogNormal": {"mu": 0.0, "sigma": 1.0},
        "Cauchy": {"x0": 0.0, "gamma": 1.0},
        "Weibull": {"k": 1.5, "lam": 1.0},
        "Frechet": {"alpha": 2.0, "s": 1.0, "m": 0.0},
        "GEV_Frechet": {"xi": 0.3, "mu": 0.0, "sigma": 1.0},
        "GeneralizedPareto": {"xi": 0.5, "sigma": 1.0, "mu": 0.0},
        "BurrXII": {"c": 2.0, "k": 1.0, "s": 1.0},
        "LogLogistic": {"kappa": 1.0, "lam": 1.0},
        "InverseGamma": {"alpha": 3.0, "beta": 1.0},
        "BetaPrime": {"a": 2.0, "b": 5.0, "s": 1.0},
        "Zipf": {"s": 2.0},
        "YuleSimon": {"rho": 2.0},
        "DiscretePareto": {"alpha": 2.0, "k_min": 1},
    }

    def __init__(self, iterations: int = 100):
        """
        Initialize the benchmark suite.

        Args:
            iterations: Number of iterations to run for each benchmark
        """
        self.iterations = iterations
        self.results: list[BenchmarkResult] = []

    def benchmark_pdf(self, dist_name: str, params: dict[str, Any]) -> BenchmarkResult:
        """
        Benchmark PDF evaluation for a distribution.

        Args:
            dist_name: Name of the distribution class
            params: Distribution parameters

        Returns:
            BenchmarkResult containing timing statistics
        """
        dist_class = getattr(heavytails, dist_name)
        dist = dist_class(**params)

        # Generate test points
        if dist_name in ["Zipf", "YuleSimon", "DiscretePareto"]:
            test_points = list(range(1, 21))
        else:
            test_points = [1.0 + i * 0.5 for i in range(20)]

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            for x in test_points:
                with contextlib.suppress(ValueError, ZeroDivisionError, OverflowError):
                    dist.pdf(x)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._create_result(dist_name, "pdf", times, params)

    def benchmark_cdf(self, dist_name: str, params: dict[str, Any]) -> BenchmarkResult:
        """
        Benchmark CDF evaluation for a distribution.

        Args:
            dist_name: Name of the distribution class
            params: Distribution parameters

        Returns:
            BenchmarkResult containing timing statistics
        """
        dist_class = getattr(heavytails, dist_name)
        dist = dist_class(**params)

        # Generate test points based on support
        if dist_name in ["Zipf", "YuleSimon", "DiscretePareto"]:
            test_points = list(range(1, 21))
        elif dist_name == "Frechet":
            # Frechet has support x > m, use m + positive values
            m = params.get("m", 0.0)
            test_points = [m + 1.0 + i * 0.5 for i in range(20)]
        else:
            test_points = [1.0 + i * 0.5 for i in range(20)]

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            for x in test_points:
                with contextlib.suppress(
                    ValueError, ZeroDivisionError, OverflowError, AttributeError
                ):
                    dist.cdf(x)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._create_result(dist_name, "cdf", times, params)

    def benchmark_ppf(self, dist_name: str, params: dict[str, Any]) -> BenchmarkResult:
        """
        Benchmark PPF (quantile) evaluation for a distribution.

        Args:
            dist_name: Name of the distribution class
            params: Distribution parameters

        Returns:
            BenchmarkResult containing timing statistics
        """
        # Skip discrete distributions that may not have PPF
        if dist_name in ["Zipf", "YuleSimon", "DiscretePareto"]:
            return self._create_dummy_result(dist_name, "ppf", params)

        dist_class = getattr(heavytails, dist_name)
        dist = dist_class(**params)

        # Test points in (0, 1)
        test_points = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

        times = []
        for _ in range(self.iterations):
            start = time.perf_counter()
            for u in test_points:
                with contextlib.suppress(
                    ValueError, ZeroDivisionError, OverflowError, NotImplementedError
                ):
                    dist.ppf(u)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._create_result(dist_name, "ppf", times, params)

    def benchmark_rvs(self, dist_name: str, params: dict[str, Any]) -> BenchmarkResult:
        """
        Benchmark random variate sampling for a distribution.

        Args:
            dist_name: Name of the distribution class
            params: Distribution parameters

        Returns:
            BenchmarkResult containing timing statistics
        """
        dist_class = getattr(heavytails, dist_name)
        dist = dist_class(**params)

        sample_size = 1000
        times = []

        for i in range(self.iterations):
            start = time.perf_counter()
            with contextlib.suppress(ValueError, ZeroDivisionError, OverflowError):
                dist.rvs(sample_size, seed=42 + i)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        return self._create_result(dist_name, "rvs", times, params)

    def run_all_benchmarks(self) -> list[BenchmarkResult]:
        """
        Run all benchmarks for all distributions.

        Returns:
            List of BenchmarkResult objects
        """
        all_results = []

        for dist_name, params in self.DISTRIBUTIONS.items():
            print(f"Benchmarking {dist_name}...")

            # Benchmark each operation
            try:
                all_results.append(self.benchmark_pdf(dist_name, params))
                all_results.append(self.benchmark_cdf(dist_name, params))
                all_results.append(self.benchmark_ppf(dist_name, params))
                all_results.append(self.benchmark_rvs(dist_name, params))
            except Exception as e:
                print(f"  Warning: Error benchmarking {dist_name}: {e}")
                continue

        self.results = all_results
        return all_results

    def _create_result(
        self, dist_name: str, operation: str, times: list[float], params: dict[str, Any]
    ) -> BenchmarkResult:
        """Create a BenchmarkResult from timing data."""
        if not times:
            times = [0.0]

        mean_time = sum(times) / len(times)
        variance = sum((t - mean_time) ** 2 for t in times) / len(times)
        std_time = variance**0.5
        min_time = min(times)
        max_time = max(times)
        ops_per_sec = 1.0 / mean_time if mean_time > 0 else 0.0

        return BenchmarkResult(
            distribution=dist_name,
            operation=operation,
            mean_time=mean_time,
            std_time=std_time,
            min_time=min_time,
            max_time=max_time,
            iterations=self.iterations,
            operations_per_second=ops_per_sec,
            parameters=params,
        )

    def _create_dummy_result(
        self, dist_name: str, operation: str, params: dict[str, Any]
    ) -> BenchmarkResult:
        """Create a dummy result for unsupported operations."""
        return BenchmarkResult(
            distribution=dist_name,
            operation=operation,
            mean_time=0.0,
            std_time=0.0,
            min_time=0.0,
            max_time=0.0,
            iterations=0,
            operations_per_second=0.0,
            parameters=params,
        )

    def save_results(self, filepath: str):
        """
        Save benchmark results to JSON file.

        Args:
            filepath: Path to output file
        """
        with Path(filepath).open("w") as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)

    def compare_with_baseline(
        self, baseline_file: str, threshold: float = 0.25
    ) -> list[dict[str, Any]]:
        """
        Compare current results with baseline.

        Args:
            baseline_file: Path to baseline JSON file
            threshold: Regression threshold (default: 25%)

        Returns:
            List of regressions detected
        """
        try:
            with Path(baseline_file).open() as f:
                baseline = json.load(f)
        except FileNotFoundError:
            print(f"Baseline file {baseline_file} not found, skipping comparison")
            return []

        regressions = []
        for current_result in self.results:
            for base_result in baseline:
                if (
                    current_result.distribution == base_result["distribution"]
                    and current_result.operation == base_result["operation"]
                    and base_result["mean_time"] > 0
                ):
                    change = (
                        current_result.mean_time - base_result["mean_time"]
                    ) / base_result["mean_time"]

                    if change > threshold:
                        regressions.append(
                            {
                                "distribution": current_result.distribution,
                                "operation": current_result.operation,
                                "change": change,
                                "baseline_time": base_result["mean_time"],
                                "current_time": current_result.mean_time,
                            }
                        )

        return regressions


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Performance benchmarking for HeavyTails distributions"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file for benchmark results",
    )
    parser.add_argument(
        "--baseline", type=str, help="Baseline JSON file for comparison", default=None
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per benchmark (default: 50)",
    )

    args = parser.parse_args()

    print(f"Starting performance benchmarks ({args.iterations} iterations)...")
    print("-" * 60)

    # Run benchmarks
    benchmark = DistributionBenchmark(iterations=args.iterations)
    results = benchmark.run_all_benchmarks()

    print(f"\nCompleted {len(results)} benchmarks")
    print(f"Saving results to {args.output}")

    # Save results
    benchmark.save_results(args.output)

    # Compare with baseline if provided
    if args.baseline:
        print(f"\nComparing with baseline: {args.baseline}")
        regressions = benchmark.compare_with_baseline(args.baseline)

        if regressions:
            print(f"\nWarning: {len(regressions)} performance regressions detected:")
            for reg in regressions:
                print(
                    f"  {reg['distribution']}.{reg['operation']}: "
                    f"{reg['change']:.1%} slower "
                    f"({reg['baseline_time']:.6f}s -> {reg['current_time']:.6f}s)"
                )
        else:
            print("\nNo significant performance regressions detected")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
