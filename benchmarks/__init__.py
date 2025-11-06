"""
Benchmarks package for HeavyTails library.

This package contains performance benchmarks and profiling tools
for all distributions and operations in the HeavyTails library.

Modules:
- performance_tests: Main benchmark suite
- memory_profiling: Memory usage analysis (future)
- regression_tests: Performance regression detection (future)
"""

from .performance_tests import BenchmarkResult, DistributionBenchmark

__all__ = ["BenchmarkResult", "DistributionBenchmark"]
