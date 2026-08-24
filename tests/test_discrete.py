"""Tests for discrete heavy-tailed distributions."""

import math
import time
from typing import ClassVar

import numpy as np
import pytest

from heavytails.discrete import DiscretePareto, YuleSimon, Zipf
from heavytails.heavy_tails import ParameterError


class TestZipf:
    """Tests for Zipf distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        zipf = Zipf(s=2.0)
        assert zipf.s == 2.0
        assert zipf.kmax == 10_000
        assert zipf._Z > 0

    def test_initialization_custom_kmax(self) -> None:
        """Test initialization with custom kmax."""
        zipf = Zipf(s=2.5, kmax=1000)
        assert zipf.s == 2.5
        assert zipf.kmax == 1000

    def test_initialization_invalid_s(self) -> None:
        """Test initialization with invalid s parameter."""
        with pytest.raises(ParameterError, match="s>1"):
            Zipf(s=0.5)

        with pytest.raises(ParameterError, match="s>1"):
            Zipf(s=1.0)

    def test_pmf_basic(self) -> None:
        """Test basic PMF evaluation."""
        zipf = Zipf(s=2.0, kmax=100)

        # PMF should be positive for valid k
        assert zipf.pmf(1) > 0
        assert zipf.pmf(10) > 0

        # PMF should be 0 for k < 1 or k > kmax
        assert zipf.pmf(0) == 0
        assert zipf.pmf(-1) == 0
        assert zipf.pmf(101) == 0

    def test_pmf_monotonicity(self) -> None:
        """Test that PMF is monotonically decreasing."""
        zipf = Zipf(s=2.0, kmax=100)

        # PMF should decrease for increasing k
        pmf_values = [zipf.pmf(k) for k in range(1, 11)]
        assert all(
            pmf_values[i] > pmf_values[i + 1] for i in range(len(pmf_values) - 1)
        )

    def test_pmf_normalization(self) -> None:
        """Test that PMF sums to approximately 1."""
        zipf = Zipf(s=2.0, kmax=100)
        total = sum(zipf.pmf(k) for k in range(1, zipf.kmax + 1))
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        zipf = Zipf(s=2.0, kmax=100)

        # CDF should be monotonically increasing
        assert 0 < zipf.cdf(1) < 1
        assert zipf.cdf(10) > zipf.cdf(5)
        assert zipf.cdf(100) <= 1.0

    def test_cdf_boundary(self) -> None:
        """Test CDF boundary conditions."""
        zipf = Zipf(s=2.0, kmax=100)

        # CDF at kmax should be 1
        assert math.isclose(zipf.cdf(zipf.kmax), 1.0, rel_tol=1e-9)

        # CDF at values beyond kmax should also be 1
        assert math.isclose(zipf.cdf(zipf.kmax + 10), 1.0, rel_tol=1e-9)

    def test_ppf_basic(self) -> None:
        """Test basic PPF (quantile function) evaluation."""
        zipf = Zipf(s=2.0, kmax=100)

        # PPF should return valid values
        assert 1 <= zipf.ppf(0.1) <= zipf.kmax
        assert 1 <= zipf.ppf(0.5) <= zipf.kmax
        assert 1 <= zipf.ppf(0.9) <= zipf.kmax

    def test_ppf_monotonicity(self) -> None:
        """Test that PPF is monotonically increasing."""
        zipf = Zipf(s=2.0, kmax=100)

        q1 = zipf.ppf(0.1)
        q2 = zipf.ppf(0.5)
        q3 = zipf.ppf(0.9)

        assert q1 <= q2 <= q3

    def test_ppf_invalid_u(self) -> None:
        """Test PPF with invalid u values."""
        zipf = Zipf(s=2.0)

        with pytest.raises(ValueError, match="u must be in"):
            zipf.ppf(0.0)

        with pytest.raises(ValueError, match="u must be in"):
            zipf.ppf(1.0)

        with pytest.raises(ValueError, match="u must be in"):
            zipf.ppf(-0.5)

        with pytest.raises(ValueError, match="u must be in"):
            zipf.ppf(1.5)

    def test_sampling(self) -> None:
        """Test random sampling."""
        zipf = Zipf(s=2.0, kmax=100)

        samples = zipf.rvs(100, seed=42)

        assert len(samples) == 100
        assert all(1 <= x <= zipf.kmax for x in samples)
        assert all(isinstance(x, int) for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test that sampling with same seed produces same results."""
        zipf = Zipf(s=2.0, kmax=100)

        samples1 = zipf.rvs(50, seed=42)
        samples2 = zipf.rvs(50, seed=42)

        assert samples1 == samples2


class TestYuleSimon:
    """Tests for Yule-Simon distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        ys = YuleSimon(rho=1.5)
        assert ys.rho == 1.5

    def test_initialization_invalid_rho(self) -> None:
        """Test initialization with invalid rho parameter."""
        with pytest.raises(ParameterError, match="rho>0"):
            YuleSimon(rho=0.0)

        with pytest.raises(ParameterError, match="rho>0"):
            YuleSimon(rho=-1.0)

    def test_pmf_basic(self) -> None:
        """Test basic PMF evaluation."""
        ys = YuleSimon(rho=1.5)

        # PMF should be positive for k >= 1
        assert ys.pmf(1) > 0
        assert ys.pmf(5) > 0
        assert ys.pmf(10) > 0

        # PMF should be 0 for k < 1
        assert ys.pmf(0) == 0
        assert ys.pmf(-1) == 0

    def test_pmf_monotonicity(self) -> None:
        """Test that PMF is monotonically decreasing."""
        ys = YuleSimon(rho=2.0)

        # PMF should decrease for increasing k
        pmf_values = [ys.pmf(k) for k in range(1, 11)]
        assert all(
            pmf_values[i] > pmf_values[i + 1] for i in range(len(pmf_values) - 1)
        )

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        ys = YuleSimon(rho=1.5)

        # CDF should be monotonically increasing
        assert 0 < ys.cdf(1) < 1
        assert ys.cdf(10) > ys.cdf(5)
        assert ys.cdf(5) > ys.cdf(1)

    def test_cdf_monotonicity(self) -> None:
        """Test that CDF is monotonically increasing."""
        ys = YuleSimon(rho=2.0)

        cdf_values = [ys.cdf(k) for k in range(1, 21)]
        assert all(
            cdf_values[i] <= cdf_values[i + 1] for i in range(len(cdf_values) - 1)
        )

    def test_sampling(self) -> None:
        """Test random sampling."""
        ys = YuleSimon(rho=2.0)

        samples = ys.rvs(100, seed=42)

        assert len(samples) == 100
        assert all(x >= 1 for x in samples)
        assert all(isinstance(x, int) for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test that sampling with same seed produces same results."""
        ys = YuleSimon(rho=2.0)

        samples1 = ys.rvs(50, seed=42)
        samples2 = ys.rvs(50, seed=42)

        assert samples1 == samples2

    def test_pmf_formula(self) -> None:
        """Test PMF formula correctness."""
        ys = YuleSimon(rho=2.0)

        # For k=1: PMF = rho * B(1, rho+1) = rho / (rho+1)
        expected_pmf_1 = ys.rho / (ys.rho + 1)
        assert math.isclose(ys.pmf(1), expected_pmf_1, rel_tol=1e-9)


class TestDiscretePareto:
    """Tests for Discrete Pareto distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        dp = DiscretePareto(alpha=1.5)
        assert dp.alpha == 1.5
        assert dp.k_min == 1
        assert dp.k_max == 10_000
        assert dp._H > 0

    def test_initialization_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        dp = DiscretePareto(alpha=2.0, k_min=5, k_max=1000)
        assert dp.alpha == 2.0
        assert dp.k_min == 5
        assert dp.k_max == 1000

    def test_initialization_invalid_alpha(self) -> None:
        """Test initialization with invalid alpha parameter."""
        with pytest.raises(ParameterError, match="alpha>0"):
            DiscretePareto(alpha=0.0)

        with pytest.raises(ParameterError, match="alpha>0"):
            DiscretePareto(alpha=-1.0)

    def test_initialization_invalid_k_min(self) -> None:
        """Test initialization with invalid k_min parameter."""
        with pytest.raises(ParameterError, match="k_min>=1"):
            DiscretePareto(alpha=2.0, k_min=0)

        with pytest.raises(ParameterError, match="k_min>=1"):
            DiscretePareto(alpha=2.0, k_min=-1)

    def test_pmf_basic(self) -> None:
        """Test basic PMF evaluation."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        # PMF should be positive for k_min <= k <= k_max
        assert dp.pmf(1) > 0
        assert dp.pmf(10) > 0
        assert dp.pmf(100) > 0

        # PMF should be 0 for k < k_min or k > k_max
        assert dp.pmf(0) == 0
        assert dp.pmf(101) == 0

    def test_pmf_monotonicity(self) -> None:
        """Test that PMF is monotonically decreasing."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        # PMF should decrease for increasing k
        pmf_values = [dp.pmf(k) for k in range(dp.k_min, min(dp.k_min + 20, dp.k_max))]
        assert all(
            pmf_values[i] >= pmf_values[i + 1] for i in range(len(pmf_values) - 1)
        )

    def test_pmf_normalization(self) -> None:
        """Test that PMF sums to approximately 1."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)
        total = sum(dp.pmf(k) for k in range(dp.k_min, dp.k_max + 1))
        assert math.isclose(total, 1.0, rel_tol=1e-9)

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        # CDF should be monotonically increasing
        assert 0 < dp.cdf(1) <= 1
        assert dp.cdf(10) > dp.cdf(5)
        assert dp.cdf(100) <= 1.0

    def test_cdf_boundary(self) -> None:
        """Test CDF boundary conditions."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        # CDF at k_max should be 1
        assert math.isclose(dp.cdf(dp.k_max), 1.0, rel_tol=1e-9)

        # CDF at values beyond k_max should also be 1
        assert math.isclose(dp.cdf(dp.k_max + 10), 1.0, rel_tol=1e-9)

    def test_ppf_basic(self) -> None:
        """Test basic PPF evaluation."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        # PPF should return valid values
        assert dp.k_min <= dp.ppf(0.1) <= dp.k_max
        assert dp.k_min <= dp.ppf(0.5) <= dp.k_max
        assert dp.k_min <= dp.ppf(0.9) <= dp.k_max

    def test_ppf_monotonicity(self) -> None:
        """Test that PPF is monotonically increasing."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        q1 = dp.ppf(0.1)
        q2 = dp.ppf(0.5)
        q3 = dp.ppf(0.9)

        assert q1 <= q2 <= q3

    def test_sampling(self) -> None:
        """Test random sampling."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        samples = dp.rvs(100, seed=42)

        assert len(samples) == 100
        assert all(dp.k_min <= x <= dp.k_max for x in samples)
        assert all(isinstance(x, int) for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test that sampling with same seed produces same results."""
        dp = DiscretePareto(alpha=2.0, k_min=1, k_max=100)

        samples1 = dp.rvs(50, seed=42)
        samples2 = dp.rvs(50, seed=42)

        assert samples1 == samples2

    def test_custom_k_min(self) -> None:
        """Test behavior with custom k_min."""
        dp = DiscretePareto(alpha=2.0, k_min=10, k_max=100)

        # PMF should be 0 for k < k_min
        assert dp.pmf(1) == 0
        assert dp.pmf(5) == 0
        assert dp.pmf(9) == 0

        # PMF should be positive for k >= k_min
        assert dp.pmf(10) > 0
        assert dp.pmf(50) > 0

    def test_different_alpha_values(self) -> None:
        """Test with different alpha values."""
        dp1 = DiscretePareto(alpha=1.0, k_min=1, k_max=100)
        dp2 = DiscretePareto(alpha=3.0, k_min=1, k_max=100)

        # Higher alpha should give more concentrated distribution
        assert dp2.pmf(1) > dp1.pmf(1)
        assert dp2.pmf(50) < dp1.pmf(50)


class TestTheDiscreteFamiliesTakeOneValueOrMany:
    """Same contract as the continuous families since 0.5.0.

    ``Pareto.pdf([1, 2, 3])`` worked and ``Zipf.pmf([1, 2, 3])`` did not, which
    is an inconsistency a caller meets rather than reads about.
    """

    FAMILIES: ClassVar[list[object]] = [
        Zipf(s=1.5, kmax=2000),
        YuleSimon(rho=1.5),
        DiscretePareto(alpha=1.5, k_min=1, k_max=2000),
    ]

    @pytest.mark.parametrize("dist", FAMILIES, ids=lambda d: type(d).__name__)
    @pytest.mark.parametrize("method", ["pmf", "cdf"])
    def test_it_mirrors_its_input(self, dist: object, method: str) -> None:
        grid = [1, 2, 5, 50, 500]
        one_at_a_time = np.array([getattr(dist, method)(k) for k in grid])
        vectorised = np.asarray(getattr(dist, method)(grid))
        assert vectorised.shape == (len(grid),)
        assert isinstance(getattr(dist, method)(grid[0]), float)
        np.testing.assert_allclose(vectorised, one_at_a_time, rtol=1e-13, atol=1e-15)

    @pytest.mark.parametrize("dist", FAMILIES, ids=lambda d: type(d).__name__)
    def test_the_quantile_returns_integers(self, dist: object) -> None:
        probabilities = [0.01, 0.25, 0.5, 0.9]
        assert isinstance(dist.ppf(0.5), int)
        many = dist.ppf(probabilities)
        assert many.dtype.kind == "i", "a discrete quantile is an integer"
        assert list(many) == [dist.ppf(u) for u in probabilities]

    @pytest.mark.parametrize("dist", FAMILIES, ids=lambda d: type(d).__name__)
    def test_the_quantile_rejects_probabilities_outside_the_unit_interval(
        self, dist: object
    ) -> None:
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError, match=r"must be in \(0,1\)"):
                dist.ppf(bad)
        with pytest.raises(ValueError, match=r"must be in \(0,1\)"):
            dist.ppf([0.5, 1.5])


class TestTheDistributionFunctionIsZeroBelowTheSupport:
    """It was not: it returned the mass at the first support point.

    ``Zipf.cdf(k)`` clamped k up to 1 before summing, so ``cdf(0)`` -- and
    ``cdf(-5)`` -- returned P(X = 1) = 0.387 rather than 0. ``DiscretePareto``
    did the same at ``k_min``. Clamping the upper end is right, because the
    support is truncated there; clamping the lower end is not, because below
    the support the answer is zero and not the first atom.
    """

    def test_zipf_is_zero_below_one(self) -> None:
        zipf = Zipf(s=1.5, kmax=2000)
        assert zipf.pmf(1) > 0.3, "the atom this used to leak into cdf(0)"
        for k in (-5, -1, 0):
            assert zipf.cdf(k) == 0.0
        assert zipf.cdf(1) == pytest.approx(zipf.pmf(1))

    def test_discrete_pareto_is_zero_below_k_min(self) -> None:
        dist = DiscretePareto(alpha=1.5, k_min=3, k_max=500)
        assert dist.pmf(3) > 0.1
        for k in (-1, 0, 1, 2):
            assert dist.cdf(k) == 0.0
        assert dist.cdf(3) == pytest.approx(dist.pmf(3))

    def test_it_still_saturates_above_the_truncation(self) -> None:
        """The upper clamp is correct and stays."""
        zipf = Zipf(s=1.5, kmax=200)
        assert zipf.cdf(200) == pytest.approx(1.0)
        assert zipf.cdf(10_000) == pytest.approx(1.0)


class TestTheQuantileNoLongerScansTheSupport:
    def test_a_far_tail_quantile_is_cheap(self) -> None:
        """It walked the support one k at a time, so a tail quantile cost
        tens of thousands of steps -- on the distributions this library is
        for, and once per draw inside ``rvs``.
        """
        dist = Zipf(s=1.1, kmax=100_000)
        start = time.perf_counter()
        for _ in range(50):
            dist.ppf(0.999)
        elapsed = time.perf_counter() - start
        assert dist.ppf(0.999) > 50_000, "this must be a far-tail quantile"
        # The scan took about 16ms per call here; a search is microseconds.
        assert elapsed < 0.5, f"50 tail quantiles took {elapsed:.3f}s"
