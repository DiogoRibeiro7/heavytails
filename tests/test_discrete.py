"""Tests for discrete heavy-tailed distributions."""

import math

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

        with pytest.raises(ValueError, match="u in"):
            zipf.ppf(0.0)

        with pytest.raises(ValueError, match="u in"):
            zipf.ppf(1.0)

        with pytest.raises(ValueError, match="u in"):
            zipf.ppf(-0.5)

        with pytest.raises(ValueError, match="u in"):
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
