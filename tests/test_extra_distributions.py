"""Tests for extra heavy-tailed distributions."""

import math

import pytest

from heavytails.extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
    _betainc_reg,
    _gammainc_lower_reg,
    _log_beta,
    _ppf_monotone,
)
from heavytails.heavy_tails import ParameterError


class TestHelperFunctions:
    """Tests for helper special functions."""

    def test_log_beta_basic(self) -> None:
        """Test log beta function."""
        # B(1,1) = 1, so log B(1,1) = 0
        result = _log_beta(1.0, 1.0)
        assert math.isclose(result, 0.0, abs_tol=1e-10)

        # B(2,2) = 1/6, so log B(2,2) = log(1/6)
        result = _log_beta(2.0, 2.0)
        expected = math.log(1.0 / 6.0)
        assert math.isclose(result, expected, rel_tol=1e-10)

    def test_betainc_reg_boundaries(self) -> None:
        """Test regularized incomplete beta at boundaries."""
        # I_0(a,b) = 0
        assert _betainc_reg(1.0, 1.0, 0.0) == 0.0

        # I_1(a,b) = 1
        assert _betainc_reg(1.0, 1.0, 1.0) == 1.0

    def test_betainc_reg_invalid_x(self) -> None:
        """Test regularized incomplete beta with invalid x."""
        with pytest.raises(ValueError, match="x must be in"):
            _betainc_reg(1.0, 1.0, -0.1)

        with pytest.raises(ValueError, match="x must be in"):
            _betainc_reg(1.0, 1.0, 1.5)

    def test_betainc_reg_invalid_params(self) -> None:
        """Test regularized incomplete beta with invalid parameters."""
        with pytest.raises(ValueError, match="a,b must be > 0"):
            _betainc_reg(0.0, 1.0, 0.5)

        with pytest.raises(ValueError, match="a,b must be > 0"):
            _betainc_reg(1.0, -1.0, 0.5)

    def test_betainc_reg_symmetry(self) -> None:
        """Test symmetry property of regularized incomplete beta."""
        # I_x(a,b) = 1 - I_{1-x}(b,a)
        a, b, x = 2.0, 3.0, 0.4
        i1 = _betainc_reg(a, b, x)
        i2 = _betainc_reg(b, a, 1.0 - x)
        assert math.isclose(i1, 1.0 - i2, rel_tol=1e-10)

    def test_gammainc_lower_reg_boundaries(self) -> None:
        """Test regularized lower incomplete gamma at boundaries."""
        # P(a,0) = 0
        assert _gammainc_lower_reg(1.0, 0.0) == 0.0

    def test_gammainc_lower_reg_invalid_params(self) -> None:
        """Test regularized lower incomplete gamma with invalid parameters."""
        with pytest.raises(ValueError, match="a must be >0"):
            _gammainc_lower_reg(0.0, 1.0)

        with pytest.raises(ValueError, match="x>=0"):
            _gammainc_lower_reg(1.0, -1.0)

    def test_gammainc_lower_reg_series(self) -> None:
        """Test series branch of regularized lower incomplete gamma."""
        # For x < a + 1, uses series
        result = _gammainc_lower_reg(5.0, 2.0)
        assert 0 < result < 1

    def test_gammainc_lower_reg_cf(self) -> None:
        """Test continued fraction branch of regularized lower incomplete gamma."""
        # For x >= a + 1, uses continued fraction
        result = _gammainc_lower_reg(2.0, 5.0)
        assert 0 < result < 1

    def test_ppf_monotone_basic(self) -> None:
        """Test generic PPF monotone inversion."""

        def simple_cdf(x: float) -> float:
            # CDF of uniform(0,1)
            return max(0.0, min(1.0, x))

        result = _ppf_monotone(simple_cdf, 0.0, 1.0, 0.5)
        assert math.isclose(result, 0.5, rel_tol=1e-6)

    def test_ppf_monotone_with_pdf(self) -> None:
        """Test generic PPF with PDF provided (Newton method)."""

        def simple_cdf(x: float) -> float:
            return max(0.0, min(1.0, x))

        def simple_pdf(x: float) -> float:
            return 1.0 if 0 <= x <= 1 else 0.0

        result = _ppf_monotone(simple_cdf, 0.0, 1.0, 0.7, pdf=simple_pdf)
        assert math.isclose(result, 0.7, rel_tol=1e-6)

    def test_ppf_monotone_invalid_u(self) -> None:
        """Test PPF monotone with invalid u."""

        def simple_cdf(x: float) -> float:
            return max(0.0, min(1.0, x))

        with pytest.raises(ValueError, match="u must be in"):
            _ppf_monotone(simple_cdf, 0.0, 1.0, 0.0)

        with pytest.raises(ValueError, match="u must be in"):
            _ppf_monotone(simple_cdf, 0.0, 1.0, 1.0)


class TestGeneralizedPareto:
    """Tests for Generalized Pareto Distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        gpd = GeneralizedPareto(xi=0.5, sigma=2.0, mu=1.0)
        assert gpd.xi == 0.5
        assert gpd.sigma == 2.0
        assert gpd.mu == 1.0

    def test_initialization_invalid_sigma(self) -> None:
        """Test initialization with invalid sigma."""
        with pytest.raises(ParameterError, match="sigma>0"):
            GeneralizedPareto(xi=0.5, sigma=0.0)

        with pytest.raises(ParameterError, match="sigma>0"):
            GeneralizedPareto(xi=0.5, sigma=-1.0)

    def test_pdf_basic(self) -> None:
        """Test basic PDF evaluation."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        assert gpd.pdf(1.0) > 0
        assert gpd.pdf(2.0) > 0

    def test_pdf_outside_support(self) -> None:
        """Test PDF outside support for negative xi."""
        # For negative xi, there's a bounded support
        gpd = GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0)
        # Upper bound is mu - sigma/xi = 0 - 1/(-0.5) = 2.0
        # So values > 2.0 should have pdf = 0
        assert gpd.pdf(3.0) == 0.0

    def test_pdf_xi_zero(self) -> None:
        """Test PDF when xi=0 (exponential case)."""
        gpd = GeneralizedPareto(xi=0.0, sigma=1.0, mu=0.0)
        # Should reduce to exponential
        assert gpd.pdf(1.0) > 0
        assert gpd.pdf(2.0) < gpd.pdf(1.0)  # Should be decreasing

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        assert 0 < gpd.cdf(1.0) < 1
        assert gpd.cdf(2.0) > gpd.cdf(1.0)

    def test_cdf_xi_zero(self) -> None:
        """Test CDF when xi=0 (exponential case)."""
        gpd = GeneralizedPareto(xi=0.0, sigma=1.0, mu=0.0)
        x = 1.0
        expected = 1.0 - math.exp(-x)
        assert math.isclose(gpd.cdf(x), expected, rel_tol=1e-10)

    def test_sf_basic(self) -> None:
        """Test survival function."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        x = 2.0
        assert math.isclose(gpd.sf(x), 1.0 - gpd.cdf(x), rel_tol=1e-10)

    def test_ppf_basic(self) -> None:
        """Test basic PPF evaluation."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        q = gpd.ppf(0.5)
        assert q > 0

    def test_ppf_cdf_inverse(self) -> None:
        """Test that PPF and CDF are inverses."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        u = 0.7
        x = gpd.ppf(u)
        assert math.isclose(gpd.cdf(x), u, rel_tol=1e-6)

    def test_ppf_xi_zero(self) -> None:
        """Test PPF when xi=0."""
        gpd = GeneralizedPareto(xi=0.0, sigma=1.0, mu=0.0)
        u = 0.5
        expected = -1.0 * math.log(1.0 - u)
        assert math.isclose(gpd.ppf(u), expected, rel_tol=1e-10)

    def test_ppf_invalid_u(self) -> None:
        """Test PPF with invalid u."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            gpd.ppf(0.0)

        with pytest.raises(ValueError, match="u must be in"):
            gpd.ppf(1.0)

    def test_sampling(self) -> None:
        """Test random sampling."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        samples = gpd.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x >= gpd.mu for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test sampling reproducibility."""
        gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
        samples1 = gpd.rvs(50, seed=42)
        samples2 = gpd.rvs(50, seed=42)
        assert samples1 == samples2


class TestBurrXII:
    """Tests for Burr Type XII distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        burr = BurrXII(c=1.5, k=2.0, s=3.0)
        assert burr.c == 1.5
        assert burr.k == 2.0
        assert burr.s == 3.0

    def test_initialization_invalid_params(self) -> None:
        """Test initialization with invalid parameters."""
        with pytest.raises(ParameterError, match="c>0"):
            BurrXII(c=0.0, k=1.0)

        with pytest.raises(ParameterError, match="k>0"):
            BurrXII(c=1.0, k=0.0)

        with pytest.raises(ParameterError, match="s>0"):
            BurrXII(c=1.0, k=1.0, s=0.0)

    def test_pdf_basic(self) -> None:
        """Test basic PDF evaluation."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        assert burr.pdf(1.0) > 0
        assert burr.pdf(2.0) > 0
        assert burr.pdf(-1.0) == 0.0
        assert burr.pdf(0.0) == 0.0

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        assert burr.cdf(0.0) == 0.0
        assert burr.cdf(-1.0) == 0.0
        assert 0 < burr.cdf(1.0) < 1
        assert burr.cdf(2.0) > burr.cdf(1.0)

    def test_sf_basic(self) -> None:
        """Test survival function."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        assert burr.sf(0.0) == 1.0
        assert burr.sf(-1.0) == 1.0
        x = 2.0
        assert math.isclose(burr.sf(x), 1.0 - burr.cdf(x), rel_tol=1e-10)

    def test_ppf_basic(self) -> None:
        """Test basic PPF evaluation."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        q = burr.ppf(0.5)
        assert q > 0

    def test_ppf_cdf_inverse(self) -> None:
        """Test that PPF and CDF are inverses."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        u = 0.7
        x = burr.ppf(u)
        assert math.isclose(burr.cdf(x), u, rel_tol=1e-6)

    def test_ppf_invalid_u(self) -> None:
        """Test PPF with invalid u."""
        burr = BurrXII(c=1.5, k=2.0)
        with pytest.raises(ValueError, match="u must be in"):
            burr.ppf(0.0)

        with pytest.raises(ValueError, match="u must be in"):
            burr.ppf(1.0)

    def test_sampling(self) -> None:
        """Test random sampling."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        samples = burr.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x > 0 for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test sampling reproducibility."""
        burr = BurrXII(c=1.5, k=2.0, s=1.0)
        samples1 = burr.rvs(50, seed=42)
        samples2 = burr.rvs(50, seed=42)
        assert samples1 == samples2


class TestLogLogistic:
    """Tests for Log-Logistic distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        ll = LogLogistic(kappa=1.5, lam=2.0)
        assert ll.kappa == 1.5
        assert ll.lam == 2.0

    def test_initialization_invalid_params(self) -> None:
        """Test initialization with invalid parameters."""
        with pytest.raises(ParameterError, match="kappa>0"):
            LogLogistic(kappa=0.0, lam=1.0)

        with pytest.raises(ParameterError, match="lam>0"):
            LogLogistic(kappa=1.0, lam=0.0)

    def test_pdf_basic(self) -> None:
        """Test basic PDF evaluation."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        assert ll.pdf(1.0) > 0
        assert ll.pdf(2.0) > 0
        assert ll.pdf(-1.0) == 0.0
        assert ll.pdf(0.0) == 0.0

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        assert ll.cdf(0.0) == 0.0
        assert ll.cdf(-1.0) == 0.0
        assert 0 < ll.cdf(1.0) < 1
        assert ll.cdf(2.0) > ll.cdf(1.0)

    def test_sf_basic(self) -> None:
        """Test survival function."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        assert ll.sf(0.0) == 1.0
        assert ll.sf(-1.0) == 1.0
        x = 2.0
        assert math.isclose(ll.sf(x), 1.0 - ll.cdf(x), rel_tol=1e-10)

    def test_ppf_basic(self) -> None:
        """Test basic PPF evaluation."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        q = ll.ppf(0.5)
        assert q > 0

    def test_ppf_cdf_inverse(self) -> None:
        """Test that PPF and CDF are inverses."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        u = 0.7
        x = ll.ppf(u)
        assert math.isclose(ll.cdf(x), u, rel_tol=1e-6)

    def test_ppf_invalid_u(self) -> None:
        """Test PPF with invalid u."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            ll.ppf(0.0)

        with pytest.raises(ValueError, match="u must be in"):
            ll.ppf(1.0)

    def test_sampling(self) -> None:
        """Test random sampling."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        samples = ll.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x > 0 for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test sampling reproducibility."""
        ll = LogLogistic(kappa=1.5, lam=1.0)
        samples1 = ll.rvs(50, seed=42)
        samples2 = ll.rvs(50, seed=42)
        assert samples1 == samples2


class TestInverseGamma:
    """Tests for Inverse-Gamma distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        assert ig.alpha == 2.0
        assert ig.beta == 3.0

    def test_initialization_invalid_params(self) -> None:
        """Test initialization with invalid parameters."""
        with pytest.raises(ParameterError, match="alpha>0"):
            InverseGamma(alpha=0.0, beta=1.0)

        with pytest.raises(ParameterError, match="beta>0"):
            InverseGamma(alpha=1.0, beta=0.0)

    def test_pdf_basic(self) -> None:
        """Test basic PDF evaluation."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        assert ig.pdf(1.0) > 0
        assert ig.pdf(2.0) > 0
        assert ig.pdf(-1.0) == 0.0
        assert ig.pdf(0.0) == 0.0

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        assert ig.cdf(0.0) == 0.0
        assert ig.cdf(-1.0) == 0.0
        assert 0 < ig.cdf(1.0) < 1
        assert ig.cdf(2.0) > ig.cdf(1.0)

    def test_sf_basic(self) -> None:
        """Test survival function."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        x = 2.0
        assert math.isclose(ig.sf(x), 1.0 - ig.cdf(x), rel_tol=1e-10)

    def test_ppf_basic(self) -> None:
        """Test basic PPF evaluation."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        q = ig.ppf(0.5)
        assert q > 0

    def test_ppf_cdf_inverse(self) -> None:
        """Test that PPF and CDF are inverses."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        u = 0.7
        x = ig.ppf(u)
        assert math.isclose(ig.cdf(x), u, rel_tol=1e-5)

    def test_ppf_invalid_u(self) -> None:
        """Test PPF with invalid u."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        with pytest.raises(ValueError, match="u must be in"):
            ig.ppf(0.0)

        with pytest.raises(ValueError, match="u must be in"):
            ig.ppf(1.0)

    def test_sampling(self) -> None:
        """Test random sampling."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        samples = ig.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x > 0 for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test sampling reproducibility."""
        ig = InverseGamma(alpha=2.0, beta=3.0)
        samples1 = ig.rvs(50, seed=42)
        samples2 = ig.rvs(50, seed=42)
        assert samples1 == samples2


class TestBetaPrime:
    """Tests for Beta-Prime distribution."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.5)
        assert bp.a == 2.0
        assert bp.b == 3.0
        assert bp.s == 1.5

    def test_initialization_invalid_params(self) -> None:
        """Test initialization with invalid parameters."""
        with pytest.raises(ParameterError, match="a>0"):
            BetaPrime(a=0.0, b=1.0)

        with pytest.raises(ParameterError, match="b>0"):
            BetaPrime(a=1.0, b=0.0)

        with pytest.raises(ParameterError, match="s>0"):
            BetaPrime(a=1.0, b=1.0, s=0.0)

    def test_pdf_basic(self) -> None:
        """Test basic PDF evaluation."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        assert bp.pdf(1.0) > 0
        assert bp.pdf(2.0) > 0
        assert bp.pdf(-1.0) == 0.0
        assert bp.pdf(0.0) == 0.0

    def test_cdf_basic(self) -> None:
        """Test basic CDF evaluation."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        assert bp.cdf(0.0) == 0.0
        assert bp.cdf(-1.0) == 0.0
        assert 0 < bp.cdf(1.0) < 1
        assert bp.cdf(2.0) > bp.cdf(1.0)

    def test_sf_basic(self) -> None:
        """Test survival function."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        x = 2.0
        assert math.isclose(bp.sf(x), 1.0 - bp.cdf(x), rel_tol=1e-10)

    def test_ppf_basic(self) -> None:
        """Test basic PPF evaluation."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        q = bp.ppf(0.5)
        assert q > 0

    def test_ppf_cdf_inverse(self) -> None:
        """Test that PPF and CDF are inverses."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        u = 0.7
        x = bp.ppf(u)
        assert math.isclose(bp.cdf(x), u, rel_tol=1e-5)

    def test_ppf_invalid_u(self) -> None:
        """Test PPF with invalid u."""
        bp = BetaPrime(a=2.0, b=3.0)
        with pytest.raises(ValueError, match="u must be in"):
            bp.ppf(0.0)

        with pytest.raises(ValueError, match="u must be in"):
            bp.ppf(1.0)

    def test_sampling(self) -> None:
        """Test random sampling."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        samples = bp.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x > 0 for x in samples)

    def test_sampling_reproducibility(self) -> None:
        """Test sampling reproducibility."""
        bp = BetaPrime(a=2.0, b=3.0, s=1.0)
        samples1 = bp.rvs(50, seed=42)
        samples2 = bp.rvs(50, seed=42)
        assert samples1 == samples2
