"""Comprehensive tests for heavy_tails.py module."""


import pytest

from heavytails.heavy_tails import (
    RNG,
    Cauchy,
    Frechet,
    LogNormal,
    ParameterError,
    Pareto,
    StudentT,
    Weibull,
)


class TestRNG:
    """Test random number generator utilities."""

    def test_rng_initialization_with_seed(self):
        """Test RNG initialization with seed."""
        rng = RNG(seed=42)
        assert rng.rng is not None

    def test_rng_initialization_without_seed(self):
        """Test RNG initialization without seed."""
        rng = RNG()
        assert rng.rng is not None

    def test_uniform_0_1(self):
        """Test uniform random number generation."""
        rng = RNG(seed=42)
        u = rng.uniform_0_1()
        assert 0.0 < u < 1.0

    def test_uniform_0_1_bounds(self):
        """Test that uniform_0_1 avoids exact 0 and 1."""
        rng = RNG(seed=42)
        # Generate many samples and check they're all in (0, 1)
        samples = [rng.uniform_0_1() for _ in range(1000)]
        assert all(0.0 < u < 1.0 for u in samples)
        assert all(u > 1e-16 for u in samples)
        assert all(u < 1.0 - 1e-16 for u in samples)

    def test_standard_normal(self):
        """Test standard normal generation."""
        rng = RNG(seed=42)
        z = rng.standard_normal()
        assert isinstance(z, float)

    def test_standard_normal_statistics(self):
        """Test that standard normal has reasonable statistics."""
        rng = RNG(seed=42)
        samples = [rng.standard_normal() for _ in range(1000)]
        mean = sum(samples) / len(samples)
        variance = sum((x - mean) ** 2 for x in samples) / len(samples)
        # Mean should be close to 0, variance close to 1
        assert abs(mean) < 0.2
        assert abs(variance - 1.0) < 0.2

    def test_gamma_valid_parameters(self):
        """Test gamma distribution with valid parameters."""
        rng = RNG(seed=42)
        x = rng.gamma(shape_k=2.0, scale_theta=1.0)
        assert x > 0

    def test_gamma_invalid_shape(self):
        """Test gamma with invalid shape parameter."""
        rng = RNG(seed=42)
        with pytest.raises(ParameterError, match="Gamma requires"):
            rng.gamma(shape_k=0.0, scale_theta=1.0)

    def test_gamma_invalid_scale(self):
        """Test gamma with invalid scale parameter."""
        rng = RNG(seed=42)
        with pytest.raises(ParameterError, match="Gamma requires"):
            rng.gamma(shape_k=2.0, scale_theta=0.0)

    def test_gamma_shape_less_than_one(self):
        """Test gamma with shape < 1 (uses boost method)."""
        rng = RNG(seed=42)
        x = rng.gamma(shape_k=0.5, scale_theta=1.0)
        assert x > 0

    def test_gamma_shape_greater_than_one(self):
        """Test gamma with shape >= 1 (direct method)."""
        rng = RNG(seed=42)
        x = rng.gamma(shape_k=3.0, scale_theta=2.0)
        assert x > 0

    def test_gamma_mt_loop_coverage(self):
        """Test that gamma_mt rejection loop is covered."""
        rng = RNG(seed=42)
        # Generate multiple samples to hit rejection cases
        samples = [rng.gamma(shape_k=5.0, scale_theta=1.0) for _ in range(100)]
        assert all(x > 0 for x in samples)

    def test_chisquare_valid(self):
        """Test chi-square distribution."""
        rng = RNG(seed=42)
        x = rng.chisquare(df=5.0)
        assert x > 0

    def test_chisquare_invalid_df(self):
        """Test chi-square with invalid degrees of freedom."""
        rng = RNG(seed=42)
        with pytest.raises(ParameterError, match="Chi-square requires"):
            rng.chisquare(df=0.0)

    def test_chisquare_negative_df(self):
        """Test chi-square with negative degrees of freedom."""
        rng = RNG(seed=42)
        with pytest.raises(ParameterError, match="Chi-square requires"):
            rng.chisquare(df=-1.0)


class TestPareto:
    """Test Pareto distribution."""

    def test_pareto_initialization_valid(self):
        """Test Pareto with valid parameters."""
        dist = Pareto(alpha=2.5, xm=1.0)
        assert dist.alpha == 2.5
        assert dist.xm == 1.0

    def test_pareto_initialization_invalid_alpha(self):
        """Test Pareto with invalid alpha."""
        with pytest.raises(ParameterError, match="Pareto requires"):
            Pareto(alpha=0.0, xm=1.0)

    def test_pareto_initialization_negative_alpha(self):
        """Test Pareto with negative alpha."""
        with pytest.raises(ParameterError, match="Pareto requires"):
            Pareto(alpha=-1.0, xm=1.0)

    def test_pareto_initialization_invalid_xm(self):
        """Test Pareto with invalid xm."""
        with pytest.raises(ParameterError, match="Pareto requires"):
            Pareto(alpha=2.5, xm=0.0)

    def test_pareto_pdf_below_xm(self):
        """Test PDF returns 0 for x < xm."""
        dist = Pareto(alpha=2.5, xm=1.0)
        assert dist.pdf(0.5) == 0.0

    def test_pareto_pdf_at_xm(self):
        """Test PDF at x = xm."""
        dist = Pareto(alpha=2.5, xm=1.0)
        pdf_val = dist.pdf(1.0)
        assert pdf_val > 0

    def test_pareto_cdf_below_xm(self):
        """Test CDF returns 0 for x < xm."""
        dist = Pareto(alpha=2.5, xm=1.0)
        assert dist.cdf(0.5) == 0.0

    def test_pareto_sf_below_xm(self):
        """Test survival function returns 1 for x < xm."""
        dist = Pareto(alpha=2.5, xm=1.0)
        assert dist.sf(0.5) == 1.0

    def test_pareto_sf_above_xm(self):
        """Test survival function above xm."""
        dist = Pareto(alpha=2.5, xm=1.0)
        sf_val = dist.sf(2.0)
        assert 0 < sf_val < 1

    def test_pareto_ppf_invalid_u_zero(self):
        """Test PPF with u = 0."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(0.0)

    def test_pareto_ppf_invalid_u_one(self):
        """Test PPF with u = 1."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(1.0)

    def test_pareto_ppf_invalid_u_negative(self):
        """Test PPF with negative u."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(-0.1)

    def test_pareto_ppf_invalid_u_greater_than_one(self):
        """Test PPF with u > 1."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(1.5)

    def test_pareto_rvs_invalid_n_zero(self):
        """Test rvs with n = 0."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            dist.rvs(0)

    def test_pareto_rvs_invalid_n_negative(self):
        """Test rvs with negative n."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            dist.rvs(-5)

    def test_pareto_rvs_invalid_n_float(self):
        """Test rvs with float n."""
        dist = Pareto(alpha=2.5, xm=1.0)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            dist.rvs(5.5)  # type: ignore


class TestCauchy:
    """Test Cauchy distribution."""

    def test_cauchy_initialization_valid(self):
        """Test Cauchy with valid parameters."""
        dist = Cauchy(x0=0.0, gamma=1.0)
        assert dist.x0 == 0.0
        assert dist.gamma == 1.0

    def test_cauchy_initialization_invalid_gamma(self):
        """Test Cauchy with invalid gamma."""
        with pytest.raises(ParameterError, match="Cauchy requires"):
            Cauchy(x0=0.0, gamma=0.0)

    def test_cauchy_initialization_negative_gamma(self):
        """Test Cauchy with negative gamma."""
        with pytest.raises(ParameterError, match="Cauchy requires"):
            Cauchy(x0=0.0, gamma=-1.0)

    def test_cauchy_ppf_invalid_u_zero(self):
        """Test Cauchy PPF with u = 0."""
        dist = Cauchy(x0=0.0, gamma=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(0.0)

    def test_cauchy_ppf_invalid_u_one(self):
        """Test Cauchy PPF with u = 1."""
        dist = Cauchy(x0=0.0, gamma=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(1.0)


class TestStudentT:
    """Test Student-t distribution."""

    def test_studentt_initialization_valid(self):
        """Test Student-t with valid parameters."""
        dist = StudentT(nu=5.0)
        assert dist.nu == 5.0

    def test_studentt_initialization_invalid_nu(self):
        """Test Student-t with invalid nu."""
        with pytest.raises(ParameterError, match="Student-t requires"):
            StudentT(nu=0.0)

    def test_studentt_initialization_negative_nu(self):
        """Test Student-t with negative nu."""
        with pytest.raises(ParameterError, match="Student-t requires"):
            StudentT(nu=-1.0)

    def test_studentt_pdf(self):
        """Test Student-t PDF calculation."""
        dist = StudentT(nu=5.0)
        pdf_val = dist.pdf(0.0)
        assert pdf_val > 0

    def test_studentt_pdf_tails(self):
        """Test Student-t PDF in the tails."""
        dist = StudentT(nu=5.0)
        pdf_val = dist.pdf(5.0)
        assert pdf_val > 0


class TestLogNormal:
    """Test LogNormal distribution."""

    def test_lognormal_initialization_valid(self):
        """Test LogNormal with valid parameters."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        assert dist.mu == 0.0
        assert dist.sigma == 1.0

    def test_lognormal_initialization_invalid_sigma(self):
        """Test LogNormal with invalid sigma."""
        with pytest.raises(ParameterError, match="LogNormal requires"):
            LogNormal(mu=0.0, sigma=0.0)

    def test_lognormal_initialization_negative_sigma(self):
        """Test LogNormal with negative sigma."""
        with pytest.raises(ParameterError, match="LogNormal requires"):
            LogNormal(mu=0.0, sigma=-1.0)

    def test_lognormal_pdf_negative_x(self):
        """Test LogNormal PDF with negative x."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        assert dist.pdf(-1.0) == 0.0

    def test_lognormal_pdf_zero_x(self):
        """Test LogNormal PDF with x = 0."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        assert dist.pdf(0.0) == 0.0

    def test_lognormal_cdf_negative_x(self):
        """Test LogNormal CDF with negative x."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        assert dist.cdf(-1.0) == 0.0

    def test_lognormal_cdf_zero_x(self):
        """Test LogNormal CDF with x = 0."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        assert dist.cdf(0.0) == 0.0

    def test_lognormal_sf(self):
        """Test LogNormal survival function."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        sf_val = dist.sf(1.0)
        cdf_val = dist.cdf(1.0)
        assert abs(sf_val + cdf_val - 1.0) < 1e-10

    def test_lognormal_ppf_invalid_u_zero(self):
        """Test LogNormal PPF with u = 0."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(0.0)

    def test_lognormal_ppf_invalid_u_one(self):
        """Test LogNormal PPF with u = 1."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(1.0)

    def test_lognormal_rvs(self):
        """Test LogNormal random sampling."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        samples = dist.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x > 0 for x in samples)


class TestWeibull:
    """Test Weibull distribution."""

    def test_weibull_initialization_valid(self):
        """Test Weibull with valid parameters."""
        dist = Weibull(k=2.0, lam=1.0)
        assert dist.k == 2.0
        assert dist.lam == 1.0

    def test_weibull_initialization_invalid_k(self):
        """Test Weibull with invalid k."""
        with pytest.raises(ParameterError, match="Weibull requires"):
            Weibull(k=0.0, lam=1.0)

    def test_weibull_initialization_negative_k(self):
        """Test Weibull with negative k."""
        with pytest.raises(ParameterError, match="Weibull requires"):
            Weibull(k=-1.0, lam=1.0)

    def test_weibull_initialization_invalid_lam(self):
        """Test Weibull with invalid lam."""
        with pytest.raises(ParameterError, match="Weibull requires"):
            Weibull(k=2.0, lam=0.0)

    def test_weibull_pdf_negative_x(self):
        """Test Weibull PDF with negative x."""
        dist = Weibull(k=2.0, lam=1.0)
        assert dist.pdf(-1.0) == 0.0

    def test_weibull_pdf_positive_x(self):
        """Test Weibull PDF with positive x."""
        dist = Weibull(k=2.0, lam=1.0)
        pdf_val = dist.pdf(1.0)
        assert pdf_val > 0

    def test_weibull_cdf_negative_x(self):
        """Test Weibull CDF with negative x."""
        dist = Weibull(k=2.0, lam=1.0)
        assert dist.cdf(-1.0) == 0.0

    def test_weibull_cdf_positive_x(self):
        """Test Weibull CDF with positive x."""
        dist = Weibull(k=2.0, lam=1.0)
        cdf_val = dist.cdf(1.0)
        assert 0 < cdf_val < 1

    def test_weibull_sf_negative_x(self):
        """Test Weibull survival function with negative x."""
        dist = Weibull(k=2.0, lam=1.0)
        assert dist.sf(-1.0) == 1.0

    def test_weibull_sf_positive_x(self):
        """Test Weibull survival function with positive x."""
        dist = Weibull(k=2.0, lam=1.0)
        sf_val = dist.sf(1.0)
        assert 0 < sf_val < 1

    def test_weibull_ppf_valid(self):
        """Test Weibull PPF with valid u."""
        dist = Weibull(k=2.0, lam=1.0)
        x = dist.ppf(0.5)
        assert x > 0

    def test_weibull_ppf_invalid_u_zero(self):
        """Test Weibull PPF with u = 0."""
        dist = Weibull(k=2.0, lam=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(0.0)

    def test_weibull_ppf_invalid_u_one(self):
        """Test Weibull PPF with u = 1."""
        dist = Weibull(k=2.0, lam=1.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(1.0)

    def test_weibull_rvs(self):
        """Test Weibull random sampling."""
        dist = Weibull(k=2.0, lam=1.0)
        samples = dist.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x >= 0 for x in samples)


class TestFrechet:
    """Test Frechet distribution."""

    def test_frechet_initialization_valid(self):
        """Test Frechet with valid parameters."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        assert dist.alpha == 2.0
        assert dist.s == 1.0
        assert dist.m == 0.0

    def test_frechet_initialization_invalid_alpha(self):
        """Test Frechet with invalid alpha."""
        with pytest.raises(ParameterError, match="Frechet requires"):
            Frechet(alpha=0.0, s=1.0, m=0.0)

    def test_frechet_initialization_negative_alpha(self):
        """Test Frechet with negative alpha."""
        with pytest.raises(ParameterError, match="Frechet requires"):
            Frechet(alpha=-1.0, s=1.0, m=0.0)

    def test_frechet_initialization_invalid_s(self):
        """Test Frechet with invalid s."""
        with pytest.raises(ParameterError, match="Frechet requires"):
            Frechet(alpha=2.0, s=0.0, m=0.0)

    def test_frechet_pdf_below_m(self):
        """Test Frechet PDF with x <= m."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        assert dist.pdf(0.0) == 0.0
        assert dist.pdf(-1.0) == 0.0

    def test_frechet_pdf_above_m(self):
        """Test Frechet PDF with x > m."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        pdf_val = dist.pdf(1.0)
        assert pdf_val > 0

    def test_frechet_cdf_below_m(self):
        """Test Frechet CDF with x <= m."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        assert dist.cdf(0.0) == 0.0
        assert dist.cdf(-1.0) == 0.0

    def test_frechet_cdf_above_m(self):
        """Test Frechet CDF with x > m."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        cdf_val = dist.cdf(1.0)
        assert 0 < cdf_val < 1

    def test_frechet_ppf_valid(self):
        """Test Frechet PPF with valid u."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        x = dist.ppf(0.5)
        assert x > 0

    def test_frechet_ppf_invalid_u_zero(self):
        """Test Frechet PPF with u = 0."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(0.0)

    def test_frechet_ppf_invalid_u_one(self):
        """Test Frechet PPF with u = 1."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        with pytest.raises(ValueError, match="u must be in"):
            dist.ppf(1.0)

    def test_frechet_rvs(self):
        """Test Frechet random sampling."""
        dist = Frechet(alpha=2.0, s=1.0, m=0.0)
        samples = dist.rvs(100, seed=42)
        assert len(samples) == 100
        assert all(x > 0 for x in samples)


class TestDistributionFunctionality:
    """Test actual functionality of distributions (not just error cases)."""

    def test_pareto_pdf_cdf_consistency(self):
        """Test that Pareto PDF and CDF are consistent."""
        dist = Pareto(alpha=2.5, xm=1.0)
        x = 2.0
        assert dist.pdf(x) > 0
        assert 0 < dist.cdf(x) < 1

    def test_pareto_rvs_within_support(self):
        """Test that Pareto samples are within support."""
        dist = Pareto(alpha=2.5, xm=1.0)
        samples = dist.rvs(100, seed=42)
        assert all(x >= dist.xm for x in samples)

    def test_cauchy_pdf_symmetric(self):
        """Test Cauchy PDF symmetry."""
        dist = Cauchy(x0=0.0, gamma=1.0)
        assert abs(dist.pdf(1.0) - dist.pdf(-1.0)) < 1e-10

    def test_cauchy_cdf_at_median(self):
        """Test Cauchy CDF at median."""
        dist = Cauchy(x0=0.0, gamma=1.0)
        assert abs(dist.cdf(0.0) - 0.5) < 1e-10

    def test_cauchy_rvs_sampling(self):
        """Test Cauchy random sampling."""
        dist = Cauchy(x0=0.0, gamma=1.0)
        samples = dist.rvs(100, seed=42)
        assert len(samples) == 100

    def test_studentt_rvs_sampling(self):
        """Test Student-t random sampling."""
        dist = StudentT(nu=5.0)
        samples = dist.rvs(100, seed=42)
        assert len(samples) == 100

    def test_lognormal_ppf_cdf_inverse(self):
        """Test that LogNormal PPF is inverse of CDF."""
        dist = LogNormal(mu=0.0, sigma=1.0)
        u = 0.5
        x = dist.ppf(u)
        u_back = dist.cdf(x)
        assert abs(u - u_back) < 1e-6

    def test_pareto_ppf_cdf_inverse(self):
        """Test that Pareto PPF is inverse of CDF."""
        dist = Pareto(alpha=2.5, xm=1.0)
        u = 0.5
        x = dist.ppf(u)
        u_back = dist.cdf(x)
        assert abs(u - u_back) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
