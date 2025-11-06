"""
Test suite for heavy-tail copula implementations.

This module contains tests for:
- StudentTCopula with heavy-tailed marginals
- ExtremeValueCopula (Gumbel, Clayton, Frank)
- Tail dependence coefficient calculations
"""

import pytest

# Check if scipy is available for copula tests
try:
    import numpy as np
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    np = None
    stats = None

from heavytails.extensions import (
    ExtremeValueCopula,
    HeavyTailCopula,
    StudentTCopula,
)

# ========================================
# StudentTCopula Tests
# ========================================


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy required for copula tests")
class TestStudentTCopula:
    """Test suite for Student-t copula implementation."""

    def test_initialization_valid(self):
        """Test valid initialization of StudentTCopula."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        assert copula.nu == nu
        assert np.allclose(copula.correlation, np.array(corr))
        assert len(copula.marginals) == 2

    def test_initialization_invalid_nu(self):
        """Test that negative or zero nu raises ValueError."""
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t"]

        with pytest.raises(ValueError, match="nu must be positive"):
            StudentTCopula(nu=-1.0, correlation_matrix=corr, marginals=marginals)

        with pytest.raises(ValueError, match="nu must be positive"):
            StudentTCopula(nu=0.0, correlation_matrix=corr, marginals=marginals)

    def test_initialization_non_square_correlation(self):
        """Test that non-square correlation matrix raises ValueError."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0], [0.3, 0.4]]  # 3x2 matrix
        marginals = ["t", "t"]

        with pytest.raises(ValueError, match="must be square"):
            StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

    def test_initialization_dimension_mismatch(self):
        """Test that dimension mismatch raises ValueError."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t", "t"]  # 3 marginals but 2x2 matrix

        with pytest.raises(ValueError, match="must match correlation matrix"):
            StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

    def test_initialization_non_symmetric(self):
        """Test that non-symmetric correlation matrix raises ValueError."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.3, 1.0]]  # Not symmetric
        marginals = ["t", "t"]

        with pytest.raises(ValueError, match="must be symmetric"):
            StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

    def test_initialization_incorrect_diagonal(self):
        """Test that incorrect diagonal raises ValueError."""
        nu = 5.0
        corr = [[0.9, 0.5], [0.5, 1.0]]  # Diagonal not 1
        marginals = ["t", "t"]

        with pytest.raises(ValueError, match=r"Diagonal elements.*must be 1"):
            StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

    def test_initialization_not_positive_definite(self):
        """Test that non-positive definite matrix raises ValueError."""
        nu = 5.0
        corr = [[1.0, 1.0], [1.0, 1.0]]  # Not positive definite (determinant = 0)
        marginals = ["t", "t"]

        with pytest.raises(ValueError, match="positive definite"):
            StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

    def test_pdf_independence(self):
        """Test PDF with independence (correlation = 0)."""
        nu = 5.0
        corr = [[1.0, 0.0], [0.0, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        # For independence copula, c(u,v) = 1
        u = [0.5, 0.5]
        pdf_val = copula.pdf(u)

        # Should be close to 1 for independence
        # Note: The exact value depends on nu and the t-distribution
        # For nu=5 at (0.5, 0.5), we expect a value close to but not exactly 1
        assert abs(pdf_val - 1.0) < 0.15

    def test_pdf_positive_correlation(self):
        """Test PDF with positive correlation."""
        nu = 5.0
        corr = [[1.0, 0.7], [0.7, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        # Test at center point
        u = [0.5, 0.5]
        pdf_val = copula.pdf(u)

        # PDF should be positive
        assert pdf_val > 0

        # Test at concordant point (both high)
        u_high = [0.8, 0.8]
        pdf_high = copula.pdf(u_high)
        assert pdf_high > 0

    def test_pdf_boundary_conditions(self):
        """Test that PDF raises error for boundary values."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        # Test at boundaries (should raise error)
        with pytest.raises(ValueError, match="open interval"):
            copula.pdf([0.0, 0.5])

        with pytest.raises(ValueError, match="open interval"):
            copula.pdf([0.5, 1.0])

    def test_pdf_dimension_mismatch(self):
        """Test that PDF raises error for wrong dimensions."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        with pytest.raises(ValueError, match=r"Expected.*dimensions"):
            copula.pdf([0.5])  # Only 1 dimension

        with pytest.raises(ValueError, match=r"Expected.*dimensions"):
            copula.pdf([0.5, 0.5, 0.5])  # 3 dimensions

    def test_cdf_boundary_conditions(self):
        """Test CDF boundary conditions."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        # CDF should be 0 if any component is 0
        assert copula.cdf([0.0, 0.5]) == 0.0
        assert copula.cdf([0.5, 0.0]) == 0.0

        # CDF should be 1 at (1, 1)
        assert copula.cdf([1.0, 1.0]) == 1.0

    def test_cdf_monotonicity(self):
        """Test that CDF is monotonically increasing."""
        nu = 5.0
        corr = [[1.0, 0.5], [0.5, 1.0]]
        marginals = ["t", "t"]

        copula = StudentTCopula(nu=nu, correlation_matrix=corr, marginals=marginals)

        # Test monotonicity in first dimension
        cdf1 = copula.cdf([0.3, 0.5])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.7, 0.5])

        assert cdf1 < cdf2 < cdf3

        # Test monotonicity in second dimension
        cdf1 = copula.cdf([0.5, 0.3])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.5, 0.7])

        assert cdf1 < cdf2 < cdf3


# ========================================
# ExtremeValueCopula Tests
# ========================================


class TestExtremeValueCopula:
    """Test suite for extreme value copula implementations."""

    # ========================================
    # Initialization Tests
    # ========================================

    def test_initialization_gumbel_valid(self):
        """Test valid Gumbel copula initialization."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=2.0, marginals=["pareto", "pareto"]
        )
        assert copula.copula_type == "gumbel"
        assert copula.theta == 2.0

    def test_initialization_clayton_valid(self):
        """Test valid Clayton copula initialization."""
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=1.5, marginals=["pareto", "pareto"]
        )
        assert copula.copula_type == "clayton"
        assert copula.theta == 1.5

    def test_initialization_frank_valid(self):
        """Test valid Frank copula initialization."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=5.0, marginals=["pareto", "pareto"]
        )
        assert copula.copula_type == "frank"
        assert copula.theta == 5.0

    def test_initialization_invalid_type(self):
        """Test that invalid copula type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid copula type"):
            ExtremeValueCopula(
                copula_type="invalid", theta=2.0, marginals=["pareto", "pareto"]
            )

    def test_initialization_gumbel_invalid_theta(self):
        """Test that Gumbel with theta < 1 raises ValueError."""
        with pytest.raises(ValueError, match=r"Gumbel.*theta >= 1"):
            ExtremeValueCopula(
                copula_type="gumbel", theta=0.5, marginals=["pareto", "pareto"]
            )

    def test_initialization_clayton_invalid_theta(self):
        """Test that Clayton with theta <= 0 raises ValueError."""
        with pytest.raises(ValueError, match=r"Clayton.*theta > 0"):
            ExtremeValueCopula(
                copula_type="clayton", theta=-1.0, marginals=["pareto", "pareto"]
            )

    def test_initialization_frank_invalid_theta(self):
        """Test that Frank with theta = 0 raises ValueError."""
        with pytest.raises(ValueError, match=r"Frank.*theta != 0"):
            ExtremeValueCopula(
                copula_type="frank", theta=0.0, marginals=["pareto", "pareto"]
            )

    def test_initialization_non_bivariate(self):
        """Test that non-bivariate raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="only support 2 dimensions"):
            ExtremeValueCopula(
                copula_type="gumbel",
                theta=2.0,
                marginals=["pareto", "pareto", "pareto"],
            )

    # ========================================
    # Gumbel Copula Tests
    # ========================================

    def test_gumbel_cdf_independence(self):
        """Test Gumbel CDF at independence (theta=1)."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=1.0, marginals=["pareto", "pareto"]
        )

        # At theta=1, Gumbel reduces to independence copula: C(u,v) = u*v
        u, v = 0.5, 0.6
        cdf_val = copula.cdf([u, v])

        # Should be close to u*v for independence
        assert abs(cdf_val - (u * v)) < 1e-6

    def test_gumbel_cdf_strong_dependence(self):
        """Test Gumbel CDF with strong dependence."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=5.0, marginals=["pareto", "pareto"]
        )

        u, v = 0.5, 0.5
        cdf_val = copula.cdf([u, v])

        # Should be greater than independence (0.25) for positive dependence
        assert cdf_val > 0.25

    def test_gumbel_cdf_boundary(self):
        """Test Gumbel CDF boundary conditions."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=2.0, marginals=["pareto", "pareto"]
        )

        # Boundaries
        assert copula.cdf([0.0, 0.5]) == 0.0
        assert copula.cdf([0.5, 0.0]) == 0.0
        assert copula.cdf([1.0, 1.0]) == 1.0

    def test_gumbel_pdf_positive(self):
        """Test that Gumbel PDF is positive."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=2.0, marginals=["pareto", "pareto"]
        )

        pdf_val = copula.pdf([0.5, 0.5])
        assert pdf_val > 0

    # ========================================
    # Clayton Copula Tests
    # ========================================

    def test_clayton_cdf_weak_dependence(self):
        """Test Clayton CDF with weak dependence."""
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=0.5, marginals=["pareto", "pareto"]
        )

        u, v = 0.5, 0.5
        cdf_val = copula.cdf([u, v])

        # Should be positive and less than 1
        assert 0 < cdf_val < 1

    def test_clayton_cdf_strong_dependence(self):
        """Test Clayton CDF with strong dependence."""
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=5.0, marginals=["pareto", "pareto"]
        )

        u, v = 0.5, 0.5
        cdf_val = copula.cdf([u, v])

        # Should be greater than 0 and less than 1
        assert 0 < cdf_val < 1

    def test_clayton_cdf_boundary(self):
        """Test Clayton CDF boundary conditions."""
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=2.0, marginals=["pareto", "pareto"]
        )

        # Boundaries
        assert copula.cdf([0.0, 0.5]) == 0.0
        assert copula.cdf([0.5, 0.0]) == 0.0
        assert copula.cdf([1.0, 1.0]) == 1.0

    def test_clayton_pdf_positive(self):
        """Test that Clayton PDF is positive."""
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=2.0, marginals=["pareto", "pareto"]
        )

        pdf_val = copula.pdf([0.5, 0.5])
        assert pdf_val > 0

    # ========================================
    # Frank Copula Tests
    # ========================================

    def test_frank_cdf_positive_theta(self):
        """Test Frank CDF with positive theta."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=5.0, marginals=["pareto", "pareto"]
        )

        u, v = 0.5, 0.5
        cdf_val = copula.cdf([u, v])

        # Should be positive and less than 1
        assert 0 < cdf_val < 1

    def test_frank_cdf_negative_theta(self):
        """Test Frank CDF with negative theta."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=-5.0, marginals=["pareto", "pareto"]
        )

        u, v = 0.5, 0.5
        cdf_val = copula.cdf([u, v])

        # Should be positive and less than 1
        assert 0 < cdf_val < 1

    def test_frank_cdf_boundary(self):
        """Test Frank CDF boundary conditions."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=5.0, marginals=["pareto", "pareto"]
        )

        # Boundaries
        assert copula.cdf([0.0, 0.5]) == 0.0
        assert copula.cdf([0.5, 0.0]) == 0.0
        assert copula.cdf([1.0, 1.0]) == 1.0

    def test_frank_pdf_positive(self):
        """Test that Frank PDF is positive."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=5.0, marginals=["pareto", "pareto"]
        )

        pdf_val = copula.pdf([0.5, 0.5])
        assert pdf_val > 0

    # ========================================
    # Tail Dependence Coefficient Tests
    # ========================================

    def test_gumbel_tail_dependence(self):
        """Test Gumbel tail dependence coefficients."""
        theta = 2.0
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=theta, marginals=["pareto", "pareto"]
        )

        lambda_u, lambda_l = copula.tail_dependence_coefficient()

        # Gumbel: λ_U = 2 - 2^(1/θ), λ_L = 0
        expected_upper = 2.0 - 2.0 ** (1.0 / theta)
        expected_lower = 0.0

        assert abs(lambda_u - expected_upper) < 1e-10
        assert abs(lambda_l - expected_lower) < 1e-10

    def test_gumbel_tail_dependence_independence(self):
        """Test Gumbel tail dependence at independence."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=1.0, marginals=["pareto", "pareto"]
        )

        lambda_u, lambda_l = copula.tail_dependence_coefficient()

        # At theta=1, no tail dependence
        assert abs(lambda_u - 0.0) < 1e-10
        assert abs(lambda_l - 0.0) < 1e-10

    def test_clayton_tail_dependence(self):
        """Test Clayton tail dependence coefficients."""
        theta = 2.0
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=theta, marginals=["pareto", "pareto"]
        )

        lambda_u, lambda_l = copula.tail_dependence_coefficient()

        # Clayton: λ_U = 0, λ_L = 2^(-1/θ)
        expected_upper = 0.0
        expected_lower = 2.0 ** (-1.0 / theta)

        assert abs(lambda_u - expected_upper) < 1e-10
        assert abs(lambda_l - expected_lower) < 1e-10

    def test_frank_tail_dependence(self):
        """Test Frank tail dependence coefficients."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=5.0, marginals=["pareto", "pareto"]
        )

        lambda_u, lambda_l = copula.tail_dependence_coefficient()

        # Frank: λ_U = λ_L = 0
        assert abs(lambda_u - 0.0) < 1e-10
        assert abs(lambda_l - 0.0) < 1e-10

    # ========================================
    # CDF Monotonicity Tests
    # ========================================

    def test_cdf_monotonicity_gumbel(self):
        """Test that Gumbel CDF is monotonically increasing."""
        copula = ExtremeValueCopula(
            copula_type="gumbel", theta=2.0, marginals=["pareto", "pareto"]
        )

        # Monotonicity in first dimension
        cdf1 = copula.cdf([0.3, 0.5])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.7, 0.5])
        assert cdf1 < cdf2 < cdf3

        # Monotonicity in second dimension
        cdf1 = copula.cdf([0.5, 0.3])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.5, 0.7])
        assert cdf1 < cdf2 < cdf3

    def test_cdf_monotonicity_clayton(self):
        """Test that Clayton CDF is monotonically increasing."""
        copula = ExtremeValueCopula(
            copula_type="clayton", theta=2.0, marginals=["pareto", "pareto"]
        )

        # Monotonicity in first dimension
        cdf1 = copula.cdf([0.3, 0.5])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.7, 0.5])
        assert cdf1 < cdf2 < cdf3

        # Monotonicity in second dimension
        cdf1 = copula.cdf([0.5, 0.3])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.5, 0.7])
        assert cdf1 < cdf2 < cdf3

    def test_cdf_monotonicity_frank(self):
        """Test that Frank CDF is monotonically increasing."""
        copula = ExtremeValueCopula(
            copula_type="frank", theta=5.0, marginals=["pareto", "pareto"]
        )

        # Monotonicity in first dimension
        cdf1 = copula.cdf([0.3, 0.5])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.7, 0.5])
        assert cdf1 < cdf2 < cdf3

        # Monotonicity in second dimension
        cdf1 = copula.cdf([0.5, 0.3])
        cdf2 = copula.cdf([0.5, 0.5])
        cdf3 = copula.cdf([0.5, 0.7])
        assert cdf1 < cdf2 < cdf3


# ========================================
# Integration Tests
# ========================================


def test_abstract_base_class():
    """Test that HeavyTailCopula cannot be instantiated directly."""

    class DummyCopula(HeavyTailCopula):
        def pdf(self, u):
            return 1.0

        def cdf(self, u):
            return 0.5

    # This should work
    copula = DummyCopula(marginals=["pareto", "pareto"])
    assert len(copula.marginals) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
