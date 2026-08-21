"""Accuracy bounds for the special functions, against arbitrary precision.

The library implements the regularized incomplete beta and the regularized
lower incomplete gamma from the standard library alone, so nothing else
constrains their accuracy. These tests pin the bounds measured by
``scripts/special_function_accuracy.py``, which means a regression fails the
build rather than only changing a number in the documentation.

`mpmath` is a development dependency and is not needed to use the library.
"""

from __future__ import annotations

import math

import pytest

from heavytails import InverseGamma, StudentT
from heavytails._special import _betainc_reg, _gammainc_lower_reg

mpmath = pytest.importorskip("mpmath", reason="mpmath is a development dependency")

# Far beyond double precision, so the reference is exact for our purposes.
mpmath.mp.dps = 50

# The bounds the sweep reports, with a little headroom. Tightening these when
# the implementation improves is expected; loosening them is a regression and
# should be argued for in review.
BETA_TOLERANCE = 1e-11
GAMMA_TOLERANCE = 1e-11


def _beta_reference(a: float, b: float, x: float) -> float:
    return float(
        mpmath.betainc(mpmath.mpf(a), mpmath.mpf(b), 0, mpmath.mpf(x), regularized=True)
    )


def _gamma_reference(a: float, x: float) -> float:
    return float(mpmath.gammainc(mpmath.mpf(a), 0, mpmath.mpf(x), regularized=True))


def _relative_error(got: float, exact: float) -> float:
    if exact == 0.0:
        return abs(got)
    return abs(got - exact) / abs(exact)


class TestIncompleteBetaAccuracy:
    """Regularized incomplete beta against mpmath."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (0.5, 0.5),
            (0.5, 1.5),
            (1.0, 1.0),
            (2.0, 3.0),
            (5.0, 0.5),
            (0.05, 0.05),
            (15.0, 0.5),
            (100.0, 0.5),
            (1000.0, 0.5),
        ],
    )
    @pytest.mark.parametrize(
        "x", [1e-12, 1e-8, 1e-4, 0.1, 0.5, 0.9, 1 - 1e-4, 1 - 1e-8, 1 - 1e-12]
    )
    def test_within_tolerance(self, a: float, b: float, x: float) -> None:
        got = _betainc_reg(a, b, x)
        exact = _beta_reference(a, b, x)
        assert _relative_error(got, exact) < BETA_TOLERANCE

    @pytest.mark.parametrize(("a", "b"), [(0.5, 0.5), (2.0, 3.0), (15.0, 0.5)])
    def test_accuracy_does_not_degrade_at_the_switch_point(
        self, a: float, b: float
    ) -> None:
        """The implementation mirrors the problem above (a+1)/(a+b+2).

        A badly chosen switch shows up as an error spike on one side. Both
        sides must meet the same bound.
        """
        boundary = (a + 1.0) / (a + b + 2.0)
        for x in (boundary - 1e-6, boundary + 1e-6):
            if not 0.0 < x < 1.0:
                continue
            assert (
                _relative_error(_betainc_reg(a, b, x), _beta_reference(a, b, x))
                < BETA_TOLERANCE
            )

    def test_endpoints_are_exact(self) -> None:
        assert _betainc_reg(2.0, 3.0, 0.0) == 0.0
        assert _betainc_reg(2.0, 3.0, 1.0) == 1.0

    def test_matches_the_closed_form_for_half_half(self) -> None:
        """I_x(1/2, 1/2) = (2/pi) arcsin(sqrt(x))."""
        for x in (0.05, 0.25, 0.5, 0.8, 0.99):
            expected = 2.0 / math.pi * math.asin(math.sqrt(x))
            assert _betainc_reg(0.5, 0.5, x) == pytest.approx(expected, rel=1e-13)


class TestIncompleteGammaAccuracy:
    """Regularized lower incomplete gamma against mpmath."""

    @pytest.mark.parametrize("a", [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0, 1000.0])
    @pytest.mark.parametrize("ratio", [1e-6, 0.1, 0.5, 1.0, 1.05, 2.0, 5.0, 20.0])
    def test_within_tolerance(self, a: float, ratio: float) -> None:
        x = a * ratio
        got = _gammainc_lower_reg(a, x)
        exact = _gamma_reference(a, x)
        assert _relative_error(got, exact) < GAMMA_TOLERANCE

    @pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
    def test_accuracy_does_not_degrade_at_the_switch_point(self, a: float) -> None:
        """The implementation switches from the series to the continued
        fraction at x = a + 1.

        This is where the continued fraction was previously returning a
        completely wrong value: `P(20, 21)` came back as 0.0 against a true
        0.6157, because the Lentz recurrence was missing its leading term and
        its `b` was shifted by one.
        """
        for x in (a + 1.0 - 1e-6, a + 1.0 + 1e-6):
            assert (
                _relative_error(_gammainc_lower_reg(a, x), _gamma_reference(a, x))
                < GAMMA_TOLERANCE
            )

    @pytest.mark.parametrize(("a", "x"), [(20.0, 21.0), (2.0, 3.0), (5.0, 6.0)])
    def test_continued_fraction_regression(self, a: float, x: float) -> None:
        """Direct regression test for the bug the accuracy sweep found."""
        assert _gammainc_lower_reg(a, x) == pytest.approx(
            _gamma_reference(a, x), rel=1e-12
        )

    def test_is_zero_at_the_origin(self) -> None:
        assert _gammainc_lower_reg(2.0, 0.0) == 0.0

    def test_matches_the_closed_form_for_unit_shape(self) -> None:
        """P(1, x) = 1 - exp(-x)."""
        for x in (0.1, 1.0, 2.0, 5.0, 20.0):
            assert _gammainc_lower_reg(1.0, x) == pytest.approx(
                1.0 - math.exp(-x), rel=1e-13
            )

    @pytest.mark.parametrize(("a", "x"), [(0.0, 1.0), (-1.0, 1.0), (1.0, -1.0)])
    def test_rejects_invalid_arguments(self, a: float, x: float) -> None:
        with pytest.raises(ValueError):
            _gammainc_lower_reg(a, x)


class TestDistributionsThatDependOnThem:
    """The bug reached users through InverseGamma, so check that path too."""

    @pytest.mark.parametrize(
        ("alpha", "beta"), [(2.0, 1.0), (3.0, 2.0), (0.5, 1.0), (5.0, 3.0)]
    )
    @pytest.mark.parametrize("x", [0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 10.0])
    def test_inverse_gamma_cdf(self, alpha: float, beta: float, x: float) -> None:
        """InverseGamma.cdf was wrong by factors of 2 to 17 in the lower tail.

        The CDF is P(alpha, beta/x), so every x below roughly beta/(alpha+1)
        landed in the broken continued-fraction branch.
        """
        # F(x) = Q(alpha, beta/x) = 1 - P(alpha, beta/x).
        got = InverseGamma(alpha=alpha, beta=beta).cdf(x)
        exact = 1.0 - _gamma_reference(alpha, beta / x)
        assert _relative_error(got, exact) < 1e-10

    def test_inverse_gamma_cdf_is_monotone(self) -> None:
        dist = InverseGamma(alpha=2.0, beta=1.0)
        values = [dist.cdf(x) for x in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 10.0, 100.0)]
        assert values == sorted(values)

    @pytest.mark.parametrize("nu", [1.0, 2.5, 5.0, 30.0])
    @pytest.mark.parametrize("x", [-10.0, -1.0, 0.5, 3.0, 25.0])
    def test_student_t_cdf(self, nu: float, x: float) -> None:
        """Student-t reaches the incomplete beta rather than the gamma."""
        got = StudentT(nu=nu).cdf(x)
        y = nu / (nu + x * x)
        half = 0.5 * _beta_reference(nu / 2.0, 0.5, y)
        exact = 1.0 - half if x >= 0 else half
        assert abs(got - exact) < 1e-12
