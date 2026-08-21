"""Tests for the shared special functions in :mod:`heavytails._special`."""

from __future__ import annotations

import math

import pytest

from heavytails import extra_distributions as extra
from heavytails._special import (
    _betainc_reg,
    _betaincinv_reg,
    _gammainc_lower_reg,
    _log_beta,
)


class TestBetaIncInv:
    """The regularized incomplete beta inverse."""

    @pytest.mark.parametrize(
        ("a", "b"),
        [(0.5, 0.5), (1.0, 1.0), (2.0, 3.0), (0.5, 2.5), (15.0, 0.5), (50.0, 0.5)],
    )
    @pytest.mark.parametrize(
        "p", [1e-12, 1e-8, 1e-4, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-6]
    )
    def test_round_trip(self, a: float, b: float, p: float) -> None:
        """I_y(a,b) should return p for the y the inverse produced."""
        y = _betaincinv_reg(a, b, p)
        assert 0.0 <= y <= 1.0
        assert _betainc_reg(a, b, y) == pytest.approx(p, rel=1e-9)

    def test_boundaries(self) -> None:
        """p = 0 and p = 1 map to the endpoints exactly."""
        assert _betaincinv_reg(2.0, 3.0, 0.0) == 0.0
        assert _betaincinv_reg(2.0, 3.0, 1.0) == 1.0

    def test_tiny_target_keeps_relative_precision(self) -> None:
        """A very small p must not collapse to zero.

        This is the case a plain bisection on [0, 1] gets wrong: it converges to
        a fixed absolute precision and leaves a y of order 1e-15 with barely one
        correct digit.
        """
        p = 1e-14
        y = _betaincinv_reg(0.5, 0.5, p)
        assert y > 0.0
        assert _betainc_reg(0.5, 0.5, y) == pytest.approx(p, rel=1e-8)

    def test_matches_closed_form_for_half_half(self) -> None:
        """I_y(1/2, 1/2) = (2/pi) arcsin(sqrt(y)) has an exact inverse."""
        for p in (0.1, 0.3, 0.5, 0.8, 0.95):
            expected = math.sin(p * math.pi / 2.0) ** 2
            assert _betaincinv_reg(0.5, 0.5, p) == pytest.approx(expected, rel=1e-10)

    def test_matches_closed_form_for_unit_shapes(self) -> None:
        """I_y(1,1) = y, so the inverse is the identity."""
        for p in (0.01, 0.25, 0.5, 0.99):
            assert _betaincinv_reg(1.0, 1.0, p) == pytest.approx(p, rel=1e-12)

    @pytest.mark.parametrize(
        ("a", "b", "p"), [(0.0, 1.0, 0.5), (1.0, -1.0, 0.5), (1.0, 1.0, 1.5)]
    )
    def test_rejects_invalid_arguments(self, a: float, b: float, p: float) -> None:
        """Out-of-range shapes or probabilities raise."""
        with pytest.raises(ValueError):
            _betaincinv_reg(a, b, p)


class TestSpecialFunctionsStillExported:
    """The helpers moved modules; the old import path must keep working."""

    def test_reexported_from_extra_distributions(self) -> None:
        """Existing code importing these from extra_distributions is unaffected."""
        assert extra._log_beta is _log_beta
        assert extra._betainc_reg is _betainc_reg
        assert extra._gammainc_lower_reg is _gammainc_lower_reg
