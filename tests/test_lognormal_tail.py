"""Far-tail behaviour of the LogNormal quantile and survival functions.

Both methods used to fail in the region the library exists to model: ``ppf``
raised ``OverflowError`` where the true quantile is simply not representable,
and ``sf`` was computed as ``1 - cdf(x)``, which collapses to exactly zero once
``cdf(x)`` rounds to 1.0.
"""

from __future__ import annotations

import math

import pytest

from heavytails import LogNormal


class TestPpfOverflow:
    """ppf reports inf rather than raising when the quantile is unrepresentable."""

    @pytest.mark.parametrize(
        ("mu", "sigma", "u"),
        [
            (1000.0, 1.0, 0.5),
            (750.0, 1.0, 0.99),
            (800.0, 5.0, 0.999),
            (0.0, 400.0, 0.999),
        ],
    )
    def test_returns_inf_beyond_the_float_range(
        self, mu: float, sigma: float, u: float
    ) -> None:
        """exp(mu + sigma*z) overflows here, and inf is the right answer.

        The median of LogNormal(mu=1000) is exp(1000). That value genuinely
        exceeds the float range, so reporting inf is correct; raising makes the
        failure look like a mistake by the caller and breaks parameter sweeps.
        """
        assert LogNormal(mu=mu, sigma=sigma).ppf(u) == math.inf

    @pytest.mark.parametrize(
        ("mu", "sigma", "u", "expected"),
        [
            (0.0, 1.0, 0.5, 1.0),
            (0.0, 1.0, 0.999, 21.982183979583034),
            (100.0, 1.0, 0.99, 2.752759277584903e44),
        ],
    )
    def test_finite_quantiles_are_unchanged(
        self, mu: float, sigma: float, u: float, expected: float
    ) -> None:
        """The fix must not perturb any quantile that was already representable."""
        assert LogNormal(mu=mu, sigma=sigma).ppf(u) == pytest.approx(
            expected, rel=1e-12
        )

    def test_sweeping_parameters_no_longer_raises(self) -> None:
        """A grid sweep is the case the exception used to break."""
        values = [
            LogNormal(mu=float(mu), sigma=1.0).ppf(0.99) for mu in range(0, 1200, 100)
        ]
        assert all(v > 0 for v in values)
        assert any(math.isinf(v) for v in values), "expected the sweep to reach inf"
        assert math.isfinite(values[0])

    @pytest.mark.parametrize("u", [0.0, 1.0, -0.1, 1.5])
    def test_still_rejects_u_outside_the_open_unit_interval(self, u: float) -> None:
        """Invalid input must keep raising; only the overflow path changed."""
        with pytest.raises(ValueError):
            LogNormal(mu=0.0, sigma=1.0).ppf(u)


class TestSurvivalFunctionAccuracy:
    """sf is computed with erfc rather than as 1 - cdf."""

    @pytest.mark.parametrize("x", [1e3, 1e5, 1e8, 1e12])
    def test_sf_stays_positive_where_one_minus_cdf_underflows(self, x: float) -> None:
        """Past x = 1e5 the naive form is exactly zero and carries no information."""
        ln = LogNormal(mu=0.0, sigma=1.0)
        assert ln.sf(x) > 0.0
        assert math.isfinite(ln.sf(x))

    def test_one_minus_cdf_really_does_underflow_here(self) -> None:
        """Pins the premise of the test above.

        If this ever fails, the comparison has stopped exercising what it claims
        and the tests above are no longer meaningful.
        """
        ln = LogNormal(mu=0.0, sigma=1.0)
        assert 1.0 - ln.cdf(1e5) == 0.0
        assert ln.sf(1e5) > 0.0

    @pytest.mark.parametrize(
        ("x", "expected"),
        [
            (10.0, 0.010651099341700122),
            (100.0, 2.060643395971714e-06),
            (1000.0, 2.461912018815488e-12),
            (100000.0, 5.677979296840896e-31),
        ],
    )
    def test_sf_matches_reference_values(self, x: float, expected: float) -> None:
        """Reference values from scipy.stats.lognorm(s=1, scale=1).sf."""
        assert LogNormal(mu=0.0, sigma=1.0).sf(x) == pytest.approx(expected, rel=1e-13)

    def test_sf_complements_cdf_where_both_are_representable(self) -> None:
        """The two must still agree in the range where the naive form works."""
        ln = LogNormal(mu=0.0, sigma=1.0)
        for x in (0.1, 0.5, 1.0, 2.0, 10.0):
            assert ln.cdf(x) + ln.sf(x) == pytest.approx(1.0, abs=1e-12)

    def test_sf_is_one_below_the_support(self) -> None:
        """The support is x > 0."""
        ln = LogNormal(mu=0.0, sigma=1.0)
        assert ln.sf(0.0) == 1.0
        assert ln.sf(-5.0) == 1.0

    def test_sf_is_monotone(self) -> None:
        """The survival function never increases."""
        ln = LogNormal(mu=0.0, sigma=1.0)
        values = [ln.sf(x) for x in (0.1, 1.0, 10.0, 100.0, 1e4, 1e8)]
        assert values == sorted(values, reverse=True)
