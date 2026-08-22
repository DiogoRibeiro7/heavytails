"""Direct inversion of the incomplete beta and gamma, and what it replaced.

Two families had no closed-form quantile and solved against their own
distribution function instead, paying a continued fraction per iteration and
reaching about four correct digits in the far tail. Both are now inverted
directly, which is faster where it is faster and far more accurate everywhere.

The speed claim is checked by **counting continued-fraction evaluations**
rather than by timing anything. A wall-clock assertion would flake under CI
load and would say nothing about why; the evaluation count is exactly the
quantity the change reduces, and it is deterministic.
"""

from __future__ import annotations

from itertools import pairwise
import math

import pytest

from heavytails._special import (
    _betainc_reg,
    _betaincinv_reg,
    _gammainc_lower_reg,
    _gammainc_upper_reg,
    _gammaincinv_reg,
)
from heavytails.extra_distributions import BetaPrime, InverseGamma

mpmath = pytest.importorskip("mpmath")
mpmath.mp.dps = 50

QUANTILES = [
    1e-12,
    1e-9,
    1e-6,
    1e-3,
    0.01,
    0.1,
    0.25,
    0.5,
    0.75,
    0.9,
    0.99,
    0.999,
    1 - 1e-6,
    1 - 1e-9,
]


def _upper_reference(a: float, x: float) -> float:
    """Q(a, x) at 50 digits, computed as the upper integral."""
    return float(mpmath.gammainc(mpmath.mpf(a), mpmath.mpf(x), regularized=True))


def _relative(got: float, exact: float) -> float:
    if exact == 0.0:
        return abs(got)
    return abs(got - exact) / abs(exact)


class TestUpperIncompleteGamma:
    @pytest.mark.parametrize("a", [0.3, 1.0, 2.0, 5.0, 20.0])
    @pytest.mark.parametrize("x", [0.1, 1.0, 5.0, 30.0, 100.0])
    def test_it_matches_mpmath(self, a: float, x: float) -> None:
        assert _relative(_gammainc_upper_reg(a, x), _upper_reference(a, x)) < 1e-12

    @pytest.mark.parametrize(("a", "x"), [(2.0, 50.0), (3.0, 100.0), (0.5, 40.0)])
    def test_it_resolves_what_one_minus_p_cannot(self, a: float, x: float) -> None:
        """``1 - P`` returns exactly zero here; the true value is not zero.

        This is the entire reason the function exists. The subtraction is done
        in double precision, so anything below about 1e-16 is gone, and the far
        upper tail is precisely where a heavy-tailed distribution is
        interesting.
        """
        assert 1.0 - _gammainc_lower_reg(a, x) == 0.0
        direct = _gammainc_upper_reg(a, x)
        assert direct > 0.0
        assert _relative(direct, _upper_reference(a, x)) < 1e-12

    @pytest.mark.parametrize("a", [0.5, 2.0, 10.0])
    def test_it_complements_the_lower_one_where_both_are_representable(
        self, a: float
    ) -> None:
        for x in [0.5, 1.0, 3.0, 8.0]:
            total = _gammainc_upper_reg(a, x) + _gammainc_lower_reg(a, x)
            assert total == pytest.approx(1.0, abs=1e-14)

    def test_the_edges(self) -> None:
        assert _gammainc_upper_reg(2.0, 0.0) == 1.0
        assert _gammainc_upper_reg(2.0, 1e5) == 0.0  # underflows honestly
        with pytest.raises(ValueError, match="a must be"):
            _gammainc_upper_reg(0.0, 1.0)
        with pytest.raises(ValueError, match="a must be"):
            _gammainc_upper_reg(1.0, -1.0)


class TestIncompleteGammaInverse:
    @pytest.mark.parametrize("a", [0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
    @pytest.mark.parametrize("p", [1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9, 0.99])
    def test_the_lower_inverse_round_trips(self, a: float, p: float) -> None:
        assert _relative(_gammainc_lower_reg(a, _gammaincinv_reg(a, p)), p) < 1e-11

    @pytest.mark.parametrize("a", [0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0])
    @pytest.mark.parametrize("p", [1e-12, 1e-6, 1e-3, 0.1, 0.5, 0.9])
    def test_the_upper_inverse_round_trips(self, a: float, p: float) -> None:
        x = _gammaincinv_reg(a, p, upper=True)
        assert _relative(_gammainc_upper_reg(a, x), p) < 1e-11

    def test_the_two_modes_agree_where_both_are_well_conditioned(self) -> None:
        for a in [0.5, 2.0, 10.0]:
            for p in [0.1, 0.3, 0.5]:
                assert _gammaincinv_reg(a, p) == pytest.approx(
                    _gammaincinv_reg(a, 1.0 - p, upper=True), rel=1e-9
                )

    def test_it_increases_with_the_probability(self) -> None:
        values = [_gammaincinv_reg(2.0, p) for p in [1e-6, 1e-3, 0.1, 0.5, 0.9, 0.99]]
        assert all(b > a for a, b in pairwise(values))

    def test_the_edges(self) -> None:
        assert _gammaincinv_reg(2.0, 0.0) == 0.0
        assert _gammaincinv_reg(2.0, 1.0) == math.inf
        assert _gammaincinv_reg(2.0, 1.0, upper=True) == 0.0
        assert _gammaincinv_reg(2.0, 0.0, upper=True) == math.inf
        with pytest.raises(ValueError, match="p must be"):
            _gammaincinv_reg(2.0, 1.5)
        with pytest.raises(ValueError, match="a must be"):
            _gammaincinv_reg(0.0, 0.5)


class TestTheSafeguardThatWasNotSafeguarding:
    def test_the_case_that_exposed_it(self) -> None:
        """``I_y(50, 0.3) = 1e-3``: the bisection fallback returned the input.

        The Newton loop narrowed its bracket *after* computing the midpoint to
        fall back to, so on the first iteration -- where the starting point is
        the midpoint by construction -- the fallback produced that same point,
        the no-progress check fired, and the routine declared convergence with
        a relative error of 0.18.

        It was invisible while the bracketing phase bisected to 1e-13 first,
        because "converged at the midpoint" was then true to thirteen digits.
        Anything that makes the bracketing cheaper exposes it, so the ordering
        is pinned here rather than left to be rediscovered.
        """
        y = _betaincinv_reg(50.0, 0.3, 1e-3)
        assert _relative(_betainc_reg(50.0, 0.3, y), 1e-3) < 1e-12

    @pytest.mark.parametrize("a", [0.3, 1.0, 2.0, 10.0, 50.0, 200.0])
    @pytest.mark.parametrize("b", [0.3, 1.0, 3.0, 50.0])
    @pytest.mark.parametrize("p", [1e-12, 1e-6, 1e-3, 0.01, 0.3, 0.7, 0.99])
    def test_the_inverse_round_trips_across_the_parameter_grid(
        self, a: float, b: float, p: float
    ) -> None:
        y = _betaincinv_reg(a, b, p)
        assert _relative(_betainc_reg(a, b, y), p) < 1e-9


class TestEvaluationCounts:
    """The speed claim, as a count rather than a clock."""

    def test_the_beta_inverse_no_longer_bisects_the_whole_exponent_range(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """It used to spend about 63 evaluations before Newton began.

        Blind bisection of ``log y`` over a range of 745 down to a width of
        1e-13 needs log2(745/1e-13) of them, every one a continued fraction.
        Seeding from the small-y asymptote instead arrives in the right place
        immediately.
        """
        import heavytails._special as special  # noqa: PLC0415

        calls = 0
        original = special._betainc_reg

        def counted(a: float, b: float, x: float) -> float:
            nonlocal calls
            calls += 1
            return original(a, b, x)

        monkeypatch.setattr(special, "_betainc_reg", counted)
        special._betaincinv_reg(1.5, 0.5, 1e-6)
        assert calls < 30

    def test_the_student_t_quantile_is_cheap_in_the_far_tail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tail is where the old code was most wasteful and most needed."""
        import heavytails._special as special  # noqa: PLC0415

        calls = 0
        original = special._betainc_reg

        def counted(a: float, b: float, x: float) -> float:
            nonlocal calls
            calls += 1
            return original(a, b, x)

        monkeypatch.setattr(special, "_betainc_reg", counted)
        special._betaincinv_reg(1.5, 0.5, 1e-14)
        assert calls < 30


class TestInverseGammaQuantile:
    @pytest.mark.parametrize(
        "dist",
        [
            InverseGamma(alpha=2.0, beta=1.0),
            InverseGamma(alpha=0.5, beta=3.0),
            InverseGamma(alpha=5.0, beta=0.1),
        ],
    )
    @pytest.mark.parametrize("u", QUANTILES)
    def test_the_quantile_inverts_the_distribution_function(
        self, dist: InverseGamma, u: float
    ) -> None:
        """Round trips to fourteen digits where the solver managed about four.

        The old path bracketed and solved against ``cdf``, which was itself
        computing ``1 - P`` and had already thrown the lower tail away.
        """
        assert _relative(dist.cdf(dist.ppf(u)), u) < 1e-11

    @pytest.mark.parametrize("dist", [InverseGamma(alpha=2.0, beta=1.0)])
    def test_the_lower_tail_is_no_longer_lost(self, dist: InverseGamma) -> None:
        """``cdf`` returned exactly zero here. The true value is not zero."""
        exact = _upper_reference(dist.alpha, dist.beta / 0.02)
        assert exact > 0.0
        assert _relative(dist.cdf(0.02), exact) < 1e-12

    def test_the_distribution_function_and_its_complement_sum_to_one(self) -> None:
        dist = InverseGamma(alpha=2.0, beta=1.0)
        for x in [0.02, 0.1, 0.5, 1.0, 5.0, 100.0]:
            assert dist.cdf(x) + dist.sf(x) == pytest.approx(1.0, abs=1e-15)

    def test_the_quantile_increases(self) -> None:
        dist = InverseGamma(alpha=2.0, beta=1.0)
        values = [dist.ppf(u) for u in QUANTILES]
        assert all(b >= a for a, b in pairwise(values))

    def test_it_rejects_probabilities_outside_the_open_unit_interval(self) -> None:
        dist = InverseGamma(alpha=2.0, beta=1.0)
        for u in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="u must be"):
                dist.ppf(u)

    def test_sampling_still_agrees_with_the_quantile_function(self) -> None:
        """``rvs`` uses gamma variates, not inverse transform.

        Worth pinning, because the issue this closes assumed otherwise: the
        quantile cost was never paid per variate for this family. The two
        routes must still describe the same distribution.
        """
        dist = InverseGamma(alpha=3.0, beta=2.0)
        sample = sorted(dist.rvs(20_000, seed=1))
        for u in [0.1, 0.25, 0.5, 0.75, 0.9]:
            empirical = sample[int(u * len(sample)) - 1]
            assert empirical == pytest.approx(dist.ppf(u), rel=0.06)


class TestBetaPrimeQuantile:
    @pytest.mark.parametrize(
        "dist",
        [
            BetaPrime(a=2.0, b=3.0, s=1.0),
            BetaPrime(a=0.5, b=0.5, s=2.0),
            BetaPrime(a=5.0, b=1.5, s=0.3),
        ],
    )
    @pytest.mark.parametrize("u", QUANTILES)
    def test_the_quantile_inverts_the_distribution_function(
        self, dist: BetaPrime, u: float
    ) -> None:
        """Worst relative error was 5.6e-4 in the far tail; now around 1e-15.

        Checked against ``sf`` above the median. Not to be kind to the
        implementation: ``cdf`` works from ``z = x/(x+s)``, which rounds to
        exactly 1 once ``x`` passes about 1e17, so it cannot represent the
        answer there at any accuracy. ``sf`` works from ``s/(x+s)``, which is
        the small quantity and stays exact, so it is the only side that can
        express the result being checked.
        """
        x = dist.ppf(u)
        if u <= 0.5:
            assert _relative(dist.cdf(x), u) < 1e-11
        else:
            assert _relative(dist.sf(x), 1.0 - u) < 1e-11

    def test_the_upper_tail_uses_the_mirrored_problem(self) -> None:
        """``x = s z/(1-z)`` cancels catastrophically as ``z`` approaches one.

        Solving for ``1 - z`` directly makes that the quantity computed rather
        than the quantity subtracted, which is what keeps the upper tail
        accurate at all.
        """
        dist = BetaPrime(a=2.0, b=3.0, s=1.0)
        for u in [1 - 1e-6, 1 - 1e-9, 1 - 1e-12]:
            assert _relative(dist.sf(dist.ppf(u)), 1.0 - u) < 1e-9

    def test_the_survival_function_survives_where_the_complement_cannot(self) -> None:
        """``1 - cdf`` is exactly zero here. The true probability is 1e-9."""
        dist = BetaPrime(a=0.5, b=0.5, s=2.0)
        x = dist.ppf(1 - 1e-9)
        assert 1.0 - dist.cdf(x) == 0.0
        assert dist.sf(x) == pytest.approx(1e-9, rel=1e-9)

    def test_the_two_sides_agree_where_both_are_representable(self) -> None:
        dist = BetaPrime(a=2.0, b=3.0, s=1.0)
        for x in [0.1, 0.5, 1.0, 5.0, 50.0]:
            assert dist.cdf(x) + dist.sf(x) == pytest.approx(1.0, abs=1e-14)

    def test_the_quantile_increases(self) -> None:
        dist = BetaPrime(a=2.0, b=3.0, s=1.0)
        values = [dist.ppf(u) for u in QUANTILES]
        assert all(b >= a for a, b in pairwise(values))

    def test_the_scale_factors_out(self) -> None:
        """``s`` is a pure scale, so the quantile must be linear in it."""
        base = BetaPrime(a=2.0, b=3.0, s=1.0)
        scaled = BetaPrime(a=2.0, b=3.0, s=7.0)
        for u in [0.1, 0.5, 0.9, 0.999]:
            assert scaled.ppf(u) == pytest.approx(7.0 * base.ppf(u), rel=1e-12)

    def test_it_rejects_probabilities_outside_the_open_unit_interval(self) -> None:
        dist = BetaPrime(a=2.0, b=3.0, s=1.0)
        for u in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="u must be"):
                dist.ppf(u)

    def test_sampling_still_agrees_with_the_quantile_function(self) -> None:
        dist = BetaPrime(a=2.0, b=3.0, s=1.0)
        sample = sorted(dist.rvs(20_000, seed=2))
        for u in [0.1, 0.25, 0.5, 0.75, 0.9]:
            empirical = sample[int(u * len(sample)) - 1]
            assert empirical == pytest.approx(dist.ppf(u), rel=0.06)
