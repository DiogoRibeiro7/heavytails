"""The NumPy fast path must agree with the scalar methods exactly.

Every kernel is a hand transcription of a scalar method, which is the whole
risk: a transcription can drop a guard, mirror a branch the wrong way, or use
a mathematically equal form that rounds differently. So the tests compare
element by element to a budget of a few units in the last place -- not a
relative tolerance, which would let a genuine formula error hide inside it.

Eight of the ten kernels are bit-identical to their scalar counterparts at
every point tested, because NumPy and :mod:`math` call the same library for the
functions involved. BurrXII and LogLogistic are not: NumPy evaluates ``**``
over an array through a different route, and about one point in two thousand
differs in the last bits, by at most 4 ULP. The budget is 8, which is tight
enough that any real error fails immediately.

The arrays deliberately include the guard regions: below the support, at the
support boundary, at the endpoints of a bounded distribution, and far into both
tails. A kernel that is right in the middle and wrong at the edges is the
likely failure, and testing only sensible values would miss it.
"""

from __future__ import annotations

import math

import pytest

from heavytails import (
    Cauchy,
    Frechet,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)
from heavytails.extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
)
from heavytails.vectorized import accelerated, cdf, pdf, ppf, sf

np = pytest.importorskip("numpy", reason="the fast path needs numpy")

# Units in the last place. Measured worst case is 4, on BurrXII.pdf.
ULP_BUDGET = 8


def _within_budget(fast: object, scalar: object) -> bool:
    """Whether two arrays agree to within the ULP budget, elementwise.

    Infinities and NaNs must match exactly -- those are guard outcomes, not
    arithmetic, and a disagreement there is a dropped branch rather than a
    rounding difference.
    """
    fast_a, scalar_a = np.asarray(fast, dtype=float), np.asarray(scalar, dtype=float)
    if fast_a.shape != scalar_a.shape:
        return False
    special = ~np.isfinite(scalar_a)
    if not np.array_equal(fast_a[special], scalar_a[special], equal_nan=True):
        return False
    finite = ~special
    if not finite.any():
        return True
    difference = np.abs(fast_a[finite] - scalar_a[finite])
    return bool(np.all(difference <= ULP_BUDGET * np.spacing(np.abs(scalar_a[finite]))))


# Families with a kernel, and points spanning their guard regions.
ACCELERATED = [
    pytest.param(
        Pareto(alpha=2.5, xm=2.0),
        [-5.0, 0.0, 1.0, 1.999, 2.0, 2.5, 10.0, 1e6, 1e15],
        id="Pareto",
    ),
    pytest.param(
        Cauchy(x0=1.0, gamma=0.5),
        [-1e15, -1e6, -3.0, 0.4, 0.99, 1.0, 1.01, 1.6, 3.0, 1e6, 1e15],
        id="Cauchy",
    ),
    pytest.param(
        Weibull(k=0.7, lam=2.0),
        [-5.0, -1e-12, 0.0, 1e-12, 0.5, 2.0, 20.0, 1e6],
        id="Weibull",
    ),
    pytest.param(
        Frechet(alpha=2.0, s=1.5, m=0.5),
        [-5.0, 0.0, 0.4999, 0.5, 0.5001, 2.0, 100.0, 1e8],
        id="Frechet",
    ),
    pytest.param(
        GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0),
        # The support starts at mu - sigma/xi = -2, so -2 is the boundary.
        [-1e6, -3.0, -2.001, -2.0, -1.999, 0.0, 5.0, 1e6],
        id="GEV_Frechet",
    ),
    pytest.param(
        GeneralizedPareto(xi=0.4, sigma=1.0, mu=1.0),
        [-5.0, 0.0, 0.999, 1.0, 1.001, 3.0, 1e6, 1e15],
        id="GPD-heavy",
    ),
    pytest.param(
        GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0),
        # Bounded above at mu - sigma/xi = 2, where the scalar cdf returns 1
        # rather than 0. Getting that constant backwards is the obvious slip.
        [-1.0, 0.0, 0.5, 1.9, 1.999, 2.0, 2.001, 5.0],
        id="GPD-bounded",
    ),
    pytest.param(
        GeneralizedPareto(xi=0.0, sigma=2.0, mu=0.0),
        [-1.0, 0.0, 0.5, 3.0, 50.0],
        id="GPD-exponential",
    ),
    pytest.param(
        BurrXII(c=2.0, k=1.5, s=1.0),
        [-1.0, 0.0, 1e-12, 0.5, 1.0, 100.0, 1e8],
        id="BurrXII",
    ),
    pytest.param(
        LogLogistic(kappa=2.0, lam=1.0),
        [-1.0, 0.0, 1e-12, 0.5, 1.0, 100.0, 1e8],
        id="LogLogistic",
    ),
]

# Families with no kernel, which must still work through the fallback.
FALLBACK = [
    pytest.param(LogNormal(mu=0.0, sigma=1.0), [1e-6, 0.5, 1.0, 10.0], id="LogNormal"),
    pytest.param(StudentT(nu=3.0), [-10.0, -1.0, 0.0, 1.0, 10.0], id="StudentT"),
    pytest.param(
        InverseGamma(alpha=2.0, beta=1.0), [0.05, 0.5, 1.0, 20.0], id="InverseGamma"
    ),
    pytest.param(
        BetaPrime(a=2.0, b=3.0, s=1.0), [0.01, 0.5, 1.0, 50.0], id="BetaPrime"
    ),
]

PROBABILITIES = [
    1e-12,
    1e-9,
    1e-6,
    0.001,
    0.24,
    0.25,
    0.26,
    0.5,
    0.74,
    0.76,
    0.99,
    1 - 1e-9,
]


class TestNumpyAndMathAgree:
    """The premise the exactness claim rests on, checked directly.

    If a platform's NumPy and libm disagree, every kernel comparison below
    fails for the same reason and this test says what it is.
    """

    @pytest.mark.parametrize(
        ("numpy_fn", "math_fn", "low", "high"),
        [
            (np.log, math.log, 1e-8, 1e8),
            (np.exp, math.exp, -20.0, 20.0),
            (np.log1p, math.log1p, -0.999, 100.0),
            (np.expm1, math.expm1, -20.0, 20.0),
            (np.arctan, math.atan, -1e6, 1e6),
            (np.tan, math.tan, -1.5, 1.5),
        ],
    )
    def test_they_are_bit_identical(
        self, numpy_fn: object, math_fn: object, low: float, high: float
    ) -> None:
        values = np.random.default_rng(0).uniform(low, high, 20_000)
        assert np.array_equal(
            numpy_fn(values),  # type: ignore[operator]
            np.array([math_fn(v) for v in values]),  # type: ignore[operator]
        )

    def test_exponentiation_is_bit_identical(self) -> None:
        values = np.random.default_rng(1).uniform(1.0, 100.0, 20_000)
        assert np.array_equal(values**2.5, np.array([v**2.5 for v in values]))


class TestTheFastPathMatchesTheScalarPath:
    @pytest.mark.parametrize(("dist", "points"), ACCELERATED)
    @pytest.mark.parametrize("name", ["pdf", "cdf", "sf"])
    def test_exactly_on_the_guard_regions(
        self, dist: object, points: list[float], name: str
    ) -> None:
        """Not to a tolerance. Every element, including outside the support."""
        assert accelerated(dist, name)
        fast = {"pdf": pdf, "cdf": cdf, "sf": sf}[name](dist, points)
        scalar = np.array([getattr(dist, name)(p) for p in points])
        assert _within_budget(fast, scalar), (
            f"{type(dist).__name__}.{name} differs beyond {ULP_BUDGET} ULP at "
            f"{[p for p, a, b in zip(points, fast, scalar, strict=True) if a != b]}"
        )

    @pytest.mark.parametrize(("dist", "points"), ACCELERATED)
    def test_the_quantile_matches_exactly(
        self, dist: object, points: list[float]
    ) -> None:
        del points
        fast = ppf(dist, PROBABILITIES)
        scalar = np.array([dist.ppf(p) for p in PROBABILITIES])  # type: ignore[attr-defined]
        assert _within_budget(fast, scalar)

    @pytest.mark.parametrize(("dist", "points"), ACCELERATED)
    def test_it_matches_on_a_large_random_sample(
        self, dist: object, points: list[float]
    ) -> None:
        """The hand-picked points above are where a slip is likely; this is
        where volume would expose one that is not."""
        del points
        sample = dist.rvs(5_000, seed=1)  # type: ignore[attr-defined]
        for name, function in (("pdf", pdf), ("cdf", cdf), ("sf", sf)):
            fast = function(dist, sample)
            scalar = np.array([getattr(dist, name)(v) for v in sample])
            assert _within_budget(fast, scalar), f"{name} differs"


class TestTheFallbackPath:
    @pytest.mark.parametrize(("dist", "points"), FALLBACK)
    def test_families_without_a_kernel_still_work(
        self, dist: object, points: list[float]
    ) -> None:
        """Correct by construction: the fallback *is* the scalar method."""
        for name, function in (("pdf", pdf), ("cdf", cdf), ("sf", sf)):
            assert not accelerated(dist, name)
            # The fallback *is* the scalar method, so this is exact.
            assert np.array_equal(
                function(dist, points),
                np.array([getattr(dist, name)(p) for p in points]),
            )

    @pytest.mark.parametrize(("dist", "points"), FALLBACK)
    def test_the_quantile_falls_back_too(
        self, dist: object, points: list[float]
    ) -> None:
        del points
        assert np.array_equal(
            ppf(dist, PROBABILITIES),
            np.array([dist.ppf(p) for p in PROBABILITIES]),  # type: ignore[attr-defined]
        )

    def test_which_families_are_accelerated_is_reported_honestly(self) -> None:
        """NumPy has no error function and no incomplete beta or gamma.

        Claiming a speedup for these would be worse than not having one: a
        caller sizing a job on the strength of it would be badly wrong.
        """
        assert accelerated(Pareto(alpha=2.0, xm=1.0), "cdf")
        for dist in (
            LogNormal(mu=0.0, sigma=1.0),
            StudentT(nu=3.0),
            InverseGamma(alpha=2.0, beta=1.0),
            BetaPrime(a=2.0, b=3.0, s=1.0),
        ):
            assert not accelerated(dist, "cdf")

    def test_an_unknown_method_is_not_claimed_as_accelerated(self) -> None:
        assert not accelerated(Pareto(alpha=2.0, xm=1.0), "nonsense")


class TestInputHandling:
    def test_it_accepts_any_sequence(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        expected = [dist.cdf(v) for v in (1.0, 2.0, 4.0)]
        for form in ([1.0, 2.0, 4.0], (1.0, 2.0, 4.0), np.array([1.0, 2.0, 4.0])):
            assert np.array_equal(cdf(dist, form), np.array(expected))

    def test_integers_are_accepted(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        assert np.array_equal(cdf(dist, [1, 2, 4]), cdf(dist, [1.0, 2.0, 4.0]))

    def test_an_empty_input_gives_an_empty_result(self) -> None:
        assert len(cdf(Pareto(alpha=2.0, xm=1.0), [])) == 0

    def test_a_single_point_works(self) -> None:
        dist = Pareto(alpha=2.0, xm=1.0)
        assert cdf(dist, [3.0])[0] == dist.cdf(3.0)

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5, float("nan")])
    def test_a_probability_outside_the_unit_interval_is_rejected(
        self, bad: float
    ) -> None:
        """Validated up front rather than partway through the loop.

        The scalar method raises when it reaches the offending value, having
        already computed everything before it. Checking first means the work is
        not started, and the message can name what was wrong.
        """
        with pytest.raises(ValueError, match="must be in"):
            ppf(Pareto(alpha=2.0, xm=1.0), [0.5, bad, 0.9])

    def test_the_error_names_the_offending_value_and_counts_the_rest(self) -> None:
        with pytest.raises(ValueError, match="2 other"):
            ppf(Pareto(alpha=2.0, xm=1.0), [0.5, 1.5, 2.5, 3.5])

    def test_a_valid_input_is_not_rejected(self) -> None:
        assert len(ppf(Pareto(alpha=2.0, xm=1.0), [1e-15, 0.5, 1 - 1e-15])) == 3


class TestConsistencyWithTheRestOfTheLibrary:
    @pytest.mark.parametrize(("dist", "points"), ACCELERATED)
    def test_the_two_probabilities_still_bracket_the_quantile(
        self, dist: object, points: list[float]
    ) -> None:
        del points
        quantiles = ppf(dist, [0.1, 0.5, 0.9])
        assert np.all(np.diff(quantiles) > 0)
        recovered = cdf(dist, quantiles)
        assert np.allclose(recovered, [0.1, 0.5, 0.9], rtol=1e-9)

    @pytest.mark.parametrize(("dist", "points"), ACCELERATED)
    def test_the_density_is_never_negative(
        self, dist: object, points: list[float]
    ) -> None:
        assert np.all(pdf(dist, points) >= 0.0)

    @pytest.mark.parametrize(("dist", "points"), ACCELERATED)
    def test_probabilities_stay_in_the_unit_interval(
        self, dist: object, points: list[float]
    ) -> None:
        for function in (cdf, sf):
            values = function(dist, points)
            assert np.all(values >= 0.0)
            assert np.all(values <= 1.0)
