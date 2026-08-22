"""Every distribution method mirrors the kind of thing it was given.

A number in gives a float out; anything array-like in gives an array out, with
the same shape. That is the whole contract, and these tests hold all twelve
continuous families to it at once rather than trusting each conversion.

The agreement test is the one that matters. Each method is now written once,
against NumPy, and evaluated the same way for one point as for a million --
a 0-d array and an n-d array run the identical expression. So scalar and array
results are compared for *exact* equality here, not to a tolerance. A tolerance
is what the previous arrangement needed, when a hand-written kernel sat beside
each scalar method and the two could drift; there is nothing left to drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from heavytails.extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
)
from heavytails.heavy_tails import (
    RNG,
    Cauchy,
    Frechet,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)

# Parameters chosen to sit inside each support with a real tail, and to cover
# both signs of the GPD shape -- the bounded case has an upper endpoint, which
# is where a mask is easiest to get wrong.
DISTRIBUTIONS = [
    Pareto(alpha=2.0, xm=1.0),
    Cauchy(x0=0.0, gamma=1.0),
    StudentT(nu=3.0),
    LogNormal(mu=0.0, sigma=1.0),
    Weibull(k=0.7, lam=1.0),
    Frechet(alpha=2.0),
    GEV_Frechet(xi=0.5),
    GeneralizedPareto(xi=0.4, sigma=1.0, mu=0.0),
    GeneralizedPareto(xi=-0.25, sigma=1.0, mu=0.0),
    GeneralizedPareto(xi=0.0, sigma=1.0, mu=0.0),
    BurrXII(c=2.0, k=1.5),
    LogLogistic(2.0, 1.0),
    InverseGamma(alpha=2.0, beta=1.0),
    BetaPrime(a=2.0, b=3.0),
]

IDS = [
    f"{type(d).__name__}(xi={d.xi:g})"
    if isinstance(d, GeneralizedPareto)
    else type(d).__name__
    for d in DISTRIBUTIONS
]

# Spans the support and steps outside it on both sides, since the guards are
# what the array rewrite had to reproduce.
POINTS = [-3.0, -1.0, 0.0, 0.25, 1.0, 1.5, 4.0, 100.0, 1e6]
PROBABILITIES = [1e-12, 1e-6, 0.001, 0.25, 0.5, 0.75, 0.999, 1 - 1e-9]


def _grid(method: str) -> list[float]:
    return PROBABILITIES if method == "ppf" else POINTS


@pytest.fixture(params=DISTRIBUTIONS, ids=IDS)
def distribution(request: pytest.FixtureRequest) -> object:
    return request.param


@pytest.mark.parametrize("method", ["pdf", "cdf", "sf", "ppf"])
class TestTheInputKindIsMirrored:
    def test_a_number_gives_a_float(self, distribution: object, method: str) -> None:
        got = getattr(distribution, method)(_grid(method)[3])
        assert isinstance(got, float)
        assert not isinstance(got, np.ndarray)

    def test_a_list_gives_an_array(self, distribution: object, method: str) -> None:
        grid = _grid(method)
        got = getattr(distribution, method)(grid)
        assert isinstance(got, np.ndarray)
        assert got.shape == (len(grid),)

    def test_the_shape_is_preserved(self, distribution: object, method: str) -> None:
        """Including shapes no caller is likely to pass.

        A method that reshapes, ravels, or returns a bare list would pass a
        one-dimensional test and fail here, and a caller evaluating a density
        over a mesh would get back something it could not use.
        """
        grid = np.array(_grid(method)[:8]).reshape(2, 4)
        assert getattr(distribution, method)(grid).shape == (2, 4)

    def test_an_empty_array_gives_an_empty_array(
        self, distribution: object, method: str
    ) -> None:
        got = getattr(distribution, method)(np.array([]))
        assert isinstance(got, np.ndarray)
        assert got.shape == (0,)

    def test_the_array_result_equals_the_scalar_one(
        self, distribution: object, method: str
    ) -> None:
        grid = _grid(method)
        vectorised = np.asarray(getattr(distribution, method)(grid))
        one_at_a_time = np.array([getattr(distribution, method)(v) for v in grid])
        # Exact, not approximate: there is one implementation now.
        np.testing.assert_array_equal(vectorised, one_at_a_time)


class TestProbabilitiesAreChecked:
    """``ppf`` rejects the whole input before computing any of it.

    The scalar version raised on the offending value because it met it while
    working. An array version that checked as it went would return a partial
    answer or raise having already done the work, and would not be able to say
    which value was wrong -- so the check happens up front and names one.
    """

    @pytest.mark.parametrize("bad", [0.0, 1.0, -0.5, 1.5, np.nan])
    def test_a_bad_scalar_is_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match=r"u must be in \(0,1\)"):
            Pareto(alpha=2.0).ppf(bad)

    def test_a_bad_value_anywhere_in_an_array_is_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"u must be in \(0,1\); got 1.5"):
            Pareto(alpha=2.0).ppf([0.1, 0.2, 1.5, 0.4])

    def test_the_message_counts_the_rest(self) -> None:
        with pytest.raises(ValueError, match=r"got 1.5 and 1 other"):
            Pareto(alpha=2.0).ppf([0.1, 1.5, 2.5])


class TestTheQuantileFunctionInverts:
    """``ppf`` and ``cdf`` undo each other, checked on arrays in one call.

    Worth stating for its own sake, and it also catches the failure the mask
    rewrite could plausibly produce: a guard that silently substitutes a value
    rather than masking it shows up here as a quantile that does not come back.
    """

    def test_round_trip(self, distribution: object) -> None:
        u = np.array(PROBABILITIES)
        x = distribution.ppf(u)
        finite = np.isfinite(x)
        back = distribution.cdf(x[finite])
        np.testing.assert_allclose(back, u[finite], rtol=1e-8, atol=1e-12)


class TestTheSurvivalFunctionComplementsTheDistributionFunction:
    def test_they_sum_to_one_where_both_are_resolvable(
        self, distribution: object
    ) -> None:
        """Away from the tails, where neither has lost its digits.

        Only the middle is checked. In the far tail the sum is *supposed* to
        differ from 1 in the last bits, because the small side is computed
        directly rather than by subtraction -- which is the point of computing
        it that way, and is asserted properly in the reference-value tests.
        """
        x = distribution.ppf(np.array([0.1, 0.25, 0.5, 0.75, 0.9]))
        np.testing.assert_allclose(
            distribution.cdf(x) + distribution.sf(x), 1.0, rtol=1e-9
        )


class TestSamplingInABatchMatchesSamplingOneAtATime:
    """The batched sampler must draw the same variates as the loop it replaced.

    It draws the same uniforms -- the generator is untouched -- and inverts
    them with the same expression, so the samples agree. Not always to the
    last bit: NumPy's vectorised ``log`` and ``pow`` round differently from its
    scalar ones, so a handful of draws land one ULP away. The budget below is
    a few ULP, which is far tighter than anything that could hide a wrong
    quantile function and loose enough to survive that.
    """

    @pytest.mark.parametrize(
        "distribution",
        [
            Pareto(alpha=2.0, xm=1.0),
            Cauchy(x0=0.0, gamma=1.0),
            Weibull(k=0.7, lam=1.0),
            Frechet(alpha=2.0),
            GEV_Frechet(xi=0.5),
            GeneralizedPareto(xi=0.4),
            BurrXII(c=2.0, k=1.5),
            LogLogistic(2.0, 1.0),
        ],
        ids=lambda d: type(d).__name__,
    )
    def test_the_variates_agree(self, distribution: object) -> None:
        batched = np.array(distribution.rvs(2000, seed=42))
        rng = RNG(42)
        one_at_a_time = np.array([distribution._rvs_one(rng) for _ in range(2000)])
        np.testing.assert_allclose(batched, one_at_a_time, rtol=8e-16, atol=0.0)

    def test_the_seed_still_reproduces(self) -> None:
        assert Pareto(alpha=2.0).rvs(50, seed=7) == Pareto(alpha=2.0).rvs(50, seed=7)

    def test_a_batch_is_a_list_of_floats(self) -> None:
        """``rvs`` still returns a list.

        The methods mirror their input; ``rvs`` has no input to mirror, and
        returning an array from it would quietly turn ``sample_a + sample_b``
        from concatenation into elementwise addition wherever anyone had
        written that. Changing it is its own decision, not a side effect of
        this one.
        """
        sample = Pareto(alpha=2.0).rvs(5, seed=1)
        assert isinstance(sample, list)
        assert all(isinstance(v, float) for v in sample)
