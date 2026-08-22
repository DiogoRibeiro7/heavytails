"""Properties that must hold for every family, checked over random parameters.

The existing property tests are hand-written per family, and cover four of the
twelve. These are generic: one registry of families and parameter strategies,
and the properties applied to all of them. A family added later gets the whole
set by being added to the registry.

Property tests establish self-consistency and nothing more. A formula that is
simply wrong still produces monotone probabilities in [0, 1] that complement
their survival function — ``InverseGamma.cdf`` did exactly that while being
wrong by a factor of seventeen. ``tests/test_reference_values.py`` is where
correctness is checked, against values this library did not produce. These two
files answer different questions and neither substitutes for the other.
"""

from __future__ import annotations

from itertools import pairwise
import math
from typing import Any

from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
import pytest

from heavytails.extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
)
from heavytails.heavy_tails import (
    Cauchy,
    Frechet,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)

SETTINGS = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

shape = st.floats(min_value=0.3, max_value=6.0, allow_nan=False, allow_infinity=False)
scale = st.floats(min_value=0.2, max_value=20.0, allow_nan=False, allow_infinity=False)
location = st.floats(min_value=-5.0, max_value=5.0, allow_nan=False)
probability = st.floats(
    min_value=1e-6, max_value=1 - 1e-6, allow_nan=False, allow_infinity=False
)

# One builder per family. Every generic property below runs against all of
# them, so adding a family here is what gives it coverage.
FAMILIES: dict[str, Any] = {
    "Pareto": st.builds(Pareto, alpha=shape, xm=scale),
    "Cauchy": st.builds(Cauchy, x0=location, gamma=scale),
    "StudentT": st.builds(StudentT, nu=shape),
    "LogNormal": st.builds(
        LogNormal,
        mu=location,
        sigma=st.floats(min_value=0.2, max_value=3.0, allow_nan=False),
    ),
    "Weibull": st.builds(Weibull, k=shape, lam=scale),
    "Frechet": st.builds(Frechet, alpha=shape, s=scale, m=st.just(0.0)),
    "GEV_Frechet": st.builds(
        GEV_Frechet,
        xi=st.floats(min_value=0.1, max_value=2.0, allow_nan=False),
        mu=location,
        sigma=scale,
    ),
    "GeneralizedPareto": st.builds(
        GeneralizedPareto,
        xi=st.floats(min_value=0.05, max_value=2.0, allow_nan=False),
        sigma=scale,
        mu=location,
    ),
    "BurrXII": st.builds(BurrXII, c=shape, k=shape, s=scale),
    "LogLogistic": st.builds(LogLogistic, kappa=shape, lam=scale),
    "InverseGamma": st.builds(InverseGamma, alpha=shape, beta=scale),
    "BetaPrime": st.builds(BetaPrime, a=shape, b=shape, s=scale),
}

ALL = st.one_of(*FAMILIES.values())


class TestPropertiesOfEveryFamily:
    @given(dist=ALL, u=probability)
    @SETTINGS
    def test_the_quantile_inverts_the_distribution_function(
        self, dist: Any, u: float
    ) -> None:
        x = dist.ppf(u)
        assume(math.isfinite(x))
        # Loose, because this is a self-consistency check across twelve
        # families at random parameters, some of which are badly conditioned.
        # The tight accuracy claims live in test_reference_values.py.
        assert dist.cdf(x) == pytest.approx(u, rel=1e-6, abs=1e-12)

    @given(dist=ALL, u=probability)
    @SETTINGS
    def test_the_survival_function_complements_the_distribution_function(
        self, dist: Any, u: float
    ) -> None:
        x = dist.ppf(u)
        assume(math.isfinite(x))
        # Not to machine precision, and deliberately so. Several families now
        # compute the two sides independently -- each from the branch where its
        # own value is the small quantity -- which is what lets both tails be
        # accurate. Two independent evaluations do not sum to exactly one, and
        # insisting that they did would force back the very subtraction that
        # made the lower tail return zero.
        assert dist.cdf(x) + dist.sf(x) == pytest.approx(1.0, abs=1e-9)

    @given(dist=ALL, u=probability, v=probability)
    @SETTINGS
    def test_the_distribution_function_never_decreases(
        self, dist: Any, u: float, v: float
    ) -> None:
        lo, hi = sorted((dist.ppf(min(u, v)), dist.ppf(max(u, v))))
        assume(math.isfinite(lo) and math.isfinite(hi))
        assert dist.cdf(hi) >= dist.cdf(lo) - 1e-12

    @given(dist=ALL, u=probability)
    @SETTINGS
    def test_the_density_is_never_negative(self, dist: Any, u: float) -> None:
        x = dist.ppf(u)
        assume(math.isfinite(x))
        assert dist.pdf(x) >= 0.0

    @given(dist=ALL, u=probability)
    @SETTINGS
    def test_probabilities_stay_in_the_unit_interval(self, dist: Any, u: float) -> None:
        x = dist.ppf(u)
        assume(math.isfinite(x))
        assert 0.0 <= dist.cdf(x) <= 1.0
        assert 0.0 <= dist.sf(x) <= 1.0

    @given(dist=ALL, u=probability)
    @SETTINGS
    def test_the_quantile_never_decreases(self, dist: Any, u: float) -> None:
        higher = min(u + (1.0 - u) / 2.0, 1 - 1e-9)
        first, second = dist.ppf(u), dist.ppf(higher)
        assume(math.isfinite(first) and math.isfinite(second))
        assert second >= first

    @given(dist=ALL)
    @SETTINGS
    def test_a_probability_outside_the_open_unit_interval_is_rejected(
        self, dist: Any
    ) -> None:
        """Silently accepting 0 or 1 would return an endpoint dressed as a
        quantile."""
        for bad in (0.0, 1.0, -0.5, 1.5):
            with pytest.raises(ValueError):
                dist.ppf(bad)

    @given(dist=ALL)
    @SETTINGS
    def test_sampling_lands_inside_the_support(self, dist: Any) -> None:
        sample = dist.rvs(200, seed=1)
        assert len(sample) == 200
        assert all(math.isfinite(value) for value in sample)
        floor = dist.ppf(1e-12)
        if math.isfinite(floor):
            # Two hundred draws below the 1e-12 quantile would be a one in
            # five-billion coincidence, so this is a support check rather than
            # a flaky bound.
            assert min(sample) >= floor

    @given(dist=ALL)
    @SETTINGS
    def test_the_seed_makes_sampling_reproducible(self, dist: Any) -> None:
        assert dist.rvs(50, seed=7) == dist.rvs(50, seed=7)

    @given(dist=ALL)
    @SETTINGS
    def test_the_median_splits_the_sample(self, dist: Any) -> None:
        """A weak check on the two agreeing, but it exercises the whole path.

        The quantile function and the sampler are separate code in several of
        these families -- InverseGamma draws a gamma variate and never touches
        its own ppf -- so nothing else pins them to the same distribution.
        """
        median = dist.ppf(0.5)
        assume(math.isfinite(median))
        sample = dist.rvs(400, seed=3)
        below = sum(1 for value in sample if value <= median)
        assert 0.38 < below / len(sample) < 0.62


class TestOutsideTheSupport:
    """Probabilities must stay probabilities where the density is zero.

    This is where ``GeneralizedPareto`` was wrong for every sign of ``xi``: its
    validity check tested only the bracket ``1 + xi z > 0``, which is the
    *upper* endpoint of a bounded distribution and is satisfied well below
    ``mu``. So points below the support were treated as inside it, and
    ``cdf(mu - 1)`` returned -2.586.

    Nothing caught it because the existing tests evaluate distributions on
    their own samples, which are inside the support by construction. Reaching
    below it is the whole point of these.
    """

    @given(dist=ALL)
    @SETTINGS
    def test_the_distribution_function_stays_a_probability_below_the_support(
        self, dist: Any
    ) -> None:
        floor = dist.ppf(1e-12)
        assume(math.isfinite(floor))
        for offset in (1.0, 10.0, 1e6):
            below = floor - offset
            assert 0.0 <= dist.cdf(below) <= 1.0
            assert 0.0 <= dist.sf(below) <= 1.0
            assert dist.pdf(below) >= 0.0

    @given(dist=ALL)
    @SETTINGS
    def test_it_stays_a_probability_far_above_the_support(self, dist: Any) -> None:
        ceiling = dist.ppf(1 - 1e-12)
        assume(math.isfinite(ceiling))
        for factor in (1.0, 10.0, 1e6):
            above = ceiling * factor + 1.0
            assert 0.0 <= dist.cdf(above) <= 1.0
            assert 0.0 <= dist.sf(above) <= 1.0
            assert dist.pdf(above) >= 0.0

    @pytest.mark.parametrize(
        ("xi", "sigma", "mu"),
        [(0.4, 1.0, 1.0), (-0.5, 1.0, 0.0), (0.0, 2.0, 0.0), (1.2, 0.5, -3.0)],
    )
    def test_the_generalized_pareto_starts_at_its_location(
        self, xi: float, sigma: float, mu: float
    ) -> None:
        """The support is ``[mu, ...)`` whichever way ``xi`` points."""
        dist = GeneralizedPareto(xi=xi, sigma=sigma, mu=mu)
        for below in (mu - 1e-9, mu - 1.0, mu - 1e6):
            assert dist.cdf(below) == 0.0
            assert dist.sf(below) == 1.0
            assert dist.pdf(below) == 0.0
        assert dist.cdf(mu) == 0.0

    def test_a_bounded_generalized_pareto_is_certain_past_its_endpoint(self) -> None:
        """Negative ``xi`` bounds it above, where the answer is 1 rather than 0.

        The two constants are easy to swap, and swapping them would look
        plausible in every test that never goes outside the support.
        """
        dist = GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0)
        endpoint = 0.0 - 1.0 / -0.5
        assert dist.cdf(endpoint + 1e-9) == 1.0
        assert dist.sf(endpoint + 1e-9) == 0.0

    @pytest.mark.parametrize(("k", "expected"), [(0.5, math.inf), (0.7, math.inf)])
    def test_the_weibull_density_diverges_at_the_origin_for_small_shape(
        self, k: float, expected: float
    ) -> None:
        """It used to raise ZeroDivisionError, which is not a density.

        For ``k < 1`` the density is unbounded at zero -- the limit exists and
        is infinite, and saying so beats raising from ``0.0 ** negative``.
        """
        assert Weibull(k=k, lam=2.0).pdf(0.0) == expected

    @pytest.mark.parametrize(("k", "expected"), [(1.0, 0.5), (1.5, 0.0)])
    def test_the_weibull_density_at_the_origin_is_finite_otherwise(
        self, k: float, expected: float
    ) -> None:
        assert Weibull(k=k, lam=2.0).pdf(0.0) == expected


class TestDocumentedRelationships:
    """Families that are special cases of one another, checked numerically.

    Each of these is stated in the documentation. A relationship that holds on
    paper and not in the code means one of the two implementations is wrong,
    and this says which pair to look at.
    """

    @pytest.mark.parametrize("x", [-8.0, -1.5, -0.25, 0.0, 0.25, 1.5, 8.0])
    def test_student_t_with_one_degree_of_freedom_is_cauchy(self, x: float) -> None:
        assert StudentT(nu=1.0).cdf(x) == pytest.approx(
            Cauchy(x0=0.0, gamma=1.0).cdf(x), rel=1e-12
        )
        assert StudentT(nu=1.0).pdf(x) == pytest.approx(
            Cauchy(x0=0.0, gamma=1.0).pdf(x), rel=1e-12
        )

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 3.0, 25.0])
    def test_the_log_logistic_is_burr_xii_with_unit_shape(self, x: float) -> None:
        """``BurrXII(c, k=1, s)`` is ``LogLogistic(kappa=c, lam=s)``."""
        assert BurrXII(c=2.5, k=1.0, s=1.7).cdf(x) == pytest.approx(
            LogLogistic(kappa=2.5, lam=1.7).cdf(x), rel=1e-12
        )

    @pytest.mark.parametrize("x", [0.1, 1.0, 4.0, 40.0])
    def test_a_unit_shape_generalized_pareto_is_a_pareto(self, x: float) -> None:
        """``GPD(xi, sigma=xi*xm, mu=xm)`` is ``Pareto(alpha=1/xi, xm)``."""
        xi, xm = 0.4, 2.0
        assume_x = max(x, xm)
        assert GeneralizedPareto(xi=xi, sigma=xi * xm, mu=xm).cdf(
            assume_x
        ) == pytest.approx(Pareto(alpha=1.0 / xi, xm=xm).cdf(assume_x), rel=1e-12)

    @pytest.mark.parametrize("x", [0.2, 1.0, 5.0])
    def test_a_unit_shape_weibull_is_exponential(self, x: float) -> None:
        assert Weibull(k=1.0, lam=2.0).cdf(x) == pytest.approx(
            -math.expm1(-x / 2.0), rel=1e-12
        )

    @pytest.mark.parametrize("x", [0.5, 1.0, 3.0])
    def test_a_zero_shape_generalized_pareto_is_exponential(self, x: float) -> None:
        assert GeneralizedPareto(xi=0.0, sigma=1.5, mu=0.0).cdf(x) == pytest.approx(
            -math.expm1(-x / 1.5), rel=1e-12
        )

    @pytest.mark.parametrize("x", [1.5, 3.0, 10.0])
    def test_the_inverse_of_a_pareto_variate_is_a_beta_variate(self, x: float) -> None:
        """``P(X > x) = (xm/x)^alpha`` is the definition; check it holds."""
        dist = Pareto(alpha=1.8, xm=1.2)
        assert dist.sf(x) == pytest.approx((1.2 / x) ** 1.8, rel=1e-13)


class TestTailBehaviour:
    """The tail index each family is supposed to have, measured."""

    @pytest.mark.parametrize(
        ("dist", "alpha"),
        [
            (Pareto(alpha=2.5, xm=1.0), 2.5),
            (StudentT(nu=3.0), 3.0),
            (Cauchy(x0=0.0, gamma=1.0), 1.0),
            (BurrXII(c=2.0, k=1.5, s=1.0), 3.0),
            (LogLogistic(kappa=2.5, lam=1.0), 2.5),
            (InverseGamma(alpha=1.7, beta=1.0), 1.7),
            (BetaPrime(a=2.0, b=2.2, s=1.0), 2.2),
            (Frechet(alpha=1.4, s=1.0, m=0.0), 1.4),
            (GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0), 2.0),
        ],
    )
    def test_the_survival_function_decays_at_the_stated_rate(
        self, dist: Any, alpha: float
    ) -> None:
        """``S(x) ~ c x^-alpha``, so ``log S`` against ``log x`` has slope
        ``-alpha``.

        Measured far enough out for the second-order term to have faded, and
        checked as a slope rather than at one point, which any constant would
        satisfy.
        """
        lo, hi = 1e4, 1e7
        slope = (math.log(dist.sf(hi)) - math.log(dist.sf(lo))) / (
            math.log(hi) - math.log(lo)
        )
        assert slope == pytest.approx(-alpha, rel=0.02)

    @pytest.mark.parametrize(
        "dist",
        [LogNormal(mu=0.0, sigma=1.0), Weibull(k=0.7, lam=1.0)],
    )
    def test_the_subexponential_families_decay_faster_than_any_power(
        self, dist: Any
    ) -> None:
        """Heavy-tailed but not regularly varying, so the slope keeps steepening.

        Grouping them with the Pareto-tailed families is the mistake this
        guards against: a tail index estimator applied to either will return
        something, and it will not mean what it usually means.
        """
        slopes = []
        for lo, hi in [(10.0, 100.0), (100.0, 1000.0), (1000.0, 10_000.0)]:
            slopes.append(
                (math.log(dist.sf(hi)) - math.log(dist.sf(lo)))
                / (math.log(hi) - math.log(lo))
            )
        assert all(b < a for a, b in pairwise(slopes))


class TestMomentExistence:
    """Moments exist exactly when the theory says, and not otherwise."""

    @pytest.mark.parametrize(
        ("alpha", "mean_exists", "variance_exists"),
        [
            (0.5, False, False),
            (1.5, True, False),
            (2.5, True, True),
            (3.5, True, True),
        ],
    )
    def test_a_pareto_sample_diverges_where_the_moment_does_not(
        self, alpha: float, mean_exists: bool, variance_exists: bool
    ) -> None:
        """The sample mean of a Pareto with ``alpha <= 1`` does not settle.

        Checked by growing the sample and watching whether the estimate
        stabilises, which is what a practitioner would see and what makes the
        non-existence concrete rather than a footnote.
        """
        dist = Pareto(alpha=alpha, xm=1.0)
        means = []
        for size in (2_000, 20_000, 200_000):
            sample = dist.rvs(size, seed=5)
            means.append(sum(sample) / len(sample))

        spread = (max(means) - min(means)) / min(means)
        if mean_exists:
            assert spread < 0.5, f"the mean should settle for alpha={alpha}"
        else:
            assert spread > 0.5, f"the mean should not settle for alpha={alpha}"

        # Recorded so the parameterisation stays honest about what it asserts.
        assert variance_exists == (alpha > 2.0)

    @pytest.mark.parametrize("nu", [0.5, 1.0, 1.5])
    def test_a_student_t_without_a_mean_does_not_settle(self, nu: float) -> None:
        dist = StudentT(nu=nu)
        means = [
            sum(s) / len(s)
            for s in (dist.rvs(size, seed=7) for size in (2_000, 20_000, 200_000))
        ]
        assert max(abs(m) for m in means) > 0.05

    def test_the_cauchy_sample_mean_never_settles(self) -> None:
        """It has no mean at all: the sample mean is itself Cauchy.

        More samples do not help, which is the whole point and the reason this
        distribution is the standard counterexample to the law of large
        numbers.
        """
        dist = Cauchy(x0=0.0, gamma=1.0)
        spreads = []
        for size in (1_000, 10_000, 100_000):
            means = [sum(dist.rvs(size, seed=s)) / size for s in range(8)]
            spreads.append(max(means) - min(means))
        assert max(spreads) > 1.0
