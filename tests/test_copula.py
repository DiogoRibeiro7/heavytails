"""Copulas, and what they say about joint extremes.

Three things are checked, and the third is the one that matters:

**The margins are uniform.** That is the definition of a copula, and a sampler
that gets the dependence right and the margins wrong is not one.

**The sampler produces the copula it claims.** For Gumbel and Galambos there is
a closed-form distribution function to compare the empirical one against, which
turns "the sample looks dependent" into an actual check. The Marshall-Olkin
stable construction and the conditional inversion are both easy to get subtly
wrong and neither would look wrong in a scatter plot.

**The empirical estimator is biased, and the tests say by how much.** On a
Gaussian copula whose true tail dependence is exactly zero it reports 0.26 at
the 0.99 level. A test that merely checked the estimator "recovers" tail
dependence on a t copula would pass while hiding that, so there is a test
asserting the bias exists instead.
"""

from __future__ import annotations

from itertools import pairwise
import math

import pytest

from heavytails.copula import (
    GalambosCopula,
    GaussianCopula,
    GumbelCopula,
    StudentTCopula,
    empirical_tail_dependence,
)
from heavytails.heavy_tails import ParameterError

CORRELATION = [[1.0, 0.6], [0.6, 1.0]]

ALL_COPULAS = [
    pytest.param(GaussianCopula(CORRELATION), id="Gaussian"),
    pytest.param(StudentTCopula(nu=4.0, correlation=CORRELATION), id="StudentT"),
    pytest.param(GumbelCopula(theta=2.0), id="Gumbel"),
    pytest.param(GalambosCopula(theta=1.5), id="Galambos"),
]

WITH_CDF = [
    pytest.param(GumbelCopula(theta=1.5), id="Gumbel-1.5"),
    pytest.param(GumbelCopula(theta=4.0), id="Gumbel-4"),
    pytest.param(GalambosCopula(theta=1.0), id="Galambos-1"),
    pytest.param(GalambosCopula(theta=3.0), id="Galambos-3"),
]


class TestTheMarginsAreUniform:
    """The defining property, so it is checked for every sampler."""

    @pytest.mark.parametrize("copula", ALL_COPULAS)
    def test_each_component_is_uniform(self, copula: object) -> None:
        draws = copula.rvs(30_000, seed=1)  # type: ignore[attr-defined]
        for component in (0, 1):
            values = [draw[component] for draw in draws]
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9):
                share = sum(1 for v in values if v <= quantile) / len(values)
                assert share == pytest.approx(quantile, abs=0.012)

    @pytest.mark.parametrize("copula", ALL_COPULAS)
    def test_every_draw_is_inside_the_unit_square(self, copula: object) -> None:
        for draw in copula.rvs(2_000, seed=2):  # type: ignore[attr-defined]
            assert len(draw) == 2
            assert all(0.0 < value < 1.0 for value in draw)


class TestTheSamplerMatchesTheDistributionFunction:
    """Where there is a closed form, the sample is checked against it.

    This is what separates a correct sampler from one that merely produces
    dependent-looking points.
    """

    @pytest.mark.parametrize("copula", WITH_CDF)
    def test_the_empirical_distribution_function_agrees(self, copula: object) -> None:
        draws = copula.rvs(40_000, seed=3)  # type: ignore[attr-defined]
        for u in (0.2, 0.5, 0.8):
            for v in (0.2, 0.5, 0.8):
                empirical = sum(1 for a, b in draws if a <= u and b <= v) / len(draws)
                assert empirical == pytest.approx(
                    copula.cdf([u, v]),
                    abs=0.012,  # type: ignore[attr-defined]
                )

    def test_independence_is_reproduced_exactly(self) -> None:
        """``theta = 1`` is the independence copula, where ``C(u,v) = uv``."""
        copula = GumbelCopula(theta=1.0)
        for u, v in ((0.3, 0.7), (0.5, 0.5), (0.9, 0.2)):
            assert copula.cdf([u, v]) == pytest.approx(u * v, rel=1e-12)
            assert copula.pdf([u, v]) == pytest.approx(1.0, rel=1e-12)


class TestTheDensitiesAreTheDerivativesTheyClaim:
    """Each density is checked against a finite difference of its own cdf.

    A copula density is a mixed second partial derivative written out by hand,
    which is exactly the kind of algebra that produces something plausible and
    wrong. Points where the density is very small are excluded: the second
    difference loses all its precision there and at one of them it comes out
    negative, so the check would be measuring the arithmetic.
    """

    @pytest.mark.parametrize("copula", WITH_CDF)
    @pytest.mark.parametrize(
        ("u", "v"), [(0.3, 0.4), (0.5, 0.5), (0.7, 0.6), (0.9, 0.85)]
    )
    def test_the_density_matches_a_finite_difference(
        self, copula: object, u: float, v: float
    ) -> None:
        step = 1e-5
        numeric = (
            copula.cdf([u + step, v + step])  # type: ignore[attr-defined]
            - copula.cdf([u + step, v - step])  # type: ignore[attr-defined]
            - copula.cdf([u - step, v + step])  # type: ignore[attr-defined]
            + copula.cdf([u - step, v - step])  # type: ignore[attr-defined]
        ) / (4 * step * step)
        assert copula.pdf([u, v]) == pytest.approx(numeric, rel=1e-5)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("copula", ALL_COPULAS)
    def test_the_density_integrates_to_one(self, copula: object) -> None:
        """Over the unit square, on a midpoint grid avoiding the corners."""
        steps = 300
        width = 1.0 / steps
        total = sum(
            copula.pdf([(i + 0.5) * width, (j + 0.5) * width])  # type: ignore[attr-defined]
            * width
            * width
            for i in range(steps)
            for j in range(steps)
        )
        assert total == pytest.approx(1.0, abs=0.05)


class TestTailDependence:
    @pytest.mark.parametrize(
        ("copula", "expected_upper", "expected_lower"),
        [
            (GaussianCopula(CORRELATION), 0.0, 0.0),
            (GumbelCopula(theta=2.0), 2 - math.sqrt(2), 0.0),
            (GalambosCopula(theta=1.0), 0.5, 0.0),
        ],
    )
    def test_the_closed_forms(
        self, copula: object, expected_upper: float, expected_lower: float
    ) -> None:
        result = copula.tail_dependence()  # type: ignore[attr-defined]
        assert result["upper"] == pytest.approx(expected_upper, rel=1e-12)
        assert result["lower"] == pytest.approx(expected_lower, abs=1e-15)

    def test_the_gaussian_has_none_at_any_correlation(self) -> None:
        """The property that makes it the wrong default for joint risk.

        Correlation 0.99 and still asymptotically independent.
        """
        for rho in (0.0, 0.5, 0.9, 0.99):
            copula = GaussianCopula([[1.0, rho], [rho, 1.0]])
            assert copula.tail_dependence() == {"lower": 0.0, "upper": 0.0}

    def test_the_student_t_has_it_at_zero_correlation(self) -> None:
        """The contrast, and the reason to prefer the t."""
        copula = StudentTCopula(nu=4.0, correlation=[[1.0, 0.0], [0.0, 1.0]])
        assert copula.tail_dependence()["upper"] > 0.07

    def test_it_is_symmetric_for_the_student_t(self) -> None:
        copula = StudentTCopula(nu=3.0, correlation=CORRELATION)
        result = copula.tail_dependence()
        assert result["upper"] == result["lower"]

    def test_the_extreme_value_copulas_are_upper_only(self) -> None:
        """Joint crashes without joint booms, which is the asymmetry."""
        for copula in (GumbelCopula(theta=3.0), GalambosCopula(theta=2.0)):
            result = copula.tail_dependence()
            assert result["upper"] > 0.5
            assert result["lower"] == 0.0

    def test_gumbel_strengthens_with_theta(self) -> None:
        values = [
            GumbelCopula(theta=theta).tail_dependence()["upper"]
            for theta in (1.0, 1.5, 2.0, 5.0, 20.0)
        ]
        assert all(b > a for a, b in pairwise(values))
        assert values[0] == 0.0

    def test_galambos_strengthens_with_theta(self) -> None:
        values = [
            GalambosCopula(theta=theta).tail_dependence()["upper"]
            for theta in (0.2, 0.5, 1.0, 3.0, 10.0)
        ]
        assert all(b > a for a, b in pairwise(values))

    def test_the_two_extreme_value_families_can_agree_on_the_coefficient(
        self,
    ) -> None:
        """Which is the point of having both: the coefficient does not
        determine the copula.

        Gumbel at ``theta`` and Galambos at the matching parameter share an
        upper coefficient while putting different mass elsewhere.
        """
        gumbel = GumbelCopula(theta=2.0)
        target = gumbel.tail_dependence()["upper"]
        # 2**(-1/theta) = target  =>  theta = -1 / log2(target)
        galambos = GalambosCopula(theta=-1.0 / math.log2(target))
        assert galambos.tail_dependence()["upper"] == pytest.approx(target, rel=1e-12)
        assert gumbel.cdf([0.3, 0.6]) != pytest.approx(
            galambos.cdf([0.3, 0.6]), rel=1e-6
        )


class TestTheEmpiricalEstimator:
    def test_it_is_biased_upwards_where_the_truth_is_zero(self) -> None:
        """The caveat that matters most, asserted rather than only written down.

        A Gaussian copula has no tail dependence at all. The estimator reports
        a quarter of it at the 0.99 level on sixty thousand pairs, and a test
        that only checked it "recovers" dependence on a t copula would pass
        while hiding that.
        """
        draws = GaussianCopula([[1.0, 0.7], [0.7, 1.0]]).rvs(60_000, seed=1)
        result = empirical_tail_dependence(
            [a for a, _ in draws], [b for _, b in draws], level=0.99
        )
        assert result["upper"] > 0.15

    def test_the_bias_falls_with_the_level_without_reaching_zero(self) -> None:
        """It converges, far too slowly to be useful for the question people
        ask of it."""
        draws = GaussianCopula([[1.0, 0.7], [0.7, 1.0]]).rvs(200_000, seed=2)
        x = [a for a, _ in draws]
        y = [b for _, b in draws]
        estimates = [
            empirical_tail_dependence(x, y, level=level)["upper"]
            for level in (0.90, 0.95, 0.99, 0.999)
        ]
        assert all(b < a for a, b in pairwise(estimates))
        assert estimates[-1] > 0.05

    def test_it_ranks_copulas_correctly_at_a_fixed_level(self) -> None:
        """What it is actually good for: comparing, not measuring."""
        level = 0.99
        estimates = {}
        for name, copula in (
            ("gaussian", GaussianCopula(CORRELATION)),
            ("t", StudentTCopula(nu=3.0, correlation=CORRELATION)),
            ("gumbel", GumbelCopula(theta=3.0)),
        ):
            draws = copula.rvs(60_000, seed=3)
            estimates[name] = empirical_tail_dependence(
                [a for a, _ in draws], [b for _, b in draws], level=level
            )["upper"]
        assert estimates["gaussian"] < estimates["t"] < estimates["gumbel"]

    def test_it_is_invariant_to_monotone_rescaling(self) -> None:
        """It works on ranks, so it is a statement about the copula and not
        about the margins."""
        draws = StudentTCopula(nu=4.0, correlation=CORRELATION).rvs(20_000, seed=4)
        x = [a for a, _ in draws]
        y = [b for _, b in draws]
        plain = empirical_tail_dependence(x, y, level=0.95)
        rescaled = empirical_tail_dependence(
            [math.exp(10 * a) for a in x], [b**3 for b in y], level=0.95
        )
        assert plain["upper"] == pytest.approx(rescaled["upper"], rel=1e-12)

    def test_it_reports_how_many_pairs_the_estimate_rests_on(self) -> None:
        draws = GumbelCopula(theta=2.0).rvs(10_000, seed=5)
        result = empirical_tail_dependence(
            [a for a, _ in draws], [b for _, b in draws], level=0.99
        )
        assert result["n"] == 10_000
        assert 80 <= result["n_upper_exceedances"] <= 120

    def test_a_level_leaving_too_few_pairs_is_refused(self) -> None:
        """Rather than returning a ratio of two very small integers."""
        draws = GumbelCopula(theta=2.0).rvs(100, seed=6)
        with pytest.raises(ValueError, match="too few"):
            empirical_tail_dependence(
                [a for a, _ in draws], [b for _, b in draws], level=0.999
            )

    def test_mismatched_or_short_series_are_refused(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            empirical_tail_dependence([1.0, 2.0], [1.0])
        with pytest.raises(ValueError, match="at least 20"):
            empirical_tail_dependence([1.0] * 10, [2.0] * 10)

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5])
    def test_a_bad_level_is_refused(self, level: float) -> None:
        with pytest.raises(ValueError, match="level must be"):
            empirical_tail_dependence(
                [float(i) for i in range(50)],
                [float(i) for i in range(50)],
                level=level,
            )


class TestValidation:
    @pytest.mark.parametrize(
        "matrix",
        [[[1.0, 2.0], [2.0, 1.0]], [[2.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.4, 1.0]]],
    )
    def test_a_bad_correlation_matrix_is_refused(
        self, matrix: list[list[float]]
    ) -> None:
        with pytest.raises(ParameterError):
            GaussianCopula(matrix)

    def test_the_diagonal_must_be_ones(self) -> None:
        with pytest.raises(ParameterError, match="ones on the diagonal"):
            GaussianCopula([[2.0, 0.0], [0.0, 2.0]])

    @pytest.mark.parametrize("theta", [0.5, 0.0, -1.0])
    def test_gumbel_refuses_theta_below_one(self, theta: float) -> None:
        with pytest.raises(ParameterError, match="theta >= 1"):
            GumbelCopula(theta=theta)

    @pytest.mark.parametrize("theta", [0.0, -1.0])
    def test_galambos_refuses_a_non_positive_theta(self, theta: float) -> None:
        with pytest.raises(ParameterError, match="theta > 0"):
            GalambosCopula(theta=theta)

    @pytest.mark.parametrize("nu", [0.0, -1.0, float("inf")])
    def test_the_t_copula_refuses_a_bad_degrees_of_freedom(self, nu: float) -> None:
        with pytest.raises(ParameterError, match="nu > 0"):
            StudentTCopula(nu=nu, correlation=CORRELATION)

    @pytest.mark.parametrize("point", [[0.0, 0.5], [0.5, 1.0], [-0.1, 0.5], [0.5]])
    def test_a_point_outside_the_open_unit_square_is_refused(
        self, point: list[float]
    ) -> None:
        with pytest.raises(ValueError):
            GumbelCopula(theta=2.0).pdf(point)

    @pytest.mark.parametrize("copula", ALL_COPULAS)
    def test_a_bad_draw_count_is_refused(self, copula: object) -> None:
        with pytest.raises(ValueError):
            copula.rvs(0)  # type: ignore[attr-defined]

    @pytest.mark.parametrize("copula", ALL_COPULAS)
    def test_the_seed_makes_sampling_reproducible(self, copula: object) -> None:
        assert copula.rvs(200, seed=7) == copula.rvs(200, seed=7)  # type: ignore[attr-defined]
