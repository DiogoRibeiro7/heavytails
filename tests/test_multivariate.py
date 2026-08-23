"""Multivariate heavy-tailed distributions.

The elliptical family offers unusually many exact identities, and the tests use
them rather than settling for "close to a simulation":

- a Cholesky factor times its transpose is the matrix it came from;
- marginals of a multivariate t are multivariate t with the same degrees of
  freedom, and its one-dimensional marginal is the univariate t already in the
  library;
- the Mahalanobis distance of a multivariate t obeys an exact F distribution;
- the multivariate t tends to the multivariate normal as the degrees of freedom
  grow;
- the coefficient of tail dependence has a closed form, which is checked
  against simulation because the formula is the part most easily got wrong.

That last one matters most. A wrong tail-dependence formula produces numbers
that are in [0,1], increase with correlation and decrease with the degrees of
freedom -- everything one would check by eye -- while being the wrong numbers.
"""

from __future__ import annotations

from itertools import pairwise
import math
import statistics

import numpy as np
import pytest

from heavytails import StudentT
from heavytails.heavy_tails import RNG, ParameterError
from heavytails.multivariate import (
    MultivariateNormal,
    MultivariateStudentT,
    cholesky,
    fit_multivariate_t,
    tail_dependence_coefficient,
)

IDENTITY = [[1.0, 0.0], [0.0, 1.0]]
CORRELATED = [[1.0, 0.5], [0.5, 2.0]]


def _matmul_transpose(lower: list[list[float]]) -> list[list[float]]:
    """``L L'``, to check a factorisation against what it factored."""
    size = len(lower)
    return [
        [sum(lower[i][k] * lower[j][k] for k in range(size)) for j in range(size)]
        for i in range(size)
    ]


class TestCholesky:
    @pytest.mark.parametrize(
        "matrix",
        [
            [[4.0]],
            [[4.0, 2.0], [2.0, 5.0]],
            [[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],
            [[1.0, 0.9, 0.8], [0.9, 1.0, 0.9], [0.8, 0.9, 1.0]],
        ],
    )
    def test_the_factor_reproduces_the_matrix(self, matrix: list[list[float]]) -> None:
        product = _matmul_transpose(cholesky(matrix))
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                assert product[i][j] == pytest.approx(value, abs=1e-12)

    def test_the_factor_is_lower_triangular(self) -> None:
        lower = cholesky([[4.0, 2.0], [2.0, 5.0]])
        assert lower[0][1] == 0.0

    def test_a_non_positive_definite_matrix_is_rejected_with_the_pivot(self) -> None:
        """ "Not positive definite" alone is hard to act on; the entry is not."""
        with pytest.raises(ParameterError, match=r"pivot at \(1,1\)"):
            cholesky([[1.0, 2.0], [2.0, 1.0]])

    def test_a_singular_matrix_is_rejected(self) -> None:
        """Perfectly correlated components have no density to speak of."""
        with pytest.raises(ParameterError, match="positive definite"):
            cholesky([[1.0, 1.0], [1.0, 1.0]])

    def test_an_asymmetric_matrix_is_rejected(self) -> None:
        with pytest.raises(ParameterError, match="symmetric"):
            cholesky([[1.0, 0.5], [0.4, 1.0]])

    def test_a_non_square_matrix_is_rejected(self) -> None:
        with pytest.raises(ParameterError, match="square"):
            cholesky([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def test_an_empty_matrix_is_rejected(self) -> None:
        with pytest.raises(ParameterError, match="empty"):
            cholesky([])


class TestTheDensity:
    def test_the_standard_normal_density_at_the_origin(self) -> None:
        normal = MultivariateNormal(mu=[0.0, 0.0], sigma=IDENTITY)
        assert normal.pdf([0.0, 0.0]) == pytest.approx(1 / (2 * math.pi), rel=1e-12)

    def test_one_dimensional_marginals_are_the_univariate_families(self) -> None:
        """The multivariate code must agree with the code already there.

        Not a formality: the normalising constant is the part of a multivariate
        density that is easy to get subtly wrong, and in one dimension it has a
        known answer.
        """
        t = MultivariateStudentT(nu=4.0, mu=[0.0], sigma=[[1.0]])
        for x in (-3.0, -0.5, 0.0, 0.5, 3.0):
            assert t.pdf([x]) == pytest.approx(StudentT(nu=4.0).pdf(x), rel=1e-12)

    @pytest.mark.parametrize("nu", [2.5, 5.0, 30.0])
    def test_the_density_integrates_to_one(self, nu: float) -> None:
        """Quadrature over a grid wide enough to hold the mass.

        The heavier the tail the more escapes the grid, so the tolerance
        follows the degrees of freedom rather than being one loose number that
        covers the worst case.
        """
        t = MultivariateStudentT(nu=nu, mu=[0.0, 0.0], sigma=IDENTITY)
        limit, steps = 60.0, 400
        width = 2 * limit / steps
        total = sum(
            t.pdf(
                [
                    -limit + (i + 0.5) * width,
                    -limit + (j + 0.5) * width,
                ]
            )
            * width
            * width
            for i in range(steps)
            for j in range(steps)
        )
        assert total == pytest.approx(1.0, abs=0.02)

    def test_the_log_density_survives_where_the_density_underflows(self) -> None:
        """A point can be far enough out to underflow and still be of interest.

        This is why the log is the primitive rather than something derived.
        """
        normal = MultivariateNormal(
            mu=[0.0] * 5,
            sigma=[[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)],
        )
        far = [40.0] * 5
        assert normal.pdf(far) == 0.0
        assert math.isfinite(normal.logpdf(far))
        assert normal.logpdf(far) < -3000.0

    def test_the_density_falls_away_from_the_centre(self) -> None:
        t = MultivariateStudentT(nu=4.0, mu=[1.0, -1.0], sigma=CORRELATED)
        values = [t.pdf([1.0 + d, -1.0]) for d in (0.0, 0.5, 2.0, 10.0)]
        assert all(b < a for a, b in pairwise(values))

    def test_the_scale_matrix_is_not_the_covariance(self) -> None:
        """A confusion the class is explicit about, so the test is too."""
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        assert t.covariance()[0][0] == pytest.approx(5.0 / 3.0)

    @pytest.mark.parametrize("nu", [0.5, 1.0, 2.0])
    def test_the_covariance_is_refused_where_it_does_not_exist(self, nu: float) -> None:
        t = MultivariateStudentT(nu=nu, mu=[0.0, 0.0], sigma=IDENTITY)
        with pytest.raises(ValueError, match="only for nu > 2"):
            t.covariance()


class TestMahalanobis:
    def test_it_is_zero_at_the_centre(self) -> None:
        t = MultivariateStudentT(nu=4.0, mu=[1.0, -1.0], sigma=CORRELATED)
        assert t.mahalanobis([1.0, -1.0]) == pytest.approx(0.0, abs=1e-12)

    def test_it_matches_the_explicit_quadratic_form(self) -> None:
        """Checked against the inverse computed by hand, for a 2x2 where that
        is easy: the point of the Cholesky route is speed and conditioning, not
        a different answer."""
        sigma = [[2.0, 0.5], [0.5, 1.0]]
        determinant = 2.0 * 1.0 - 0.5 * 0.5
        inverse = [
            [1.0 / determinant, -0.5 / determinant],
            [-0.5 / determinant, 2.0 / determinant],
        ]
        t = MultivariateStudentT(nu=4.0, mu=[0.0, 0.0], sigma=sigma)
        for point in ([1.0, 0.0], [0.0, 1.0], [1.5, -2.5]):
            expected = sum(
                point[i] * inverse[i][j] * point[j] for i in range(2) for j in range(2)
            )
            assert t.mahalanobis(point) == pytest.approx(expected, rel=1e-12)

    def test_it_follows_the_exact_f_distribution(self) -> None:
        """For a multivariate t, ``q / d`` is exactly ``F(d, nu)``.

        An identity rather than an approximation, so its median is a sharp
        check on the whole sampling path -- the Cholesky factor, the chi-square
        mixing variable and the normal draw together.
        """
        dim, nu = 2, 6.0
        t = MultivariateStudentT(nu=nu, mu=[0.0, 0.0], sigma=CORRELATED)
        distances = [t.mahalanobis(draw) / dim for draw in t.rvs(40_000, seed=1)]
        # Median of F(2, 6) is 0.7972 to four places.
        assert statistics.median(distances) == pytest.approx(0.7972, rel=0.05)

    def test_a_point_of_the_wrong_length_is_rejected(self) -> None:
        t = MultivariateStudentT(nu=4.0, mu=[0.0, 0.0], sigma=IDENTITY)
        with pytest.raises(ValueError, match="2 components"):
            t.mahalanobis([1.0, 2.0, 3.0])


class TestSampling:
    def test_the_sample_covariance_matches_the_theoretical_one(self) -> None:
        t = MultivariateStudentT(nu=8.0, mu=[1.0, -2.0], sigma=CORRELATED)
        draws = t.rvs(60_000, seed=2)
        expected = t.covariance()
        first = [d[0] for d in draws]
        second = [d[1] for d in draws]
        assert statistics.fmean(first) == pytest.approx(1.0, abs=0.05)
        assert statistics.fmean(second) == pytest.approx(-2.0, abs=0.08)
        assert statistics.pvariance(first) == pytest.approx(expected[0][0], rel=0.10)
        assert statistics.covariance(first, second) == pytest.approx(
            expected[0][1], rel=0.15
        )

    def test_the_normal_sample_covariance_matches_its_scale_matrix(self) -> None:
        """For the normal the two are the same object, which is the contrast."""
        normal = MultivariateNormal(mu=[0.0, 0.0], sigma=CORRELATED)
        draws = normal.rvs(60_000, seed=3)
        assert statistics.covariance(
            [d[0] for d in draws], [d[1] for d in draws]
        ) == pytest.approx(0.5, rel=0.08)

    def test_the_seed_makes_it_reproducible(self) -> None:
        t = MultivariateStudentT(nu=4.0, mu=[0.0, 0.0], sigma=IDENTITY)
        assert t.rvs(50, seed=9) == t.rvs(50, seed=9)

    def test_a_bad_count_is_rejected(self) -> None:
        t = MultivariateStudentT(nu=4.0, mu=[0.0, 0.0], sigma=IDENTITY)
        with pytest.raises(ValueError, match="positive integer"):
            t.rvs(0)

    def test_draws_have_the_right_shape(self) -> None:
        t = MultivariateStudentT(
            nu=4.0,
            mu=[0.0] * 3,
            sigma=[[1.0 if i == j else 0.2 for j in range(3)] for i in range(3)],
        )
        draws = t.rvs(10, seed=1)
        assert len(draws) == 10
        assert all(len(d) == 3 for d in draws)


class TestMarginals:
    def test_a_one_dimensional_marginal_is_the_univariate_t(self) -> None:
        t = MultivariateStudentT(nu=4.0, mu=[0.5, -1.0], sigma=CORRELATED)
        marginal = t.marginal([0])
        for x in (-2.0, 0.0, 1.5):
            expected = StudentT(nu=4.0).pdf((x - 0.5) / 1.0)
            assert marginal.pdf([x]) == pytest.approx(expected, rel=1e-12)

    def test_it_keeps_the_requested_block_of_the_scale_matrix(self) -> None:
        sigma = [[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 3.0]]
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 1.0, 2.0], sigma=sigma)
        marginal = t.marginal([0, 2])
        assert marginal.mu == [0.0, 2.0]
        assert marginal.sigma == [[1.0, 0.3], [0.3, 3.0]]
        assert marginal.nu == 5.0

    def test_it_agrees_with_simulating_and_dropping_a_component(self) -> None:
        """The identity is exact, so the check is on the sampling too."""
        sigma = [[1.0, 0.2, 0.3], [0.2, 2.0, 0.4], [0.3, 0.4, 3.0]]
        t = MultivariateStudentT(nu=8.0, mu=[0.0, 1.0, 2.0], sigma=sigma)
        dropped = [[d[0], d[2]] for d in t.rvs(40_000, seed=4)]
        direct = t.marginal([0, 2]).rvs(40_000, seed=4)
        for index in (0, 1):
            assert statistics.fmean([d[index] for d in dropped]) == pytest.approx(
                statistics.fmean([d[index] for d in direct]), abs=0.1
            )

    def test_the_order_of_the_indices_is_respected(self) -> None:
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 1.0], sigma=CORRELATED)
        assert t.marginal([1, 0]).mu == [1.0, 0.0]

    @pytest.mark.parametrize("indices", [[], [0, 0], [5], [-1]])
    def test_bad_indices_are_rejected(self, indices: list[int]) -> None:
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 1.0], sigma=CORRELATED)
        with pytest.raises(ValueError):
            t.marginal(indices)


class TestTheNormalLimit:
    @pytest.mark.parametrize("point", [[1.0, -0.5], [2.5, 2.5], [-3.0, 1.0]])
    def test_large_degrees_of_freedom_give_the_normal_density(
        self, point: list[float]
    ) -> None:
        """``nu -> infinity`` is the Gaussian, and the approach is visible.

        Away from the centre. At the mode the two densities already agree to
        machine precision by ``nu = 10``, after which the difference is
        rounding in ``lgamma`` rather than the limit, and it drifts upwards --
        so a monotonicity check there measures the arithmetic, not the
        mathematics.
        """
        normal = MultivariateNormal(mu=[0.0, 0.0], sigma=CORRELATED)
        errors = [
            abs(
                MultivariateStudentT(nu=nu, mu=[0.0, 0.0], sigma=CORRELATED).pdf(point)
                - normal.pdf(point)
            )
            for nu in (10.0, 100.0, 1000.0, 10000.0)
        ]
        assert all(b < a for a, b in pairwise(errors))
        assert errors[-1] < 1e-4


class TestTailDependence:
    def test_it_matches_simulation(self) -> None:
        """The formula is the part most easily got wrong, so it is checked
        against the definition rather than against itself.

        Estimated at a finite level, which approaches the limit from above, so
        the simulated value sits slightly high. That direction is expected and
        the tolerance is one-sided about it.
        """
        for nu, rho in ((4.0, 0.5), (4.0, 0.0), (3.0, 0.7)):
            t = MultivariateStudentT(
                nu=nu, mu=[0.0, 0.0], sigma=[[1.0, rho], [rho, 1.0]]
            )
            draws = t.rvs(200_000, seed=1)
            threshold = StudentT(nu=nu).ppf(0.995)
            exceed = [d for d in draws if d[0] > threshold]
            both = sum(1 for d in exceed if d[1] > threshold)
            simulated = both / len(exceed)
            closed = tail_dependence_coefficient(nu, rho)
            assert simulated == pytest.approx(closed, abs=0.04)
            assert simulated >= closed - 0.02

    def test_it_is_positive_at_zero_correlation(self) -> None:
        """The property that makes the t worth using over the Gaussian.

        Uncorrelated t components still crash together, because they share the
        mixing variable.
        """
        assert tail_dependence_coefficient(nu=4.0, rho=0.0) > 0.07

    def test_it_vanishes_as_the_tail_lightens(self) -> None:
        values = [
            tail_dependence_coefficient(nu=nu, rho=0.5)
            for nu in (2.0, 4.0, 10.0, 30.0, 100.0)
        ]
        assert all(b < a for a, b in pairwise(values))
        assert values[-1] < 1e-4

    def test_it_increases_with_correlation(self) -> None:
        values = [
            tail_dependence_coefficient(nu=4.0, rho=rho)
            for rho in (-0.5, 0.0, 0.5, 0.9)
        ]
        assert all(b > a for a, b in pairwise(values))

    def test_the_extremes_of_correlation(self) -> None:
        assert tail_dependence_coefficient(nu=4.0, rho=1.0) == 1.0
        assert tail_dependence_coefficient(nu=4.0, rho=-1.0) == 0.0

    def test_it_stays_a_probability(self) -> None:
        for nu in (0.5, 1.0, 4.0, 50.0):
            for rho in (-0.99, -0.5, 0.0, 0.5, 0.99):
                assert 0.0 <= tail_dependence_coefficient(nu, rho) <= 1.0

    @pytest.mark.parametrize(
        ("nu", "rho"), [(0.0, 0.5), (-1.0, 0.5), (4.0, 1.5), (4.0, -2.0)]
    )
    def test_bad_arguments_are_rejected(self, nu: float, rho: float) -> None:
        with pytest.raises(ValueError):
            tail_dependence_coefficient(nu, rho)


class TestFitting:
    def test_it_recovers_the_parameters_it_was_given(self) -> None:
        source = MultivariateStudentT(
            nu=5.0, mu=[1.0, -1.0], sigma=[[1.0, 0.3], [0.3, 2.0]]
        )
        result = fit_multivariate_t(source.rvs(8_000, seed=1), nu=5.0)
        fitted = result["distribution"]
        assert result["converged"]
        assert fitted.mu[0] == pytest.approx(1.0, abs=0.08)
        assert fitted.mu[1] == pytest.approx(-1.0, abs=0.12)
        assert fitted.sigma[0][0] == pytest.approx(1.0, rel=0.12)
        assert fitted.sigma[0][1] == pytest.approx(0.3, abs=0.12)

    def test_the_weights_are_what_makes_it_robust(self) -> None:
        """A Gaussian fit is dragged by an outlier; a t fit is not.

        The EM weights are ``(nu + d) / (nu + q)``, so a point far out in
        Mahalanobis distance gets less say automatically. That is the whole
        reason to fit a t rather than a normal to contaminated data, and it is
        worth a test that contrasts the two rather than a comment.
        """
        source = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        clean = source.rvs(4_000, seed=2)
        # Both outliers pull the same way. Opposite ones would cancel in the
        # arithmetic mean and the comparison would show nothing, which is how
        # the first version of this test managed to prove neither.
        contaminated = [*clean, [500.0, 400.0], [450.0, 500.0]]

        clean_mean = [sum(row[j] for row in clean) / len(clean) for j in (0, 1)]
        robust = fit_multivariate_t(contaminated, nu=5.0)["distribution"]
        naive = [
            sum(row[j] for row in contaminated) / len(contaminated) for j in (0, 1)
        ]

        for j in (0, 1):
            # The robust fit barely notices two points in four thousand.
            assert abs(robust.mu[j] - clean_mean[j]) < 0.05
            # The arithmetic mean is dragged an order of magnitude further.
            assert abs(naive[j] - clean_mean[j]) > 0.2
            assert abs(robust.mu[j] - clean_mean[j]) < abs(naive[j] - clean_mean[j])

    def test_choosing_the_degrees_of_freedom_by_profile_likelihood(self) -> None:
        """Coarser than the other two parameters, and documented as such."""
        source = MultivariateStudentT(nu=4.0, mu=[0.0, 0.0], sigma=IDENTITY)
        result = fit_multivariate_t(source.rvs(6_000, seed=5))
        assert 2.5 <= result["nu"] <= 10.0
        assert result["distribution"].nu == result["nu"]

    def test_the_reported_likelihood_is_the_fitted_one(self) -> None:
        source = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        data = source.rvs(2_000, seed=6)
        result = fit_multivariate_t(data, nu=5.0)
        recomputed = sum(result["distribution"].logpdf(row) for row in data)
        assert result["log_likelihood"] == pytest.approx(recomputed, rel=1e-12)

    def test_it_beats_the_wrong_degrees_of_freedom_on_likelihood(self) -> None:
        source = MultivariateStudentT(nu=3.0, mu=[0.0, 0.0], sigma=IDENTITY)
        data = source.rvs(6_000, seed=7)
        right = fit_multivariate_t(data, nu=3.0)["log_likelihood"]
        wrong = fit_multivariate_t(data, nu=50.0)["log_likelihood"]
        assert right > wrong

    def test_too_few_observations_is_refused(self) -> None:
        """Fewer rows than columns leaves a singular scale matrix."""
        with pytest.raises(ValueError, match="singular"):
            fit_multivariate_t([[1.0, 2.0], [3.0, 4.0]], nu=5.0)

    def test_ragged_data_is_refused(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            fit_multivariate_t([[1.0, 2.0], [3.0], [1.0, 1.0]], nu=5.0)

    def test_empty_data_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            fit_multivariate_t([], nu=5.0)

    def test_a_constant_component_is_refused(self) -> None:
        rows = [[float(i), 3.0] for i in range(20)]
        with pytest.raises(ValueError, match="no variation"):
            fit_multivariate_t(rows, nu=5.0)


class TestTheMonteCarloDistributionFunction:
    def test_it_reports_a_standard_error(self) -> None:
        """A simulated probability without one cannot be acted on."""
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        result = t.cdf_monte_carlo([0.0, 0.0], n=20_000, seed=1)
        assert set(result) == {"probability", "standard_error", "n"}
        assert result["standard_error"] > 0.0

    def test_independent_components_give_the_product_of_the_marginals(self) -> None:
        """With a diagonal scale matrix the components are uncorrelated but
        **not** independent -- they share the mixing variable -- so the answer
        at the centre is a quarter only because of symmetry, not independence.
        """
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        result = t.cdf_monte_carlo([0.0, 0.0], n=60_000, seed=2)
        assert abs(result["probability"] - 0.25) < 4 * result["standard_error"]

    def test_the_error_shrinks_with_the_sample(self) -> None:
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        errors = [
            t.cdf_monte_carlo([1.0, 1.0], n=n, seed=3)["standard_error"]
            for n in (5_000, 20_000, 80_000)
        ]
        assert all(b < a for a, b in pairwise(errors))

    def test_it_validates_its_arguments(self) -> None:
        t = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0], sigma=IDENTITY)
        with pytest.raises(ValueError, match="2 components"):
            t.cdf_monte_carlo([0.0], n=100)
        with pytest.raises(ValueError, match="positive integer"):
            t.cdf_monte_carlo([0.0, 0.0], n=0)


class TestValidation:
    def test_mismatched_dimensions_are_rejected(self) -> None:
        with pytest.raises(ParameterError, match="mu has length"):
            MultivariateStudentT(nu=4.0, mu=[0.0, 0.0, 0.0], sigma=IDENTITY)

    @pytest.mark.parametrize("nu", [0.0, -1.0, float("inf"), float("nan")])
    def test_a_bad_degrees_of_freedom_is_rejected(self, nu: float) -> None:
        with pytest.raises(ParameterError, match="nu > 0"):
            MultivariateStudentT(nu=nu, mu=[0.0, 0.0], sigma=IDENTITY)

    def test_the_dimension_is_reported(self) -> None:
        t = MultivariateStudentT(
            nu=4.0,
            mu=[0.0] * 4,
            sigma=[[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)],
        )
        assert t.dim == 4


class TestBatchedSamplingMatchesTheDefinition:
    """``rvs`` is the batched form of ``_one``, and must stay that.

    ``_one`` is the scale-mixture construction written out a draw at a time.
    ``rvs`` does the same arithmetic with one matrix product instead of n
    triangular ones, which is worth doing -- the density and the quadratic
    form are where the time went, and this is the same idea applied to
    sampling -- but it means there are two statements of the construction in
    the file. This is what keeps them one statement in effect.

    Not to the bit. A matrix product associates its sums differently from the
    running total ``_one`` writes, so they agree to a few units in the last
    place and not exactly.
    """

    @pytest.mark.parametrize(
        "distribution",
        [
            MultivariateNormal(mu=[0.0, 0.0], sigma=[[1.0, 0.4], [0.4, 1.0]]),
            MultivariateStudentT(
                nu=4.0,
                mu=[1.0, -2.0, 0.5],
                sigma=[[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
            ),
        ],
        ids=["normal", "student-t"],
    )
    def test_the_draws_agree(self, distribution: object) -> None:
        batched = np.array(distribution.rvs(500, seed=11))
        rng = RNG(11)
        one_at_a_time = np.array([distribution._one(rng) for _ in range(500)])
        np.testing.assert_allclose(batched, one_at_a_time, rtol=1e-13, atol=1e-14)

    def test_a_seed_still_reproduces(self) -> None:
        normal = MultivariateNormal(mu=[0.0, 0.0], sigma=[[1.0, 0.0], [0.0, 1.0]])
        assert normal.rvs(20, seed=5) == normal.rvs(20, seed=5)

    def test_it_still_returns_lists_of_floats(self) -> None:
        normal = MultivariateNormal(mu=[0.0, 0.0], sigma=[[1.0, 0.0], [0.0, 1.0]])
        sample = normal.rvs(3, seed=1)
        assert isinstance(sample, list)
        assert all(isinstance(row, list) for row in sample)
        assert all(isinstance(value, float) for row in sample for value in row)
