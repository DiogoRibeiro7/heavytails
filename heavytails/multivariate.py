"""Multivariate heavy-tailed distributions, in the elliptical family.

Correlation is not the question when heavy-tailed variables are modelled
jointly. The question is whether their extremes arrive together, and that is a
property of the joint tail rather than of the covariance. Two series can be
almost uncorrelated and still crash on the same day.

The distributions here are **normal scale mixtures**: a Gaussian vector divided
by the square root of an independent positive variable. That single
construction gives both members below, and the mixing variable is what makes
the tail heavy::

    X = mu + L Z / sqrt(S)

with ``L`` a Cholesky factor of the scale matrix, ``Z`` standard normal, and
``S`` the mixing variable. ``S = 1`` gives the multivariate normal; ``S = W/nu``
with ``W`` chi-square on ``nu`` degrees of freedom gives the multivariate
Student-t.

Two things this deliberately does not do:

**There is no distribution function.** The multivariate t has no closed-form
cdf in dimension two or above; every implementation computes it by numerical
integration or simulation. :meth:`MultivariateStudentT.cdf_monte_carlo` is
offered instead, and it reports a standard error, because a probability
estimated by simulation without one is not usable.

**The linear algebra is pure Python.** Cholesky is cubic in the dimension and
happens once per instance; each density evaluation is quadratic. That is
comfortable at the dimensions this is for -- tail dependence is usually asked
about a handful of series at a time -- and would be the wrong choice at
hundreds. :mod:`heavytails.vectorized` is where a NumPy path would go if that
ever mattered.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from heavytails.heavy_tails import RNG, ParameterError, StudentT

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Elliptical",
    "MultivariateNormal",
    "MultivariateStudentT",
    "cholesky",
    "fit_multivariate_t",
    "tail_dependence_coefficient",
]

Matrix = list[list[float]]
Vector = list[float]


# --------------------------- Linear algebra ---------------------------------- #


def cholesky(matrix: Sequence[Sequence[float]]) -> Matrix:
    """
    Lower-triangular ``L`` with ``L L' = matrix``.

    The factorisation doubles as the validity check on a scale matrix: it
    succeeds exactly when the matrix is symmetric positive definite, which is
    what a covariance or scale matrix has to be. A separate eigenvalue test
    would be more work and would answer the same question less directly.

    Args:
        matrix: Square, symmetric, positive definite.

    Returns:
        The lower-triangular factor, as a list of rows.

    Raises:
        ParameterError: If the matrix is not square, not symmetric, or not
            positive definite -- with which entry failed, since "not positive
            definite" on its own is a hard message to act on.

    Examples:
        >>> cholesky([[4.0, 2.0], [2.0, 5.0]])
        [[2.0, 0.0], [1.0, 2.0]]
    """
    size = len(matrix)
    if size == 0:
        raise ParameterError("the scale matrix must not be empty.")
    for i, row in enumerate(matrix):
        if len(row) != size:
            raise ParameterError(f"the scale matrix must be square; row {i} is not.")
    for i in range(size):
        for j in range(i):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12 * max(1.0, abs(matrix[i][j])):
                raise ParameterError(
                    f"the scale matrix must be symmetric; entries ({i},{j}) and "
                    f"({j},{i}) differ."
                )

    lower: Matrix = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i + 1):
            total = sum(lower[i][k] * lower[j][k] for k in range(j))
            if i == j:
                pivot = matrix[i][i] - total
                if pivot <= 0.0:
                    raise ParameterError(
                        f"the scale matrix is not positive definite: the pivot at "
                        f"({i},{i}) is {pivot:.6g}, which is not positive."
                    )
                lower[i][j] = math.sqrt(pivot)
            else:
                lower[i][j] = (matrix[i][j] - total) / lower[j][j]
    return lower


def _log_determinant(lower: Matrix) -> float:
    """``log |L L'|``, which is twice the log of the diagonal product."""
    return 2.0 * sum(math.log(lower[i][i]) for i in range(len(lower)))


# --------------------------- Elliptical family ------------------------------- #


@dataclass(frozen=True)
class Elliptical(ABC):
    """
    A normal scale mixture: ``X = mu + L Z / sqrt(S)``.

    Subclasses supply two things: the density kernel in the quadratic form, and
    how to draw the mixing variable. Everything else -- the Cholesky factor, the
    Mahalanobis distance, sampling, marginals -- follows from the construction
    and is shared.

    Args:
        mu: Location vector.
        sigma: Scale matrix, symmetric positive definite. For the normal this is
            the covariance; for the Student-t it is **not**, since the
            covariance is ``nu / (nu - 2)`` times it and exists only above two
            degrees of freedom.

    Raises:
        ParameterError: If the dimensions disagree or the scale matrix is not
            positive definite.
    """

    mu: Vector
    sigma: Matrix
    _lower: Matrix = field(init=False, repr=False, compare=False)
    _lower_array: Any = field(init=False, repr=False, compare=False)
    _mu_array: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.mu) != len(self.sigma):
            raise ParameterError(
                f"mu has length {len(self.mu)} but the scale matrix is "
                f"{len(self.sigma)} by {len(self.sigma)}."
            )
        object.__setattr__(self, "_lower", cholesky(self.sigma))
        # Array copies of both, built once. The factorisation is the expensive
        # part of constructing one of these and it is already done above; what
        # these avoid is rebuilding an array from lists on every call, which
        # for a fit that evaluates the quadratic form ten thousand times per
        # iteration is most of the work.
        object.__setattr__(self, "_lower_array", np.array(self._lower, dtype=float))
        object.__setattr__(self, "_mu_array", np.array(self.mu, dtype=float))

    @property
    def dim(self) -> int:
        """Number of components."""
        return len(self.mu)

    def mahalanobis(self, x: Any) -> Any:
        """
        The quadratic form ``(x - mu)' Sigma^-1 (x - mu)``.

        Computed by forward substitution on the Cholesky factor rather than by
        inverting the matrix, which is both faster and better conditioned.

        Takes one point or many, and mirrors what it was given: a single point
        gives a float, an ``(n, dim)`` array of them gives an array of n. The
        substitution loops over the dimension, which is small, and does each of
        its steps across every observation at once -- the opposite of the
        arrangement it replaced, which looped over the observations and did
        each one's substitution in the interpreter.

        Args:
            x: A point with as many components as the distribution, or an array
                of such points whose last axis is the components.

        Returns:
            The squared Mahalanobis distance, which is non-negative.

        Raises:
            ValueError: If the last axis does not match the dimension.
        """
        points = np.asarray(x, dtype=float)
        if points.ndim == 0 or points.shape[-1] != self.dim:
            got = points.shape[-1] if points.ndim else 0
            raise ValueError(f"x must have {self.dim} components, got {got}.")
        single = points.ndim == 1

        rows = points.reshape(-1, self.dim)
        centred = (rows - self._mu_array).T
        lower = self._lower_array
        solved = np.empty_like(centred)
        for i in range(self.dim):
            solved[i] = (centred[i] - lower[i, :i] @ solved[:i]) / lower[i, i]
        quadratic = np.einsum("ij,ij->j", solved, solved)

        if single:
            return float(quadratic[0])
        return quadratic.reshape(points.shape[:-1])

    def logpdf(self, x: Any) -> Any:
        """Log density at ``x``, for one point or many.

        The log is the primitive rather than an afterthought: a multivariate
        density in even a few dimensions underflows to zero well before the
        point stops being interesting, and fitting works on the log anyway.
        """
        return self._log_kernel(self.mahalanobis(x)) - 0.5 * _log_determinant(
            self._lower
        )

    def pdf(self, x: Any) -> Any:
        """Density at ``x``, or zero where it underflows."""
        value = self.logpdf(x)
        if isinstance(value, float):
            return math.exp(value) if value > -745.0 else 0.0
        with np.errstate(under="ignore"):
            return np.where(value > -745.0, np.exp(value), 0.0)

    def rvs(self, n: int, seed: int | None = None) -> list[Vector]:
        """
        Draw ``n`` independent vectors.

        Args:
            n: Number of draws, positive.
            seed: Seed for reproducibility.

        Returns:
            A list of ``n`` vectors.

        Raises:
            ValueError: If ``n`` is not a positive integer.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = RNG(seed)

        # The generator is Python's and each draw depends on the last, so the
        # draws stay in a loop and in the order the one-at-a-time version used
        # them: dim normals, then the mixing variable, per observation. That
        # order is what makes a seeded sample reproduce. What moves out of the
        # loop is the multiplication by the factor, which becomes one matrix
        # product instead of n triangular ones.
        normals = np.empty((n, self.dim), dtype=float)
        scales = np.empty(n, dtype=float)
        for index in range(n):
            for component in range(self.dim):
                normals[index, component] = rng.standard_normal()
            scales[index] = math.sqrt(self._mixing(rng))

        draws = self._mu_array + (normals @ self._lower_array.T) / scales[:, None]
        return [[float(value) for value in row] for row in draws]

    def _one(self, rng: RNG) -> Vector:
        """One draw, by the scale-mixture construction.

        The definition, kept in the plainest form it can be written in.
        :meth:`rvs` is the batched form of exactly this and is what callers
        get; the two are held together by a test rather than by anyone
        remembering, because the association order of the matrix product means
        they agree to about 4e-15 and not to the bit.
        """
        normal = [rng.standard_normal() for _ in range(self.dim)]
        scale = math.sqrt(self._mixing(rng))
        return [
            self.mu[i]
            + sum(self._lower[i][k] * normal[k] for k in range(i + 1)) / scale
            for i in range(self.dim)
        ]

    def marginal(self, indices: Sequence[int]) -> Elliptical:
        """
        The distribution of the named components.

        Marginals of an elliptical distribution stay in the family, with the
        corresponding sub-block of the scale matrix and the same mixing
        variable. That is an exact identity rather than an approximation, and
        the test suite checks it against simulation.

        Args:
            indices: Which components to keep, in the order wanted.

        Returns:
            The same kind of distribution, of lower dimension.

        Raises:
            ValueError: If an index is out of range or repeated.
        """
        if not indices:
            raise ValueError("at least one component must be kept.")
        if len(set(indices)) != len(indices):
            raise ValueError("indices must not repeat.")
        for index in indices:
            if not 0 <= index < self.dim:
                raise ValueError(f"index {index} is outside 0..{self.dim - 1}.")
        return self._rebuild(
            [self.mu[i] for i in indices],
            [[self.sigma[i][j] for j in indices] for i in indices],
        )

    @abstractmethod
    def _log_kernel(self, quadratic: Any) -> Any:
        """Log density as a function of the quadratic form, normalisation
        included except for the determinant. Elementwise over an array."""

    @abstractmethod
    def _mixing(self, rng: RNG) -> float:
        """Draw the mixing variable ``S``; the normal draw is divided by its
        square root."""

    @abstractmethod
    def _rebuild(self, mu: Vector, sigma: Matrix) -> Elliptical:
        """Construct the same family at a different dimension."""


@dataclass(frozen=True)
class MultivariateNormal(Elliptical):
    """
    Multivariate normal, present as the reference member of the family.

    Not heavy-tailed, and included anyway: it is the ``nu -> infinity`` limit of
    the Student-t and therefore the thing to check that limit against, and its
    **zero tail dependence** is the contrast that makes the t interesting. Two
    jointly normal variables with correlation 0.9 still have asymptotically
    independent extremes.

    Examples:
        >>> normal = MultivariateNormal(mu=[0.0, 0.0], sigma=[[1.0, 0.0], [0.0, 1.0]])
        >>> round(normal.pdf([0.0, 0.0]), 6)
        0.159155
        >>> round(1 / (2 * math.pi), 6)
        0.159155
    """

    def _log_kernel(self, quadratic: Any) -> Any:
        return -0.5 * (self.dim * math.log(2.0 * math.pi) + quadratic)

    def _mixing(self, _rng: RNG) -> float:
        return 1.0

    def _rebuild(self, mu: Vector, sigma: Matrix) -> MultivariateNormal:
        return MultivariateNormal(mu=mu, sigma=sigma)


@dataclass(frozen=True)
class MultivariateStudentT(Elliptical):
    """
    Multivariate Student-t, the workhorse of joint heavy-tail modelling.

    The tail index is ``nu`` in every direction, and -- unlike the normal --
    its extremes arrive together: the coefficient of tail dependence is
    positive for any correlation, including zero. That is the property the
    whole elliptical apparatus is here to provide, and
    :func:`tail_dependence_coefficient` gives it in closed form.

    Args:
        nu: Degrees of freedom, positive. Moments of order ``nu`` and above do
            not exist, so the covariance exists only above two.
        mu: Location vector.
        sigma: **Scale** matrix, not the covariance. The covariance is
            ``nu / (nu - 2)`` times it where it exists at all.

    Raises:
        ParameterError: If ``nu`` is not positive, or the scale matrix is not
            positive definite.

    Examples:
        >>> t = MultivariateStudentT(nu=4.0, mu=[0.0, 0.0],
        ...                          sigma=[[1.0, 0.5], [0.5, 1.0]])
        >>> t.dim
        2
        >>> round(t.mahalanobis([1.0, 1.0]), 6)
        1.333333

        Its marginals are univariate t with the same degrees of freedom:

        >>> from heavytails import StudentT
        >>> round(t.marginal([0]).pdf([1.5]), 8) == round(StudentT(nu=4.0).pdf(1.5), 8)
        True
    """

    nu: float = 1.0

    def __post_init__(self) -> None:
        if not (self.nu > 0.0) or not math.isfinite(self.nu):
            raise ParameterError("MultivariateStudentT requires a finite nu > 0.")
        super().__post_init__()

    def _log_kernel(self, quadratic: Any) -> Any:
        half = 0.5 * (self.nu + self.dim)
        constant = (
            math.lgamma(half)
            - math.lgamma(0.5 * self.nu)
            - 0.5 * self.dim * math.log(self.nu * math.pi)
        )
        # np.log1p rather than math.log1p so this works elementwise; on a float
        # it returns a numpy scalar, which the float() in mahalanobis has
        # already made unnecessary to worry about upstream, so the result is
        # cast back here to keep logpdf(one point) a plain float.
        term = np.log1p(np.asarray(quadratic, dtype=float) / self.nu)
        result = constant - half * term
        if isinstance(quadratic, float):
            return float(result)
        return result

    def _mixing(self, rng: RNG) -> float:
        return rng.chisquare(self.nu) / self.nu

    def _rebuild(self, mu: Vector, sigma: Matrix) -> MultivariateStudentT:
        return MultivariateStudentT(nu=self.nu, mu=mu, sigma=sigma)

    def covariance(self) -> Matrix:
        """
        The covariance matrix, which is not the scale matrix.

        Returns:
            ``nu / (nu - 2)`` times the scale matrix.

        Raises:
            ValueError: If ``nu`` is two or below, where no covariance exists.
                Returning the scale matrix instead would be the natural mistake
                and would understate the spread by an unbounded factor.
        """
        if self.nu <= 2.0:
            raise ValueError(
                f"the covariance of a multivariate t exists only for nu > 2; "
                f"this has nu = {self.nu}."
            )
        factor = self.nu / (self.nu - 2.0)
        return [[factor * value for value in row] for row in self.sigma]

    def cdf_monte_carlo(
        self,
        upper: Sequence[float],
        n: int = 100_000,
        seed: int | None = None,
    ) -> dict[str, float]:
        """
        Estimate ``P(X <= upper)`` by simulation, with a standard error.

        The multivariate t has no closed-form distribution function in
        dimension two or above. Every library that offers one computes it
        numerically, and this says so in its name rather than presenting an
        estimate as an exact value.

        The standard error is returned alongside, because a simulated
        probability without one cannot be acted on -- and it is largest in
        relative terms exactly where the probability is smallest, which is the
        corner anyone modelling joint extremes is asking about.

        Args:
            upper: Componentwise upper limits.
            n: Number of draws.
            seed: Seed for reproducibility.

        Returns:
            ``probability``, ``standard_error`` and ``n``.

        Raises:
            ValueError: If ``upper`` has the wrong length or ``n`` is not
                positive.

        Examples:
            >>> t = MultivariateStudentT(nu=5.0, mu=[0.0, 0.0],
            ...                          sigma=[[1.0, 0.0], [0.0, 1.0]])
            >>> result = t.cdf_monte_carlo([0.0, 0.0], n=20000, seed=1)
            >>> abs(result["probability"] - 0.25) < 4 * result["standard_error"]
            True
        """
        if len(upper) != self.dim:
            raise ValueError(f"upper must have {self.dim} components.")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        draws = np.asarray(self.rvs(n, seed=seed), dtype=float)
        hits = int(
            np.count_nonzero(np.all(draws <= np.asarray(upper, dtype=float), axis=1))
        )
        probability = hits / n
        return {
            "probability": probability,
            "standard_error": math.sqrt(probability * (1.0 - probability) / n),
            "n": float(n),
        }


# --------------------------- Tail dependence --------------------------------- #


def tail_dependence_coefficient(nu: float, rho: float) -> float:
    """
    Coefficient of tail dependence for a bivariate Student-t, in closed form.

    The probability that one component is extreme *given* the other is, in the
    limit::

        lambda = 2 T_{nu+1}( -sqrt( (nu+1)(1-rho) / (1+rho) ) )

    with ``T`` the univariate t distribution function. Upper and lower are
    equal by symmetry.

    Two properties are worth knowing before using correlation as a proxy for
    joint risk:

    **It is positive even at zero correlation.** Uncorrelated t variables still
    have extremes that arrive together, because they share the same mixing
    variable -- the market-wide shock, in the usual reading.

    **It goes to zero as nu grows.** The Gaussian limit has no tail dependence
    at any correlation short of one, which is exactly why a Gaussian copula
    understates joint extremes.

    Args:
        nu: Degrees of freedom, positive.
        rho: Correlation of the scale matrix, in [-1, 1].

    Returns:
        The coefficient, in [0, 1].

    Raises:
        ValueError: If ``nu`` is not positive or ``rho`` is outside [-1, 1].

    Examples:
        >>> round(tail_dependence_coefficient(nu=4.0, rho=0.5), 6)
        0.25317

        Positive even when the components are uncorrelated:

        >>> round(tail_dependence_coefficient(nu=4.0, rho=0.0), 6)
        0.075587

        And vanishing as the tail lightens:

        >>> round(tail_dependence_coefficient(nu=30.0, rho=0.5), 6)
        0.003047
    """
    if not (nu > 0.0) or not math.isfinite(nu):
        raise ValueError("nu must be finite and positive.")
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must be in [-1,1].")
    if rho == 1.0:
        return 1.0
    if rho == -1.0:
        return 0.0
    argument = -math.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * StudentT(nu=nu + 1.0).cdf(argument))


# --------------------------- Fitting ----------------------------------------- #


def fit_multivariate_t(
    data: Sequence[Sequence[float]],
    nu: float | None = None,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict[str, Any]:
    """
    Fit a multivariate Student-t by expectation-maximisation.

    The scale-mixture construction is what makes this easy: treat the mixing
    variable as missing data, and each iteration is a *weighted* mean and
    covariance with weights ``(nu + d) / (nu + q_i)``. Those weights are the
    whole robustness story -- a point far out in Mahalanobis distance gets less
    say, automatically, which is why a t fit is not dragged around by outliers
    the way a Gaussian one is.

    With ``nu`` left unset it is chosen by profile likelihood over a grid. That
    is a coarser answer than the other two parameters get, and it is honest
    about the shape of the problem: the profile likelihood in ``nu`` is flat
    enough that a fitted value of 6 rather than 8 usually says more about the
    sample than about the population.

    Args:
        data: Rows of observations, all the same length.
        nu: Degrees of freedom, or None to select by profile likelihood.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on the location and scale.

    Returns:
        ``distribution``, ``nu``, ``iterations``, ``log_likelihood`` and
        ``converged``.

    Raises:
        ValueError: If the data is empty, ragged, or has fewer rows than
            columns -- in which case the scale matrix is singular and no fit
            exists.

    Examples:
        >>> source = MultivariateStudentT(nu=5.0, mu=[1.0, -1.0],
        ...                               sigma=[[1.0, 0.3], [0.3, 2.0]])
        >>> fit = fit_multivariate_t(source.rvs(4000, seed=1), nu=5.0)
        >>> fit["converged"]
        True
        >>> all(abs(a - b) < 0.15 for a, b in zip(fit["distribution"].mu, [1.0, -1.0]))
        True
    """
    rows = [list(row) for row in data]
    if not rows:
        raise ValueError("data must not be empty.")
    dim = len(rows[0])
    if any(len(row) != dim for row in rows):
        raise ValueError("every observation must have the same length.")
    if len(rows) <= dim:
        raise ValueError(
            f"fitting {dim} dimensions needs more than {dim} observations; "
            f"got {len(rows)}. The scale matrix would be singular."
        )

    if nu is None:
        best = None
        for candidate in (2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0, 100.0):
            trial = fit_multivariate_t(rows, nu=candidate, max_iter=max_iter, tol=tol)
            if best is None or trial["log_likelihood"] > best["log_likelihood"]:
                best = trial
        assert best is not None
        return best

    if not (nu > 0.0):
        raise ValueError("nu must be positive.")

    n = len(rows)
    # One array for the whole sample, built once rather than per iteration.
    # Every quantity below is a reduction over its rows, and each used to be a
    # comprehension over n in the interpreter -- run once per EM step, and the
    # whole fit run once per candidate nu when the degrees of freedom are being
    # chosen, so the interpreter was doing the same pass some thousands of
    # times over.
    observations = np.array(rows, dtype=float)
    mu_array = observations.mean(axis=0)
    centred = observations - mu_array
    sigma_array = centred.T @ centred / n

    for i in range(dim):
        if sigma_array[i, i] <= 0.0:
            raise ValueError(f"component {i} has no variation to fit.")

    mu = [float(value) for value in mu_array]
    sigma = [[float(value) for value in row] for row in sigma_array]

    converged = False
    iterations = 0
    for step in range(1, max_iter + 1):
        iterations = step
        current = MultivariateStudentT(nu=nu, mu=mu, sigma=sigma)

        # The quadratic form for every observation in one call.
        weights = (nu + dim) / (nu + current.mahalanobis(observations))
        total = float(weights.sum())

        new_mu_array = (weights @ observations) / total
        deviations = observations - new_mu_array
        # The weighted scatter as one matrix product. Written out, this is the
        # dim-by-dim double loop it replaces, each entry of which walked the
        # whole sample.
        new_sigma_array = (deviations * weights[:, None]).T @ deviations / n

        shift = max(
            float(np.max(np.abs(new_mu_array - mu_array))),
            float(np.max(np.abs(new_sigma_array - sigma_array))),
        )
        mu_array, sigma_array = new_mu_array, new_sigma_array
        mu = [float(value) for value in mu_array]
        sigma = [[float(value) for value in row] for row in sigma_array]
        if shift < tol:
            converged = True
            break

    fitted = MultivariateStudentT(nu=nu, mu=mu, sigma=sigma)
    return {
        "distribution": fitted,
        "nu": nu,
        "iterations": iterations,
        "log_likelihood": float(np.sum(fitted.logpdf(observations))),
        "converged": converged,
    }
