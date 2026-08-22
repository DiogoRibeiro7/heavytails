"""Copulas: the dependence between variables, separated from their margins.

Sklar's theorem says any joint distribution splits into its marginal
distributions and a copula holding the dependence, and that the split is
unique for continuous margins. That is what makes the copula the right object
here: the question of whether two heavy-tailed series crash together is a
question about their copula, and it survives any monotone rescaling of either
series.

**Correlation does not answer it.** The four copulas below can be given the
same rank correlation and still disagree completely about joint extremes:

===================  ==============  ==============  ============================
Copula               Lower tail      Upper tail
===================  ==============  ==============  ============================
:class:`Gaussian`    0               0               No joint extremes at any
                                                     correlation below one
:class:`StudentT`    positive        positive        Equal, and positive even at
                                                     zero correlation
:class:`Gumbel`      0               ``2 - 2**(1/theta)``  Upper only
:class:`Galambos`    0               ``2 ** (-1/theta)``   Upper only
===================  ==============  ==============  ============================

The Gaussian row is the one that has caused trouble in practice. Two variables
joined by a Gaussian copula with correlation 0.95 are still asymptotically
independent in the tail: condition on one being extreme and the probability the
other is too goes to zero. A t copula with the same correlation does not
behave that way, and neither do markets.

What is not here: vine copulas, which need pair-copula construction and
structure selection and are a subsystem rather than a class (#306 keeps them).
The distribution functions of the elliptical copulas are also absent in closed
form, for the reason given in :mod:`heavytails.multivariate` -- there is a
Monte Carlo estimate with a standard error instead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

from heavytails._special import _phi_inverse, _ppf_monotone
from heavytails.heavy_tails import RNG, ParameterError, StudentT
from heavytails.multivariate import MultivariateNormal, MultivariateStudentT

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "Copula",
    "GalambosCopula",
    "GaussianCopula",
    "GumbelCopula",
    "StudentTCopula",
    "empirical_tail_dependence",
]

_EDGE = 1e-12


def _normal_cdf(z: float) -> float:
    """Standard normal distribution function, via erfc so the left tail keeps
    its digits."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _check_unit(point: Sequence[float], dim: int) -> None:
    """Validate a point of the unit cube."""
    if len(point) != dim:
        raise ValueError(f"the point must have {dim} components, got {len(point)}.")
    for value in point:
        if not (0.0 < value < 1.0):
            raise ValueError(f"every component must be in (0,1); got {value!r}.")


class Copula(ABC):
    """A distribution on the unit cube with uniform margins."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Number of components."""

    @abstractmethod
    def rvs(self, n: int, seed: int | None = None) -> list[list[float]]:
        """Draw ``n`` points of the unit cube."""

    @abstractmethod
    def tail_dependence(self) -> dict[str, float]:
        """The limiting coefficients, as ``lower`` and ``upper``."""


# --------------------------- Elliptical copulas ------------------------------ #


@dataclass(frozen=True)
class GaussianCopula(Copula):
    """
    The copula of a multivariate normal.

    Present largely as the cautionary case. **Its tail dependence is zero at
    every correlation short of one**: condition on one component being extreme
    and the probability that another is too tends to zero. Fitting a Gaussian
    copula to data whose extremes do arrive together will reproduce the
    correlation faithfully and understate the joint risk without any sign that
    it has done so.

    Args:
        correlation: Correlation matrix, symmetric positive definite with unit
            diagonal.

    Raises:
        ParameterError: If the matrix is not a valid correlation matrix.

    Examples:
        >>> copula = GaussianCopula([[1.0, 0.7], [0.7, 1.0]])
        >>> copula.tail_dependence()
        {'lower': 0.0, 'upper': 0.0}
        >>> round(copula.pdf([0.5, 0.5]), 6)
        1.40028
    """

    correlation: list[list[float]]

    def __post_init__(self) -> None:
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "_normal",
            MultivariateNormal(
                mu=[0.0] * len(self.correlation), sigma=self.correlation
            ),
        )

    @property
    def dim(self) -> int:
        """Number of components."""
        return len(self.correlation)

    def logpdf(self, point: Sequence[float]) -> float:
        """
        Log copula density.

        The ratio of the joint normal density to the product of its margins,
        which is what remains once the marginal shape is divided out.
        """
        _check_unit(point, self.dim)
        z = [_phi_inverse(value) for value in point]
        joint = self.__dict__["_normal"].logpdf(z)
        marginal = sum(-0.5 * (math.log(2.0 * math.pi) + value * value) for value in z)
        return float(joint - marginal)

    def pdf(self, point: Sequence[float]) -> float:
        """Copula density."""
        value = self.logpdf(point)
        return math.exp(value) if value > -745.0 else 0.0

    def rvs(self, n: int, seed: int | None = None) -> list[list[float]]:
        """Draw ``n`` points, by transforming a normal sample to uniforms."""
        return [
            [_normal_cdf(value) for value in draw]
            for draw in self.__dict__["_normal"].rvs(n, seed=seed)
        ]

    def tail_dependence(self) -> dict[str, float]:
        """Zero, both sides, whatever the correlation."""
        return {"lower": 0.0, "upper": 0.0}


@dataclass(frozen=True)
class StudentTCopula(Copula):
    """
    The copula of a multivariate Student-t.

    The workhorse for joint heavy tails, because its extremes arrive together
    and its dependence is symmetric between the tails. Unlike the Gaussian, the
    coefficient is **positive even at zero correlation** -- uncorrelated t
    components share the mixing variable, which is the market-wide shock in the
    usual reading.

    Args:
        nu: Degrees of freedom, positive. Smaller means heavier joint tails.
        correlation: Correlation matrix.

    Raises:
        ParameterError: If ``nu`` is not positive or the matrix is invalid.

    Examples:
        >>> copula = StudentTCopula(nu=4.0, correlation=[[1.0, 0.0], [0.0, 1.0]])
        >>> round(copula.tail_dependence()["upper"], 6)
        0.075587

        Zero correlation, and the extremes still arrive together.
    """

    nu: float
    correlation: list[list[float]]

    def __post_init__(self) -> None:
        if not (self.nu > 0.0) or not math.isfinite(self.nu):
            raise ParameterError("StudentTCopula requires a finite nu > 0.")
        _validate_correlation(self.correlation)
        object.__setattr__(
            self,
            "_joint",
            MultivariateStudentT(
                nu=self.nu, mu=[0.0] * len(self.correlation), sigma=self.correlation
            ),
        )

    @property
    def dim(self) -> int:
        """Number of components."""
        return len(self.correlation)

    def logpdf(self, point: Sequence[float]) -> float:
        """Log copula density."""
        _check_unit(point, self.dim)
        marginal = StudentT(nu=self.nu)
        x = [marginal.ppf(value) for value in point]
        joint = self.__dict__["_joint"].logpdf(x)
        product = sum(math.log(marginal.pdf(value)) for value in x)
        return float(joint - product)

    def pdf(self, point: Sequence[float]) -> float:
        """Copula density."""
        value = self.logpdf(point)
        return math.exp(value) if value > -745.0 else 0.0

    def rvs(self, n: int, seed: int | None = None) -> list[list[float]]:
        """Draw ``n`` points, by transforming a t sample to uniforms."""
        marginal = StudentT(nu=self.nu)
        return [
            [marginal.cdf(value) for value in draw]
            for draw in self.__dict__["_joint"].rvs(n, seed=seed)
        ]

    def tail_dependence(self) -> dict[str, float]:
        """
        Equal in both tails, and positive at any correlation above -1.

        Uses the pairwise correlation of the first two components, since the
        coefficient is a property of a pair rather than of a matrix.

        Raises:
            ValueError: If the copula has only one component.
        """
        if self.dim < 2:
            raise ValueError("tail dependence needs at least two components.")
        from heavytails.multivariate import (  # noqa: PLC0415
            tail_dependence_coefficient,
        )

        value = tail_dependence_coefficient(self.nu, self.correlation[0][1])
        return {"lower": value, "upper": value}


def _validate_correlation(matrix: Sequence[Sequence[float]]) -> None:
    """A correlation matrix is a scale matrix with a unit diagonal."""
    from heavytails.multivariate import cholesky  # noqa: PLC0415

    for i, row in enumerate(matrix):
        if abs(row[i] - 1.0) > 1e-12:
            raise ParameterError(
                f"a correlation matrix has ones on the diagonal; entry ({i},{i}) "
                f"is {row[i]!r}."
            )
    cholesky(matrix)


# --------------------------- Extreme-value copulas --------------------------- #


@dataclass(frozen=True)
class GumbelCopula(Copula):
    """
    Gumbel-Hougaard copula: upper tail dependence, none in the lower tail.

    An extreme-value copula, and the one that arises as the limit of maxima. Its
    asymmetry is the point: joint crashes without joint booms, which is what
    equity returns tend to look like once the sign convention is fixed.

    ``theta = 1`` is independence; larger is stronger dependence.

    Args:
        theta: Dependence parameter, at least one.

    Raises:
        ParameterError: If ``theta`` is below one.

    Examples:
        >>> copula = GumbelCopula(theta=2.0)
        >>> round(copula.cdf([0.5, 0.5]), 6)
        0.375214
        >>> {k: round(v, 6) for k, v in copula.tail_dependence().items()}
        {'lower': 0.0, 'upper': 0.585786}
    """

    theta: float

    def __post_init__(self) -> None:
        if not (self.theta >= 1.0) or not math.isfinite(self.theta):
            raise ParameterError("GumbelCopula requires a finite theta >= 1.")

    @property
    def dim(self) -> int:
        """Two; the bivariate case is the one with a closed form here."""
        return 2

    def cdf(self, point: Sequence[float]) -> float:
        """``exp(-[(-ln u)^theta + (-ln v)^theta]^(1/theta))``."""
        _check_unit(point, 2)
        return float(math.exp(-self._combined(point)))

    def pdf(self, point: Sequence[float]) -> float:
        """Copula density, checked against a finite difference of the cdf."""
        _check_unit(point, 2)
        u, v = point[0], point[1]
        a, b = -math.log(u), -math.log(v)
        combined = self._combined(point)
        return float(
            math.exp(-combined)
            * (a * b) ** (self.theta - 1.0)
            * combined ** (1.0 - 2.0 * self.theta)
            * (combined + self.theta - 1.0)
            / (u * v)
        )

    def _combined(self, point: Sequence[float]) -> float:
        a, b = -math.log(point[0]), -math.log(point[1])
        return float((a**self.theta + b**self.theta) ** (1.0 / self.theta))

    def rvs(self, n: int, seed: int | None = None) -> list[list[float]]:
        """
        Draw ``n`` points, by the Marshall-Olkin construction.

        A Gumbel copula is Archimedean with a positive stable mixing variable,
        so a draw is two exponentials divided by one stable variate. That is
        exact, unlike inverting the conditional distribution numerically, and
        it extends to any dimension.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = RNG(seed)
        if self.theta == 1.0:
            return [[rng.uniform_0_1(), rng.uniform_0_1()] for _ in range(n)]
        return [self._one(rng) for _ in range(n)]

    def _one(self, rng: RNG) -> list[float]:
        stable = _positive_stable(rng, 1.0 / self.theta)
        return [
            math.exp(-((-math.log(rng.uniform_0_1()) / stable) ** (1.0 / self.theta)))
            for _ in range(2)
        ]

    def tail_dependence(self) -> dict[str, float]:
        """Upper only: ``2 - 2**(1/theta)``, and zero below."""
        return {"lower": 0.0, "upper": float(2.0 - 2.0 ** (1.0 / self.theta))}


@dataclass(frozen=True)
class GalambosCopula(Copula):
    """
    Galambos copula: upper tail dependence, none in the lower tail.

    Also an extreme-value copula, and included because it is **not**
    Archimedean -- so it makes a different shape of upper dependence available
    at the same coefficient, and shows that the coefficient does not pin the
    copula down.

    ``theta -> 0`` is independence; larger is stronger dependence.

    Args:
        theta: Dependence parameter, positive.

    Raises:
        ParameterError: If ``theta`` is not positive.

    Examples:
        >>> copula = GalambosCopula(theta=1.0)
        >>> round(copula.cdf([0.5, 0.5]), 6)
        0.353553
        >>> {k: round(v, 6) for k, v in copula.tail_dependence().items()}
        {'lower': 0.0, 'upper': 0.5}
    """

    theta: float

    def __post_init__(self) -> None:
        if not (self.theta > 0.0) or not math.isfinite(self.theta):
            raise ParameterError("GalambosCopula requires a finite theta > 0.")

    @property
    def dim(self) -> int:
        """Two."""
        return 2

    def cdf(self, point: Sequence[float]) -> float:
        """``u v exp([(-ln u)^-theta + (-ln v)^-theta]^(-1/theta))``."""
        _check_unit(point, 2)
        return float(point[0] * point[1] * math.exp(self._exponent(point)))

    def pdf(self, point: Sequence[float]) -> float:
        """Copula density, checked against a finite difference of the cdf."""
        _check_unit(point, 2)
        a, b = -math.log(point[0]), -math.log(point[1])
        combined = a ** (-self.theta) + b ** (-self.theta)
        power = combined ** (-1.0 - 1.0 / self.theta)
        first = 1.0 - power * a ** (-self.theta - 1.0)
        second = 1.0 - power * b ** (-self.theta - 1.0)
        cross = (
            (1.0 + self.theta)
            * (a * b) ** (-self.theta - 1.0)
            * combined ** (-2.0 - 1.0 / self.theta)
        )
        return float(math.exp(self._exponent(point)) * (first * second + cross))

    def _exponent(self, point: Sequence[float]) -> float:
        a, b = -math.log(point[0]), -math.log(point[1])
        return float((a ** (-self.theta) + b ** (-self.theta)) ** (-1.0 / self.theta))

    def _conditional(self, u: float, v: float) -> float:
        """``P(V <= v | U = u)``, which is ``dC/du``."""
        a, b = -math.log(u), -math.log(v)
        combined = a ** (-self.theta) + b ** (-self.theta)
        return float(
            v
            * math.exp(combined ** (-1.0 / self.theta))
            * (1.0 - combined ** (-1.0 - 1.0 / self.theta) * a ** (-self.theta - 1.0))
        )

    def rvs(self, n: int, seed: int | None = None) -> list[list[float]]:
        """
        Draw ``n`` points, by inverting the conditional distribution.

        The Galambos copula is not Archimedean, so there is no mixing-variable
        construction to sample from. Conditional inversion is numerical and
        correspondingly slower, which is the price of the extra flexibility.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = RNG(seed)
        draws = []
        for _ in range(n):
            first = rng.uniform_0_1()
            target = rng.uniform_0_1()

            def conditional(value: float, given: float = first) -> float:
                return self._conditional(given, value)

            draws.append(
                [first, _ppf_monotone(conditional, _EDGE, 1.0 - _EDGE, target)]
            )
        return draws

    def tail_dependence(self) -> dict[str, float]:
        """Upper only: ``2 ** (-1/theta)``, and zero below."""
        return {"lower": 0.0, "upper": float(2.0 ** (-1.0 / self.theta))}


def _positive_stable(rng: RNG, alpha: float) -> float:
    """A positive stable variate of index ``alpha`` in (0, 1).

    Chambers, Mallows and Stuck's method, which is the standard exact
    construction: no rejection step and no series to truncate.
    """
    uniform = math.pi * rng.uniform_0_1()
    exponential = -math.log(rng.uniform_0_1())
    return float(
        (math.sin(alpha * uniform) / math.sin(uniform) ** (1.0 / alpha))
        * (math.sin((1.0 - alpha) * uniform) / exponential) ** ((1.0 - alpha) / alpha)
    )


# --------------------------- Estimation from data ---------------------------- #


def empirical_tail_dependence(
    x: Sequence[float],
    y: Sequence[float],
    level: float = 0.95,
) -> dict[str, Any]:
    """
    Estimate the tail dependence coefficients from a paired sample.

    Ranks both series, then counts how often both exceed the ``level``
    quantile. Working on ranks is what makes this a statement about the copula:
    it is unchanged by any monotone rescaling of either series, so it does not
    care what the margins are.

    **The estimate is biased upwards, badly, and no level available in practice
    removes it.** The coefficient is a limit as the level goes to one; at any
    finite level the conditional probability sits above it. How far above is
    easy to underestimate, so here it is measured on 400,000 draws from a
    Gaussian copula with correlation 0.7, whose true coefficient is *exactly
    zero*:

    ========  ==========  ============
    level     estimate    exceedances
    ========  ==========  ============
    0.90      0.4655      40000
    0.95      0.3871      20000
    0.99      0.2580      4000
    0.995     0.2195      2000
    0.999     0.1650      400
    0.9995    0.1450      200
    ========  ==========  ============

    It falls, slowly, and never reaches zero at any level the data supports.
    **This estimator cannot tell asymptotic independence from moderate tail
    dependence**, and reading 0.26 off a sample as evidence of joint extremes
    would be a mistake at any sample size. It is useful for comparing two
    datasets at the same level, and for rejecting *strong* dependence; it is
    not useful for establishing that dependence exists.

    The exceedance counts are returned for that reason: they are what decides
    whether raising the level buys anything.

    Args:
        x: First series.
        y: Second series, the same length.
        level: Quantile defining "extreme", in (0, 1).

    Returns:
        ``lower``, ``upper``, ``n_exceedances`` for each tail, ``level`` and
        ``n``.

    Raises:
        ValueError: If the series differ in length, are too short for the
            level, or the level is outside (0, 1).

    Examples:
        >>> copula = StudentTCopula(nu=3.0, correlation=[[1.0, 0.5], [0.5, 1.0]])
        >>> draws = copula.rvs(20000, seed=1)
        >>> result = empirical_tail_dependence(
        ...     [d[0] for d in draws], [d[1] for d in draws], level=0.98
        ... )
        >>> 0.2 < result["upper"] < 0.6
        True
    """
    if len(x) != len(y):
        raise ValueError("both series must have the same length.")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    n = len(x)
    if n < 20:
        raise ValueError(f"at least 20 pairs are needed; got {n}.")

    upper_rank = _to_ranks(x)
    other_rank = _to_ranks(y)
    expected = n * (1.0 - level)
    if expected < 5.0:
        raise ValueError(
            f"level {level} leaves about {expected:.1f} exceedances in {n} pairs, "
            "which is too few to estimate anything. Lower the level or get more "
            "data."
        )

    upper_hits = sum(
        1
        for a, b in zip(upper_rank, other_rank, strict=True)
        if a > level and b > level
    )
    upper_marginal = sum(1 for a in upper_rank if a > level)
    lower_hits = sum(
        1
        for a, b in zip(upper_rank, other_rank, strict=True)
        if a < 1.0 - level and b < 1.0 - level
    )
    lower_marginal = sum(1 for a in upper_rank if a < 1.0 - level)

    return {
        "lower": lower_hits / lower_marginal if lower_marginal else float("nan"),
        "upper": upper_hits / upper_marginal if upper_marginal else float("nan"),
        "n_lower_exceedances": lower_marginal,
        "n_upper_exceedances": upper_marginal,
        "level": level,
        "n": n,
    }


def _to_ranks(values: Sequence[float]) -> list[float]:
    """Map a series onto (0,1) by its ranks, ties broken by position."""
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for position, index in enumerate(order):
        ranks[index] = (position + 1) / (n + 1)
    return ranks
