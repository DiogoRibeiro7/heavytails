# extra_distributions.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from heavytails._array import as_array, check_probabilities, elementwise, restore

# The special functions live in heavytails._special so that heavy_tails.py can
# use them too without creating an import cycle. They are re-exported here
# because they were public-by-convention at this location first.
from heavytails._special import (
    _betainc_reg,
    _betaincinv_reg,
    _gammainc_lower_reg,
    _gammainc_upper_reg,
    _gammaincinv_reg,
    _log_beta,
    _ppf_monotone,
)
from heavytails.heavy_tails import (
    RNG,
    InverseTransformSampling,
    ParameterError,
    Samplable,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "BetaPrime",
    "BurrXII",
    "GeneralizedPareto",
    "InverseGamma",
    "LogLogistic",
    "_betainc_reg",
    "_gammainc_lower_reg",
    "_log_beta",
    "_ppf_monotone",
]


# =============================================================================
# Distributions
# =============================================================================


@dataclass(frozen=True)
class GeneralizedPareto(InverseTransformSampling):
    """
    Generalized Pareto Distribution (GPD) with shape xi, scale sigma>0, location mu.

    Support:
        x >= mu if xi >= 0  (heavy-tailed when xi>0)
        mu <= x <= mu - sigma/xi if xi < 0 (bounded tail; not heavy)

    CDF:
        F(x) = 1 - (1 + xi (x-mu)/sigma)^(-1/xi), valid where bracket > 0
    PDF:
        f(x) = (1/sigma) * (1 + xi z)^(-1/xi - 1),  z=(x-mu)/sigma
    PPF:
        x = mu + (sigma/xi) * ( (1-u)^(-xi) - 1 )      if xi != 0
        x = mu - sigma * ln(1-u)                     if xi = 0 (exponential limit)

    Examples:
        The limit law for exceedances over a high threshold, which is what
        makes peaks-over-threshold work at all:

        >>> gpd = GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0)
        >>> round(gpd.sf(1.0), 6)
        0.444444
        >>> round(gpd.ppf(0.5), 6)
        0.828427

        Positive ``xi`` gives a Pareto tail with index ``1 / xi``; negative
        ``xi`` bounds it above at ``mu - sigma / xi``, past which nothing
        falls:

        >>> GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0).cdf(2.0)
        1.0
        >>> gpd.cdf(-1.0)
        0.0
    """

    xi: float
    sigma: float = 1.0
    mu: float = 0.0

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise ParameterError("GPD requires sigma>0.")

    def _valid(self, x: Any) -> Any:
        """Whether ``x`` is inside the support, elementwise.

        Both halves matter. The bracket condition alone is the *upper* endpoint
        for a bounded (negative xi) distribution, and it is satisfied well
        below ``mu`` for every sign of xi -- so on its own it let points below
        the support through and produced probabilities like -2.586.
        """
        values = np.asarray(x, dtype=float)
        bracket = 1.0 + self.xi * ((values - self.mu) / self.sigma)
        return (values >= self.mu) & (bracket > 0.0)

    def _inner(self, z: Any) -> Any:
        """``xi z``, held above -1 wherever ``1 + xi z`` is not positive.

        This returns the argument to ``log1p`` rather than ``1 + xi z``
        itself, and that is the whole point of it. Forming the sum and taking
        an ordinary logarithm throws away the small quantity exactly as
        subtracting from one would: at x = 1e-9 the distribution function then
        comes back as 1.0000000820e-09 for a true 1e-09, wrong in the eighth
        digit, which is the same error this file already avoids twice by hand.

        The substitute value never reaches an answer -- every caller masks it
        away -- but it has to be something ``log1p`` and a power will accept
        quietly, because NumPy evaluates the whole array before the mask
        selects from it. A scalar implementation could return early instead;
        an array one computes the invalid entries and then discards them.
        """
        inner = self.xi * z
        return np.where(inner > -1.0, inner, 0.0)

    def pdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        z = (values - self.mu) / self.sigma
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.xi == 0.0:
                density = np.exp(-z) / self.sigma
            else:
                density = (1.0 / self.sigma) * (1.0 + self._inner(z)) ** (
                    -1.0 / self.xi - 1.0
                )
        return restore(np.where(self._valid(values), density, 0.0), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        z = (values - self.mu) / self.sigma
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.xi == 0.0:
                probability = -np.expm1(-z)
            else:
                # -expm1(-log1p(xi z)/xi) rather than 1 - (1 + xi z)**(-1/xi).
                # Both the power and the subtraction lose the small quantity:
                # at x = 1e-9 the naive form returns 1.0000000827e-09 for a
                # true 1e-09, wrong in the eighth digit, and the lower tail of
                # a GPD is a real part of it.
                probability = -np.expm1(-np.log1p(self._inner(z)) / self.xi)
        outside = np.where(values < self.mu, 0.0, 1.0)
        return restore(np.where(self._valid(values), probability, outside), scalar)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function, from the power itself rather than ``1 - cdf``.

        The distribution function approaches 1 in the tail, so subtracting it
        from 1 keeps only the digits the tail has already lost. This is the
        tail the distribution exists to describe, so it is computed directly.
        """
        values, scalar = as_array(x)
        z = (values - self.mu) / self.sigma
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            if self.xi == 0.0:
                survival = np.exp(-z)
            else:
                survival = np.exp(-np.log1p(self._inner(z)) / self.xi)
        outside = np.where(values < self.mu, 1.0, 0.0)
        return restore(np.where(self._valid(values), survival, outside), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore"):
            if self.xi == 0.0:
                quantile = self.mu - self.sigma * np.log1p(-values)
            else:
                # expm1(-xi log1p(-u)) rather than (1-u)**-xi - 1. Both the
                # subtraction inside the power and the one after it cancel for
                # small u: the textbook form returns 1.0000000827e-09 where
                # the answer is 1.0000000007e-09, wrong in the eighth digit,
                # and the lower tail of a GPD is not a corner case.
                quantile = self.mu + (self.sigma / self.xi) * np.expm1(
                    -self.xi * np.log1p(-values)
                )
        return restore(quantile, scalar)


@dataclass(frozen=True)
class BurrXII(InverseTransformSampling):
    """
    Burr Type XII with shapes c>0, k>0 and scale s>0.

    CDF: F(x) = 1 - (1 + (x/s)^c)^(-k),   x > 0
    PDF: f(x) = (ck/s) * (x/s)^(c-1) * (1 + (x/s)^c)^(-k-1)
    PPF: x = s * ( (1 - u)^(-1/k) - 1 )^(1/c)

    Examples:
        The tail index is the product ``c * k``, so shape and scale can be
        traded against each other:

        >>> burr = BurrXII(c=2.0, k=1.0, s=1.0)
        >>> burr.cdf(1.0)
        0.5
        >>> round(burr.sf(3.0), 10)
        0.1
        >>> round(BurrXII(c=1.0, k=2.0, s=1.0).sf(3.0), 6)
        0.0625
    """

    c: float
    k: float
    s: float = 1.0

    def __post_init__(self) -> None:
        if not (self.c > 0 and self.k > 0 and self.s > 0):
            raise ParameterError("BurrXII requires c>0, k>0, s>0.")

    def _z(self, values: Any) -> Any:
        """``(x/s)**c`` on the positive half line, zero elsewhere.

        The base is guarded rather than the result: a negative base under a
        fractional power is a NaN, and a NaN does not stay in the entry that
        produced it once it meets an addition.
        """
        positive = np.where(values > 0.0, values, 0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return (positive / self.s) ** self.c

    def pdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        z = self._z(values)
        positive = np.where(values > 0.0, values, 1.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            density = (
                (self.c * self.k / self.s)
                * (positive / self.s) ** (self.c - 1.0)
                * (1.0 + z) ** (-self.k - 1.0)
            )
        return restore(np.where(values > 0.0, density, 0.0), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        # See GeneralizedPareto.cdf for why this is not 1 - (1+z)**-k.
        with np.errstate(over="ignore", invalid="ignore"):
            probability = -np.expm1(-self.k * np.log1p(self._z(values)))
        return restore(np.where(values > 0.0, probability, 0.0), scalar)

    def sf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            survival = (1.0 + self._z(values)) ** (-self.k)
        return restore(np.where(values > 0.0, survival, 1.0), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore", divide="ignore"):
            # See GeneralizedPareto.ppf: expm1 of a log1p keeps the lower tail,
            # where (1-u)**(-1/k) - 1 loses all but about seven digits.
            quantile = self.s * np.expm1(-np.log1p(-values) / self.k) ** (1.0 / self.c)
        return restore(quantile, scalar)


@dataclass(frozen=True)
class LogLogistic(InverseTransformSampling):
    """
    Log-Logistic (Fisk) with shape kappa>0 and scale lambda_>0 (support x>0).
    CDF: F(x) = 1 / (1 + (lambda_/x)^kappa) = (x^kappa) / (x^kappa + lambda_^kappa)
    PDF: f(x) = (kappa/lambda_) (x/lambda_)^(kappa-1) / (1 + (x/lambda_)^kappa)^2
    PPF: x = lambda_ * (u/(1-u))^(1/kappa)

    Examples:
        The median is ``lam`` exactly, whatever the shape:

        >>> loglogistic = LogLogistic(kappa=2.0, lam=3.0)
        >>> loglogistic.cdf(3.0)
        0.5
        >>> round(loglogistic.ppf(0.5), 10)
        3.0

        The tail index is ``kappa``:

        >>> round(loglogistic.sf(30.0), 6)
        0.009901
    """

    kappa: float
    lam: float = 1.0

    def __post_init__(self) -> None:
        if not (self.kappa > 0 and self.lam > 0):
            raise ParameterError("LogLogistic requires kappa>0 and lam>0.")

    def _z(self, values: Any) -> Any:
        """``(x/lam)**kappa`` on the positive half line. See BurrXII._z."""
        positive = np.where(values > 0.0, values, 0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            return (positive / self.lam) ** self.kappa

    def pdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        z = self._z(values)
        positive = np.where(values > 0.0, values, 1.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            density = (
                (self.kappa / self.lam)
                * (positive / self.lam) ** (self.kappa - 1.0)
                / (1.0 + z) ** 2
            )
        return restore(np.where(values > 0.0, density, 0.0), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        z = self._z(values)
        with np.errstate(over="ignore", invalid="ignore"):
            probability = z / (1.0 + z)
        return restore(np.where(values > 0.0, probability, 0.0), scalar)

    def sf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        with np.errstate(over="ignore", invalid="ignore"):
            survival = 1.0 / (1.0 + self._z(values))
        return restore(np.where(values > 0.0, survival, 1.0), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore", divide="ignore"):
            quantile = self.lam * (values / (1.0 - values)) ** (1.0 / self.kappa)
        return restore(quantile, scalar)


@dataclass(frozen=True)
class InverseGamma(Samplable):
    """
    Inverse-Gamma with shape alpha>0 and scale β>0 (support x>0).
    PDF: f(x) = β^alpha / Gamma(alpha) * x^{-alpha-1} * exp(-β/x)
    CDF: F(x) = Q(alpha, β/x) = Gamma(alpha, β/x) / Gamma(alpha)  (regularized upper gamma)
         where Q = 1 - P and P is the regularized lower gamma.
    Sampling: If G ~ Gamma(alpha, scale=1), then X = β / G has InvGamma(alpha, β).

    Examples:
        Its tail index is ``alpha``, and its *lower* tail is the interesting
        numerical case: the probability there is far too small to reach by
        subtracting from one.

        >>> inverse_gamma = InverseGamma(alpha=2.0, beta=1.0)
        >>> round(inverse_gamma.ppf(0.5), 6)
        0.595824
        >>> f"{inverse_gamma.cdf(0.02):.6e}"
        '9.836624e-21'
        >>> round(inverse_gamma.sf(1.0), 6)
        0.264241
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.beta > 0):
            raise ParameterError("InverseGamma requires alpha>0 and beta>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density.

        Elementary, so it is a single NumPy expression. The probabilities
        below are not -- they need the incomplete gamma, which NumPy does not
        have -- and they go one element at a time through
        :func:`~heavytails._array.elementwise`.
        """
        values, scalar = as_array(x)
        a, b = self.alpha, self.beta
        positive = np.where(values > 0.0, values, 1.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            density = (
                (b**a / math.exp(math.lgamma(a)))
                * positive ** (-a - 1.0)
                * np.exp(-b / positive)
            )
        return restore(np.where(values > 0.0, density, 0.0), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        return restore(elementwise(self._cdf_one, values), scalar)

    def _cdf_one(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        a, b = self.alpha, self.beta
        # F(x) = Q(a, b/x), taken from the upper incomplete gamma rather than
        # as 1 - P. For small x the probability is tiny and 1 - P returns
        # exactly zero: at alpha=2, x=0.02 the true value is 9.8e-21 and the
        # subtraction gives nothing at all. The lower tail of a heavy-tailed
        # distribution is not the place to lose every digit.
        return _gammainc_upper_reg(a, b / x)

    def sf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        return restore(elementwise(self._sf_one, values), scalar)

    def _sf_one(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        # The mirror of the above: here P is the small quantity.
        return _gammainc_lower_reg(self.alpha, self.beta / x)

    def ppf(self, u: ArrayLike) -> Any:
        values, scalar = as_array(u)
        check_probabilities(values)
        return restore(elementwise(self._ppf_one, values), scalar)

    def _ppf_one(self, u: float) -> float:
        """Quantile function, by inverting the incomplete gamma directly.

        ``F(x) = Q(alpha, beta/x)``, so the quantile is ``beta`` divided by the
        inverse incomplete gamma. That replaces a bracket-and-solve against the
        distribution function, which paid a continued fraction per iteration
        and reached only about seven digits in the far tail.

        Which of the two inverses is used depends on the side, and that choice
        is the whole point: for small ``u`` the small quantity is ``u`` itself,
        and going through the lower inverse would mean forming ``1 - u`` and
        throwing away exactly the precision the tail is being asked about.
        """
        if u <= 0.5:
            y = _gammaincinv_reg(self.alpha, u, upper=True)
        else:
            y = _gammaincinv_reg(self.alpha, 1.0 - u)
        if y <= 0.0:
            return math.inf
        return float(self.beta / y)

    def _rvs_one(self, rng: RNG) -> float:
        g = rng.gamma(shape_k=self.alpha, scale_theta=1.0)
        return self.beta / g


@dataclass(frozen=True)
class BetaPrime(Samplable):
    """
    Beta-Prime (a.k.a. Inverse-Beta, Pearson Type VI) with shapes a>0, b>0 and scale s>0.

    PDF: f(x) = 1 / (s * B(a,b)) * (x/s)^(a-1) * (1 + x/s)^(-a-b),  x>0
    CDF: F(x) = I_{ y }(a,b) with y = x / (x + s)  (regularized incomplete beta)
    PPF: No closed form in general -> monotone numeric inversion.
    Sampling: If U~Gamma(a,1), V~Gamma(b,1), then X = s * U/V ~ BetaPrime(a,b,s).

    Examples:
        A ratio of two gamma variates, with tail index ``b``:

        >>> beta_prime = BetaPrime(a=2.0, b=3.0, s=1.0)
        >>> round(beta_prime.cdf(1.0), 10)
        0.6875
        >>> round(beta_prime.ppf(0.5), 6)
        0.627942

        The survival function stays accurate where one minus the distribution
        function would have run out of digits:

        >>> f"{beta_prime.sf(1e6):.6e}"
        '3.999985e-18'
    """

    a: float
    b: float
    s: float = 1.0

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > 0 and self.s > 0):
            raise ParameterError("BetaPrime requires a>0, b>0, s>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density.

        Elementary, so it is one NumPy expression. The probabilities below
        need the incomplete beta, which NumPy does not have, and go one
        element at a time through :func:`~heavytails._array.elementwise`.
        """
        values, scalar = as_array(x)
        a, b, s = self.a, self.b, self.s
        z = np.where(values > 0.0, values, 1.0) / s
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            density = (
                math.exp(-(math.log(s) + _log_beta(a, b)))
                * z ** (a - 1.0)
                * (1.0 + z) ** (-(a + b))
            )
        return restore(np.where(values > 0.0, density, 0.0), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        return restore(elementwise(self._cdf_one, values), scalar)

    def _cdf_one(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        y = x / (x + self.s)
        return _betainc_reg(self.a, self.b, y)

    def sf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        return restore(elementwise(self._sf_one, values), scalar)

    def _sf_one(self, x: float) -> float:
        """Survival function, from the mirrored incomplete beta.

        ``1 - I_z(a,b)`` is ``I_{1-z}(b,a)``, and ``1 - z`` is ``s/(x+s)``,
        which is computed rather than subtracted. It matters in the far upper
        tail: ``z = x/(x+s)`` rounds to exactly 1 once ``x`` reaches about
        1e17, so the complement of the distribution function is zero there
        while the true probability is around 1e-9. That tail is the reason
        anyone reaches for this distribution.
        """
        if x <= 0.0:
            return 1.0
        if x <= self.s:
            # Below the scale the survival probability is the large one, and
            # the mirrored form would be the mistake: x + s rounds to s when x
            # is small enough, so s/(x+s) becomes exactly 1 and the answer
            # comes back as 1 with the interesting part gone. At x = 1e-16 with
            # s = 1 that turned a survival of 0.999999 into 1.
            return 1.0 - self._cdf_one(x)
        return _betainc_reg(self.b, self.a, self.s / (x + self.s))

    def ppf(self, u: ArrayLike) -> Any:
        values, scalar = as_array(u)
        check_probabilities(values)
        return restore(elementwise(self._ppf_one, values), scalar)

    def _ppf_one(self, u: float) -> float:
        """Quantile function, by inverting the incomplete beta directly.

        If ``Y`` is ``Beta(a, b)`` then ``s Y / (1 - Y)`` is this distribution,
        so the quantile follows from the inverse incomplete beta with no
        solver against the distribution function at all.

        The side matters. Near ``u = 1`` the beta quantile approaches one, and
        ``1 - z`` there is a subtraction of two nearly equal numbers, so the
        upper tail would come back with a handful of digits. Solving the
        mirrored problem instead makes ``1 - z`` the quantity that is computed
        rather than the quantity that is cancelled.
        """
        if u <= 0.5:
            z = _betaincinv_reg(self.a, self.b, u)
            if z >= 1.0:
                return math.inf
            return float(self.s * z / (1.0 - z))
        w = _betaincinv_reg(self.b, self.a, 1.0 - u)
        if w <= 0.0:
            return math.inf
        return float(self.s * (1.0 - w) / w)

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.gamma(shape_k=self.a, scale_theta=1.0)
        v = rng.gamma(shape_k=self.b, scale_theta=1.0)
        return self.s * (u / v)


# =============================================================================
# Minimal self-test / examples
# =============================================================================


def _demo() -> None:
    seed = 123

    gpd = GeneralizedPareto(xi=0.3, sigma=2.0, mu=1.0)
    print("GPD ppf(0.99) =", gpd.ppf(0.99), "cdf(ppf) =", gpd.cdf(gpd.ppf(0.99)))
    print("GPD samples:", gpd.rvs(5, seed))

    burr = BurrXII(c=1.2, k=2.5, s=3.0)
    print("Burr ppf(0.9) =", burr.ppf(0.9))
    print("Burr samples:", burr.rvs(5, seed))

    fisk = LogLogistic(kappa=1.5, lam=2.0)
    print("Fisk ppf(0.95) =", fisk.ppf(0.95))
    print("Fisk samples:", fisk.rvs(5, seed))

    invg = InverseGamma(alpha=3.5, beta=2.0)
    xq = invg.ppf(0.9)
    print("InvGamma ppf(0.9)=", xq, "cdf(ppf)=", invg.cdf(xq))
    print("InvGamma samples:", invg.rvs(5, seed))

    bp = BetaPrime(a=2.0, b=3.0, s=1.0)
    xq = bp.ppf(0.9)
    print("BetaPrime ppf(0.9)=", xq, "cdf(ppf)=", bp.cdf(xq))
    print("BetaPrime samples:", bp.rvs(5, seed))


if __name__ == "__main__":
    _demo()
