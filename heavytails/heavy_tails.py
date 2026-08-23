# heavy_tails.py
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from heavytails._array import (
    as_array,
    check_probabilities,
    elementwise,
    restore,
    select,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
import random

from heavytails._special import _betainc_reg, _betaincinv_reg, _phi_inverse

# ----------------------------- Utilities ------------------------------------ #


class ParameterError(ValueError):
    """Raised when distribution parameters are invalid."""


class RNG:
    """
    Thin wrapper around random.Random for reproducibility and isolation.

    Attributes
    ----------
    rng : random.Random
        Underlying random number generator.

    Examples:
        Seeded, which is what makes a simulation study checkable rather than
        merely plausible:

        >>> RNG(42).uniform_0_1() == RNG(42).uniform_0_1()
        True
    """

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def uniform_0_1(self) -> float:
        """U ~ Uniform(0,1) in (0,1), clipped away from exact 0 and 1 for log/ppf stability."""
        # Avoid 0 and 1 to prevent log(0) or tan(pi*(U-0.5)) exploding from exactly 0.5
        u = self.rng.random()
        eps = 1e-16
        return min(max(u, eps), 1.0 - eps)

    def standard_normal(self) -> float:
        """Z ~ N(0,1). Uses Python stdlib Box-Muller via random.gauss (Ziggurat internally)."""
        return self.rng.gauss(0.0, 1.0)

    # ---------------------- Gamma / Chi-square samplers ---------------------- #
    def gamma(self, shape_k: float, scale_theta: float = 1.0) -> float:
        """
        X ~ Gamma(k, θ) with k>0, θ>0 using Marsaglia-Tsang (2000).
        Works for all k>0 (uses boost for k<1).

        References
        ----------
        G. Marsaglia and W. W. Tsang (2000). A Simple Method for Generating Gamma Variables.
        ACM Transactions on Mathematical Software 26(3):363-372.
        """
        if not (shape_k > 0 and scale_theta > 0):
            raise ParameterError("Gamma requires shape k>0 and scale θ>0.")

        k = shape_k
        if k < 1.0:
            # Boost: sample from Gamma(k+1, 1) then * U^(1/k)
            x = self._gamma_mt(k + 1.0)
            u = self.uniform_0_1()
            return float(scale_theta * (x * (u ** (1.0 / k))))
        else:
            return float(scale_theta * self._gamma_mt(k))

    def _gamma_mt(self, k: float) -> float:
        """Marsaglia-Tsang core for k >= 1, unit scale."""
        d = k - 1.0 / 3.0
        c = 1.0 / math.sqrt(9.0 * d)
        while True:
            z = self.standard_normal()
            v = 1.0 + c * z
            if v <= 0.0:
                continue
            v = v * v * v
            u = self.uniform_0_1()
            # Squeeze / acceptance tests
            if u < 1.0 - 0.0331 * (z**4):
                return d * v
            if math.log(u) < 0.5 * z * z + d * (1.0 - v + math.log(v)):
                return d * v

    def chisquare(self, df: float) -> float:
        """χ²(df) via Gamma(k=df/2, θ=2)."""
        if df <= 0:
            raise ParameterError("Chi-square requires df > 0.")
        return self.gamma(shape_k=df / 2.0, scale_theta=2.0)


# --------------------------- Base mixin -------------------------------------- #


class Samplable:
    """Mixin to provide vectorized sampling with a given RNG.

    Examples:
        Every distribution here draws through it, so the same seed gives the
        same sample:

        >>> Pareto(alpha=2.0, xm=1.0).rvs(3, seed=1) == Pareto(
        ...     alpha=2.0, xm=1.0
        ... ).rvs(3, seed=1)
        True
    """

    def rvs(self, n: int, seed: int | None = None) -> list[float]:
        """
        Draw n IID variates. Subclasses implement ._rvs_one(rng), or
        ._rvs_many(rng, n) when the whole sample can be drawn at once.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = RNG(seed)
        return self._rvs_many(rng, n)

    def _rvs_many(self, rng: RNG, n: int) -> list[float]:
        """
        Draw the whole sample.

        The default calls ._rvs_one n times, which is correct for any family
        and is what a family with no batched form should keep doing.
        :class:`InverseTransformSampling` overrides it.
        """
        return [self._rvs_one(rng) for _ in range(n)]

    def _rvs_one(self, rng: RNG) -> float:
        """Override in subclasses."""
        raise NotImplementedError


class InverseTransformSampling(Samplable):
    """
    Sampling by feeding uniforms through the quantile function.

    Draws the uniforms in a loop -- the generator is Python's and each draw
    depends on the one before, so there is nothing to vectorise there -- and
    then inverts all of them in a single call.

    That second half is where the time was going. Inverting 50,000 uniforms
    one at a time costs about 0.30s, because it is 50,000 separate dispatches
    through :func:`~heavytails._array.as_array` and back; inverting them in one
    call costs about 0.8ms.

    The uniforms are the same either way, so a seeded sample is reproducible
    and its distribution is unchanged. The variates are *almost* the same:
    NumPy's vectorised loops for ``log`` and ``pow`` round differently from its
    scalar ones, so a few draws land one ULP away from what the scalar path
    gave. For Weibull(k=0.5) that is 14 values in 20,000, worst case a relative
    1.99e-16, with the sample mean unchanged to twelve digits. Tests that pin
    seeded variates to the last bit will see it; nothing measuring a
    distribution will.

    Examples:
        >>> Pareto(alpha=2.0, xm=1.0).rvs(3, seed=7) == Pareto(
        ...     alpha=2.0, xm=1.0
        ... ).rvs(3, seed=7)
        True
    """

    def ppf(self, u: ArrayLike) -> Any:
        """The quantile function, which every family mixing this in defines.

        Declared here because the batched sampler below is written in terms of
        it: the mixin is not merely a helper, it is a statement that the family
        samples by inversion and therefore has a quantile function to invert.
        """
        raise NotImplementedError

    def _rvs_many(self, rng: RNG, n: int) -> list[float]:
        uniforms = np.fromiter(
            (rng.uniform_0_1() for _ in range(n)), dtype=float, count=n
        )
        return [float(value) for value in np.asarray(self.ppf(uniforms))]

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        # ppf mirrors its input, so a scalar in gives a float out. The cast
        # says so to the type checker, which only sees the array-or-scalar
        # return type.
        return float(self.ppf(u))


# --------------------------- Distributions ----------------------------------- #


@dataclass(frozen=True)
class Pareto(InverseTransformSampling):
    """
    Pareto Type I with scale xm>0 and shape alpha>0.
    PDF: f(x) = alpha x_m^alpha / x^{alpha+1},  x >= x_m
    CDF: F(x) = 1 - (x_m / x)^alpha
    PPF: F^{-1}(u) = x_m * (1 - u)^{-1/alpha}

    Examples:
        The defining property is a straight line on a log-log tail plot: the
        survival function falls by a factor of ``10 ** alpha`` per decade.

        >>> pareto = Pareto(alpha=2.0, xm=1.0)
        >>> round(pareto.sf(10.0), 10)
        0.01
        >>> round(pareto.sf(100.0), 10)
        0.0001
        >>> pareto.cdf(2.0)
        0.75
        >>> round(pareto.ppf(0.5), 6)
        1.414214
    """

    alpha: float
    xm: float = 1.0

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.xm > 0):
            raise ParameterError("Pareto requires alpha>0 and xm>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density. A number in gives a number out, a sequence gives an array."""
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            density = self.alpha * self.xm**self.alpha / values ** (self.alpha + 1.0)
        return restore(select(values < self.xm, 0.0, density), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        """Distribution function."""
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            below = 1.0 - (self.xm / values) ** self.alpha
        return restore(select(values < self.xm, 0.0, below), scalar)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function.

        Computed directly rather than as ``1 - cdf``, which returns exactly
        zero once the survival probability drops below the spacing of one.
        """
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            above = (self.xm / values) ** self.alpha
        return restore(select(values < self.xm, 1.0, above), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function.

        Returns ``inf`` where the quantile exceeds the float range. At extreme
        parameters the true value is genuinely not representable, and reporting
        it is more useful than raising, which aborts a parameter sweep at the
        first point that overflows.

        Raises:
            ValueError: If any probability is outside (0, 1). The whole input
                is checked before any work starts, so the message can name the
                offending value instead of failing part way through.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore"):
            quantile = self.xm * (1.0 - values) ** (-1.0 / self.alpha)
        return restore(quantile, scalar)


@dataclass(frozen=True)
class Cauchy(InverseTransformSampling):
    """
    Cauchy(location x0, scale gamma>0).
    PDF: f(x) = [1/πgamma] * [1 / (1 + ((x-x0)/gamma)^2)]
    CDF: F(x) = 0.5 + (1/π) * arctan((x - x0)/gamma)
    PPF: x = x0 + gamma * tan(π(u - 0.5))

    Examples:
        Symmetric, and heavy enough to have no mean at all -- the sample mean
        of Cauchy draws is itself Cauchy, so more data does not help.

        >>> cauchy = Cauchy(x0=0.0, gamma=1.0)
        >>> cauchy.cdf(0.0)
        0.5
        >>> round(cauchy.ppf(0.75), 10)
        1.0
        >>> round(cauchy.pdf(0.0), 6)
        0.31831

        The tail decays like ``1/x``, so the survival function times ``x``
        tends to ``1/pi``:

        >>> round(cauchy.sf(1e6) * 1e6, 5)
        0.31831
    """

    x0: float = 0.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not (self.gamma > 0):
            raise ParameterError("Cauchy requires scale gamma>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density."""
        values, scalar = as_array(x)
        z = (values - self.x0) / self.gamma
        return restore(1.0 / (math.pi * self.gamma * (1.0 + z * z)), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        """Distribution function.

        For very negative ``z`` the arctangent approaches ``-pi/2`` and adding
        one half cancels it away, leaving about seven digits of a probability
        that is the whole point of the left tail. ``arctan(-1/z)`` is the same
        number with its argument near zero, where it is exact.
        """
        values, scalar = as_array(x)
        z = (values - self.x0) / self.gamma
        with np.errstate(divide="ignore", invalid="ignore"):
            lower = np.arctan(-1.0 / z) / math.pi
            upper = 1.0 - np.arctan(1.0 / z) / math.pi
        middle = 0.5 + np.arctan(z) / math.pi
        return restore(select(z < -1.0, lower, select(z > 1.0, upper, middle)), scalar)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function 1 - CDF.

        Computed as ``atan(1/z)/pi`` in the upper tail rather than as
        ``1 - cdf(x)``. For large z the latter subtracts two numbers that agree
        to every displayed digit and collapses to exactly zero, whereas
        ``atan(1/z) -> 1/z`` keeps full relative precision.
        """
        values, scalar = as_array(x)
        z = (values - self.x0) / self.gamma
        with np.errstate(divide="ignore", invalid="ignore"):
            positive = np.arctan(1.0 / z) / math.pi
        return restore(select(z > 0.0, positive, 0.5 - np.arctan(z) / math.pi), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function.

        Returns ``inf`` when the quantile exceeds the float range. At extreme
        parameters the true value is genuinely not representable, and reporting
        it is more useful than raising, which aborts a parameter sweep at the
        first point that overflows.

        Uses the cotangent form in the tails. The textbook
        ``tan(pi (u - 1/2))`` puts its argument next to ``+-pi/2``, where the
        tangent is arbitrarily steep, so a rounding error of one ulp in the
        argument becomes an enormous error in the result: at ``u = 1e-9`` it
        returns -318309868.8 where the answer is -318309886.2, wrong in the
        eighth digit. ``cot(pi u)`` has its argument near zero instead, where
        it is computed exactly, and the tail is the only part of a Cauchy
        anyone cares about.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(divide="ignore", invalid="ignore"):
            low = self.x0 - self.gamma / np.tan(math.pi * values)
            high = self.x0 + self.gamma / np.tan(math.pi * (1.0 - values))
        middle = self.x0 + self.gamma * np.tan(math.pi * (values - 0.5))
        return restore(
            select(values < 0.25, low, select(values > 0.75, high, middle)),
            scalar,
        )


@dataclass(frozen=True)
class StudentT(Samplable):
    """
    Student's t with degrees of freedom nu>0.
    PDF: f(x) = Gamma((nu+1)/2) / [ sqrt(nuπ) Gamma(nu/2) ] * (1 + x^2/nu)^(-(nu+1)/2)
    Sampling: X = Z / sqrt(Y/nu) with Z~N(0,1), Y~χ²(nu)

    The CDF, survival function and quantile function are expressed through the
    regularized incomplete beta function in ``heavytails._special``, so no
    third-party dependency is required.

    Examples:
        The tail index is ``nu``, so moments below it exist and the rest do
        not. One degree of freedom is the Cauchy exactly:

        >>> round(StudentT(nu=1.0).cdf(1.0), 10)
        0.75
        >>> round(Cauchy(x0=0.0, gamma=1.0).cdf(1.0), 10)
        0.75

        More degrees of freedom means a lighter tail:

        >>> round(StudentT(nu=1.0).sf(10.0), 6)
        0.031726
        >>> round(StudentT(nu=4.0).sf(10.0), 6)
        0.000281
    """

    nu: float

    def __post_init__(self) -> None:
        if not (self.nu > 0):
            raise ParameterError("Student-t requires nu>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density. Elementary, so this one vectorises."""
        values, scalar = as_array(x)
        nu = self.nu
        constant = math.gamma((nu + 1.0) / 2.0) / (
            math.sqrt(nu * math.pi) * math.gamma(nu / 2.0)
        )
        density = constant * (1.0 + values * values / nu) ** (-(nu + 1.0) / 2.0)
        return restore(density, scalar)

    def _tail_half(self, x: float) -> float:
        """Return P(|X| > |x|) / 2 = I_{nu/(nu+x^2)}(nu/2, 1/2) / 2.

        This is the building block for both the CDF and the survival function.
        Working with it directly keeps the far tail accurate, because it never
        forms the difference of two nearly equal numbers.
        """
        nu = self.nu
        # x**2 can overflow for very large |x|; nu / (nu + x*x) -> 0 there, and
        # the incomplete beta is continuous at 0, so clamping is exact enough.
        try:
            y = nu / (nu + x * x)
        except OverflowError:  # pragma: no cover - only for |x| near the float limit
            y = 0.0
        return 0.5 * _betainc_reg(nu / 2.0, 0.5, y)

    def cdf(self, x: ArrayLike) -> Any:
        """Distribution function, via the regularized incomplete beta.

        Evaluated one point at a time. NumPy has no incomplete beta, so there
        is no expression to vectorise and the loop is the implementation
        rather than a stand-in for one. :meth:`pdf` is elementary and does
        vectorise.
        """
        values, scalar = as_array(x)
        return restore(elementwise(self._cdf_one, values), scalar)

    def _cdf_one(self, x: float) -> float:
        half = self._tail_half(x)
        return float(1.0 - half) if x >= 0.0 else float(half)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function, computed directly for tail accuracy."""
        values, scalar = as_array(x)
        return restore(elementwise(self._sf_one, values), scalar)

    def _sf_one(self, x: float) -> float:
        half = self._tail_half(x)
        return float(half) if x >= 0.0 else float(1.0 - half)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function, obtained by inverting the incomplete beta.

        Uses the symmetry of the Student-t about zero so that only the upper
        half is solved, then inverts ``I_y(nu/2, 1/2) = 2(1-u)`` directly.
        Inverting in ``y`` rather than in ``x`` is what keeps the far tail
        accurate: a quantile solver working to an absolute tolerance on the CDF
        loses most of its digits where the density is vanishingly small.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        return restore(elementwise(self._ppf_one, values), scalar)

    def _ppf_one(self, u: float) -> float:
        """One quantile, by inverting the incomplete beta."""
        if u == 0.5:
            return 0.0

        nu = self.nu
        # Both halves reduce to the same inversion because the density is
        # symmetric: the tail probability beyond |x| is I_y(nu/2, 1/2) / 2 with
        # y = nu / (nu + x^2). Reflecting the lower half as `ppf(1 - u)` would
        # be wrong for tiny u, where `1 - u` rounds to exactly 1.0 and the
        # quantile information is lost entirely.
        if u < 0.5:
            tail, sign = 2.0 * u, -1.0
        else:
            tail, sign = 2.0 * (1.0 - u), 1.0

        y = _betaincinv_reg(nu / 2.0, 0.5, tail)
        if y <= 0.0:  # pragma: no cover - u indistinguishable from 0.0 or 1.0
            return sign * math.inf
        return float(sign * math.sqrt(nu * (1.0 / y - 1.0)))

    def _rvs_one(self, rng: RNG) -> float:
        z = rng.standard_normal()
        y = rng.chisquare(self.nu)
        return z / math.sqrt(y / self.nu)


@dataclass(frozen=True)
class LogNormal(Samplable):
    """
    LogNormal with underlying Normal(mu, sigma^2), sigma>0.
    PDF: f(x) = [1/(x sigma sqrt(2π))] * exp( -(ln x - mu)^2 / (2sigma^2) ), x>0
    CDF: F(x) = 0.5 * [1 + erf( (ln x - mu) / (sigma sqrt(2)) )], x>0

    Examples:
        The median is ``exp(mu)`` exactly:

        >>> lognormal = LogNormal(mu=0.0, sigma=1.0)
        >>> lognormal.ppf(0.5)
        1.0
        >>> round(lognormal.cdf(1.0), 10)
        0.5

        Heavy-tailed but **not** regularly varying: the tail decays faster
        than any power, so a tail index estimator applied to it returns
        something that does not mean what it usually means.

        >>> round(lognormal.sf(10.0), 6)
        0.010651
    """

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise ParameterError("LogNormal requires sigma>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density. Elementary, so this one vectorises."""
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (np.log(values) - self.mu) / self.sigma
            density = np.exp(-0.5 * z * z) / (
                values * self.sigma * math.sqrt(2.0 * math.pi)
            )
        return restore(select(values <= 0.0, 0.0, density), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        """Distribution function.

        Evaluated one point at a time: it needs the complementary error
        function and NumPy has none, so the loop is the implementation rather
        than a stand-in. :meth:`pdf` is elementary and does vectorise.
        """
        values, scalar = as_array(x)
        return restore(elementwise(self._cdf_one, values), scalar)

    def _cdf_one(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (math.log(x) - self.mu) / (self.sigma * math.sqrt(2.0))
        # 0.5*erfc(-z), not 0.5*(1 + erf(z)): for z a few units negative the
        # erf term is -0.99999999999, and the sum keeps five digits of a
        # probability whose leading digit is all anyone wanted.
        return 0.5 * math.erfc(-z)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function 1 - CDF, computed with ``erfc`` for tail accuracy.

        ``1 - cdf(x)`` collapses to exactly zero once ``cdf(x)`` rounds to 1.0,
        which happens well inside the range of interest. ``erfc`` is accurate
        for large arguments and keeps the true decay.
        """
        values, scalar = as_array(x)
        return restore(elementwise(self._sf_one, values), scalar)

    def _sf_one(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        z = (math.log(x) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * math.erfc(z)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function.

        Returns ``inf`` when the quantile exceeds the float range rather than
        raising. For large ``mu`` the answer is genuinely not representable --
        the median of ``LogNormal(mu=1000)`` is ``exp(1000)`` -- and ``inf`` is
        the correct value to report for it.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        return restore(elementwise(self._ppf_one, values), scalar)

    def _ppf_one(self, u: float) -> float:
        # The normal quantile needs erfinv, which is not in the standard
        # library and not in NumPy either, so this stays a scalar routine.
        z = _phi_inverse(u)
        try:
            return math.exp(self.mu + self.sigma * z)
        except OverflowError:
            return math.inf

    def _rvs_one(self, rng: RNG) -> float:
        z = rng.standard_normal()
        return math.exp(self.mu + self.sigma * z)


@dataclass(frozen=True)
class Weibull(InverseTransformSampling):
    """
    Weibull(k, lambda_) with shape k>0 and scale lambda_>0.
    PDF: f(x) = (k/lambda_) (x/lambda_)^{k-1} exp(-(x/lambda_)^k), x>=0
    CDF: F(x) = 1 - exp(-(x/lambda_)^k), x>=0
    PPF: x = lambda_ * (-ln(1-u))^{1/k}
    Heavy-tailed for k in (0,1) (subexponential, slower than exponential decay).

    Examples:
        Shape one is the exponential distribution, whose survival function at
        the scale is ``1/e``:

        >>> round(Weibull(k=1.0, lam=1.0).sf(1.0), 6)
        0.367879

        Below one it is heavy-tailed, and the density is unbounded at the
        origin:

        >>> Weibull(k=0.5, lam=1.0).pdf(0.0)
        inf
        >>> round(Weibull(k=0.5, lam=1.0).sf(4.0), 6)
        0.135335
    """

    k: float
    lam: float = 1.0

    def __post_init__(self) -> None:
        if not (self.k > 0 and self.lam > 0):
            raise ParameterError("Weibull requires k>0 and lambda_>0.")

    def pdf(self, x: ArrayLike) -> Any:
        """Density, which diverges at the origin for shape below one."""
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (values / self.lam) ** self.k
            density = (
                (self.k / self.lam) * (values / self.lam) ** (self.k - 1.0) * np.exp(-z)
            )
        if self.k < 1.0:
            # The limit at the origin. The expression above is 0 ** negative
            # there, which is an error rather than a statement about the
            # density.
            density = select(values == 0.0, np.inf, density)
        return restore(select(values < 0.0, 0.0, density), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        """Distribution function, via ``-expm1`` so the lower tail survives."""
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            below = -np.expm1(-((values / self.lam) ** self.k))
        return restore(select(values < 0.0, 0.0, below), scalar)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function: 1 - CDF(x)."""
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            above = np.exp(-((values / self.lam) ** self.k))
        return restore(select(values < 0.0, 1.0, above), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function.

        Returns ``inf`` when the quantile exceeds the float range. At extreme
        parameters the true value is genuinely not representable, and reporting
        it is more useful than raising, which aborts a parameter sweep at the
        first point that overflows.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore"):
            quantile = self.lam * (-np.log1p(-values)) ** (1.0 / self.k)
        return restore(quantile, scalar)


@dataclass(frozen=True)
class Frechet(InverseTransformSampling):
    """
    Fréchet(alpha, s, m): heavy-tailed extreme-value distribution.
    Support x > m. alpha>0 (shape), s>0 (scale), m (location).
    CDF: F(x) = exp( - ((x - m)/s)^(-alpha) ), x>m
    PDF: f(x) = (alpha/s) * ((x - m)/s)^(-alpha-1) * exp( - ((x - m)/s)^(-alpha) ), x>m
    PPF: x = m + s * [ -ln(u) ]^{-1/alpha}

    Examples:
        The limit law for maxima of Pareto-tailed data, so ``cdf(m + s)`` is
        ``exp(-1)`` whatever the shape:

        >>> round(Frechet(alpha=2.0, s=1.0, m=0.0).cdf(1.0), 6)
        0.367879
        >>> round(Frechet(alpha=5.0, s=1.0, m=0.0).cdf(1.0), 6)
        0.367879
        >>> round(Frechet(alpha=2.0, s=1.0, m=0.0).ppf(0.5), 6)
        1.201122
    """

    alpha: float
    s: float = 1.0
    m: float = 0.0

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.s > 0):
            raise ParameterError("Frechet requires alpha>0 and s>0.")

    def pdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (values - self.m) / self.s
            density = (
                (self.alpha / self.s)
                * z ** (-(self.alpha + 1.0))
                * np.exp(-(z ** (-self.alpha)))
            )
        return restore(select(values <= self.m, 0.0, density), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            below = np.exp(-(((values - self.m) / self.s) ** (-self.alpha)))
        return restore(select(values <= self.m, 0.0, below), scalar)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function 1 - CDF, via ``-expm1`` for tail accuracy.

        In the upper tail the exponent tends to zero and ``exp`` of it rounds to
        exactly 1.0, so ``1 - cdf(x)`` would yield 0. ``-expm1(t)`` is accurate
        for small t and preserves the true decay.
        """
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            above = -np.expm1(-(((values - self.m) / self.s) ** (-self.alpha)))
        return restore(select(values <= self.m, 1.0, above), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function.

        Returns ``inf`` when the quantile exceeds the float range. At extreme
        parameters the true value is genuinely not representable, and reporting
        it is more useful than raising, which aborts a parameter sweep at the
        first point that overflows.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore"):
            quantile = self.m + self.s * (-np.log(values)) ** (-1.0 / self.alpha)
        return restore(quantile, scalar)


@dataclass(frozen=True)
class GEV_Frechet(InverseTransformSampling):
    """
    Generalized Extreme Value (Fréchet-type) with xi>0, mu (loc), sigma>0 (scale).
    Heavy-tailed when xi>0.

    CDF: F(x) = exp( -[1 + xi ( (x-mu)/sigma )]^(-1/xi) ), for 1 + xi (x-mu)/sigma > 0
    PDF: f(x) = (1/sigma) * [1 + xi z]^(-1/xi - 1) * exp( -[1 + xi z]^(-1/xi) ), z=(x-mu)/sigma
    PPF: x = mu + (sigma/xi) * ( (-ln u)^(-xi) - 1 )

    Examples:
        The generalized extreme value distribution in its heavy-tailed branch,
        parameterised by ``xi = 1 / alpha``:

        >>> gev = GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0)
        >>> round(gev.cdf(1.0), 6)
        0.64118
        >>> round(gev.ppf(0.5), 6)
        0.402245

        Its support starts at ``mu - sigma / xi``, and nothing falls below it:

        >>> gev.cdf(-2.0)
        0.0
    """

    xi: float
    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if not (self.xi > 0 and self.sigma > 0):
            raise ParameterError(
                "GEV_Frechet requires xi>0 and sigma>0 (heavy-tailed branch)."
            )

    def _valid(self, x: float) -> bool:
        return (1.0 + self.xi * ((x - self.mu) / self.sigma)) > 0.0

    def pdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = 1.0 + self.xi * (values - self.mu) / self.sigma
            density = (
                (1.0 / self.sigma)
                * t ** (-1.0 / self.xi - 1.0)
                * np.exp(-(t ** (-1.0 / self.xi)))
            )
        return restore(select(t > 0.0, density, 0.0), scalar)

    def cdf(self, x: ArrayLike) -> Any:
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = 1.0 + self.xi * (values - self.mu) / self.sigma
            below = np.exp(-(t ** (-1.0 / self.xi)))
        return restore(select(t > 0.0, below, 0.0), scalar)

    def sf(self, x: ArrayLike) -> Any:
        """Survival function 1 - CDF, via ``-expm1`` for tail accuracy.

        See :meth:`Frechet.sf`: forming ``1 - exp(-t)`` for small t loses every
        significant digit, while ``-expm1(-t)`` does not.
        """
        values, scalar = as_array(x)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = 1.0 + self.xi * (values - self.mu) / self.sigma
            above = -np.expm1(-(t ** (-1.0 / self.xi)))
        return restore(select(t > 0.0, above, 1.0), scalar)

    def ppf(self, u: ArrayLike) -> Any:
        """Quantile function.

        Returns ``inf`` when the quantile exceeds the float range. At extreme
        parameters the true value is genuinely not representable, and reporting
        it is more useful than raising, which aborts a parameter sweep at the
        first point that overflows.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        with np.errstate(over="ignore"):
            quantile = self.mu + (self.sigma / self.xi) * (
                (-np.log(values)) ** (-self.xi) - 1.0
            )
        return restore(quantile, scalar)


def _demo() -> None:
    """Basic checks and example usage. Run `python heavy_tails.py`."""
    rng_seed = 123

    pareto = Pareto(alpha=1.5, xm=1.0)
    cauchy = Cauchy(x0=0.0, gamma=1.0)
    studt = StudentT(nu=3.0)
    lgn = LogNormal(mu=0.0, sigma=1.0)
    weib = Weibull(k=0.7, lam=2.0)  # heavy-tailed regime k<1
    frech = Frechet(alpha=2.5, s=1.0, m=0.0)
    gev = GEV_Frechet(xi=0.3, mu=0.0, sigma=1.0)

    # Example: single values
    x = 2.5
    print("Pareto PDF/CDF at x=2.5:", pareto.pdf(x), pareto.cdf(x))
    print("Cauchy CDF at x=1.0:", cauchy.cdf(1.0))
    print("t_3 PDF at x=0:", studt.pdf(0.0))
    print("LogNormal CDF at x=1:", lgn.cdf(1.0))
    print("Weibull(k=0.7) SF at x=5:", 1.0 - weib.cdf(5.0))
    print("Frechet PPF(u=0.99):", frech.ppf(0.99))
    print("GEV_Frechet PPF(u=0.99):", gev.ppf(0.99))

    # Sampling sanity checks
    n = 5
    print("Pareto samples:", pareto.rvs(n, seed=rng_seed))
    print("Cauchy samples:", cauchy.rvs(n, seed=rng_seed))
    print("Student-t samples:", studt.rvs(n, seed=rng_seed))
    print("LogNormal samples:", lgn.rvs(n, seed=rng_seed))
    print("Weibull samples:", weib.rvs(n, seed=rng_seed))
    print("Frechet samples:", frech.rvs(n, seed=rng_seed))
    print("GEV_Frechet samples:", gev.rvs(n, seed=rng_seed))


if __name__ == "__main__":
    _demo()
