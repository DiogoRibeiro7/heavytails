"""Actuarial layer: compound distributions, aggregate losses and reinsurance.

An insurance portfolio produces a random number of claims of random size, and
the quantity that matters is their total::

    S = X_1 + X_2 + ... + X_N

with ``N`` the claim count and ``X_i`` the individual losses. This module builds
the distribution of ``S`` from a frequency model and a severity distribution,
and prices the reinsurance structures written on top of it.

Two routes to the aggregate distribution, with different failure modes:

**Panjer recursion** (:func:`panjer_recursion`) is exact given a discretised
severity, and gives the whole distribution rather than a sample. It is the
right tool when the severity is not too heavy and the expected claim count is
moderate. It has two limitations that this module reports rather than hides:
the severity grid is finite, so mass past its end is lost and counted; and the
recursion starts from ``P_N(f_0)``, which underflows to zero for a large
expected count, taking the whole distribution with it.

**Simulation** (:func:`simulate_aggregate_loss`) has neither limitation and
converges slowly, which is the trade. For a genuinely heavy-tailed severity it
is usually the honest choice, and :class:`EmpiricalAggregate` plugs the sample
into the same interface so the two routes are interchangeable.

A warning that applies to both, and to every actuarial text that recommends a
normal or translated-gamma approximation to the aggregate: **for a severity
with tail index ``alpha <= 2`` the variance of ``S`` does not exist**, and for
``alpha <= 1`` neither does its mean. :func:`compound_moments` reports ``inf``
in those cases. An approximation matching two moments cannot match moments
that are not there.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any

from heavytails._special import _gammainc_lower_reg
from heavytails.heavy_tails import RNG, ParameterError

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "AggregateLoss",
    "Binomial",
    "EmpiricalAggregate",
    "LayeredSeverity",
    "NegativeBinomial",
    "Poisson",
    "PolicyTerms",
    "compound_moments",
    "discretise_severity",
    "excess_of_loss_premium",
    "limited_expected_value",
    "panjer_recursion",
    "simulate_aggregate_loss",
]

# --------------------------- Frequency models -------------------------------- #
#
# All three below are the (a,b,0) class: p_k / p_{k-1} = a + b/k. That identity
# is exactly what makes the Panjer recursion possible, so the class is not an
# arbitrary selection of convenient distributions -- it is the set the
# recursion applies to.


@dataclass(frozen=True)
class Poisson:
    """
    Poisson claim count with rate ``lam``.

    The default frequency model, and the one where variance equals mean. Real
    portfolios are usually over-dispersed, which is what
    :class:`NegativeBinomial` is for.

    Args:
        lam: Expected number of claims per period, positive.

    Raises:
        ParameterError: If ``lam`` is not positive.

    Examples:
        >>> Poisson(lam=3.0).mean()
        3.0
        >>> round(Poisson(lam=3.0).pmf(2), 6)
        0.224042
    """

    lam: float

    def __post_init__(self) -> None:
        if not (self.lam > 0.0) or not math.isfinite(self.lam):
            raise ParameterError("Poisson requires a finite lam > 0.")

    def pmf(self, k: int) -> float:
        """Probability of exactly ``k`` claims."""
        if k < 0:
            return 0.0
        return math.exp(-self.lam + k * math.log(self.lam) - math.lgamma(k + 1))

    def mean(self) -> float:
        """Expected claim count."""
        return float(self.lam)

    def variance(self) -> float:
        """Variance of the claim count, equal to the mean."""
        return float(self.lam)

    def pgf(self, s: float) -> float:
        """Probability generating function ``E[s^N] = exp(lam (s - 1))``."""
        return math.exp(self.lam * (s - 1.0))

    def panjer_ab(self) -> tuple[float, float]:
        """The ``(a, b)`` pair of the recursion class."""
        return (0.0, float(self.lam))

    def thin(self, probability: float) -> Poisson:
        """
        Keep each claim independently with ``probability``.

        Thinning a Poisson gives a Poisson, which is what lets a per-payment
        severity be paired with a reduced claim count. See
        :class:`LayeredSeverity` for why that matters.
        """
        _check_probability(probability)
        return Poisson(lam=self.lam * probability)

    def draw(self, rng: RNG) -> int:
        """One claim count from a caller-supplied stream."""
        return _poisson_variate(rng, self.lam)

    def rvs(self, n: int, seed: int | None = None) -> list[int]:
        """Draw ``n`` independent claim counts."""
        rng = RNG(seed)
        return [self.draw(rng) for _ in range(n)]


@dataclass(frozen=True)
class NegativeBinomial:
    """
    Negative binomial claim count, in the actuarial ``(r, beta)`` parameters.

    Mean ``r*beta`` and variance ``r*beta*(1 + beta)``, so the variance always
    exceeds the mean. That over-dispersion is the reason to prefer it to
    :class:`Poisson`: it arises exactly when the rate itself is uncertain, since
    a gamma-mixed Poisson is negative binomial.

    Args:
        r: Positive shape. Need not be an integer.
        beta: Positive scale.

    Raises:
        ParameterError: If either parameter is not positive.

    Examples:
        >>> nb = NegativeBinomial(r=2.0, beta=1.5)
        >>> nb.mean(), nb.variance()
        (3.0, 7.5)
    """

    r: float
    beta: float

    def __post_init__(self) -> None:
        if not (self.r > 0.0) or not math.isfinite(self.r):
            raise ParameterError("NegativeBinomial requires a finite r > 0.")
        if not (self.beta > 0.0) or not math.isfinite(self.beta):
            raise ParameterError("NegativeBinomial requires a finite beta > 0.")

    def pmf(self, k: int) -> float:
        """Probability of exactly ``k`` claims."""
        if k < 0:
            return 0.0
        p = self.beta / (1.0 + self.beta)
        return math.exp(
            math.lgamma(self.r + k)
            - math.lgamma(self.r)
            - math.lgamma(k + 1)
            + k * math.log(p)
            - self.r * math.log1p(self.beta)
        )

    def mean(self) -> float:
        """Expected claim count."""
        return float(self.r * self.beta)

    def variance(self) -> float:
        """Variance of the claim count, always above the mean."""
        return float(self.r * self.beta * (1.0 + self.beta))

    def pgf(self, s: float) -> float:
        """Probability generating function ``(1 - beta(s - 1))^(-r)``."""
        base = 1.0 - self.beta * (s - 1.0)
        if base <= 0.0:
            raise ValueError("pgf argument outside the radius of convergence.")
        return float(base**-self.r)

    def panjer_ab(self) -> tuple[float, float]:
        """The ``(a, b)`` pair of the recursion class."""
        a = self.beta / (1.0 + self.beta)
        return (a, (self.r - 1.0) * a)

    def thin(self, probability: float) -> NegativeBinomial:
        """
        Keep each claim independently with ``probability``.

        Scales ``beta``, which follows from the gamma mixture: thinning the
        Poisson layer scales the mixing gamma's scale by the same factor.
        """
        _check_probability(probability)
        return NegativeBinomial(r=self.r, beta=self.beta * probability)

    def draw(self, rng: RNG) -> int:
        """One claim count from a caller-supplied stream."""
        return _poisson_variate(rng, rng.gamma(shape_k=self.r, scale_theta=self.beta))

    def rvs(self, n: int, seed: int | None = None) -> list[int]:
        """Draw ``n`` independent claim counts, via the gamma-Poisson mixture."""
        rng = RNG(seed)
        return [self.draw(rng) for _ in range(n)]


@dataclass(frozen=True)
class Binomial:
    """
    Binomial claim count: ``m`` risks each producing a claim with probability
    ``p``.

    Under-dispersed, with variance below the mean, so it fits a closed group of
    policies where at most one claim per policy is possible. Included because it
    completes the ``(a,b,0)`` class the Panjer recursion is defined on.

    Args:
        m: Number of risks, a positive integer.
        p: Claim probability per risk, in (0, 1).

    Raises:
        ParameterError: If ``m`` is not a positive integer or ``p`` is outside
            (0, 1).

    Examples:
        >>> Binomial(m=10, p=0.2).mean()
        2.0
    """

    m: int
    p: float

    def __post_init__(self) -> None:
        if not isinstance(self.m, int) or self.m <= 0:
            raise ParameterError("Binomial requires an integer m > 0.")
        if not (0.0 < self.p < 1.0):
            raise ParameterError("Binomial requires p in (0,1).")

    def pmf(self, k: int) -> float:
        """Probability of exactly ``k`` claims."""
        if k < 0 or k > self.m:
            return 0.0
        return math.exp(
            math.lgamma(self.m + 1)
            - math.lgamma(k + 1)
            - math.lgamma(self.m - k + 1)
            + k * math.log(self.p)
            + (self.m - k) * math.log1p(-self.p)
        )

    def mean(self) -> float:
        """Expected claim count."""
        return float(self.m * self.p)

    def variance(self) -> float:
        """Variance of the claim count, always below the mean."""
        return float(self.m * self.p * (1.0 - self.p))

    def pgf(self, s: float) -> float:
        """Probability generating function ``(1 + p(s - 1))^m``."""
        return float((1.0 + self.p * (s - 1.0)) ** self.m)

    def panjer_ab(self) -> tuple[float, float]:
        """The ``(a, b)`` pair of the recursion class. ``a`` is negative here."""
        a = -self.p / (1.0 - self.p)
        return (a, -(self.m + 1) * a)

    def thin(self, probability: float) -> Binomial:
        """Keep each claim independently with ``probability``."""
        _check_probability(probability)
        return Binomial(m=self.m, p=self.p * probability)

    def draw(self, rng: RNG) -> int:
        """One claim count from a caller-supplied stream."""
        return sum(1 for _ in range(self.m) if rng.uniform_0_1() < self.p)

    def rvs(self, n: int, seed: int | None = None) -> list[int]:
        """Draw ``n`` independent claim counts."""
        rng = RNG(seed)
        return [self.draw(rng) for _ in range(n)]


def _check_probability(probability: float) -> None:
    """Validate a thinning probability."""
    if not (0.0 < probability <= 1.0):
        raise ValueError("Thinning probability must be in (0,1].")


def _poisson_variate(rng: RNG, lam: float) -> int:
    """One Poisson variate.

    Knuth's multiplication method below 30, where it is fast and exact, and
    Hoermann's transformed rejection above it, where the multiplication method
    would need ``lam`` iterations and its ``exp(-lam)`` factor would underflow.
    """
    if lam <= 0.0:
        return 0
    if lam < 30.0:
        target = math.exp(-lam)
        product = rng.uniform_0_1()
        count = 0
        while product > target:
            count += 1
            product *= rng.uniform_0_1()
        return count

    # Hoermann (1993), transformed rejection with a squeeze.
    b = 0.931 + 2.53 * math.sqrt(lam)
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2.0)
    log_lam = math.log(lam)
    while True:
        u = rng.uniform_0_1() - 0.5
        v = rng.uniform_0_1()
        us = 0.5 - abs(u)
        k = math.floor((2.0 * a / us + b) * u + lam + 0.43)
        if us >= 0.07 and v <= v_r:
            return k
        if k < 0 or (us < 0.013 and v > us):
            continue
        if math.log(v * inv_alpha / (a / (us * us) + b)) <= (
            k * log_lam - lam - math.lgamma(k + 1)
        ):
            return k


# --------------------------- Policy structures ------------------------------- #


@dataclass(frozen=True)
class PolicyTerms:
    """
    Deductible, limit and coinsurance applied to a single loss.

    The payment on a loss ``x`` is ``coinsurance * min(max(x - deductible, 0),
    limit)``. Note that ``limit`` caps the *excess over the deductible*, not the
    loss: a policy described as "1M excess of 100k" has ``deductible=100_000``
    and ``limit=1_000_000``, and pays at most 1M. Where a limit is quoted as a
    maximum covered loss ``u`` instead, pass ``limit=u - deductible``.

    Args:
        deductible: Loss retained by the insured, non-negative.
        limit: Largest excess covered, or ``None`` for unlimited.
        coinsurance: Share of the covered excess paid, in (0, 1].

    Raises:
        ParameterError: If any parameter is outside its range.

    Examples:
        >>> terms = PolicyTerms(deductible=100.0, limit=500.0)
        >>> terms.payment(50.0), terms.payment(300.0), terms.payment(9999.0)
        (0.0, 200.0, 500.0)
    """

    deductible: float = 0.0
    limit: float | None = None
    coinsurance: float = 1.0

    def __post_init__(self) -> None:
        if self.deductible < 0.0 or not math.isfinite(self.deductible):
            raise ParameterError("deductible must be finite and non-negative.")
        if self.limit is not None and not (self.limit > 0.0):
            raise ParameterError("limit must be positive when given.")
        if not (0.0 < self.coinsurance <= 1.0):
            raise ParameterError("coinsurance must be in (0,1].")

    def payment(self, loss: float) -> float:
        """The amount paid on a loss of size ``loss``."""
        excess = max(loss - self.deductible, 0.0)
        if self.limit is not None:
            excess = min(excess, self.limit)
        return float(self.coinsurance * excess)

    @property
    def max_payment(self) -> float:
        """Largest possible payment, ``inf`` when there is no limit."""
        if self.limit is None:
            return math.inf
        return float(self.coinsurance * self.limit)

    @property
    def upper_loss(self) -> float:
        """The loss at which the limit binds, ``inf`` when there is none."""
        if self.limit is None:
            return math.inf
        return float(self.deductible + self.limit)


@dataclass(frozen=True)
class LayeredSeverity:
    """
    A severity distribution with policy terms applied.

    Two bases, and choosing the wrong one is the classic error in this
    calculation:

    ``"per-loss"``
        The payment on every loss, including the zero paid on losses below the
        deductible. Has an atom of size ``F(deductible)`` at zero, and pairs
        with the *original* claim frequency.

    ``"per-payment"``
        The payment conditional on there being one. No atom at zero, a larger
        mean, and pairs with a frequency *thinned* by the probability of
        exceeding the deductible.

    Both pairings describe the same aggregate loss, and the test suite asserts
    they agree. Mixing them -- per-payment severity with unthinned frequency --
    overstates the expected aggregate by a factor of ``1/S(deductible)``, which
    for a high deductible is very large and looks plausible.

    Args:
        severity: Ground-up loss distribution, with ``cdf`` and ``ppf``.
        terms: The policy structure.
        basis: ``"per-loss"`` or ``"per-payment"``.

    Raises:
        ValueError: If ``basis`` is neither, or if the deductible is so high
            that no loss can exceed it on a per-payment basis.

    Examples:
        >>> from heavytails import Pareto
        >>> layer = LayeredSeverity(Pareto(alpha=2.0, xm=1.0),
        ...                         PolicyTerms(deductible=2.0, limit=8.0))
        >>> round(layer.mean(), 6)
        0.4
    """

    severity: Any
    terms: PolicyTerms = field(default_factory=PolicyTerms)
    basis: str = "per-loss"

    def __post_init__(self) -> None:
        if self.basis not in {"per-loss", "per-payment"}:
            raise ValueError(
                f"Unknown basis {self.basis!r}. Available: per-loss, per-payment"
            )
        if self.basis == "per-payment" and self.exceedance_probability <= 0.0:
            raise ValueError(
                "No loss can exceed the deductible, so there is no per-payment "
                "distribution. Use basis='per-loss', which is a point mass at zero."
            )

    @property
    def exceedance_probability(self) -> float:
        """Probability a ground-up loss exceeds the deductible.

        This is the thinning factor: the frequency of *payments* is the
        frequency of losses multiplied by this.
        """
        return float(1.0 - self.severity.cdf(self.terms.deductible))

    def cdf(self, y: float) -> float:
        """Distribution function of the payment."""
        if y < 0.0:
            return 0.0
        if y >= self.terms.max_payment:
            return 1.0
        loss = self.terms.deductible + y / self.terms.coinsurance
        below = float(self.severity.cdf(loss))
        if self.basis == "per-loss":
            return below
        floor = float(self.severity.cdf(self.terms.deductible))
        return (below - floor) / (1.0 - floor)

    def sf(self, y: float) -> float:
        """Survival function of the payment."""
        return 1.0 - self.cdf(y)

    def ppf(self, u: float) -> float:
        """Quantile function of the payment."""
        if not (0.0 <= u <= 1.0):
            raise ValueError("u must be in [0,1].")
        floor = float(self.severity.cdf(self.terms.deductible))
        if self.basis == "per-loss":
            if u <= floor:
                return 0.0
            target = u
        else:
            target = floor + u * (1.0 - floor)
        if target >= 1.0:
            return self.terms.max_payment
        loss = float(self.severity.ppf(target))
        return self.terms.payment(loss)

    def lev(self, t: float) -> float:
        """The layer's own limited expected value ``E[min(Y, t)]``.

        Censoring the *payment* at ``t`` is the same as censoring the loss at
        ``d + t/c``, so this is a difference of the severity's limited expected
        values and stays exact::

            E[Y ^ t] = c * (E[X ^ (d + t/c)] - E[X ^ d])

        :func:`limited_expected_value` picks this up automatically. It matters:
        computing it numerically instead means integrating a quantile function
        with a large atom at zero, and the mean-preserving discretisation takes
        a *second difference* of the result, which amplifies any error in it by
        orders of magnitude.
        """
        if t < 0.0:
            raise ValueError("t must be non-negative.")
        cap = min(t, self.terms.max_payment)
        loss = (
            math.inf
            if math.isinf(cap)
            else self.terms.deductible + cap / self.terms.coinsurance
        )
        upper = limited_expected_value(self.severity, min(loss, self.terms.upper_loss))
        lower = limited_expected_value(self.severity, self.terms.deductible)
        expected = self.terms.coinsurance * (upper - lower)
        if self.basis == "per-payment":
            return float(expected / self.exceedance_probability)
        return float(expected)

    def mean(self) -> float:
        """Expected payment, the limited expected value with no censoring."""
        return self.lev(math.inf)

    def rvs(self, n: int, seed: int | None = None) -> list[float]:
        """Draw ``n`` independent payments."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = RNG(seed)
        return [self._one(rng) for _ in range(n)]

    def _one(self, rng: RNG) -> float:
        """One payment, by inversion so the conditioning is exact."""
        u = rng.uniform_0_1()
        if self.basis == "per-loss":
            return self.terms.payment(float(self.severity.ppf(u)))
        floor = float(self.severity.cdf(self.terms.deductible))
        return self.terms.payment(float(self.severity.ppf(floor + u * (1.0 - floor))))


# --------------------------- Limited expected value -------------------------- #


def limited_expected_value(dist: Any, d: float, nodes: int = 512) -> float:
    """
    The limited expected value ``E[min(X, d)]``.

    The workhorse of layer pricing: the expected cost of the layer from ``a`` to
    ``b`` is ``E[X ^ b] - E[X ^ a]``, so every deductible, limit and excess-of-
    loss premium reduces to a difference of these.

    It is finite even when ``E[X]`` is not, which is the reason to compute the
    layer this way rather than by integrating the severity: a Pareto with
    ``alpha <= 1`` has no mean, but every bounded layer on it still has a
    perfectly finite price.

    Closed forms are used for Pareto, LogNormal, Weibull and the generalized
    Pareto. Anything else is integrated as ``int_0^F(d) ppf(u) du + d*S(d)``,
    which is a bounded smooth integrand on a finite interval even when the
    density is not.

    Args:
        dist: Severity distribution with ``cdf`` and ``ppf``.
        d: Censoring point, non-negative. May be ``inf``, giving the mean.
        nodes: Quadrature nodes for the numeric fallback.

    Returns:
        ``E[min(X, d)]``. Possibly ``inf`` when ``d`` is infinite and the mean
        does not exist.

    Raises:
        ValueError: If ``d`` is negative.

    Examples:
        >>> from heavytails import Pareto
        >>> round(limited_expected_value(Pareto(alpha=2.0, xm=1.0), 10.0), 6)
        1.9
        >>> limited_expected_value(Pareto(alpha=2.0, xm=1.0), float("inf"))
        2.0
    """
    if d < 0.0:
        raise ValueError("d must be non-negative.")
    if d == 0.0:
        return 0.0

    own = getattr(dist, "lev", None)
    if callable(own):
        return float(own(d))

    analytic = _analytic_lev(dist, d)
    if analytic is not None:
        return analytic

    return _numeric_lev(dist, d, nodes)


def _analytic_lev(dist: Any, d: float) -> float | None:
    """Closed-form limited expected value, or None when there is not one here."""
    name = type(dist).__name__

    if name == "Pareto":
        if d <= dist.xm:
            return float(d)
        if math.isinf(d):
            return (
                math.inf
                if dist.alpha <= 1.0
                else float(dist.xm * dist.alpha / (dist.alpha - 1.0))
            )
        if abs(dist.alpha - 1.0) < 1e-12:
            return float(dist.xm + dist.xm * math.log(d / dist.xm))
        power = 1.0 - dist.alpha
        return float(
            dist.xm + dist.xm**dist.alpha * (d**power - dist.xm**power) / power
        )

    if name == "LogNormal":
        whole = math.exp(dist.mu + dist.sigma * dist.sigma / 2.0)
        if math.isinf(d):
            return float(whole)
        z = (math.log(d) - dist.mu) / dist.sigma
        return float(whole * _normal_cdf(z - dist.sigma) + d * (1.0 - _normal_cdf(z)))

    if name == "Weibull":
        shape = 1.0 + 1.0 / dist.k
        whole = dist.lam * math.gamma(shape)
        if math.isinf(d):
            return float(whole)
        scaled = (d / dist.lam) ** dist.k
        lower = _gammainc_lower_reg(shape, scaled)
        return float(whole * lower + d * math.exp(-scaled))

    if name == "GeneralizedPareto":
        if d <= dist.mu:
            return float(d)
        excess = d - dist.mu
        if abs(dist.xi) < 1e-12:
            if math.isinf(d):
                return float(dist.mu + dist.sigma)
            return float(dist.mu + dist.sigma * (1.0 - math.exp(-excess / dist.sigma)))
        if math.isinf(d):
            if dist.xi >= 1.0:
                return math.inf
            return float(dist.mu + dist.sigma / (1.0 - dist.xi))
        bracket = 1.0 + dist.xi * excess / dist.sigma
        if bracket <= 0.0:  # Beyond the upper endpoint of a bounded GPD.
            return float(dist.mu + dist.sigma / (1.0 - dist.xi))
        integral = (
            dist.sigma / (1.0 - dist.xi) * (1.0 - bracket ** (1.0 - 1.0 / dist.xi))
        )
        return float(dist.mu + integral)

    return None


def _normal_cdf(x: float) -> float:
    """Standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _numeric_lev(dist: Any, d: float, nodes: int) -> float:
    """Limited expected value by quadrature on the quantile function.

    ``E[X ^ d] = int_0^F(d) ppf(u) du + d * S(d)``. Integrating in ``u`` rather
    than ``x`` bounds the integrand by ``d`` and removes the kink at the lower
    end of the support, both of which a direct integral of the survival function
    would have to cope with.
    """
    upper = 1.0 if math.isinf(d) else float(dist.cdf(d))
    if upper <= 0.0:
        return float(d)

    # An atom at zero contributes nothing to the integral, and spending nodes
    # on it wastes them: a layer with a high deductible can be zero over most
    # of the unit interval, leaving a handful of nodes to resolve the rest.
    floor = min(max(float(dist.cdf(0.0)), 0.0), upper)
    width = upper - floor

    total = 0.0
    for i in range(nodes):
        u = floor + width * (i + 0.5) / nodes
        try:
            value = float(dist.ppf(u))
        except (ValueError, OverflowError):  # pragma: no cover - guarded families
            return math.inf
        if not math.isfinite(value):
            return math.inf
        total += min(value, d)
    integral = total * width / nodes

    if math.isinf(d):
        return float(integral)
    return float(integral + d * (1.0 - upper))


# --------------------------- Discretisation ---------------------------------- #


def discretise_severity(
    severity: Any,
    h: float,
    n: int,
    method: str = "mass",
) -> tuple[list[float], float]:
    """
    Put a continuous severity onto the arithmetic grid the recursion needs.

    Panjer recursion works on a severity supported on ``0, h, 2h, ...``, so a
    continuous one must be discretised first. The choice of ``h`` is the whole
    accuracy story: too coarse and the answer is wrong, too fine and the
    recursion is quadratic in the number of points.

    Two methods:

    ``"mass"``
        Assign each grid point the probability of the interval around it. Simple
        and fast, and biased: the discrete distribution has slightly the wrong
        mean.

    ``"mean-preserving"``
        Local moment matching, which makes the discretised mean equal the
        severity's own mean exactly (up to the grid's end). Costs two limited
        expected values per point. Prefer it for pricing, where an error in the
        mean is an error in the premium.

    Args:
        severity: Distribution with ``cdf``; ``"mean-preserving"`` also needs
            whatever :func:`limited_expected_value` requires.
        h: Grid span, positive.
        n: Number of grid points, so the grid runs to ``(n-1)*h``.
        method: ``"mass"`` or ``"mean-preserving"``.

    Returns:
        The probabilities on the grid, and the mass beyond its end. **That
        second number is not decoration**: for a heavy tail it can be
        substantial, and every quantity derived from the grid is wrong by
        roughly that much.

    Raises:
        ValueError: If ``h`` is not positive, ``n`` is below 2, or ``method`` is
            unknown.

    Examples:
        >>> from heavytails import Pareto
        >>> probs, lost = discretise_severity(Pareto(alpha=2.0, xm=1.0), 0.5, 200)
        >>> round(sum(probs), 6), round(lost, 6)
        (0.999899, 0.000101)
    """
    if not (h > 0.0):
        raise ValueError("h must be positive.")
    if n < 2:
        raise ValueError("n must be at least 2.")
    if method not in {"mass", "mean-preserving"}:
        raise ValueError(f"Unknown method {method!r}. Available: mass, mean-preserving")

    if method == "mass":
        probabilities = [float(severity.cdf(0.5 * h))]
        probabilities.extend(
            float(severity.cdf((j + 0.5) * h)) - float(severity.cdf((j - 0.5) * h))
            for j in range(1, n)
        )
    else:
        levs = [limited_expected_value(severity, j * h) for j in range(n + 1)]
        probabilities = [1.0 - levs[1] / h]
        probabilities.extend(
            (2.0 * levs[j] - levs[j - 1] - levs[j + 1]) / h for j in range(1, n)
        )

    # Local moment matching takes a second difference of limited expected
    # values, so any error in them is amplified by 1/h and can turn a weight
    # negative. Clipping tiny rounding artefacts is fine; clipping a material
    # amount would leave a biased grid that still sums to something plausible,
    # so it is reported instead.
    clipped = -sum(p for p in probabilities if p < 0.0)
    if clipped > 1e-6:
        raise ArithmeticError(
            f"Local moment matching produced {clipped:.3g} of negative "
            "probability, which means the limited expected values it differences "
            "are not accurate enough at this grid span. Use a larger h, or "
            "method='mass', which cannot go negative."
        )
    probabilities = [max(p, 0.0) for p in probabilities]
    tail_mass = max(1.0 - sum(probabilities), 0.0)
    return probabilities, tail_mass


# --------------------------- Aggregate distribution -------------------------- #


@dataclass(frozen=True)
class AggregateLoss:
    """
    The distribution of the aggregate loss on an arithmetic grid.

    Produced by :func:`panjer_recursion`. Probabilities sit at ``0, h, 2h, ...``
    and ``truncated_mass`` is what fell off the end -- read it before trusting
    anything in the far tail, since a quantile above ``1 - truncated_mass``
    cannot be resolved at all and is reported as ``inf``.

    Attributes:
        h: Grid span.
        probabilities: Probability at each grid point.
        truncated_mass: Probability beyond the last grid point.
        severity_tail_mass: Severity mass lost in discretisation, propagated
            here because it is the usual cause of the above.
    """

    h: float
    probabilities: list[float]
    truncated_mass: float
    severity_tail_mass: float = 0.0

    @property
    def support(self) -> list[float]:
        """The grid points themselves."""
        return [i * self.h for i in range(len(self.probabilities))]

    def mean(self) -> float:
        """Mean of the gridded distribution.

        Below the true mean by whatever the truncation removed. Compare against
        :func:`compound_moments`, which is exact.
        """
        return float(sum(i * self.h * p for i, p in enumerate(self.probabilities)))

    def variance(self) -> float:
        """Variance of the gridded distribution, subject to the same caveat."""
        mu = self.mean()
        return float(
            sum((i * self.h - mu) ** 2 * p for i, p in enumerate(self.probabilities))
        )

    def cdf(self, x: float) -> float:
        """Probability the aggregate is at most ``x``."""
        if x < 0.0:
            return 0.0
        index = min(math.floor(x / self.h), len(self.probabilities) - 1)
        return float(sum(self.probabilities[: index + 1]))

    def sf(self, x: float) -> float:
        """Probability the aggregate exceeds ``x``."""
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        """
        Quantile of the aggregate.

        Returns ``inf`` above ``1 - truncated_mass``, because the grid holds no
        information there. Returning the last grid point instead would look like
        an answer.
        """
        if not (0.0 <= u < 1.0):
            raise ValueError("u must be in [0,1).")
        cumulative = 0.0
        for i, p in enumerate(self.probabilities):
            cumulative += p
            if cumulative >= u:
                return float(i * self.h)
        return math.inf

    def value_at_risk(self, level: float) -> float:
        """Quantile at ``level``, the aggregate value at risk."""
        return self.ppf(level)

    def expected_shortfall(self, level: float) -> float:
        """
        Mean aggregate loss given it exceeds the value at risk.

        Returns ``inf`` when the grid has been truncated inside the region being
        averaged, since the missing mass is exactly the part that matters most.
        """
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0,1).")
        var = self.value_at_risk(level)
        if math.isinf(var) or self.truncated_mass > (1.0 - level) * 1e-3:
            return math.inf
        weight = 0.0
        total = 0.0
        for i, p in enumerate(self.probabilities):
            value = i * self.h
            if value >= var:
                weight += p
                total += value * p
        if weight <= 0.0:
            return math.inf
        return float(total / weight)

    def stop_loss_premium(self, retention: float, tolerance: float = 0.01) -> float:
        """
        The stop-loss premium ``E[(S - retention)+]``.

        The expected cost of aggregate excess-of-loss cover attaching at
        ``retention``: the reinsurer pays whatever the total exceeds it.

        The truncated mass lies entirely above the retention, so the value
        computed from the grid is a **lower bound**: the part that fell off the
        end contributes at least ``truncated_mass * (grid_end - retention)``
        more. When that shortfall is a material fraction of the answer this
        returns ``inf`` rather than a number known to be too small. A heavy
        tail always truncates *something*, so demanding none would make this
        useless on exactly the distributions it is for.

        Args:
            retention: Aggregate attachment point, non-negative.
            tolerance: Largest relative shortfall to accept, 1% by default.
                Tighten it when the premium feeds a reserve rather than a
                comparison.

        Returns:
            The expected excess, or ``inf`` when the grid cannot support it.

        Raises:
            ValueError: If ``retention`` is negative.
        """
        if retention < 0.0:
            raise ValueError("retention must be non-negative.")
        premium = float(
            sum(
                (i * self.h - retention) * p
                for i, p in enumerate(self.probabilities)
                if i * self.h > retention
            )
        )
        # The truncated mass sits at or above the first point past the grid,
        # so each unit of it contributes at least that much excess. Measuring
        # from grid_end alone would report no shortfall for a retention past
        # the end, where in fact the computed premium is zero and the truth is
        # not.
        beyond = len(self.probabilities) * self.h
        shortfall = self.truncated_mass * max(beyond - retention, 0.0)
        if shortfall > tolerance * premium:
            return math.inf
        return premium


def panjer_recursion(
    frequency: Any,
    severity: Any,
    h: float,
    n: int,
    method: str = "mass",
) -> AggregateLoss:
    """
    The aggregate loss distribution by Panjer's recursion.

    For a frequency in the ``(a,b,0)`` class, where ``p_k/p_{k-1} = a + b/k``,
    the aggregate probabilities satisfy::

        g_0 = P_N(f_0)
        g_k = [sum_j (a + b*j/k) f_j g_{k-j}] / (1 - a*f_0)

    which is exact for the discretised severity, and quadratic in ``n``. It
    replaces the ``n``-fold convolution that a direct calculation would need.

    Two things go wrong, and both are detected rather than silently returned:

    **The recursion can start at zero.** ``g_0`` is the probability of no loss,
    ``exp(lam(f_0 - 1))`` for a Poisson, and for a large expected count that
    underflows to exactly zero -- after which every ``g_k`` is zero too and the
    output is not a distribution at all. This raises instead.

    **The grid can be too short.** A heavy-tailed severity puts real mass beyond
    any finite grid, and the result carries both the severity mass lost in
    discretisation and the aggregate mass lost past the end. Use
    :func:`simulate_aggregate_loss` when they are large.

    Args:
        frequency: A frequency model exposing ``panjer_ab`` and ``pgf``.
        severity: Continuous severity distribution.
        h: Grid span.
        n: Number of aggregate grid points.
        method: Discretisation method, see :func:`discretise_severity`.

    Returns:
        The aggregate distribution.

    Raises:
        ValueError: If the grid is invalid.
        ArithmeticError: If ``g_0`` underflows, with the expected count that
            caused it.

    Examples:
        >>> from heavytails import Pareto
        >>> agg = panjer_recursion(Poisson(lam=2.0), Pareto(alpha=3.0, xm=1.0),
        ...                        h=0.25, n=400)
        >>> round(agg.mean(), 3)
        2.985
    """
    a, b = frequency.panjer_ab()
    probabilities, severity_tail = discretise_severity(severity, h, n, method=method)

    g_zero = float(frequency.pgf(probabilities[0]))
    if g_zero <= 0.0:
        raise ArithmeticError(
            "The Panjer recursion underflowed at g_0: with an expected claim "
            f"count of {frequency.mean():.6g} the probability of no loss is "
            "below the smallest representable double, so the whole recursion "
            "collapses to zero. Use simulate_aggregate_loss instead, or reduce "
            "the expected count."
        )

    denominator = 1.0 - a * probabilities[0]
    aggregate = [g_zero]
    for k in range(1, n):
        total = 0.0
        for j in range(1, min(k, n - 1) + 1):
            f_j = probabilities[j]
            if f_j == 0.0:
                continue
            total += (a + b * j / k) * f_j * aggregate[k - j]
        aggregate.append(total / denominator)

    aggregate = [max(g, 0.0) for g in aggregate]
    return AggregateLoss(
        h=h,
        probabilities=aggregate,
        truncated_mass=max(1.0 - sum(aggregate), 0.0),
        severity_tail_mass=severity_tail,
    )


def compound_moments(frequency: Any, severity: Any) -> tuple[float, float]:
    """
    Exact mean and variance of the aggregate loss.

    ``E[S] = E[N] E[X]`` and ``Var[S] = E[N] Var[X] + Var[N] E[X]^2``, which
    hold whatever the frequency and severity are, and need no grid. Use them to
    check a Panjer or simulation result.

    **Both are ``inf`` for a heavy enough severity**, and that is the point of
    reporting them: a Pareto severity has no variance for ``alpha <= 2`` and no
    mean for ``alpha <= 1``. Any approximation to the aggregate that matches two
    moments -- the normal and translated-gamma approximations of the actuarial
    literature -- is inapplicable there, and this says so before it is used.

    Args:
        frequency: Frequency model with ``mean`` and ``variance``.
        severity: Severity distribution.

    Returns:
        ``(mean, variance)``, either possibly ``inf``.

    Examples:
        >>> from heavytails import Pareto
        >>> compound_moments(Poisson(lam=2.0), Pareto(alpha=1.5, xm=1.0))
        (6.0, inf)
    """
    severity_mean = limited_expected_value(severity, math.inf)
    if math.isinf(severity_mean):
        return (math.inf, math.inf)

    second = _severity_second_moment(severity)
    mean = frequency.mean() * severity_mean
    if math.isinf(second):
        return (float(mean), math.inf)

    severity_variance = second - severity_mean * severity_mean
    variance = (
        frequency.mean() * severity_variance
        + frequency.variance() * severity_mean * severity_mean
    )
    return (float(mean), float(variance))


def _severity_second_moment(severity: Any) -> float:
    """``E[X^2]``, by closed form where the family makes it easy.

    Returns ``inf`` whenever it does not exist, which for the heavy-tailed
    families here is the common case rather than the exception.
    """
    name = type(severity).__name__

    if name == "Pareto":
        if severity.alpha <= 2.0:
            return math.inf
        return float(severity.alpha * severity.xm**2 / (severity.alpha - 2.0))

    if name == "LogNormal":
        return float(math.exp(2.0 * severity.mu + 2.0 * severity.sigma**2))

    if name == "Weibull":
        return float(severity.lam**2 * math.gamma(1.0 + 2.0 / severity.k))

    if name == "GeneralizedPareto":
        if severity.xi >= 0.5:
            return math.inf
        sigma, xi, mu = severity.sigma, severity.xi, severity.mu
        centred = 2.0 * sigma**2 / ((1.0 - xi) * (1.0 - 2.0 * xi))
        shifted_mean = sigma / (1.0 - xi)
        return float(centred + 2.0 * mu * shifted_mean + mu * mu)

    # Fall back on quadrature of the squared quantile function.
    total = 0.0
    nodes = 4096
    for i in range(nodes):
        u = (i + 0.5) / nodes
        value = float(severity.ppf(u))
        if not math.isfinite(value):
            return math.inf
        total += value * value
    result = total / nodes
    return float(result) if math.isfinite(result) else math.inf


# --------------------------- Simulation -------------------------------------- #


def simulate_aggregate_loss(
    frequency: Any,
    severity: Any,
    n_sims: int,
    seed: int | None = None,
) -> list[float]:
    """
    Simulate aggregate losses directly.

    Draw a claim count, draw that many severities, add them up, repeat. Slower
    to converge than :func:`panjer_recursion` and subject to neither of its
    failure modes: no grid to truncate and no ``g_0`` to underflow. For a
    genuinely heavy-tailed severity this is usually the right route.

    Args:
        frequency: Frequency model with a ``draw(rng)`` method.
        severity: Severity with a ``ppf`` method.
        n_sims: Number of periods to simulate.
        seed: Seed for reproducibility.

    Returns:
        One aggregate loss per simulated period.

    Raises:
        ValueError: If ``n_sims`` is not a positive integer.

    Examples:
        >>> from heavytails import Pareto
        >>> losses = simulate_aggregate_loss(Poisson(lam=2.0),
        ...                                  Pareto(alpha=3.0, xm=1.0),
        ...                                  n_sims=5, seed=1)
        >>> len(losses)
        5
    """
    if not isinstance(n_sims, int) or n_sims <= 0:
        raise ValueError("n_sims must be a positive integer.")

    rng = RNG(seed)
    return [_one_period(rng, frequency, severity) for _ in range(n_sims)]


def _one_period(rng: RNG, frequency: Any, severity: Any) -> float:
    """One aggregate loss, counts and severities from a single stream.

    Drawing the count and the claim sizes from one stream is what keeps them
    independent. Seeding two streams identically would couple the number of
    claims to their sizes, which is a hard bias to notice and a nonsense model.
    """
    count = frequency.draw(rng)
    return sum(float(severity.ppf(rng.uniform_0_1())) for _ in range(count))


@dataclass(frozen=True)
class EmpiricalAggregate:
    """
    A simulated aggregate sample, wrapped so it behaves like a distribution.

    Gives simulation output the same ``cdf``/``ppf``/``sf`` interface the
    gridded result has, so the two routes are interchangeable and the functions
    in :mod:`heavytails.risk` accept either.

    Args:
        samples: Simulated aggregate losses.

    Raises:
        ValueError: If ``samples`` is empty.

    Examples:
        >>> agg = EmpiricalAggregate([1.0, 4.0, 2.0, 9.0])
        >>> agg.mean(), agg.ppf(0.5)
        (4.0, 2.0)
    """

    samples: Sequence[float]

    def __post_init__(self) -> None:
        if len(self.samples) == 0:
            raise ValueError("samples must not be empty.")
        object.__setattr__(self, "_sorted", sorted(float(s) for s in self.samples))

    @property
    def _ordered(self) -> list[float]:
        """The sample in ascending order."""
        return self.__dict__["_sorted"]  # type: ignore[no-any-return]

    def mean(self) -> float:
        """Sample mean."""
        ordered = self._ordered
        return float(sum(ordered) / len(ordered))

    def cdf(self, x: float) -> float:
        """Empirical distribution function."""
        ordered = self._ordered
        return float(bisect.bisect_right(ordered, x) / len(ordered))

    def sf(self, x: float) -> float:
        """Empirical survival function."""
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        """Empirical quantile, by the order statistic at ``ceil(u*n)``."""
        if not (0.0 < u <= 1.0):
            raise ValueError("u must be in (0,1].")
        ordered = self._ordered
        index = min(math.ceil(u * len(ordered)) - 1, len(ordered) - 1)
        return float(ordered[max(index, 0)])

    def value_at_risk(self, level: float) -> float:
        """Empirical quantile at ``level``."""
        return self.ppf(level)

    def expected_shortfall(self, level: float) -> float:
        """Mean of the sample above its ``level`` quantile."""
        if not (0.0 < level < 1.0):
            raise ValueError("level must be in (0,1).")
        ordered = self._ordered
        cut = math.ceil(level * len(ordered)) - 1
        tail = ordered[max(cut, 0) :]
        if not tail:  # pragma: no cover - cut is always inside the sample
            return math.inf
        return float(sum(tail) / len(tail))

    def stop_loss_premium(self, retention: float) -> float:
        """Empirical ``E[(S - retention)+]``."""
        if retention < 0.0:
            raise ValueError("retention must be non-negative.")
        ordered = self._ordered
        return float(sum(max(s - retention, 0.0) for s in ordered) / len(ordered))


# --------------------------- Reinsurance pricing ----------------------------- #


def excess_of_loss_premium(
    frequency: Any,
    severity: Any,
    retention: float,
    limit: float | None = None,
) -> float:
    """
    Expected annual cost of per-risk excess-of-loss reinsurance.

    The reinsurer pays ``min(max(X - retention, 0), limit)`` on each individual
    claim, so the expected cost per period is::

        E[N] * (E[X ^ (retention + limit)] - E[X ^ retention])

    exactly, with no grid and no simulation. That closed form is why
    :func:`limited_expected_value` exists.

    An unlimited layer on a severity with no mean has infinite expected cost,
    and this returns ``inf`` rather than a large number. A *limited* layer on
    the same severity is finite, which is precisely why reinsurance of
    catastrophe risk is always written with a limit.

    This is the expected loss cost, not a quotable premium: it carries no
    loading for expenses, risk margin or the reinsurer's cost of capital, and
    for a heavy-tailed layer the risk margin is the larger part.

    Args:
        frequency: Frequency model with a ``mean``.
        severity: Ground-up severity.
        retention: Attachment point per claim, non-negative.
        limit: Width of the layer, or ``None`` for unlimited.

    Returns:
        The expected cost per period.

    Raises:
        ValueError: If ``retention`` is negative or ``limit`` is not positive.

    Examples:
        >>> from heavytails import Pareto
        >>> round(excess_of_loss_premium(Poisson(lam=10.0),
        ...                              Pareto(alpha=2.0, xm=1.0),
        ...                              retention=5.0, limit=15.0), 6)
        1.5
    """
    if retention < 0.0:
        raise ValueError("retention must be non-negative.")
    if limit is not None and not (limit > 0.0):
        raise ValueError("limit must be positive when given.")

    upper = math.inf if limit is None else retention + limit
    top = limited_expected_value(severity, upper)
    if math.isinf(top):
        return math.inf
    bottom = limited_expected_value(severity, retention)
    return float(frequency.mean() * (top - bottom))
