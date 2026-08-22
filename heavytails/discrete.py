# heavytails/discrete.py
from __future__ import annotations

from dataclasses import dataclass, field
import math

from heavytails.heavy_tails import RNG, ParameterError, Samplable


@dataclass(frozen=True)
class Zipf(Samplable):
    """
    Zipf (Zeta) distribution with exponent s>1 on support k=1,2,...

    P(X=k) = k^{-s} / ζ(s)
    where ζ(s) ≈ ∑_{n=1}^∞ n^{-s}

    Attributes
    ----------
    s : float
        Exponent parameter (must be > 1)
    kmax : int
        Maximum value for truncated distribution (default: 10,000)
    _Z : float
        Normalization constant ζ(s) computed in __post_init__

    Examples:
        The discrete power law of word frequencies and city sizes. The mass
        falls as ``k ** -s``, so doubling the rank divides it by ``2 ** s``:

        >>> zipf = Zipf(s=2.0, kmax=1000)
        >>> round(zipf.pmf(1), 6)
        0.608297
        >>> round(zipf.pmf(2), 6)
        0.152074
        >>> round(zipf.pmf(1) / zipf.pmf(2), 6)
        4.0
    """

    s: float
    kmax: int = 10_000
    _Z: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and compute normalization constant."""
        if self.s <= 1:
            raise ParameterError("Zipf requires s>1.")
        # Compute the Riemann zeta function approximation (truncated at kmax)
        zeta_s = sum(n ** (-self.s) for n in range(1, self.kmax + 1))
        object.__setattr__(self, "_Z", zeta_s)

    def pmf(self, k: int) -> float:
        return float((k ** (-self.s)) / self._Z) if 1 <= k <= self.kmax else 0.0

    def cdf(self, k: int) -> float:
        k = min(max(1, k), self.kmax)
        return float(sum(n ** (-self.s) for n in range(1, k + 1)) / self._Z)

    def ppf(self, u: float) -> int:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        c, _total = 0.0, 0
        for n in range(1, self.kmax + 1):
            c += n ** (-self.s)
            if c / self._Z >= u:
                return n
        return self.kmax

    def _rvs_one(self, rng: RNG) -> int:
        return self.ppf(rng.uniform_0_1())


@dataclass(frozen=True)
class YuleSimon(Samplable):
    """
    Yule-Simon with shape rho>0 (discrete heavy tail).
    P(X=k) = rho * B(k, rho+1) = rho * Gamma(k)Gamma(rho+1) / Gamma(k+rho+1)

    Examples:
        A preferential-attachment law: the mass falls like ``k ** -(rho + 1)``,
        so the tail index is ``rho``.

        >>> yule = YuleSimon(rho=2.0)
        >>> round(yule.pmf(1), 6)
        0.666667
        >>> round(yule.pmf(2), 6)
        0.166667
        >>> round(yule.sf(10), 6)
        0.015152
    """

    rho: float

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ParameterError("rho>0 required.")

    def pmf(self, k: int) -> float:
        """Probability mass function.

        Evaluated through ``lgamma`` rather than ``gamma``. The individual
        gamma factors overflow for k around 170 even though their ratio stays
        far below one, and k that large is entirely ordinary for a heavy tail.
        """
        if k < 1:
            return 0.0
        log_pmf = (
            math.log(self.rho)
            + math.lgamma(k)
            + math.lgamma(self.rho + 1.0)
            - math.lgamma(k + self.rho + 1.0)
        )
        return float(math.exp(log_pmf))

    def sf(self, k: int) -> float:
        """Survival function P(X > k), in closed form.

        For the Yule-Simon law ``P(X > k) = k * B(k, rho + 1)``, which is
        ``k * pmf(k) / rho``. Using it avoids both the O(k) summation and the
        cancellation that ``1 - cdf(k)`` suffers in the tail.
        """
        if k < 1:
            return 1.0
        return float(k * self.pmf(k) / self.rho)

    def cdf(self, k: int) -> float:
        """Cumulative distribution function P(X <= k)."""
        if k < 1:
            return 0.0
        return float(1.0 - self.sf(k))

    def ppf(self, u: float) -> int:
        """Smallest k with ``cdf(k) >= u``.

        Found by doubling to bracket the answer and then bisecting, so the cost
        is logarithmic in k rather than linear. A linear scan is untenable here
        because the tail reaches very large k for modest u.
        """
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")

        # Double until the bracket contains the quantile.
        hi = 1
        while self.cdf(hi) < u:
            hi *= 2
            if hi > 2**62:  # pragma: no cover - u indistinguishable from 1.0
                return hi
        lo = hi // 2

        # Bisect for the smallest k satisfying the condition.
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cdf(mid) < u:
                lo = mid + 1
            else:
                hi = mid
        return int(max(lo, 1))

    def _rvs_one(self, rng: RNG) -> int:
        return self.ppf(rng.uniform_0_1())


@dataclass(frozen=True)
class DiscretePareto(Samplable):
    """
    Discrete Pareto (Zeta-type) with shape alpha>0, min k_min>=1.

    P(X=k) = (k/k_min)^(-alpha) / H_alpha(k_min,kmax)

    Attributes
    ----------
    alpha : float
        Shape parameter (must be > 0)
    k_min : int
        Minimum value of support (default: 1)
    k_max : int
        Maximum value for truncated distribution (default: 10,000)
    _H : float
        Normalization constant H_alpha computed in __post_init__

    Examples:
        The Pareto tail on the integers, truncated at ``k_max`` so the
        normalising sum is finite:

        >>> discrete_pareto = DiscretePareto(alpha=1.5, k_min=1, k_max=1000)
        >>> round(discrete_pareto.pmf(1), 6)
        0.392288
        >>> round(discrete_pareto.cdf(3), 6)
        0.606479
        >>> discrete_pareto.ppf(0.5)
        2
    """

    alpha: float
    k_min: int = 1
    k_max: int = 10_000
    _H: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate parameters and compute normalization constant."""
        if self.alpha <= 0 or self.k_min < 1:
            raise ParameterError("alpha>0, k_min>=1 required.")
        # Compute the generalized harmonic number H_alpha
        H_alpha = sum(
            (k / self.k_min) ** (-self.alpha) for k in range(self.k_min, self.k_max + 1)
        )
        object.__setattr__(self, "_H", H_alpha)

    def pmf(self, k: int) -> float:
        if k < self.k_min or k > self.k_max:
            return 0.0
        return float(((k / self.k_min) ** (-self.alpha)) / self._H)

    def cdf(self, k: int) -> float:
        k = min(max(self.k_min, k), self.k_max)
        return float(
            sum(((n / self.k_min) ** (-self.alpha)) for n in range(self.k_min, k + 1))
            / self._H
        )

    def _rvs_one(self, rng: RNG) -> int:
        return self.ppf(rng.uniform_0_1())

    def ppf(self, u: float) -> int:
        """Smallest k with ``cdf(k) >= u``."""
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        c = 0.0
        for k in range(self.k_min, self.k_max + 1):
            c += ((k / self.k_min) ** (-self.alpha)) / self._H
            if c >= u:
                return k
        return self.k_max
