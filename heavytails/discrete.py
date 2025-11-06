# heavytails/discrete.py
from __future__ import annotations

from dataclasses import dataclass
import math

from heavytails.heavy_tails import RNG, ParameterError, Samplable


@dataclass(frozen=True)
class Zipf(Samplable):
    """
    Zipf (Zeta) distribution with exponent s>1 on support k=1,2,...

    P(X=k) = k^{-s} / ζ(s)
    where ζ(s) ≈ ∑_{n=1}^∞ n^{-s}
    """

    s: float
    kmax: int = 10_000  # truncation for normalization

    def __post_init__(self) -> None:
        if self.s <= 1:
            raise ParameterError("Zipf requires s>1.")
        object.__setattr__(
            self, "_Z", sum(n ** (-self.s) for n in range(1, self.kmax + 1))
        )

    def pmf(self, k: int) -> float:
        return (k ** (-self.s)) / self._Z if 1 <= k <= self.kmax else 0.0

    def cdf(self, k: int) -> float:
        k = min(max(1, k), self.kmax)
        return sum(n ** (-self.s) for n in range(1, k + 1)) / self._Z

    def ppf(self, u: float) -> int:
        if not (0.0 < u < 1.0):
            raise ValueError("u in (0,1)")
        c, total = 0.0, 0
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
    Yule–Simon with shape ρ>0 (discrete heavy tail).
    P(X=k) = ρ * B(k, ρ+1) = ρ * Γ(k)Γ(ρ+1) / Γ(k+ρ+1)
    """

    rho: float

    def __post_init__(self) -> None:
        if self.rho <= 0:
            raise ParameterError("rho>0 required.")

    def pmf(self, k: int) -> float:
        if k < 1:
            return 0.0
        return (
            self.rho
            * math.gamma(k)
            * math.gamma(self.rho + 1)
            / math.gamma(k + self.rho + 1)
        )

    def cdf(self, k: int) -> float:
        return sum(self.pmf(i) for i in range(1, k + 1))

    def _rvs_one(self, rng: RNG) -> int:
        u = rng.uniform_0_1()
        # Inverse transform via cdf table
        c, n = 0.0, 0
        while c < u and n < 10000:
            n += 1
            c += self.pmf(n)
        return n


@dataclass(frozen=True)
class DiscretePareto(Samplable):
    """
    Discrete Pareto (Zeta-type) with shape α>0, min k_min>=1.
    P(X=k) = (k/k_min)^(-α) / H_α(k_min,kmax)
    """

    alpha: float
    k_min: int = 1
    k_max: int = 10_000

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.k_min < 1:
            raise ParameterError("alpha>0, k_min>=1 required.")
        H = sum(
            (k / self.k_min) ** (-self.alpha) for k in range(self.k_min, self.k_max + 1)
        )
        object.__setattr__(self, "_H", H)

    def pmf(self, k: int) -> float:
        if k < self.k_min or k > self.k_max:
            return 0.0
        return ((k / self.k_min) ** (-self.alpha)) / self._H

    def cdf(self, k: int) -> float:
        k = min(max(self.k_min, k), self.k_max)
        return (
            sum(((n / self.k_min) ** (-self.alpha)) for n in range(self.k_min, k + 1))
            / self._H
        )

    def _rvs_one(self, rng: RNG) -> int:
        return self.ppf(rng.uniform_0_1())

    def ppf(self, u: float) -> int:
        c = 0.0
        for k in range(self.k_min, self.k_max + 1):
            c += ((k / self.k_min) ** (-self.alpha)) / self._H
            if c >= u:
                return k
        return self.k_max
