# heavytails/discrete.py
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from heavytails._array import as_array, check_probabilities, elementwise, restore
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
    _cumulative: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate parameters and build the distribution over the support."""
        if self.s <= 1:
            raise ParameterError("Zipf requires s>1.")
        # The truncated zeta, with the running distribution function beside it.
        # The constructor already walked the whole support once to normalise;
        # keeping the cumulative sums turns `cdf` from a sum over k terms into
        # a lookup, and `ppf` from a scan over the support into a search.
        weights = np.arange(1, self.kmax + 1, dtype=float) ** (-self.s)
        object.__setattr__(self, "_Z", float(weights.sum()))
        object.__setattr__(self, "_cumulative", np.cumsum(weights) / self._Z)

    def pmf(self, k: Any) -> Any:
        values, scalar = as_array(k)
        inside = (values >= 1) & (values <= self.kmax)
        safe = np.where(inside, values, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mass = safe ** (-self.s) / self._Z
        return restore(np.where(inside, mass, 0.0), scalar)

    def cdf(self, k: Any) -> Any:
        values, scalar = as_array(k)
        index = np.clip(np.floor(values), 1, self.kmax).astype(int) - 1
        probability = self._cumulative[index]
        return restore(np.where(values < 1, 0.0, probability), scalar)

    def ppf(self, u: Any) -> Any:
        """Smallest k with ``cdf(k) >= u``, by search rather than by scanning.

        The scan this replaces walked the support one k at a time, so a
        quantile in the far tail of a kmax of ten thousand cost ten thousand
        steps -- and `rvs` pays that per draw.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        found = np.searchsorted(self._cumulative, values, side="left") + 1
        quantile = np.minimum(found, self.kmax)
        return int(quantile) if scalar else quantile.astype(int)

    def _rvs_one(self, rng: RNG) -> int:
        # ppf mirrors its input, so a scalar in gives an int out; the cast
        # says so to the type checker, which sees the array-or-scalar type.
        return int(self.ppf(rng.uniform_0_1()))


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

    def pmf(self, k: Any) -> Any:
        """Probability mass function, for one value or many.

        Evaluated through ``lgamma`` rather than ``gamma``. The individual
        gamma factors overflow for k around 170 even though their ratio stays
        far below one, and k that large is entirely ordinary for a heavy tail.

        NumPy has no ``lgamma``, so this goes one element at a time, as
        LogNormal and StudentT do for the same reason. The interface is the
        same as every other family's; the speed is not.
        """
        values, scalar = as_array(k)
        return restore(elementwise(self._pmf_one, values), scalar)

    def _pmf_one(self, k: float) -> float:
        if k < 1:
            return 0.0
        log_pmf = (
            math.log(self.rho)
            + math.lgamma(k)
            + math.lgamma(self.rho + 1.0)
            - math.lgamma(k + self.rho + 1.0)
        )
        return float(math.exp(log_pmf))

    def sf(self, k: Any) -> Any:
        """Survival function P(X > k), in closed form, for one value or many.

        For the Yule-Simon law ``P(X > k) = k * B(k, rho + 1)``, which is
        ``k * pmf(k) / rho``. Using it avoids both the O(k) summation and the
        cancellation that ``1 - cdf(k)`` suffers in the tail.
        """
        values, scalar = as_array(k)
        mass = np.asarray(self.pmf(values), dtype=float)
        survival = np.where(values < 1, 1.0, values * mass / self.rho)
        return restore(survival, scalar)

    def cdf(self, k: Any) -> Any:
        """Cumulative distribution function P(X <= k), for one value or many."""
        values, scalar = as_array(k)
        survival = np.asarray(self.sf(values), dtype=float)
        return restore(np.where(values < 1, 0.0, 1.0 - survival), scalar)

    def ppf(self, u: Any) -> Any:
        """Smallest k with ``cdf(k) >= u``, for one probability or many.

        Found by doubling to bracket the answer and then bisecting, so the cost
        is logarithmic in k rather than linear. A linear scan is untenable here
        because the tail reaches very large k for modest u.

        Unlike Zipf and DiscretePareto this support is unbounded, so there is
        no cumulative table to search; the bracket is still per probability.
        """
        values, scalar = as_array(u)
        check_probabilities(values)
        quantile = elementwise(self._ppf_one, values)
        return int(quantile) if scalar else quantile.astype(int)

    def _ppf_one(self, u: float) -> float:
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
        return float(max(lo, 1))

    def _rvs_one(self, rng: RNG) -> int:
        # ppf mirrors its input, so a scalar in gives an int out; the cast
        # says so to the type checker, which sees the array-or-scalar type.
        return int(self.ppf(rng.uniform_0_1()))


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
    _cumulative: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate parameters and compute normalization constant."""
        if self.alpha <= 0 or self.k_min < 1:
            raise ParameterError("alpha>0, k_min>=1 required.")
        # Compute the generalized harmonic number H_alpha
        H_alpha = sum(
            (k / self.k_min) ** (-self.alpha) for k in range(self.k_min, self.k_max + 1)
        )
        object.__setattr__(self, "_H", H_alpha)
        # The running distribution function over the support. See Zipf: the
        # normalisation already walks it once, and keeping the cumulative sums
        # is what lets `ppf` search instead of scan.
        support = np.arange(self.k_min, self.k_max + 1, dtype=float)
        weights = (support / self.k_min) ** (-self.alpha)
        object.__setattr__(self, "_cumulative", np.cumsum(weights) / self._H)

    def pmf(self, k: Any) -> Any:
        values, scalar = as_array(k)
        inside = (values >= self.k_min) & (values <= self.k_max)
        safe = np.where(inside, values, self.k_min)
        with np.errstate(divide="ignore", invalid="ignore"):
            mass = (safe / self.k_min) ** (-self.alpha) / self._H
        return restore(np.where(inside, mass, 0.0), scalar)

    def cdf(self, k: Any) -> Any:
        values, scalar = as_array(k)
        index = (
            np.clip(np.floor(values), self.k_min, self.k_max).astype(int) - self.k_min
        )
        probability = self._cumulative[index]
        return restore(np.where(values < self.k_min, 0.0, probability), scalar)

    def _rvs_one(self, rng: RNG) -> int:
        # ppf mirrors its input, so a scalar in gives an int out; the cast
        # says so to the type checker, which sees the array-or-scalar type.
        return int(self.ppf(rng.uniform_0_1()))

    def ppf(self, u: Any) -> Any:
        """Smallest k with ``cdf(k) >= u``, by search rather than by scanning."""
        values, scalar = as_array(u)
        check_probabilities(values)
        found = np.searchsorted(self._cumulative, values, side="left") + self.k_min
        quantile = np.minimum(found, self.k_max)
        return int(quantile) if scalar else quantile.astype(int)
