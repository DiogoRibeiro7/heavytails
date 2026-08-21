# extra_distributions.py
from __future__ import annotations

from dataclasses import dataclass
import math

# The special functions live in heavytails._special so that heavy_tails.py can
# use them too without creating an import cycle. They are re-exported here
# because they were public-by-convention at this location first.
from heavytails._special import (
    _betainc_reg,
    _gammainc_lower_reg,
    _log_beta,
    _ppf_monotone,
)
from heavytails.heavy_tails import RNG, ParameterError, Samplable

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
class GeneralizedPareto(Samplable):
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
    """

    xi: float
    sigma: float = 1.0
    mu: float = 0.0

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise ParameterError("GPD requires sigma>0.")

    def _valid(self, x: float) -> bool:
        return (1.0 + self.xi * ((x - self.mu) / self.sigma)) > 0.0

    def pdf(self, x: float) -> float:
        if not self._valid(x):
            return 0.0
        z = (x - self.mu) / self.sigma
        t = 1.0 + self.xi * z
        return (
            (1.0 / self.sigma) * (t ** (-1.0 / self.xi - 1.0))
            if self.xi != 0.0
            else (1.0 / self.sigma) * math.exp(-z)
        )

    def cdf(self, x: float) -> float:
        if not self._valid(x):
            return 0.0 if (self.xi >= 0 and x < self.mu) else 1.0
        z = (x - self.mu) / self.sigma
        if self.xi == 0.0:
            return 1.0 - math.exp(-z)
        t = 1.0 + self.xi * z
        return float(1.0 - t ** (-1.0 / self.xi))

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        if self.xi == 0.0:
            return self.mu - self.sigma * math.log(1.0 - u)
        return float(self.mu + (self.sigma / self.xi) * ((1.0 - u) ** (-self.xi) - 1.0))

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        return self.ppf(u)


@dataclass(frozen=True)
class BurrXII(Samplable):
    """
    Burr Type XII with shapes c>0, k>0 and scale s>0.

    CDF: F(x) = 1 - (1 + (x/s)^c)^(-k),   x > 0
    PDF: f(x) = (ck/s) * (x/s)^(c-1) * (1 + (x/s)^c)^(-k-1)
    PPF: x = s * ( (1 - u)^(-1/k) - 1 )^(1/c)
    """

    c: float
    k: float
    s: float = 1.0

    def __post_init__(self) -> None:
        if not (self.c > 0 and self.k > 0 and self.s > 0):
            raise ParameterError("BurrXII requires c>0, k>0, s>0.")

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (x / self.s) ** self.c
        return float(
            (self.c * self.k / self.s)
            * (x / self.s) ** (self.c - 1.0)
            * (1.0 + z) ** (-self.k - 1.0)
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (x / self.s) ** self.c
        return float(1.0 - (1.0 + z) ** (-self.k))

    def sf(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        z = (x / self.s) ** self.c
        return float((1.0 + z) ** (-self.k))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return float(self.s * (((1.0 - u) ** (-1.0 / self.k)) - 1.0) ** (1.0 / self.c))

    def _rvs_one(self, rng: RNG) -> float:
        return self.ppf(rng.uniform_0_1())


@dataclass(frozen=True)
class LogLogistic(Samplable):
    """
    Log-Logistic (Fisk) with shape kappa>0 and scale lambda_>0 (support x>0).
    CDF: F(x) = 1 / (1 + (lambda_/x)^kappa) = (x^kappa) / (x^kappa + lambda_^kappa)
    PDF: f(x) = (kappa/lambda_) (x/lambda_)^(kappa-1) / (1 + (x/lambda_)^kappa)^2
    PPF: x = lambda_ * (u/(1-u))^(1/kappa)
    """

    kappa: float
    lam: float = 1.0

    def __post_init__(self) -> None:
        if not (self.kappa > 0 and self.lam > 0):
            raise ParameterError("LogLogistic requires kappa>0 and lam>0.")

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (x / self.lam) ** self.kappa
        return float(
            (self.kappa / self.lam)
            * (x / self.lam) ** (self.kappa - 1.0)
            / (1.0 + z) ** 2
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (x / self.lam) ** self.kappa
        return float(z / (1.0 + z))

    def sf(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        z = (x / self.lam) ** self.kappa
        return float(1.0 / (1.0 + z))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return float(self.lam * (u / (1.0 - u)) ** (1.0 / self.kappa))

    def _rvs_one(self, rng: RNG) -> float:
        return self.ppf(rng.uniform_0_1())


@dataclass(frozen=True)
class InverseGamma(Samplable):
    """
    Inverse-Gamma with shape alpha>0 and scale β>0 (support x>0).
    PDF: f(x) = β^alpha / Gamma(alpha) * x^{-alpha-1} * exp(-β/x)
    CDF: F(x) = Q(alpha, β/x) = Gamma(alpha, β/x) / Gamma(alpha)  (regularized upper gamma)
         where Q = 1 - P and P is the regularized lower gamma.
    Sampling: If G ~ Gamma(alpha, scale=1), then X = β / G has InvGamma(alpha, β).
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.beta > 0):
            raise ParameterError("InverseGamma requires alpha>0 and beta>0.")

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        a, b = self.alpha, self.beta
        return float(
            (b**a / math.exp(math.lgamma(a))) * (x ** (-a - 1.0)) * math.exp(-b / x)
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        a, b = self.alpha, self.beta
        # F(x) = Q(a, b/x) = 1 - P(a, b/x)
        P = _gammainc_lower_reg(a, b / x)
        return 1.0 - P

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")

        # Solve F(x) = u on x in (0, +inf). Monotone increasing.
        def cdf_x(t: float) -> float:
            return self.cdf(t)

        # Choose a crude bracket using quantile heuristics:
        # start around mode for alpha>1: beta/(alpha+1) and expand
        a = 0.0
        b = max(1.0, self.beta / max(self.alpha + 1.0, 2.0))  # initial right
        while cdf_x(b) < u:
            b *= 2.0
            if b > 1e300:  # avoid overflow
                break
        return _ppf_monotone(cdf_x, max(1e-300, a), b, u, pdf=self.pdf)

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
    """

    a: float
    b: float
    s: float = 1.0

    def __post_init__(self) -> None:
        if not (self.a > 0 and self.b > 0 and self.s > 0):
            raise ParameterError("BetaPrime requires a>0, b>0, s>0.")

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        a, b, s = self.a, self.b, self.s
        z = x / s
        return float(
            math.exp(-(math.log(s) + _log_beta(a, b)))
            * (z ** (a - 1.0))
            * (1.0 + z) ** (-(a + b))
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        y = x / (x + self.s)
        return _betainc_reg(self.a, self.b, y)

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")

        # invert I_{x/(x+s)}(a,b) = u  -> y=u_inv, then x = s * y / (1 - y)
        # We'll solve directly for x using monotone root finding.
        def cdf_x(t: float) -> float:
            return self.cdf(t)

        # crude bracket: median is roughly s * a / b for symmetric-ish shapes.
        a0 = 0.0
        b0 = max(1e-6, self.s * (self.a / max(self.b, 1e-6)))
        # expand until bracket contains u
        while cdf_x(b0) < u:
            b0 *= 2.0
            if b0 > 1e300:
                break
        return _ppf_monotone(cdf_x, max(1e-300, a0), b0, u, pdf=self.pdf)

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
