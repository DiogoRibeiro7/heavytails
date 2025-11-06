# heavy_tails.py
from __future__ import annotations

from dataclasses import dataclass
import math
import random

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
        """Z ~ N(0,1). Uses Python stdlib Box–Muller via random.gauss (Ziggurat internally)."""
        return self.rng.gauss(0.0, 1.0)

    # ---------------------- Gamma / Chi-square samplers ---------------------- #
    def gamma(self, shape_k: float, scale_theta: float = 1.0) -> float:
        """
        X ~ Gamma(k, θ) with k>0, θ>0 using Marsaglia–Tsang (2000).
        Works for all k>0 (uses boost for k<1).

        References
        ----------
        G. Marsaglia and W. W. Tsang (2000). A Simple Method for Generating Gamma Variables.
        ACM Transactions on Mathematical Software 26(3):363–372.
        """
        if not (shape_k > 0 and scale_theta > 0):
            raise ParameterError("Gamma requires shape k>0 and scale θ>0.")

        k = shape_k
        if k < 1.0:
            # Boost: sample from Gamma(k+1, 1) then * U^(1/k)
            x = self._gamma_mt(k + 1.0)
            u = self.uniform_0_1()
            return scale_theta * (x * (u ** (1.0 / k)))
        else:
            return scale_theta * self._gamma_mt(k)

    def _gamma_mt(self, k: float) -> float:
        """Marsaglia–Tsang core for k >= 1, unit scale."""
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
    """Mixin to provide vectorized sampling with a given RNG."""

    def rvs(self, n: int, seed: int | None = None) -> list[float]:
        """
        Draw n IID variates. Subclasses must implement ._rvs_one(rng).
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        rng = RNG(seed)
        return [self._rvs_one(rng) for _ in range(n)]

    def _rvs_one(self, rng: RNG) -> float:
        """Override in subclasses."""
        raise NotImplementedError


# --------------------------- Distributions ----------------------------------- #


@dataclass(frozen=True)
class Pareto(Samplable):
    """
    Pareto Type I with scale xm>0 and shape alpha>0.
    PDF: f(x) = α x_m^α / x^{α+1},  x >= x_m
    CDF: F(x) = 1 - (x_m / x)^α
    PPF: F^{-1}(u) = x_m * (1 - u)^{-1/α}
    """

    alpha: float
    xm: float = 1.0

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.xm > 0):
            raise ParameterError("Pareto requires alpha>0 and xm>0.")

    def pdf(self, x: float) -> float:
        if x < self.xm:
            return 0.0
        return self.alpha * (self.xm**self.alpha) / (x ** (self.alpha + 1.0))

    def cdf(self, x: float) -> float:
        if x < self.xm:
            return 0.0
        return 1.0 - (self.xm / x) ** self.alpha

    def sf(self, x: float) -> float:
        """Survival function 1 - CDF."""
        if x < self.xm:
            return 1.0
        return (self.xm / x) ** self.alpha

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.xm * (1.0 - u) ** (-1.0 / self.alpha)

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        return self.ppf(u)


@dataclass(frozen=True)
class Cauchy(Samplable):
    """
    Cauchy(location x0, scale gamma>0).
    PDF: f(x) = [1/πγ] * [1 / (1 + ((x-x0)/γ)^2)]
    CDF: F(x) = 0.5 + (1/π) * arctan((x - x0)/γ)
    PPF: x = x0 + γ * tan(π(u - 0.5))
    """

    x0: float = 0.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if not (self.gamma > 0):
            raise ParameterError("Cauchy requires scale gamma>0.")

    def pdf(self, x: float) -> float:
        z = (x - self.x0) / self.gamma
        return 1.0 / (math.pi * self.gamma * (1.0 + z * z))

    def cdf(self, x: float) -> float:
        z = (x - self.x0) / self.gamma
        return 0.5 + math.atan(z) / math.pi

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.x0 + self.gamma * math.tan(math.pi * (u - 0.5))

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        return self.ppf(u)


@dataclass(frozen=True)
class StudentT(Samplable):
    """
    Student's t with degrees of freedom nu>0.
    PDF: f(x) = Γ((ν+1)/2) / [ sqrt(νπ) Γ(ν/2) ] * (1 + x^2/ν)^(-(ν+1)/2)
    Sampling: X = Z / sqrt(Y/ν) with Z~N(0,1), Y~χ²(ν)
    NOTE: CDF/PPF require special functions not in stdlib; omitted.
    """

    nu: float

    def __post_init__(self) -> None:
        if not (self.nu > 0):
            raise ParameterError("Student-t requires nu>0.")

    def pdf(self, x: float) -> float:
        nu = self.nu
        c = math.gamma((nu + 1.0) / 2.0) / (
            math.sqrt(nu * math.pi) * math.gamma(nu / 2.0)
        )
        return c * (1.0 + (x * x) / nu) ** (-(nu + 1.0) / 2.0)

    def _rvs_one(self, rng: RNG) -> float:
        z = rng.standard_normal()
        y = rng.chisquare(self.nu)
        return z / math.sqrt(y / self.nu)


@dataclass(frozen=True)
class LogNormal(Samplable):
    """
    LogNormal with underlying Normal(μ, σ^2), σ>0.
    PDF: f(x) = [1/(x σ sqrt(2π))] * exp( -(ln x - μ)^2 / (2σ^2) ), x>0
    CDF: F(x) = 0.5 * [1 + erf( (ln x - μ) / (σ sqrt(2)) )], x>0
    """

    mu: float = 0.0
    sigma: float = 1.0

    def __post_init__(self) -> None:
        if not (self.sigma > 0):
            raise ParameterError("LogNormal requires sigma>0.")

    def pdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (math.log(x) - self.mu) / self.sigma
        return math.exp(-0.5 * z * z) / (x * self.sigma * math.sqrt(2.0 * math.pi))

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (math.log(x) - self.mu) / (self.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + math.erf(z))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        # Inverse via normal quantile needs erfinv; not in stdlib.
        # Use rational approximation to Φ^{-1}(u) (Acklam’s method).
        z = _phi_inverse(u)
        return math.exp(self.mu + self.sigma * z)

    def _rvs_one(self, rng: RNG) -> float:
        z = rng.standard_normal()
        return math.exp(self.mu + self.sigma * z)


@dataclass(frozen=True)
class Weibull(Samplable):
    """
    Weibull(k, λ) with shape k>0 and scale λ>0.
    PDF: f(x) = (k/λ) (x/λ)^{k-1} exp(-(x/λ)^k), x>=0
    CDF: F(x) = 1 - exp(-(x/λ)^k), x>=0
    PPF: x = λ * (-ln(1-u))^{1/k}
    Heavy-tailed for k in (0,1) (subexponential, slower than exponential decay).
    """

    k: float
    lam: float = 1.0

    def __post_init__(self) -> None:
        if not (self.k > 0 and self.lam > 0):
            raise ParameterError("Weibull requires k>0 and λ>0.")

    def pdf(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        z = (x / self.lam) ** self.k
        return (self.k / self.lam) * (x / self.lam) ** (self.k - 1.0) * math.exp(-z)

    def cdf(self, x: float) -> float:
        if x < 0.0:
            return 0.0
        return 1.0 - math.exp(-((x / self.lam) ** self.k))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.lam * (-math.log(1.0 - u)) ** (1.0 / self.k)

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        return self.ppf(u)


@dataclass(frozen=True)
class Frechet(Samplable):
    """
    Fréchet(α, s, m): heavy-tailed extreme-value distribution.
    Support x > m. α>0 (shape), s>0 (scale), m (location).
    CDF: F(x) = exp( - ((x - m)/s)^(-α) ), x>m
    PDF: f(x) = (α/s) * ((x - m)/s)^(-α-1) * exp( - ((x - m)/s)^(-α) ), x>m
    PPF: x = m + s * [ -ln(u) ]^{-1/α}
    """

    alpha: float
    s: float = 1.0
    m: float = 0.0

    def __post_init__(self) -> None:
        if not (self.alpha > 0 and self.s > 0):
            raise ParameterError("Frechet requires alpha>0 and s>0.")

    def pdf(self, x: float) -> float:
        if x <= self.m:
            return 0.0
        z = (x - self.m) / self.s
        t = z ** (-self.alpha)
        return (self.alpha / self.s) * z ** (-(self.alpha + 1.0)) * math.exp(-t)

    def cdf(self, x: float) -> float:
        if x <= self.m:
            return 0.0
        z = (x - self.m) / self.s
        return math.exp(-(z ** (-self.alpha)))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.m + self.s * (-math.log(u)) ** (-1.0 / self.alpha)

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        return self.ppf(u)


@dataclass(frozen=True)
class GEV_Frechet(Samplable):
    """
    Generalized Extreme Value (Fréchet-type) with ξ>0, μ (loc), σ>0 (scale).
    Heavy-tailed when ξ>0.

    CDF: F(x) = exp( -[1 + ξ ( (x-μ)/σ )]^(-1/ξ) ), for 1 + ξ (x-μ)/σ > 0
    PDF: f(x) = (1/σ) * [1 + ξ z]^(-1/ξ - 1) * exp( -[1 + ξ z]^(-1/ξ) ), z=(x-μ)/σ
    PPF: x = μ + (σ/ξ) * ( (-ln u)^(-ξ) - 1 )
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

    def pdf(self, x: float) -> float:
        if not self._valid(x):
            return 0.0
        z = (x - self.mu) / self.sigma
        t = 1.0 + self.xi * z
        return (
            (1.0 / self.sigma)
            * (t ** (-1.0 / self.xi - 1.0))
            * math.exp(-(t ** (-1.0 / self.xi)))
        )

    def cdf(self, x: float) -> float:
        if not self._valid(x):
            return 0.0
        z = (x - self.mu) / self.sigma
        t = 1.0 + self.xi * z
        return math.exp(-(t ** (-1.0 / self.xi)))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.mu + (self.sigma / self.xi) * ((-math.log(u)) ** (-self.xi) - 1.0)

    def _rvs_one(self, rng: RNG) -> float:
        u = rng.uniform_0_1()
        return self.ppf(u)


# -------------- Normal quantile (Acklam’s Φ^{-1}) for LogNormal PPF ---------- #


def _phi_inverse(u: float) -> float:
    """
    Approximate the inverse standard normal CDF (quantile) for u in (0,1).
    Accuracy ~1e-9 in double precision. Based on Peter John Acklam’s method.

    Reference:
    https://web.archive.org/web/20150910002153/http://home.online.no/~pjacklam/notes/invnorm/
    """
    if not (0.0 < u < 1.0):
        raise ValueError("u must be in (0,1).")

    # Coefficients
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    # Break-points
    plow = 0.02425
    phigh = 1.0 - plow

    if u < plow:
        q = math.sqrt(-2.0 * math.log(u))
        num = ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        x = num / den
    elif u > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - u))
        num = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        den = (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        x = num / den
    else:
        q = u - 0.5
        r = q * q
        num = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        den = ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        x = num / den

    # One Halley refinement for better accuracy
    # φ(x) = standard normal pdf, Φ(x) = cdf
    def phi(x: float) -> float:
        return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

    def Phi(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    x = x - (Phi(x) - u) / max(phi(x), 1e-300)
    return x


# ----------------------------- Demo / Self-test ------------------------------ #


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
