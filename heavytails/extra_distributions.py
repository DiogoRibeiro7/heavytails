# extra_distributions.py
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math

# Reuse the small utilities from your base module
from heavy_tails import RNG, ParameterError, Samplable

# =============================================================================
# Numeric special functions (stdlib only)
# =============================================================================


def _log_beta(a: float, b: float) -> float:
    """log B(a,b) via lgamma for stability."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _betainc_reg(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta I_x(a,b) using:
      - symmetry reduction (x -> 1-x) for x > (a+1)/(a+b+2)
      - Lentz/continued-fraction for the incomplete beta function ratio
    Accuracy ~ 1e-12 in double precision for typical parameter ranges.
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError("x must be in [0,1].")
    if a <= 0 or b <= 0:
        raise ValueError("a,b must be > 0.")

    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0

    # Use symmetry to push x into the smaller side, improves convergence.
    flip = False
    if x > (a + 1.0) / (a + b + 2.0):
        flip = True
        x = 1.0 - x
        a, b = b, a  # swap

    # Compute front factor: x^a * (1-x)^b / (a * B(a,b))
    log_front = a * math.log(x) + b * math.log1p(-x) - math.log(a) - _log_beta(a, b)
    front = math.exp(log_front)

    # Continued fraction for the incomplete beta: cf = 1 / (1 + ... )
    # Lentz's algorithm
    EPS = 1e-14
    MAX_ITER = 200
    am, bm = 1.0, 1.0  # Not used directly; we implement cf in-place.
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < EPS:
        d = EPS
    d = 1.0 / d
    h = d

    for m in range(1, MAX_ITER + 1):
        m2 = 2 * m

        # even step
        num = m * (b - m) * x
        den = (a + m2 - 1.0) * (a + m2)
        aa = num / den
        d = 1.0 + aa * d
        if abs(d) < EPS:
            d = EPS
        c = 1.0 + aa / c
        if abs(c) < EPS:
            c = EPS
        d = 1.0 / d
        h *= d * c

        # odd step
        num = -(a + m) * (a + b + m) * x
        den = (a + m2) * (a + m2 + 1.0)
        aa = num / den
        d = 1.0 + aa * d
        if abs(d) < EPS:
            d = EPS
        c = 1.0 + aa / c
        if abs(c) < EPS:
            c = EPS
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < EPS:
            break
    else:
        # did not converge
        pass

    ibeta = front * h
    result = ibeta
    if flip:
        # undo symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
        result = 1.0 - ibeta
    return result


def _gammainc_lower_reg(a: float, x: float) -> float:
    """
    Regularized lower incomplete gamma P(a,x) = γ(a,x) / Γ(a).
    Uses series for x < a+1 and continued fraction for x >= a+1.
    """
    if a <= 0 or x < 0:
        raise ValueError("a must be >0 and x>=0.")

    if x == 0.0:
        return 0.0

    # Series representation (Abramowitz & Stegun 6.5.29)
    if x < a + 1.0:
        term = 1.0 / a
        summ = term
        n = 1
        while True:
            term *= x / (a + n)
            summ += term
            if abs(term) < abs(summ) * 1e-15 or n > 10000:
                break
            n += 1
        return summ * math.exp(-x + a * math.log(x) - math.lgamma(a))

    # Continued fraction (A&S 6.5.31) via Lentz's method
    # P(a,x) = 1 - e^{-x} x^a / Γ(a) * 1/CF
    # We compute Q(a,x) first through the CF, then P=1-Q
    MAX_ITER = 10_000
    EPS = 1e-14
    tiny = 1e-300

    # Initialize Lentz
    f = 1.0
    C = 1.0 / tiny
    D = 0.0

    for n in range(1, MAX_ITER + 1):
        # a_n / b_n terms; see standard CF for regularized gamma
        # Here we implement the modified Lentz for the continued fraction of Q
        # Coefficients:
        an = n * (a - n)
        bn = x + 2.0 * n - a
        # update D
        D = bn + an * D
        if abs(D) < tiny:
            D = tiny
        D = 1.0 / D
        # update C
        C = bn + an / C
        if abs(C) < tiny:
            C = tiny
        delta = C * D
        f *= delta
        if abs(delta - 1.0) < EPS:
            break
    # Q(a,x) approx:
    Q = f * math.exp(-x + a * math.log(x) - math.lgamma(a))
    P = 1.0 - Q
    # Clamp to [0,1]
    P = max(P, 0.0)
    P = min(P, 1.0)
    return P


def _ppf_monotone(
    cdf: Callable[[float], float],
    lo: float,
    hi: float,
    u: float,
    pdf: Callable[[float], float] | None = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> float:
    """
    Generic monotone inverse for continuous distributions on (lo,hi).
    Safeguarded Newton: try Newton when pdf is available and well-behaved,
    otherwise fall back to bisection.
    """
    if not (0.0 < u < 1.0):
        raise ValueError("u must be in (0,1).")
    a, b = lo, hi
    fa, fb = cdf(a), cdf(b)
    if not (fa <= u <= fb):
        # expand bounds if possible
        raise ValueError("Provided [lo, hi] does not bracket the quantile.")
    x = 0.5 * (a + b)

    for _ in range(max_iter):
        fx = cdf(x) - u
        if abs(fx) < tol:
            return x
        # Try Newton if pdf provided
        if pdf is not None:
            dfx = pdf(x)
            if dfx > 0.0:
                step = fx / dfx
                xn = x - step
                if a < xn < b:
                    x = xn
                    continue
        # Bisection fallback
        if fx > 0.0:
            b = x
        else:
            a = x
        x = 0.5 * (a + b)
    return x


# =============================================================================
# Distributions
# =============================================================================


@dataclass(frozen=True)
class GeneralizedPareto(Samplable):
    """
    Generalized Pareto Distribution (GPD) with shape ξ, scale σ>0, location μ.

    Support:
        x >= μ if ξ >= 0  (heavy-tailed when ξ>0)
        μ <= x <= μ - σ/ξ if ξ < 0 (bounded tail; not heavy)

    CDF:
        F(x) = 1 - (1 + ξ (x-μ)/σ)^(-1/ξ), valid where bracket > 0
    PDF:
        f(x) = (1/σ) * (1 + ξ z)^(-1/ξ - 1),  z=(x-μ)/σ
    PPF:
        x = μ + (σ/ξ) * ( (1-u)^(-ξ) - 1 )      if ξ != 0
        x = μ - σ * ln(1-u)                     if ξ = 0 (exponential limit)
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
        return 1.0 - t ** (-1.0 / self.xi)

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        if self.xi == 0.0:
            return self.mu - self.sigma * math.log(1.0 - u)
        return self.mu + (self.sigma / self.xi) * ((1.0 - u) ** (-self.xi) - 1.0)

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
        return (
            (self.c * self.k / self.s)
            * (x / self.s) ** (self.c - 1.0)
            * (1.0 + z) ** (-self.k - 1.0)
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (x / self.s) ** self.c
        return 1.0 - (1.0 + z) ** (-self.k)

    def sf(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        z = (x / self.s) ** self.c
        return (1.0 + z) ** (-self.k)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.s * (((1.0 - u) ** (-1.0 / self.k)) - 1.0) ** (1.0 / self.c)

    def _rvs_one(self, rng: RNG) -> float:
        return self.ppf(rng.uniform_0_1())


@dataclass(frozen=True)
class LogLogistic(Samplable):
    """
    Log-Logistic (Fisk) with shape κ>0 and scale λ>0 (support x>0).
    CDF: F(x) = 1 / (1 + (λ/x)^κ) = (x^κ) / (x^κ + λ^κ)
    PDF: f(x) = (κ/λ) (x/λ)^(κ-1) / (1 + (x/λ)^κ)^2
    PPF: x = λ * (u/(1-u))^(1/κ)
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
        return (
            (self.kappa / self.lam)
            * (x / self.lam) ** (self.kappa - 1.0)
            / (1.0 + z) ** 2
        )

    def cdf(self, x: float) -> float:
        if x <= 0.0:
            return 0.0
        z = (x / self.lam) ** self.kappa
        return z / (1.0 + z)

    def sf(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        z = (x / self.lam) ** self.kappa
        return 1.0 / (1.0 + z)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        return self.lam * (u / (1.0 - u)) ** (1.0 / self.kappa)

    def _rvs_one(self, rng: RNG) -> float:
        return self.ppf(rng.uniform_0_1())


@dataclass(frozen=True)
class InverseGamma(Samplable):
    """
    Inverse-Gamma with shape α>0 and scale β>0 (support x>0).
    PDF: f(x) = β^α / Γ(α) * x^{-α-1} * exp(-β/x)
    CDF: F(x) = Q(α, β/x) = Γ(α, β/x) / Γ(α)  (regularized upper gamma)
         where Q = 1 - P and P is the regularized lower gamma.
    Sampling: If G ~ Gamma(α, scale=1), then X = β / G has InvGamma(α, β).
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
        return (b**a / math.exp(math.lgamma(a))) * (x ** (-a - 1.0)) * math.exp(-b / x)

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
        return (
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
