# extra_distributions.py
from __future__ import annotations

from dataclasses import dataclass
import math

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
        """Whether ``x`` is inside the support.

        Both halves matter. The bracket condition alone is the *upper* endpoint
        for a bounded (negative xi) distribution, and it is satisfied well
        below ``mu`` for every sign of xi -- so on its own it let points below
        the support through and produced probabilities like -2.586.
        """
        if x < self.mu:
            return False
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
            return 0.0 if x < self.mu else 1.0
        z = (x - self.mu) / self.sigma
        if self.xi == 0.0:
            return -math.expm1(-z)
        # -expm1(-log1p(xi z)/xi) rather than 1 - (1 + xi z)**(-1/xi). Both
        # the power and the subtraction lose the small quantity: at x = 1e-9
        # the naive form returns 1.0000000827e-09 for a true 1e-09, wrong in
        # the eighth digit, and the lower tail of a GPD is a real part of it.
        return float(-math.expm1(-math.log1p(self.xi * z) / self.xi))

    def sf(self, x: float) -> float:
        return 1.0 - self.cdf(x)

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        try:
            if self.xi == 0.0:
                return self.mu - self.sigma * math.log1p(-u)
            # expm1(-xi log1p(-u)) rather than (1-u)**-xi - 1. Both the
            # subtraction inside the power and the one after it cancel for
            # small u: the textbook form returns 1.0000000827e-09 where the
            # answer is 1.0000000007e-09, wrong in the eighth digit, and the
            # lower tail of a GPD is not a corner case.
            return float(
                self.mu + (self.sigma / self.xi) * math.expm1(-self.xi * math.log1p(-u))
            )
        except OverflowError:
            return math.inf

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
        # See GeneralizedPareto.cdf for why this is not 1 - (1+z)**-k.
        return float(-math.expm1(-self.k * math.log1p(z)))

    def sf(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        z = (x / self.s) ** self.c
        return float((1.0 + z) ** (-self.k))

    def ppf(self, u: float) -> float:
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
        try:
            # See GeneralizedPareto.ppf: expm1 of a log1p keeps the lower tail,
            # where (1-u)**(-1/k) - 1 loses all but about seven digits.
            return float(
                self.s * math.expm1(-math.log1p(-u) / self.k) ** (1.0 / self.c)
            )
        except OverflowError:
            return math.inf

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
        try:
            return float(self.lam * (u / (1.0 - u)) ** (1.0 / self.kappa))
        except OverflowError:
            return math.inf

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
        # F(x) = Q(a, b/x), taken from the upper incomplete gamma rather than
        # as 1 - P. For small x the probability is tiny and 1 - P returns
        # exactly zero: at alpha=2, x=0.02 the true value is 9.8e-21 and the
        # subtraction gives nothing at all. The lower tail of a heavy-tailed
        # distribution is not the place to lose every digit.
        return _gammainc_upper_reg(a, b / x)

    def sf(self, x: float) -> float:
        if x <= 0.0:
            return 1.0
        # The mirror of the above: here P is the small quantity.
        return _gammainc_lower_reg(self.alpha, self.beta / x)

    def ppf(self, u: float) -> float:
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
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
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
            return 1.0 - self.cdf(x)
        return _betainc_reg(self.b, self.a, self.s / (x + self.s))

    def ppf(self, u: float) -> float:
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
        if not (0.0 < u < 1.0):
            raise ValueError("u must be in (0,1).")
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
