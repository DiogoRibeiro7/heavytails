"""Tail-risk metrics: value at risk, expected shortfall and Monte Carlo estimation.

These are the quantities practitioners ask for, rather than the distributions
underneath them. Two points recur and are easy to get wrong:

**Expected shortfall is infinite when the mean does not exist.** For a Pareto
tail with ``alpha <= 1`` there is no finite answer, and returning a large
number instead of ``inf`` would be worse than useless: it would look like a
result. Every function here checks and reports ``inf``.

**A Monte Carlo estimate without a standard error is not usable.** The whole
point of these metrics is the far tail, where few samples land, so
:func:`monte_carlo_tail_risk` always reports the uncertainty alongside the
estimate.
"""

from __future__ import annotations

import math
from typing import Any

from heavytails._special import _gammainc_lower_reg, _phi_inverse

__all__ = [
    "expected_shortfall",
    "mean_exists",
    "monte_carlo_tail_risk",
    "tail_conditional_expectation",
    "value_at_risk",
]


def _standard_normal_cdf(x: float) -> float:
    """Standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def mean_exists(dist: Any) -> bool:
    """Report whether a distribution has a finite mean.

    Expected shortfall is a conditional mean, so it is infinite exactly when
    this is False. The check is by family, since it depends on the parameters:
    a Pareto tail has a finite mean only for ``alpha > 1``.

    Args:
        dist: A distribution instance from this package.

    Returns:
        True if the mean is finite. Unknown families are assumed to have one,
        which is the less surprising default; the numeric path will produce a
        very large value rather than a wrong finite one if that is wrong.

    Examples:
        >>> from heavytails import Cauchy, Pareto
        >>> mean_exists(Pareto(alpha=2.0, xm=1.0))
        True
        >>> mean_exists(Pareto(alpha=0.5, xm=1.0))
        False
        >>> mean_exists(Cauchy())
        False
    """
    name = type(dist).__name__
    if name == "Cauchy":
        return False
    if name == "Pareto":
        return bool(dist.alpha > 1.0)
    if name == "StudentT":
        return bool(dist.nu > 1.0)
    if name == "Frechet":
        return bool(dist.alpha > 1.0)
    if name in {"GeneralizedPareto", "GEV_Frechet"}:
        return bool(dist.xi < 1.0)
    if name == "BurrXII":
        return bool(dist.c * dist.k > 1.0)
    if name == "LogLogistic":
        return bool(dist.kappa > 1.0)
    if name == "InverseGamma":
        return bool(dist.alpha > 1.0)
    if name == "BetaPrime":
        return bool(dist.b > 1.0)
    # LogNormal and Weibull always have a finite mean; anything unrecognised is
    # assumed to as well.
    return True


def value_at_risk(dist: Any, level: float) -> float:
    """
    Value at risk: the quantile of the loss distribution at ``level``.

    This is exactly ``dist.ppf(level)``, and exists as a named function because
    the terminology is what practitioners search for, and because pairing it
    with :func:`expected_shortfall` makes the distinction between the two
    explicit.

    Value at risk says how large a loss is exceeded with probability
    ``1 - level``. It says nothing about how large the exceedances are, which
    is what expected shortfall answers and why the two are usually reported
    together.

    Args:
        dist: A distribution with a ``ppf`` method.
        level: Confidence level in the open interval (0, 1), for example 0.99.

    Returns:
        The quantile. ``inf`` when the quantile exceeds the float range.

    Raises:
        ValueError: If ``level`` is outside (0, 1).

    Examples:
        >>> from heavytails import Pareto
        >>> round(value_at_risk(Pareto(alpha=2.0, xm=1.0), 0.99), 4)
        10.0
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    return float(dist.ppf(level))


def _analytic_expected_shortfall(dist: Any, level: float) -> float | None:
    """Closed-form expected shortfall, or None when there is not one here.

    Each formula is verified against Monte Carlo in the test suite.
    """
    name = type(dist).__name__
    var = float(dist.ppf(level))

    if name == "Pareto":
        # The conditional mean above v is v scaled by alpha/(alpha-1).
        return float(var * dist.alpha / (dist.alpha - 1.0))

    if name == "LogNormal":
        # The lognormal mean times a normal tail probability, rescaled by
        # the exceedance probability.
        z = _phi_inverse(level)
        numerator = math.exp(dist.mu + dist.sigma * dist.sigma / 2.0)
        return float(numerator * _standard_normal_cdf(dist.sigma - z) / (1.0 - level))

    if name == "GeneralizedPareto":
        # Standard peaks-over-threshold result, valid for xi below one.
        return float(
            var / (1.0 - dist.xi) + (dist.sigma - dist.xi * dist.mu) / (1.0 - dist.xi)
        )

    if name == "Weibull":
        # Uses the regularised upper incomplete gamma at shape 1 + 1/k.
        shape = 1.0 + 1.0 / dist.k
        upper = 1.0 - _gammainc_lower_reg(shape, (var / dist.lam) ** dist.k)
        return float(dist.lam * math.gamma(shape) * upper / (1.0 - level))

    return None


def _numeric_expected_shortfall(dist: Any, level: float, nodes: int) -> float:
    """Expected shortfall by quadrature on the quantile function.

    ``ES = 1/(1-p) * int_p^1 ppf(u) du``, rewritten as
    ``int_0^1 ppf(1 - (1-p)t) dt`` and then substituted ``t = s**4`` so the
    nodes cluster near ``t = 0``, which is where a heavy-tailed integrand puts
    all its mass. Without that substitution the midpoint rule badly
    underestimates the tail.
    """
    power = 4.0
    total = 0.0
    for i in range(nodes):
        s = (i + 0.5) / nodes
        t = s**power
        weight = power * s ** (power - 1.0)
        u = 1.0 - (1.0 - level) * t
        if u >= 1.0:
            continue
        total += float(dist.ppf(u)) * weight
    return total / nodes


def expected_shortfall(
    dist: Any, level: float, method: str = "auto", nodes: int = 20000
) -> float:
    """
    Expected shortfall: the mean loss given that value at risk is exceeded.

    Also called conditional value at risk. Where value at risk reports a
    threshold, this reports the average of what lies beyond it, which is the
    question that matters when the tail is heavy.

    Args:
        dist: A distribution with a ``ppf`` method.
        level: Confidence level in the open interval (0, 1).
        method: ``auto`` uses the closed form where this module has one and
            falls back to quadrature; ``analytic`` raises if there is no closed
            form; ``numeric`` always integrates.
        nodes: Quadrature nodes for the numeric path.

    Returns:
        The expected shortfall, or ``inf`` when the distribution has no finite
        mean.

    Raises:
        ValueError: If ``level`` is outside (0, 1), ``method`` is unknown, or
            ``analytic`` was requested for a family without a closed form here.

    Examples:
        >>> from heavytails import Cauchy, Pareto
        >>> round(expected_shortfall(Pareto(alpha=2.0, xm=1.0), 0.99), 4)
        20.0
        >>> expected_shortfall(Pareto(alpha=0.5, xm=1.0), 0.99)
        inf
        >>> expected_shortfall(Cauchy(), 0.99)
        inf
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    if method not in {"auto", "analytic", "numeric"}:
        raise ValueError(
            f"Unknown method {method!r}. Available: auto, analytic, numeric"
        )

    # A conditional mean cannot be finite when the mean is not.
    if not mean_exists(dist):
        return math.inf

    if method in {"auto", "analytic"}:
        closed_form = _analytic_expected_shortfall(dist, level)
        if closed_form is not None:
            return float(closed_form)
        if method == "analytic":
            raise ValueError(
                f"No closed-form expected shortfall for {type(dist).__name__} "
                "in this module; use method='auto' or 'numeric'."
            )

    return _numeric_expected_shortfall(dist, level, nodes)


def tail_conditional_expectation(dist: Any, level: float, **kwargs: Any) -> float:
    """
    Tail conditional expectation, ``E[X | X > VaR]``.

    For a continuous distribution this is the same quantity as
    :func:`expected_shortfall`, and this function delegates to it. The two
    names come from different literatures, actuarial and financial, and they
    diverge only for distributions with an atom at the quantile, which none of
    the continuous families here have.

    Args:
        dist: A distribution with a ``ppf`` method.
        level: Confidence level in the open interval (0, 1).
        **kwargs: Passed to :func:`expected_shortfall`.

    Returns:
        The tail conditional expectation.
    """
    return expected_shortfall(dist, level, **kwargs)


def monte_carlo_tail_risk(
    dist: Any,
    level: float,
    n_samples: int = 100000,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Estimate value at risk and expected shortfall by simulation, with errors.

    An estimate of a tail quantity without a standard error is not usable: the
    whole point is the region where few samples land, so the uncertainty is
    large and varies with the level. This always reports it.

    The standard error of the expected shortfall is the standard error of the
    mean of the exceedances. The standard error of the value at risk uses the
    asymptotic result for a sample quantile,
    ``sqrt(p(1-p)/n) / f(VaR)``, evaluated with the density where the
    distribution provides one.

    Args:
        dist: A distribution with ``ppf``, ``rvs`` and ideally ``pdf``.
        level: Confidence level in the open interval (0, 1).
        n_samples: Number of variates to draw.
        seed: Seed, for a reproducible estimate.

    Returns:
        Dictionary with ``value_at_risk``, ``expected_shortfall``, their
        standard errors, ``n_exceedances``, ``n_samples`` and ``level``. The
        expected shortfall is ``inf`` when the distribution has no finite mean.

    Raises:
        ValueError: If ``level`` is outside (0, 1), ``n_samples`` is too small,
            or the level leaves fewer than two exceedances to average.

    Examples:
        >>> from heavytails import Pareto
        >>> result = monte_carlo_tail_risk(
        ...     Pareto(alpha=2.0, xm=1.0), 0.99, n_samples=20000, seed=1
        ... )
        >>> result["n_exceedances"]
        200
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")

    sample = sorted(dist.rvs(n_samples, seed=seed))
    cut = int(level * n_samples)
    exceedances = sample[cut:]
    if len(exceedances) < 2:
        raise ValueError(
            f"level={level} leaves only {len(exceedances)} exceedances in "
            f"{n_samples} samples; raise n_samples or lower the level"
        )

    var = sample[cut - 1] if cut > 0 else sample[0]

    # Standard error of a sample quantile, where the density is available.
    var_error: float | None = None
    density = getattr(dist, "pdf", None)
    if callable(density):
        try:
            f = float(density(var))
            if f > 0.0:
                var_error = math.sqrt(level * (1.0 - level) / n_samples) / f
        except (ValueError, OverflowError, ZeroDivisionError):
            var_error = None

    if not mean_exists(dist):
        # The sample mean of the exceedances does not converge, so reporting it
        # with a standard error would dress up a meaningless number.
        return {
            "level": level,
            "n_samples": n_samples,
            "n_exceedances": len(exceedances),
            "value_at_risk": float(var),
            "value_at_risk_std_error": var_error,
            "expected_shortfall": math.inf,
            "expected_shortfall_std_error": None,
        }

    count = len(exceedances)
    shortfall = sum(exceedances) / count
    variance = sum((x - shortfall) ** 2 for x in exceedances) / max(count - 1, 1)

    return {
        "level": level,
        "n_samples": n_samples,
        "n_exceedances": count,
        "value_at_risk": float(var),
        "value_at_risk_std_error": var_error,
        "expected_shortfall": float(shortfall),
        "expected_shortfall_std_error": math.sqrt(variance / count),
    }
