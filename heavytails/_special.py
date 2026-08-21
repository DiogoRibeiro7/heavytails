"""Numeric special functions and root finding, implemented with the stdlib only.

These live in their own module so that both :mod:`heavytails.heavy_tails` and
:mod:`heavytails.extra_distributions` can use them without an import cycle.
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

# Spacing of the floats near 1.0; used to decide when a bracket has collapsed.
_ULP = sys.float_info.epsilon


class ConvergenceError(RuntimeError):
    """Raised when an iterative solver fails to reach the requested tolerance.

    Returning a best guess instead would be worse: the caller cannot tell an
    answer good to machine precision from one the solver simply stopped at.
    """


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
    _am, _bm = 1.0, 1.0  # Not used directly; we implement cf in-place.
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
    Regularized lower incomplete gamma P(a,x) = gamma(a,x) / Gamma(a).
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

    # Continued fraction (A&S 6.5.31) evaluated by modified Lentz.
    #
    #   Q(a,x) = e^{-x} x^a / Gamma(a) * CF,  CF = 1/(x+1-a - 1*(1-a)/(x+3-a - ...))
    #
    # The recurrence starts from b_0 = x + 1 - a with h = 1/b_0, and b advances
    # by 2 each step. Starting from h = 1 and using b = x + 2n - a instead --
    # dropping the leading term and shifting b by one -- produces a plausible
    # looking number that is simply wrong, badly enough that P came back as 0.0
    # where the true value was 0.6.
    MAX_ITER = 10_000
    EPS = 1e-14
    tiny = 1e-300

    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if abs(b) > tiny else 1.0 / tiny
    h = d

    for n in range(1, MAX_ITER + 1):
        an = -n * (n - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break

    Q = h * math.exp(-x + a * math.log(x) - math.lgamma(a))
    P = 1.0 - Q
    return min(max(P, 0.0), 1.0)


def _ppf_monotone(
    cdf: Callable[[float], float],
    lo: float,
    hi: float,
    u: float,
    pdf: Callable[[float], float] | None = None,
    max_iter: int = 100,
    tol: float = 1e-12,
) -> float:
    """Generic monotone inverse for continuous distributions on ``(lo, hi)``.

    Safeguarded Newton: a Newton step is taken whenever the density is
    available, positive, and the step stays inside the current bracket;
    otherwise the method bisects.

    The bracket is narrowed on **every** iteration from the sign of the
    residual, including the ones where a Newton step is taken. An earlier
    version only narrowed it on bisection fallbacks, so a run of accepted
    Newton steps could consume the whole iteration budget while the bracket
    stayed as wide as it started, leaving no way to tell a converged answer
    from an abandoned one.

    Parameters
    ----------
    cdf : callable
        Monotone non-decreasing distribution function.
    lo, hi : float
        Bracket known to contain the quantile.
    u : float
        Probability in the open interval (0, 1).
    pdf : callable, optional
        Density, used for the Newton steps. Bisection only, if omitted.
    max_iter : int, optional
        Maximum iterations before giving up.
    tol : float, optional
        Absolute tolerance on the residual ``cdf(x) - u``.

    Returns
    -------
    float
        The quantile.

    Raises
    ------
    ValueError
        If ``u`` is outside (0, 1) or ``[lo, hi]`` does not bracket the root.
    ConvergenceError
        If the tolerance is not met and the bracket has not collapsed to the
        resolution of the floats around it.
    """
    if not (0.0 < u < 1.0):
        raise ValueError("u must be in (0,1).")
    a, b = lo, hi
    fa, fb = cdf(a), cdf(b)
    if not (fa <= u <= fb):
        raise ValueError("Provided [lo, hi] does not bracket the quantile.")
    x = 0.5 * (a + b)

    for _ in range(max_iter):
        fx = cdf(x) - u
        if abs(fx) < tol:
            return x

        # Narrow the bracket from the sign of the residual, whatever kind of
        # step produced x. This is what guarantees progress.
        if fx > 0.0:
            b = x
        else:
            a = x

        # A bracket at the resolution of the floats around it cannot be
        # narrowed further; the density is flat here and x is as good as the
        # representation allows.
        if b - a <= 4.0 * _ULP * max(abs(a), abs(b), 1.0):
            return x

        stepped = False
        if pdf is not None:
            dfx = pdf(x)
            if dfx > 0.0:
                xn = x - fx / dfx
                if a < xn < b:
                    x = xn
                    stepped = True
        if not stepped:
            x = 0.5 * (a + b)

    if b - a <= 4.0 * _ULP * max(abs(a), abs(b), 1.0):
        return x
    raise ConvergenceError(
        f"ppf did not converge for u={u!r}: after {max_iter} iterations the "
        f"bracket is still [{a!r}, {b!r}] with residual {cdf(x) - u!r}."
    )


def _betaincinv_reg(
    a: float, b: float, p: float, *, max_iter: int = 200, tol: float = 1e-15
) -> float:
    """Inverse of the regularized incomplete beta: return y with I_y(a,b) = p.

    Solving in log-space is what makes this usable in the far tail. A plain
    bisection on ``y`` in [0, 1] converges to a fixed *absolute* precision of
    about 1e-16, which leaves a target of, say, y = 1e-15 with barely one
    correct digit. Bisecting ``t = log(y)`` instead keeps full *relative*
    precision however small y becomes, and a safeguarded Newton polish then
    reaches machine accuracy.

    Parameters
    ----------
    a, b : float
        Positive shape parameters.
    p : float
        Target value of the regularized incomplete beta, in [0, 1].
    max_iter : int, optional
        Maximum number of Newton iterations after the bracketing phase.
    tol : float, optional
        Relative convergence tolerance on ``p``.

    Returns
    -------
    float
        The value ``y`` in [0, 1] satisfying ``I_y(a, b) = p``.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if a <= 0 or b <= 0:
        raise ValueError("a,b must be > 0.")
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return 1.0

    # Symmetry reduction: I_y(a,b) = 1 - I_{1-y}(b,a). For p near 1 the root
    # sits where the incomplete beta is nearly flat, so log-space resolution in
    # y stops helping. Solving the mirrored problem puts the root back in the
    # steep region, where the small quantity 1-y is represented accurately.
    if p > 0.5:
        return 1.0 - _betaincinv_reg(b, a, 1.0 - p, max_iter=max_iter, tol=tol)

    log_b = _log_beta(a, b)

    def g(t: float) -> float:
        """I_{exp(t)}(a,b) - p."""
        return _betainc_reg(a, b, math.exp(t)) - p

    # Bracket the root in t = log(y). The lower bound is close to the smallest
    # representable exponent; the upper bound is t = 0, i.e. y = 1.
    lo, hi = -745.0, 0.0
    if g(lo) > 0.0:
        return math.exp(lo)

    # Bisection in log-space to get a tight, correctly-signed bracket.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13:
            break

    t = 0.5 * (lo + hi)

    # Newton polish on t, safeguarded by the bracket. With y = exp(t),
    # d/dt I_y(a,b) = y**a * (1-y)**(b-1) / B(a,b).
    for _ in range(max_iter):
        y = math.exp(t)
        if y >= 1.0:
            return 1.0
        err = _betainc_reg(a, b, y) - p
        if abs(err) <= tol * p:
            break
        log_deriv = a * t + (b - 1.0) * math.log1p(-y) - log_b
        if log_deriv < -745.0:
            break
        step = err / math.exp(log_deriv)
        t_new = t - step
        if not (lo < t_new < hi):
            t_new = 0.5 * (lo + hi)
        if err < 0.0:
            lo = t
        else:
            hi = t
        if abs(t_new - t) < 1e-16 * abs(t):
            t = t_new
            break
        t = t_new

    return math.exp(t)
