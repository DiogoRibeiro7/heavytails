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


def _gammainc_upper_reg(a: float, x: float) -> float:
    """
    Regularized upper incomplete gamma Q(a,x) = Gamma(a,x) / Gamma(a).

    Not simply ``1 - _gammainc_lower_reg(a, x)``. The continued fraction that
    branch uses computes Q first and returns ``1 - Q``, so for large ``x``,
    where Q is the small quantity, every digit of it is lost to the
    subtraction. Returning it directly keeps full relative precision in the far
    upper tail, which is where a heavy-tailed distribution lives.

    Parameters
    ----------
    a : float
        Positive shape parameter.
    x : float
        Non-negative argument.

    Returns
    -------
    float
        Q(a, x), in [0, 1].

    Raises
    ------
    ValueError
        If ``a`` is not positive or ``x`` is negative.
    """
    if a <= 0 or x < 0:
        raise ValueError("a must be >0 and x>=0.")
    if x == 0.0:
        return 1.0

    if x < a + 1.0:
        # Series branch: here P is the well-conditioned quantity, so Q comes
        # from the subtraction. That is the right way round -- Q is O(1) here.
        return 1.0 - _gammainc_lower_reg(a, x)

    max_iter = 10_000
    eps = 1e-14
    tiny = 1e-300

    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b if abs(b) > tiny else 1.0 / tiny
    h = d
    for n in range(1, max_iter + 1):
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
        if abs(delta - 1.0) < eps:
            break

    log_q = -x + a * math.log(x) - math.lgamma(a) + math.log(h)
    if log_q < -745.0:
        return 0.0
    return min(max(math.exp(log_q), 0.0), 1.0)


def _gammaincinv_reg(
    a: float,
    p: float,
    *,
    upper: bool = False,
    max_iter: int = 200,
    tol: float = 1e-14,
) -> float:
    """Inverse of the regularized incomplete gamma: return x with P(a,x) = p.

    With ``upper=True``, solves ``Q(a,x) = p`` instead. Both are provided
    because the caller usually knows which of the two is its small quantity,
    and going through the other one means forming ``1 - p`` and losing the
    precision that made the tail worth computing.

    Solved in ``t = log(x)`` for the same reason :func:`_betaincinv_reg` is:
    bisecting ``x`` converges to a fixed absolute precision, which leaves a
    target of 1e-15 with barely a digit, while bisecting its logarithm keeps
    relative precision however small the root becomes.

    The starting point is the small-``x`` asymptote ``P(a,x) -> x**a /
    (a Gamma(a))``, inverted, which is exact to several digits wherever the
    tail is small and never worse than a couple of log units elsewhere. That
    matters more than the speed of the inner loop: blind bisection of the
    exponent range spends sixty continued-fraction evaluations getting to
    where this arrives in one.

    Parameters
    ----------
    a : float
        Positive shape parameter.
    p : float
        Target probability in [0, 1].
    upper : bool, optional
        Solve ``Q(a,x) = p`` rather than ``P(a,x) = p``.
    max_iter : int, optional
        Maximum Newton iterations after bracketing.
    tol : float, optional
        Relative convergence tolerance on the probability.

    Returns
    -------
    float
        The value ``x >= 0`` solving the requested equation. ``inf`` when the
        root is beyond the float range.

    Raises
    ------
    ValueError
        If ``p`` is outside [0, 1] or ``a`` is not positive.
    """
    if not (0.0 <= p <= 1.0):
        raise ValueError("p must be in [0,1].")
    if a <= 0.0:
        raise ValueError("a must be > 0.")

    if upper:
        if p == 1.0:
            return 0.0
        if p == 0.0:
            return math.inf
    else:
        if p == 0.0:
            return 0.0
        if p == 1.0:
            return math.inf

    lgamma_a = math.lgamma(a)

    def g(t: float) -> float:
        """Residual, signed so that it increases with t in both modes."""
        x = math.exp(t)
        if upper:
            return p - _gammainc_upper_reg(a, x)
        return _gammainc_lower_reg(a, x) - p

    # Invert the small-x asymptote P ~ x**a / (a Gamma(a)). In upper mode the
    # equivalent small-x statement is Q ~ 1 - x**a/(a Gamma(a)), so the same
    # expression applies to 1 - p there.
    target = 1.0 - p if upper else p
    guess = 700.0 if target <= 0.0 else (math.log(target) + math.log(a) + lgamma_a) / a
    t0 = min(max(guess, -745.0), 700.0)

    g0 = g(t0)
    if g0 == 0.0:
        return math.exp(t0)

    lo, hi = t0, t0
    step = 0.5
    if g0 < 0.0:
        hi = 709.0
        while step <= 2048.0:
            probe = min(t0 + step, 709.0)
            if g(probe) >= 0.0:
                hi = probe
                break
            lo = probe
            if probe == 709.0:
                return math.inf
            step *= 2.0
    else:
        lo = -745.0
        while step <= 2048.0:
            probe = max(t0 - step, -745.0)
            if g(probe) < 0.0:
                lo = probe
                break
            hi = probe
            if probe == -745.0:
                return math.exp(probe)
            step *= 2.0

    while hi - lo > 1e-2:
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid

    t = 0.5 * (lo + hi)

    # Newton on t. With x = exp(t), d/dt P(a,x) = x**a e**-x / Gamma(a), and
    # Q differs only in sign, which the residual already carries.
    for _ in range(max_iter):
        x = math.exp(t)
        err = g(t)
        if abs(err) <= tol * max(p, 1e-300):
            break

        # Narrow before falling back to the midpoint, so the fallback is a
        # genuinely new point. See _betaincinv_reg for what the other order
        # costs.
        if err < 0.0:
            lo = t
        else:
            hi = t

        log_deriv = a * t - x - lgamma_a
        if log_deriv < -745.0:
            break
        t_new = t - err / math.exp(log_deriv)
        if not (lo < t_new < hi):
            t_new = 0.5 * (lo + hi)
        if abs(t_new - t) <= 1e-16 * abs(t):
            t = t_new
            break
        t = t_new

    return math.exp(t)


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

    # Start from the small-y asymptote rather than bisecting the whole
    # exponent range. As y -> 0, I_y(a,b) -> y**a / (a B(a,b)), so inverting
    # that gives log y directly. It is exact to six decimals for p below 1e-6
    # and within about half a log unit even at p = 0.4, against a range of 745
    # that blind bisection would have to chew through one continued-fraction
    # evaluation at a time.
    guess = (math.log(p) + math.log(a) + log_b) / a
    t0 = min(max(guess, -745.0), -1e-8)

    g0 = g(t0)
    if g0 == 0.0:
        return math.exp(t0)

    # Walk outwards from the guess until the root is bracketed. g(0) is 1 - p,
    # which is positive for every p reaching here, so t = 0 always closes the
    # bracket from above.
    lo, hi = t0, t0
    step = 0.5
    if g0 < 0.0:
        hi = 0.0
        while step <= 1024.0:
            probe = min(t0 + step, 0.0)
            if g(probe) >= 0.0:
                hi = probe
                break
            lo = probe
            if probe == 0.0:
                break
            step *= 2.0
    else:
        lo = -745.0
        while step <= 1024.0:
            probe = max(t0 - step, -745.0)
            if g(probe) < 0.0:
                lo = probe
                break
            hi = probe
            if probe == -745.0:
                return math.exp(probe)
            step *= 2.0

    # Bisect only far enough for Newton to take over. Newton converges
    # quadratically from here, so bisecting to machine precision first would
    # be paying continued-fraction evaluations for digits Newton produces in
    # two steps.
    while hi - lo > 1e-2:
        mid = 0.5 * (lo + hi)
        if g(mid) < 0.0:
            lo = mid
        else:
            hi = mid

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

        # Narrow the bracket from this residual *before* falling back to its
        # midpoint. With the other order the midpoint is computed from the
        # bracket that still contains t, so on the first iteration -- where t
        # is the midpoint by construction -- the fallback returns t itself and
        # the no-progress check below declares convergence on a point that has
        # not converged. The old fixed bisection to 1e-13 hid this by making
        # "converged at the midpoint" true anyway.
        if err < 0.0:
            lo = t
        else:
            hi = t

        log_deriv = a * t + (b - 1.0) * math.log1p(-y) - log_b
        if log_deriv < -745.0:
            break
        t_new = t - err / math.exp(log_deriv)
        if not (lo < t_new < hi):
            t_new = 0.5 * (lo + hi)
        if abs(t_new - t) <= 1e-16 * abs(t):
            t = t_new
            break
        t = t_new

    return math.exp(t)


def _phi_inverse(u: float) -> float:
    """
    Approximate the inverse standard normal CDF (quantile) for u in (0,1).
    Accuracy ~1e-9 in double precision. Based on Peter John Acklam's method.

    Reference:
    https://web.archive.org/web/20150910002153/http://home.online.no/~pjacklam/notes/invnorm/
    """
    if not (0.0 < u < 1.0):
        raise ValueError("u must be in (0,1).")

    # The standard normal quantile is antisymmetric, and the refinement below
    # is accurate only where erfc has its argument positive. Mirroring the
    # upper half onto the lower one keeps both sides at full precision instead
    # of leaving the upper tail at about 5e-11.
    if u > 0.5:
        return -_phi_inverse(1.0 - u)

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

    # One Newton refinement, which takes the rational approximation from about
    # nine digits to the full double.
    #
    # The distribution function here must come from erfc, not erf. Writing it
    # as 0.5*(1 + erf(x/sqrt(2))) is a subtraction of two nearly equal numbers
    # once x is a few units negative: at x = -7 the erf term is
    # -0.999999999999, so the sum keeps about five digits and the correction is
    # computed from noise. That is why the refinement left a relative error of
    # 4.4e-07 at u = 1e-12 -- worse than the unrefined approximation's own
    # documented accuracy -- rather than the 1e-16 it should reach.
    density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
    if density <= 0.0:  # pragma: no cover - needs |x| above 38
        return x
    tail = 0.5 * math.erfc(-x / math.sqrt(2.0))
    return x - (tail - u) / density


# ----------------------------- Demo / Self-test ------------------------------ #
