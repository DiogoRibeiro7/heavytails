"""Evaluate a distribution over many points at once, using NumPy when present.

Evaluating a density at 100,000 points through the scalar methods means 100,000
Python-level calls, and most of that time is interpreter overhead rather than
arithmetic. Expressing the same formula against NumPy arrays removes it.

Measured on the public call -- list in, array out, guards included -- at
100,000 points, by ``scripts/vectorization_benchmark.py``:

============================  ===========  ============  =========
Call                          loop (ms)    fast (ms)     speedup
============================  ===========  ============  =========
``LogLogistic.pdf``                21.6          3.37        6.4x
``GeneralizedPareto.cdf``          22.5          4.02        5.6x
``Pareto.pdf``                     16.0          3.50        4.6x
``Weibull.cdf``                    13.8          4.28        3.2x
``Cauchy.ppf``                     14.7          7.22        2.0x
============================  ===========  ============  =========

Across the 32 accelerated calls the range is 2.1x to 6.4x, median 3.9x. An
earlier draft of this table claimed 5x to 19x, which was measured on the bare
NumPy expression rather than on the call a user makes: it left out converting
the input, applying the guards, and returning an array. The benchmark script
times the public function for exactly that reason.

**The scalar methods remain the reference implementation.** Nothing here
changes them, and every kernel below is a transcription of one. The test suite
compares the two element by element, including across the guard regions below
the support where a transcription slip is likeliest.

The agreement is stronger than a tolerance and weaker than universal equality,
and it is worth stating precisely. NumPy and :mod:`math` call the same library
for ``log``, ``exp``, ``log1p``, ``expm1``, ``arctan`` and ``tan``, so eight of
the ten kernels are **bit-identical** to their scalar counterparts at every
point tested. BurrXII and LogLogistic are not: NumPy evaluates ``**`` over an
array through a different route than the scalar operator, and on roughly one
point in two thousand the last bits differ. The measured worst case is 4 units
in the last place, and the tests hold both to a budget of 8 -- tight enough
that any real transcription error fails immediately, loose enough to survive a
platform whose ``pow`` rounds differently.

Four families have no kernel and cannot have one: LogNormal needs the error
function, and StudentT, InverseGamma and BetaPrime need the incomplete beta or
gamma. NumPy provides none of these -- they live in SciPy, which this library
deliberately does not depend on. Those families fall back to the loop, which is
correct and no slower than calling the scalar method directly.

Without NumPy installed everything falls back to the loop and the results are
lists rather than arrays. That is the only difference the caller sees.

    >>> from heavytails import Pareto
    >>> from heavytails.vectorized import cdf
    >>> values = cdf(Pareto(alpha=2.0, xm=1.0), [1.0, 2.0, 10.0])
    >>> [round(float(v), 4) for v in values]
    [0.0, 0.75, 0.99]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["accelerated", "cdf", "pdf", "ppf", "sf"]

_MISSING = object()
_numpy_module: Any = _MISSING


def _numpy() -> Any:
    """Return NumPy, or None if it is not installed.

    Imported on first use and remembered, so the failure costs one attempt
    rather than one per call.
    """
    global _numpy_module  # noqa: PLW0603
    if _numpy_module is _MISSING:
        try:
            import numpy  # noqa: PLC0415
        except ModuleNotFoundError:  # pragma: no cover - needs numpy absent
            _numpy_module = None
        else:
            _numpy_module = numpy
    return _numpy_module


# --------------------------- Kernels ----------------------------------------- #
#
# Each is a transcription of the scalar method with the same name, including
# its guards. Where the scalar version branches on the argument, the kernel
# uses `where`, and the branch that would have raised or returned a constant is
# computed anyway and discarded -- so intermediate warnings are suppressed
# rather than avoided. That is the price of branchless evaluation, and it is
# why the tests check the guard regions specifically.


def _pareto_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = d.alpha * (d.xm**d.alpha) / (x ** (d.alpha + 1.0))
    return np.where(x < d.xm, 0.0, value)


def _pareto_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 1.0 - (d.xm / x) ** d.alpha
    return np.where(x < d.xm, 0.0, value)


def _pareto_sf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (d.xm / x) ** d.alpha
    return np.where(x < d.xm, 1.0, value)


def _pareto_ppf(d: Any, u: Any, _np: Any) -> Any:
    return d.xm * (1.0 - u) ** (-1.0 / d.alpha)


def _cauchy_pdf(d: Any, x: Any, _np: Any) -> Any:
    z = (x - d.x0) / d.gamma
    return 1.0 / (math.pi * d.gamma * (1.0 + z * z))


def _cauchy_cdf(d: Any, x: Any, np: Any) -> Any:
    z = (x - d.x0) / d.gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        lower = np.arctan(-1.0 / z) / math.pi
        upper = 1.0 - np.arctan(1.0 / z) / math.pi
    middle = 0.5 + np.arctan(z) / math.pi
    return np.where(z < -1.0, lower, np.where(z > 1.0, upper, middle))


def _cauchy_sf(d: Any, x: Any, np: Any) -> Any:
    z = (x - d.x0) / d.gamma
    with np.errstate(divide="ignore", invalid="ignore"):
        positive = np.arctan(1.0 / z) / math.pi
    return np.where(z > 0.0, positive, 0.5 - np.arctan(z) / math.pi)


def _cauchy_ppf(d: Any, u: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        low = d.x0 - d.gamma / np.tan(math.pi * u)
        high = d.x0 + d.gamma / np.tan(math.pi * (1.0 - u))
    middle = d.x0 + d.gamma * np.tan(math.pi * (u - 0.5))
    return np.where(u < 0.25, low, np.where(u > 0.75, high, middle))


def _weibull_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x / d.lam) ** d.k
        value = (d.k / d.lam) * (x / d.lam) ** (d.k - 1.0) * np.exp(-z)
    if d.k < 1.0:
        value = np.where(x == 0.0, np.inf, value)
    return np.where(x < 0.0, 0.0, value)


def _weibull_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = -np.expm1(-((x / d.lam) ** d.k))
    return np.where(x < 0.0, 0.0, value)


def _weibull_sf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.exp(-((x / d.lam) ** d.k))
    return np.where(x < 0.0, 1.0, value)


def _weibull_ppf(d: Any, u: Any, np: Any) -> Any:
    return d.lam * (-np.log1p(-u)) ** (1.0 / d.k)


def _frechet_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - d.m) / d.s
        t = z ** (-d.alpha)
        value = (d.alpha / d.s) * z ** (-(d.alpha + 1.0)) * np.exp(-t)
    return np.where(x <= d.m, 0.0, value)


def _frechet_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = np.exp(-(((x - d.m) / d.s) ** (-d.alpha)))
    return np.where(x <= d.m, 0.0, value)


def _frechet_sf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = -np.expm1(-(((x - d.m) / d.s) ** (-d.alpha)))
    return np.where(x <= d.m, 1.0, value)


def _frechet_ppf(d: Any, u: Any, np: Any) -> Any:
    return d.m + d.s * (-np.log(u)) ** (-1.0 / d.alpha)


def _gev_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 1.0 + d.xi * ((x - d.mu) / d.sigma)
        value = (
            (1.0 / d.sigma) * t ** (-1.0 / d.xi - 1.0) * np.exp(-(t ** (-1.0 / d.xi)))
        )
    return np.where(t > 0.0, value, 0.0)


def _gev_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 1.0 + d.xi * ((x - d.mu) / d.sigma)
        value = np.exp(-(t ** (-1.0 / d.xi)))
    return np.where(t > 0.0, value, 0.0)


def _gev_sf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        t = 1.0 + d.xi * ((x - d.mu) / d.sigma)
        value = -np.expm1(-(t ** (-1.0 / d.xi)))
    return np.where(t > 0.0, value, 1.0)


def _gev_ppf(d: Any, u: Any, np: Any) -> Any:
    return d.mu + (d.sigma / d.xi) * ((-np.log(u)) ** (-d.xi) - 1.0)


def _gpd_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - d.mu) / d.sigma
        t = 1.0 + d.xi * z
        value = (
            (1.0 / d.sigma) * t ** (-1.0 / d.xi - 1.0)
            if d.xi != 0.0
            else (1.0 / d.sigma) * np.exp(-z)
        )
    return np.where((t > 0.0) & (x >= d.mu), value, 0.0)


def _gpd_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x - d.mu) / d.sigma
        t = 1.0 + d.xi * z
        value = -np.expm1(-np.log1p(d.xi * z) / d.xi) if d.xi != 0.0 else -np.expm1(-z)
    # Below the support the answer is 0; past the upper endpoint of a bounded
    # (negative xi) GPD it is 1. Not the same constant, and the condition that
    # separates them is which side of mu the point is on.
    return np.where((t > 0.0) & (x >= d.mu), value, np.where(x < d.mu, 0.0, 1.0))


def _gpd_sf(d: Any, x: Any, np: Any) -> Any:
    # The scalar version is literally 1 - cdf, so this must be too, or the two
    # would differ in the last bit exactly where sf is small.
    return 1.0 - _gpd_cdf(d, x, np)


def _gpd_ppf(d: Any, u: Any, np: Any) -> Any:
    if d.xi == 0.0:
        return d.mu - d.sigma * np.log1p(-u)
    return d.mu + (d.sigma / d.xi) * np.expm1(-d.xi * np.log1p(-u))


def _burr_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x / d.s) ** d.c
        value = (d.c * d.k / d.s) * (x / d.s) ** (d.c - 1.0) * (1.0 + z) ** (-d.k - 1.0)
    return np.where(x <= 0.0, 0.0, value)


def _burr_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = -np.expm1(-d.k * np.log1p((x / d.s) ** d.c))
    return np.where(x <= 0.0, 0.0, value)


def _burr_sf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = (1.0 + (x / d.s) ** d.c) ** (-d.k)
    return np.where(x <= 0.0, 1.0, value)


def _burr_ppf(d: Any, u: Any, np: Any) -> Any:
    return d.s * np.expm1(-np.log1p(-u) / d.k) ** (1.0 / d.c)


def _loglogistic_pdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x / d.lam) ** d.kappa
        value = (d.kappa / d.lam) * (x / d.lam) ** (d.kappa - 1.0) / (1.0 + z) ** 2
    return np.where(x <= 0.0, 0.0, value)


def _loglogistic_cdf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (x / d.lam) ** d.kappa
        value = z / (1.0 + z)
    return np.where(x <= 0.0, 0.0, value)


def _loglogistic_sf(d: Any, x: Any, np: Any) -> Any:
    with np.errstate(divide="ignore", invalid="ignore"):
        value = 1.0 / (1.0 + (x / d.lam) ** d.kappa)
    return np.where(x <= 0.0, 1.0, value)


def _loglogistic_ppf(d: Any, u: Any, _np: Any) -> Any:
    return d.lam * (u / (1.0 - u)) ** (1.0 / d.kappa)


_KERNELS: dict[tuple[str, str], Any] = {
    ("Pareto", "pdf"): _pareto_pdf,
    ("Pareto", "cdf"): _pareto_cdf,
    ("Pareto", "sf"): _pareto_sf,
    ("Pareto", "ppf"): _pareto_ppf,
    ("Cauchy", "pdf"): _cauchy_pdf,
    ("Cauchy", "cdf"): _cauchy_cdf,
    ("Cauchy", "sf"): _cauchy_sf,
    ("Cauchy", "ppf"): _cauchy_ppf,
    ("Weibull", "pdf"): _weibull_pdf,
    ("Weibull", "cdf"): _weibull_cdf,
    ("Weibull", "sf"): _weibull_sf,
    ("Weibull", "ppf"): _weibull_ppf,
    ("Frechet", "pdf"): _frechet_pdf,
    ("Frechet", "cdf"): _frechet_cdf,
    ("Frechet", "sf"): _frechet_sf,
    ("Frechet", "ppf"): _frechet_ppf,
    ("GEV_Frechet", "pdf"): _gev_pdf,
    ("GEV_Frechet", "cdf"): _gev_cdf,
    ("GEV_Frechet", "sf"): _gev_sf,
    ("GEV_Frechet", "ppf"): _gev_ppf,
    ("GeneralizedPareto", "pdf"): _gpd_pdf,
    ("GeneralizedPareto", "cdf"): _gpd_cdf,
    ("GeneralizedPareto", "sf"): _gpd_sf,
    ("GeneralizedPareto", "ppf"): _gpd_ppf,
    ("BurrXII", "pdf"): _burr_pdf,
    ("BurrXII", "cdf"): _burr_cdf,
    ("BurrXII", "sf"): _burr_sf,
    ("BurrXII", "ppf"): _burr_ppf,
    ("LogLogistic", "pdf"): _loglogistic_pdf,
    ("LogLogistic", "cdf"): _loglogistic_cdf,
    ("LogLogistic", "sf"): _loglogistic_sf,
    ("LogLogistic", "ppf"): _loglogistic_ppf,
}


def accelerated(dist: Any, method: str) -> bool:
    """
    Report whether a NumPy kernel will be used for this call.

    Worth checking before assuming a speedup: LogNormal, StudentT, InverseGamma
    and BetaPrime have no kernel, because NumPy has neither the error function
    nor the incomplete beta and gamma. Those live in SciPy, which this library
    does not depend on.

    Args:
        dist: A distribution instance.
        method: One of ``pdf``, ``cdf``, ``sf``, ``ppf``.

    Returns:
        True if the fast path applies, False if the call will loop.

    Examples:
        >>> from heavytails import LogNormal, Pareto
        >>> accelerated(Pareto(alpha=2.0, xm=1.0), "cdf")
        True
        >>> accelerated(LogNormal(mu=0.0, sigma=1.0), "cdf")
        False
    """
    if _numpy() is None:  # pragma: no cover - needs numpy absent
        return False
    return (type(dist).__name__, method) in _KERNELS


def _evaluate(dist: Any, method: str, values: Sequence[float]) -> Any:
    """Apply ``method`` to every element, by kernel where one exists."""
    numpy = _numpy()
    if numpy is None:  # pragma: no cover - needs numpy absent
        return [getattr(dist, method)(value) for value in values]

    kernel = _KERNELS.get((type(dist).__name__, method))
    if kernel is None:
        # No kernel, so the loop -- which is exactly what the caller would have
        # written, and is the reference implementation either way. It iterates
        # `values` rather than an array built from it: converting first, then
        # looping, costs the conversion *and* makes every element a NumPy
        # scalar, which the scalar methods handle more slowly than a float.
        # Doing that turned these families' calls into a 0.6x slowdown.
        scalar = getattr(dist, method)
        return numpy.asarray([scalar(value) for value in values], dtype=float)
    array = numpy.asarray(values, dtype=float)
    return numpy.asarray(kernel(dist, array, numpy), dtype=float)


def pdf(dist: Any, values: Sequence[float]) -> Any:
    """
    Density at every point in ``values``.

    Args:
        dist: A distribution instance.
        values: Points to evaluate at.

    Returns:
        A NumPy array when NumPy is installed, a list otherwise. Both index and
        iterate the same way.

    Examples:
        >>> from heavytails import Pareto
        >>> [round(float(v), 4) for v in pdf(Pareto(alpha=2.0, xm=1.0), [0.5, 1.0, 2.0])]
        [0.0, 2.0, 0.25]
    """
    return _evaluate(dist, "pdf", values)


def cdf(dist: Any, values: Sequence[float]) -> Any:
    """
    Distribution function at every point in ``values``.

    Args:
        dist: A distribution instance.
        values: Points to evaluate at.

    Returns:
        A NumPy array when NumPy is installed, a list otherwise.
    """
    return _evaluate(dist, "cdf", values)


def sf(dist: Any, values: Sequence[float]) -> Any:
    """
    Survival function at every point in ``values``.

    Args:
        dist: A distribution instance.
        values: Points to evaluate at.

    Returns:
        A NumPy array when NumPy is installed, a list otherwise.
    """
    return _evaluate(dist, "sf", values)


def ppf(dist: Any, probabilities: Sequence[float]) -> Any:
    """
    Quantile at every probability in ``probabilities``.

    Unlike the scalar :meth:`ppf`, which raises on a probability outside the
    open unit interval, this validates the whole input first and names the
    offending value. A loop would have raised partway through and thrown away
    the work already done.

    Args:
        dist: A distribution instance.
        probabilities: Probabilities in the open interval (0, 1).

    Returns:
        A NumPy array when NumPy is installed, a list otherwise.

    Raises:
        ValueError: If any probability is outside (0, 1).

    Examples:
        >>> from heavytails import Pareto
        >>> [round(float(v), 4) for v in ppf(Pareto(alpha=2.0, xm=1.0), [0.5, 0.99])]
        [1.4142, 10.0]
    """
    numpy = _numpy()
    if numpy is None:  # pragma: no cover - needs numpy absent
        offending = [p for p in probabilities if not (0.0 < p < 1.0)]
    else:
        array = numpy.asarray(probabilities, dtype=float)
        bad = array[(array <= 0.0) | (array >= 1.0) | numpy.isnan(array)]
        offending = bad.tolist()
    if offending:
        raise ValueError(
            f"every probability must be in (0,1); got {offending[0]!r}"
            + (f" and {len(offending) - 1} other(s)" if len(offending) > 1 else "")
        )
    return _evaluate(dist, "ppf", probabilities)
