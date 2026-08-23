"""Evaluate a distribution over many points at once.

This module is a thin shim now. It exists because it was public in 0.4.0, and
everything in it forwards to the distribution methods, which take an array
directly since 0.5.0::

    >>> from heavytails import Pareto
    >>> from heavytails.vectorized import cdf
    >>> values = cdf(Pareto(alpha=2.0, xm=1.0), [1.0, 2.0, 10.0])
    >>> [round(float(v), 4) for v in values]
    [0.0, 0.75, 0.99]

``Pareto(alpha=2.0, xm=1.0).cdf([1.0, 2.0, 10.0])`` is the same call and is
what new code should write.

**What used to be here, and why it is not.** This module held a hand-written
NumPy kernel for each of 32 (family, method) pairs, each one a transcription of
the scalar method with the same name, including its guards. The scalar methods
were the reference implementation and the kernels were the fast path, and the
test suite compared the two element by element because a transcription can drop
a guard, mirror a branch the wrong way, or use a mathematically equal form that
rounds differently.

That arrangement cost more than it looks. The tolerance holding the two paths
together had to be rewritten twice before it was right -- once after asserting
bit-identity that held on Windows and failed on every Linux job, and again
after a relative budget that a single unit in the last place of ``pow`` could
exceed by four thousand. Neither was a bug in the formulas; both were the price
of having two of them.

The same shape has since turned up twice more in this library. ``streaming.py``
carried a transcription of the Hill estimator described as matching it
"operation for operation", which broke the moment the original changed its
summation. And a metadata test held a third copy of the citation title, so it
could not notice that the two files it compared had both moved. A transcription
cannot tell that it has gone stale.

So there is one implementation now. The formula lives in the method, the method
takes an array, and this module forwards to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["accelerated", "cdf", "pdf", "ppf", "sf"]


_METHODS = frozenset({"pdf", "cdf", "sf", "ppf"})

# The four families whose probabilities cannot be a NumPy expression: LogNormal
# needs the error function, and StudentT, InverseGamma and BetaPrime need the
# incomplete beta or gamma. NumPy has none of them -- they live in SciPy, which
# this library deliberately does not depend on -- so those methods evaluate one
# element at a time inside `heavytails._array.elementwise`.
#
# Their *densities* are elementary and are vectorised like everything else,
# which is why this is keyed by (family, method) and not by family.
_ELEMENTWISE = frozenset(
    (family, method)
    for family in ("LogNormal", "StudentT", "InverseGamma", "BetaPrime")
    for method in ("cdf", "sf", "ppf")
)


def accelerated(dist: Any, method: str) -> bool:
    """
    Report whether this call evaluates as a single NumPy expression.

    Worth checking before assuming a speedup: LogNormal, StudentT, InverseGamma
    and BetaPrime compute their probabilities one element at a time, because
    NumPy has neither the error function nor the incomplete beta and gamma.

    Args:
        dist: A distribution instance.
        method: One of ``pdf``, ``cdf``, ``sf``, ``ppf``.

    Returns:
        True if the call is a single NumPy expression, False if it loops.

    Note:
        This answers for the distributions in this library. A distribution
        defined elsewhere, whose methods take only scalars, is reported as
        accelerated, because there is no way to ask a method how it is written.
        Before 0.5.0 the answer came from a table of the families this module
        knew about, so an unknown class was reported as not accelerated.

    Examples:
        >>> from heavytails import LogNormal, Pareto
        >>> accelerated(Pareto(alpha=2.0, xm=1.0), "cdf")
        True
        >>> accelerated(LogNormal(mu=0.0, sigma=1.0), "cdf")
        False
        >>> accelerated(LogNormal(mu=0.0, sigma=1.0), "pdf")
        True
        >>> accelerated(Pareto(alpha=2.0, xm=1.0), "nonsense")
        False
    """
    if method not in _METHODS:
        return False
    return (type(dist).__name__, method) not in _ELEMENTWISE


def _evaluate(dist: Any, method: str, values: Sequence[float]) -> Any:
    """Apply ``method`` to every element, as one call."""
    return np.asarray(getattr(dist, method)(values), dtype=float)


def pdf(dist: Any, values: Sequence[float]) -> Any:
    """
    Density at every point in ``values``.

    Args:
        dist: A distribution instance.
        values: Points to evaluate at.

    Returns:
        An array of densities, in the order given.

    Examples:
        >>> from heavytails import Pareto
        >>> [round(float(v), 4) for v in pdf(Pareto(alpha=2.0, xm=1.0), [1.0, 2.0])]
        [2.0, 0.25]
    """
    return _evaluate(dist, "pdf", values)


def cdf(dist: Any, values: Sequence[float]) -> Any:
    """
    Distribution function at every point in ``values``.

    Args:
        dist: A distribution instance.
        values: Points to evaluate at.

    Returns:
        An array of probabilities, in the order given.

    Examples:
        >>> from heavytails import Pareto
        >>> [round(float(v), 4) for v in cdf(Pareto(alpha=2.0, xm=1.0), [2.0, 10.0])]
        [0.75, 0.99]
    """
    return _evaluate(dist, "cdf", values)


def sf(dist: Any, values: Sequence[float]) -> Any:
    """
    Survival function at every point in ``values``.

    Computed directly rather than as ``1 - cdf``, which is what keeps it
    accurate far into the tail. See the distribution methods themselves.

    Args:
        dist: A distribution instance.
        values: Points to evaluate at.

    Returns:
        An array of survival probabilities, in the order given.

    Examples:
        >>> from heavytails import Pareto
        >>> [round(float(v), 4) for v in sf(Pareto(alpha=2.0, xm=1.0), [2.0, 10.0])]
        [0.25, 0.01]
    """
    return _evaluate(dist, "sf", values)


def ppf(dist: Any, probabilities: Sequence[float]) -> Any:
    """
    Quantile at every probability in ``probabilities``.

    Args:
        dist: A distribution instance.
        probabilities: Probabilities in (0, 1).

    Returns:
        An array of quantiles, in the order given.

    Raises:
        ValueError: If any probability is outside (0, 1).

    Examples:
        >>> from heavytails import Pareto
        >>> [round(float(v), 4) for v in ppf(Pareto(alpha=2.0, xm=1.0), [0.5, 0.99])]
        [1.4142, 10.0]
    """
    # Checked here rather than left to the method, because this function has
    # promised this message and this wording since 0.4.0, and the methods
    # phrase their own refusal differently.
    array = np.asarray(probabilities, dtype=float)
    bad = array[(array <= 0.0) | (array >= 1.0) | np.isnan(array)]
    if bad.size:
        offending = bad.tolist()
        raise ValueError(
            f"every probability must be in (0,1); got {offending[0]!r}"
            + (f" and {len(offending) - 1} other(s)" if len(offending) > 1 else "")
        )
    return _evaluate(dist, "ppf", probabilities)
