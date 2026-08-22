"""Scalar-or-array dispatch, so each formula is written once.

Every distribution method accepts a number or anything array-like and returns
the same kind of thing: a float in, a float out; a sequence in, an array out.
The formula in between is written against NumPy and does not branch on which
it was given.

That is the point of this module. The alternative -- a scalar implementation
and an array implementation of the same formula -- means two things to keep in
step, and they do not stay in step. The previous arrangement had exactly that
shape, with a separate kernel per method held to a tolerance against its scalar
twin, and the tolerance itself had to be rewritten twice before it was right.

NumPy is a hard dependency. It was optional until 0.5.0, and the pure-Python
promise was worth keeping while the library was small; it stopped being worth
it once evaluating a density over a hundred thousand points meant a hundred
thousand interpreter round trips.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["as_array", "check_probabilities", "elementwise", "restore"]


def as_array(x: float | Sequence[float] | Any) -> tuple[Any, bool]:
    """
    Return ``x`` as a float array, and whether it arrived as a scalar.

    Args:
        x: A number, or anything NumPy can make an array of.

    Returns:
        The array, and True when the input was a scalar and the result should
        be unwrapped again on the way out.

    Raises:
        TypeError: If ``x`` cannot be read as floating-point numbers, which is
            reported here rather than as a confusing failure inside a formula.

    Examples:
        >>> values, scalar = as_array(0.5)
        >>> float(values), scalar
        (0.5, True)
        >>> values, scalar = as_array([0.1, 0.9])
        >>> values.shape, scalar
        ((2,), False)
    """
    array = np.asarray(x, dtype=float)
    return array, array.ndim == 0


def restore(result: Any, scalar: bool) -> Any:
    """
    Unwrap a result back to a float when the input was a scalar.

    Args:
        result: The computed array.
        scalar: Whether the caller passed a scalar, from :func:`as_array`.

    Returns:
        A float, or the array unchanged.

    Examples:
        >>> import numpy as np
        >>> restore(np.float64(2.0), True)
        2.0
        >>> restore(np.array([1.0, 2.0]), False).tolist()
        [1.0, 2.0]
    """
    if scalar:
        return float(result)
    return np.asarray(result, dtype=float)


def check_probabilities(u: Any, name: str = "u") -> None:
    """
    Reject probabilities outside the open unit interval.

    Checks the whole input before any work starts, and names an offending
    value. A loop would have raised part way through, having already computed
    everything before it and reported only that *something* was wrong.

    Args:
        u: The array to check.
        name: What to call it in the message.

    Raises:
        ValueError: If any element is outside (0, 1), or is not a number.

    Examples:
        >>> import numpy as np
        >>> check_probabilities(np.array([0.1, 0.9]))
        >>> check_probabilities(np.array([0.5, 1.0]))
        Traceback (most recent call last):
            ...
        ValueError: u must be in (0,1); got 1.0
    """
    array = np.asarray(u, dtype=float)
    bad = array[~((array > 0.0) & (array < 1.0))]
    if bad.size:
        offending = float(bad.reshape(-1)[0])
        extra = f" and {bad.size - 1} other(s)" if bad.size > 1 else ""
        raise ValueError(f"{name} must be in (0,1); got {offending}{extra}")


def elementwise(function: Any, array: Any) -> Any:
    """
    Apply a scalar function across an array, preserving its shape.

    The escape hatch for the four families whose probabilities need the error
    function or an incomplete beta or gamma. NumPy has none of those, so there
    is no expression to write and the loop is the honest implementation rather
    than a placeholder.

    It still returns an array, so those families keep the same interface as
    every other one. What they do not get is the speed: ``LogNormal.cdf``,
    ``StudentT.cdf``, ``InverseGamma.cdf`` and ``BetaPrime.cdf`` cost what they
    always did. Their **densities** are elementary and are vectorised
    normally.

    Args:
        function: A scalar callable.
        array: The values to apply it to.

    Returns:
        An array of the same shape.

    Examples:
        >>> import numpy as np
        >>> elementwise(lambda v: v * 2, np.array([1.0, 2.0])).tolist()
        [2.0, 4.0]
    """
    flat = np.asarray(array, dtype=float).reshape(-1)
    return np.array([function(float(value)) for value in flat], dtype=float).reshape(
        np.shape(array)
    )
