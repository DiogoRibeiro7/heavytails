"""Tail index estimation over a stream, without holding the sample.

The Hill estimator needs the top ``k`` order statistics. That sounds like it
rules out a streaming version, since order statistics are a global property of
the sample -- but only the top ``k + 1`` values ever matter, and which values
those are can be maintained incrementally. A min-heap of size ``k + 1`` keeps
them in ``O(k)`` memory and ``O(log k)`` time per observation, whatever the
length of the stream.

That gives an exact result, not an approximation. :class:`StreamingTailIndex`
holds the same numbers a batch estimator would sort out of the full sample, so
it returns **the same estimate to the last bit** -- which the test suite asserts
rather than assumes. Nothing is traded away except the ability to ask a
different ``k`` later.

Two semantics, and the difference matters:

:class:`StreamingTailIndex`
    The whole stream, in ``O(k)`` memory. Every observation ever seen
    contributes, so a change in the tail is diluted by everything before it.

:class:`WindowedTailIndex`
    The most recent ``window`` observations, in ``O(window)`` memory. Responds
    to a change in the tail, at a cost in memory and variance.

The memory difference is not an implementation shortcoming to be fixed later.
When the largest value in a window expires, the new maximum can be any of the
remaining ones, so a window cannot be summarised more compactly than by keeping
it. Anything claiming to track windowed order statistics in ``O(k)`` is either
approximating or wrong.
"""

from __future__ import annotations

import bisect
from collections import deque
import heapq
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "StreamingTailIndex",
    "TopK",
    "WindowedTailIndex",
]


class TopK:
    """
    The ``k`` largest values of a stream, in ``O(k)`` memory.

    A min-heap holds the retained values, so its root is the smallest of them
    and is what a new observation has to beat. That is the whole trick: the
    comparison that decides whether to keep a value is against the *smallest
    kept*, not against anything in the discarded majority.

    Args:
        k: How many values to retain, at least one.

    Raises:
        ValueError: If ``k`` is not a positive integer.

    Examples:
        >>> top = TopK(3)
        >>> top.extend([5.0, 1.0, 9.0, 3.0, 7.0])
        >>> top.descending()
        [9.0, 7.0, 5.0]
        >>> top.n_seen, len(top)
        (5, 3)
    """

    __slots__ = ("_heap", "_k", "_seen")

    def __init__(self, k: int) -> None:
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        self._k = k
        self._heap: list[float] = []
        self._seen = 0

    def push(self, value: float) -> None:
        """Offer one observation."""
        self._seen += 1
        item = float(value)
        if len(self._heap) < self._k:
            heapq.heappush(self._heap, item)
        elif item > self._heap[0]:
            heapq.heapreplace(self._heap, item)

    def extend(self, values: Iterable[float]) -> None:
        """Offer many observations."""
        for value in values:
            self.push(value)

    def descending(self) -> list[float]:
        """The retained values, largest first."""
        return sorted(self._heap, reverse=True)

    @property
    def capacity(self) -> int:
        """How many values are retained once the stream is long enough."""
        return self._k

    @property
    def n_seen(self) -> int:
        """How many observations have been offered."""
        return self._seen

    def __len__(self) -> int:
        """How many values are currently retained."""
        return len(self._heap)


def _hill(descending: list[float], k: int) -> float:
    """Hill estimator from the top ``k + 1`` order statistics.

    Written to match :func:`heavytails.tail_index.hill_estimator` operation for
    operation, so the streaming and batch results agree exactly rather than
    approximately. A different summation order would be just as correct and
    would make the equality a matter of tolerance instead of identity.
    """
    x_k = descending[k]
    return sum(math.log(descending[i] / x_k) for i in range(k)) / k


def _moment(descending: list[float], k: int) -> tuple[float, float]:
    """Dekkers-Einmahl-de Haan moment estimator, matching the batch version."""
    x_k = descending[k]
    logs = [math.log(descending[i] / x_k) for i in range(k)]
    m1 = sum(logs) / k
    m2 = sum(value**2 for value in logs) / k
    gamma = m1 + 1.0 - 0.5 * (1.0 - (m1**2) / m2) ** -1
    return gamma, 1.0 / gamma


class StreamingTailIndex:
    """
    Tail index estimators over an unbounded stream, in ``O(k)`` memory.

    Holds the top ``k + 1`` observations and nothing else. The estimate is
    exact: it is the same number a batch estimator would produce from the whole
    sample, because the estimators depend on the sample only through those
    values.

    Args:
        k: Number of top order statistics the estimators use.

    Raises:
        ValueError: If ``k`` is less than two, which no estimator here accepts.

    Examples:
        >>> from heavytails import Pareto
        >>> from heavytails.tail_index import hill_estimator
        >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        >>> stream = StreamingTailIndex(k=500)
        >>> stream.extend(data)
        >>> stream.hill() == hill_estimator(data, k=500)
        True
    """

    def __init__(self, k: int) -> None:
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer of at least 2.")
        self._k = k
        self._top = TopK(k + 1)

    def update(self, value: float) -> None:
        """Add one observation."""
        self._top.push(value)

    def extend(self, values: Iterable[float]) -> None:
        """Add many observations."""
        self._top.extend(values)

    def hill(self) -> float:
        """
        The Hill estimate of the extreme-value index.

        Returns:
            ``gamma``, equal to ``1 / alpha`` for a Pareto tail.

        Raises:
            ValueError: If fewer than ``k + 1`` observations have been seen, or
                if the retained values are not positive.
        """
        return _hill(self._ready(), self._k)

    def moment(self) -> tuple[float, float]:
        """
        The Dekkers-Einmahl-de Haan moment estimate.

        Unlike the Hill estimator this does not assume the tail is heavy, so it
        is the one to reach for when the sign of ``gamma`` is in doubt.

        Returns:
            ``(gamma, alpha)``.

        Raises:
            ValueError: If there is not yet enough data, or it is not positive.
        """
        return _moment(self._ready(), self._k)

    @property
    def threshold(self) -> float:
        """The order statistic the estimators measure exceedances above."""
        return self._ready()[self._k]

    @property
    def k(self) -> int:
        """Number of top order statistics used."""
        return self._k

    @property
    def n_seen(self) -> int:
        """How many observations have passed through."""
        return self._top.n_seen

    @property
    def ready(self) -> bool:
        """Whether enough observations have been seen to estimate anything."""
        return len(self._top) > self._k

    def _ready(self) -> list[float]:
        """The retained values, having checked they can support an estimate."""
        if not self.ready:
            raise ValueError(
                f"need at least {self._k + 1} observations for k={self._k}, "
                f"seen {self.n_seen}"
            )
        values = self._top.descending()
        if values[self._k] <= 0.0:
            raise ValueError("the tail index estimators require positive data")
        return values


class WindowedTailIndex:
    """
    The same estimators over the most recent ``window`` observations.

    The whole-stream version dilutes a change in the tail with everything that
    came before, which is the wrong behaviour for monitoring: a portfolio whose
    tail index has moved from 3 to 1.5 does not want an estimate averaging the
    two. This forgets.

    **Memory is ``O(window)``, not ``O(k)``**, and that is inherent rather than
    a gap to close later. When the largest value in the window expires, the new
    largest can be any of the survivors, so nothing smaller than the window
    itself determines the answer.

    Args:
        window: How many recent observations to keep.
        k: Number of top order statistics the estimators use, below ``window``.

    Raises:
        ValueError: If ``window`` is not a positive integer, or ``k`` is not at
            least two and strictly below ``window``.

    Examples:
        >>> from heavytails import Pareto
        >>> monitor = WindowedTailIndex(window=5000, k=400)
        >>> monitor.extend(Pareto(alpha=3.0, xm=1.0).rvs(5000, seed=1))
        >>> round(monitor.hill(), 3)  # true gamma 0.333
        0.33
        >>> monitor.extend(Pareto(alpha=1.5, xm=1.0).rvs(5000, seed=2))
        >>> round(monitor.hill(), 3)  # true gamma 0.667, the old regime gone
        0.667
    """

    def __init__(self, window: int, k: int) -> None:
        if not isinstance(window, int) or window < 2:
            raise ValueError("window must be an integer of at least 2.")
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an integer of at least 2.")
        if k >= window:
            raise ValueError(f"k must be below window; got k={k}, window={window}")
        self._window = window
        self._k = k
        self._recent: deque[float] = deque()
        # The same values kept in ascending order, so the top k + 1 are a slice
        # rather than a sort on every query. Insertion and removal are linear
        # in the window, which for the sizes this is meant for is a memmove.
        self._ordered: list[float] = []
        self._seen = 0

    def update(self, value: float) -> None:
        """Add one observation, evicting the oldest if the window is full."""
        item = float(value)
        self._seen += 1
        self._recent.append(item)
        bisect.insort(self._ordered, item)
        if len(self._recent) > self._window:
            oldest = self._recent.popleft()
            index = bisect.bisect_left(self._ordered, oldest)
            del self._ordered[index]

    def extend(self, values: Iterable[float]) -> None:
        """Add many observations."""
        for value in values:
            self.update(value)

    def hill(self) -> float:
        """The Hill estimate over the current window."""
        return _hill(self._ready(), self._k)

    def moment(self) -> tuple[float, float]:
        """The moment estimate over the current window."""
        return _moment(self._ready(), self._k)

    @property
    def threshold(self) -> float:
        """The order statistic the estimators measure exceedances above."""
        return self._ready()[self._k]

    @property
    def window(self) -> int:
        """How many recent observations are kept."""
        return self._window

    @property
    def k(self) -> int:
        """Number of top order statistics used."""
        return self._k

    @property
    def n_seen(self) -> int:
        """How many observations have passed through, including evicted ones."""
        return self._seen

    @property
    def ready(self) -> bool:
        """Whether the window holds enough observations to estimate anything."""
        return len(self._recent) > self._k

    def values(self) -> list[float]:
        """The current window contents, oldest first."""
        return list(self._recent)

    def _ready(self) -> list[float]:
        """The window's top ``k + 1``, having checked they support an estimate."""
        if not self.ready:
            raise ValueError(
                f"need at least {self._k + 1} observations in the window for "
                f"k={self._k}, have {len(self._recent)}"
            )
        top = self._ordered[-(self._k + 1) :][::-1]
        if top[self._k] <= 0.0:
            raise ValueError("the tail index estimators require positive data")
        return top
