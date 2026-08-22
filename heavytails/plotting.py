# heavytails/plotting.py
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def tail_loglog_plot(data: Sequence[float]) -> list[tuple[float, float]]:
    """
    Return (log x, log survival) pairs for tail visualization on log-log scale.
    (No plotting dependencies; returns data ready for plotting.)

    Examples:
        Returns coordinates rather than a figure, which is what keeps this
        module free of any third-party import. A power-law tail is a straight
        line here, with slope ``-alpha``.

        >>> from heavytails import Pareto
        >>> points = tail_loglog_plot(Pareto(alpha=2.0, xm=1.0).rvs(100, seed=1))
        >>> len(points)
        100
        >>> len(points[0])
        2

        See :func:`heavytails.viz.plot_tail` to draw it.
    """
    x = sorted(data)
    n = len(x)
    return [(math.log(x[i]), math.log((n - i) / n)) for i in range(n) if x[i] > 0]


def qq_pareto(data: Sequence[float]) -> list[tuple[float, float]]:
    """
    QQ plot points against Pareto quantiles.

    Examples:
        Linear if the sample is Pareto-tailed. One point fewer than the sample,
        since the largest observation has no plotting position above it.

        >>> from heavytails import Pareto
        >>> points = qq_pareto(Pareto(alpha=2.0, xm=1.0).rvs(100, seed=1))
        >>> len(points)
        99
    """
    x = sorted(data)
    n = len(x)
    return [(math.log(i / n), math.log(x[i - 1])) for i in range(1, n)]
