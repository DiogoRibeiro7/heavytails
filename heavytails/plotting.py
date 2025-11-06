# heavytails/plotting.py
from __future__ import annotations

from collections.abc import Sequence
import math


def tail_loglog_plot(data: Sequence[float]) -> list[tuple[float,float]]:
    """
    Return (log x, log survival) pairs for tail visualization on log–log scale.
    (No plotting dependencies; returns data ready for plotting.)
    """
    x = sorted(data)
    n = len(x)
    return [(math.log(x[i]), math.log((n - i) / n)) for i in range(n) if x[i] > 0]

def qq_pareto(data: Sequence[float]) -> list[tuple[float,float]]:
    """
    QQ plot points against Pareto quantiles.
    """
    x = sorted(data)
    n = len(x)
    return [(math.log(i/n), math.log(x[i-1])) for i in range(1, n)]
