# heavytails/tail_index.py
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def hill_estimator(data: Sequence[float], k: int) -> float:
    """
    Hill estimator for Pareto-type tails.

    Parameters
    ----------
    data : sequence of floats
    k : int
        number of top order statistics (1 < k < n)
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    x_k = x[k]
    return 1.0 / (sum(math.log(x[i] / x_k) for i in range(k)) / k)


def pickands_estimator(data: Sequence[float], k: int, m: int = 2) -> float:
    """
    Pickands tail index estimator (extreme-value index gamma).

    gammâ = (1 / log(m)) * log( (X_k - X_{2k}) / (X_{mk} - X_{2mk}) )
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if 4 * k * m > n:
        raise ValueError("Sample too small for Pickands estimator.")
    Xk, X2k, Xmk, X2mk = x[k - 1], x[2 * k - 1], x[m * k - 1], x[2 * m * k - 1]
    return (1.0 / math.log(m)) * math.log((Xk - X2k) / (Xmk - X2mk))


def moment_estimator(data: Sequence[float], k: int) -> tuple[float, float]:
    """
    Dekkers-Einmahl-de Haan moment estimator for tail index.

    Returns (gamma_hat, alpha_hat) where alpha = 1/gamma.
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    x_k = x[k]
    logs = [math.log(x[i] / x_k) for i in range(k)]
    M1 = sum(logs) / k
    M2 = sum(log_val**2 for log_val in logs) / k
    gamma_hat = M1 + 1.0 - 0.5 * (1.0 - (M1**2) / M2) ** -1
    return gamma_hat, 1.0 / gamma_hat
