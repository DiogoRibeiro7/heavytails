# heavytails/tail_index.py
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from heavytails._special import _phi_inverse
from heavytails.heavy_tails import RNG

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def hill_estimator(data: Sequence[float], k: int) -> float:
    """
    Hill estimator for the tail index gamma (where gamma = 1/alpha for Pareto).

    Parameters
    ----------
    data : sequence of floats
    k : int
        number of top order statistics (1 < k < n)

    Returns
    -------
    gamma : float
        The tail index estimate (gamma = 1/alpha for Pareto distributions)
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    x_k = x[k]
    # Hill estimator returns gamma (not alpha)
    return sum(math.log(x[i] / x_k) for i in range(k)) / k


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


def generalized_hill_estimator(data: Sequence[float], k: int) -> float:
    """
    Generalized Hill (UH) estimator of the extreme-value index.

    The Hill estimator is only valid for gamma > 0. The generalized Hill
    estimator applies the same log-excess averaging to the UH statistics
    ``UH_j = X_(j+1) * H_j``, where ``H_j`` is the Hill estimator on the top j
    order statistics, and is consistent for every gamma in the reals:

        gamma_hat = (1/k) * sum_{j=1}^{k} log(UH_j) - log(UH_{k+1})

    Use it when you are not certain the tail is heavy. Where Hill applies, the
    two agree closely and Hill is slightly more efficient.

    Parameters
    ----------
    data : sequence of floats
        Sample values. All must be positive.
    k : int
        Number of top order statistics to use, with ``1 < k < n - 1``. One more
        order statistic is needed than for the Hill estimator, because the
        reference term is ``UH_{k+1}``.

    Returns
    -------
    float
        The extreme-value index estimate gamma.

    Raises
    ------
    ValueError
        If k is out of range, the data is not positive, or the UH statistics
        are not positive (which happens only for degenerate samples).

    References
    ----------
    Beirlant, J., Vynckier, P., & Teugels, J. L. (1996). Excess functions and
    estimation of the extreme-value index. *Bernoulli*, 2(4), 293-318.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
    >>> round(generalized_hill_estimator(data, k=250), 1)
    0.5
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n - 1):
        raise ValueError(
            "k must satisfy 1 < k < n-1 for the generalized Hill estimator"
        )
    if x[-1] <= 0.0:
        raise ValueError("generalized Hill requires strictly positive data")

    # Prefix sums make the UH statistics O(n) instead of O(k^2):
    # H_j = (1/j) * sum_{i<j} log(x_i / x_j) = (1/j) * sum_{i<j} log(x_i) - log(x_j)
    log_x = [math.log(v) for v in x]
    prefix = [0.0] * (k + 2)
    for j in range(1, k + 2):
        prefix[j] = prefix[j - 1] + log_x[j - 1]

    log_uh = []
    for j in range(1, k + 2):
        hill_j = prefix[j] / j - log_x[j]
        if hill_j <= 0.0:
            raise ValueError(
                f"UH statistic is not positive at j={j}; the sample is degenerate "
                "in its upper tail"
            )
        log_uh.append(math.log(x[j]) + math.log(hill_j))

    return sum(log_uh[:k]) / k - log_uh[k]


def hill_plot(
    data: Sequence[float], ks: Sequence[int] | None = None
) -> list[tuple[int, float]]:
    """
    Hill estimates across a range of k, for the Hill plot.

    Every tail index estimator depends on how many upper order statistics it
    uses, and there is no universally correct choice: small k means low bias
    and high variance, large k the reverse. The standard practice is to plot
    the estimate against k and read it off a stable plateau. If there is no
    plateau, the data does not support a tail index estimate and forcing one
    produces a confident wrong answer.

    Parameters
    ----------
    data : sequence of floats
        Sample values.
    ks : sequence of ints, optional
        Values of k to evaluate. Defaults to a logarithmically spaced sweep
        from 5 to ``n // 2``, which keeps the plot readable for large samples.

    Returns
    -------
    list of (int, float)
        ``(k, gamma_hat)`` pairs, ordered by k.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
    >>> points = hill_plot(data)
    >>> all(k > 0 for k, _ in points)
    True
    """
    n = len(data)
    if n < 8:
        raise ValueError("need at least 8 observations for a Hill plot")

    if ks is None:
        upper = max(6, n // 2)
        # Logarithmic spacing: the interesting structure is at small k.
        count = min(60, upper - 4)
        ks = sorted(
            {round(5 * (upper / 5) ** (i / max(count - 1, 1))) for i in range(count)}
        )

    x = sorted(data, reverse=True)
    points = []
    for k in ks:
        if not (1 < k < n):
            continue
        x_k = x[k]
        if x_k <= 0.0:
            continue
        points.append((k, sum(math.log(x[i] / x_k) for i in range(k)) / k))
    return points


#: Estimators available to :func:`tail_index_confidence_interval`.
_POINT_ESTIMATORS: dict[str, Callable[[Sequence[float], int], float]] = {
    "hill": hill_estimator,
    "generalized_hill": generalized_hill_estimator,
    "moment": lambda d, k: moment_estimator(d, k)[0],
    "pickands": pickands_estimator,
}


def tail_index_confidence_interval(
    data: Sequence[float],
    k: int,
    *,
    estimator: str = "hill",
    level: float = 0.95,
    method: str = "asymptotic",
    n_bootstrap: int = 500,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Point estimate and confidence interval for the extreme-value index.

    A tail index without an interval is not usable: the estimate depends on a
    choice of k that no rule fixes, and the sampling variability at realistic
    sample sizes is large.

    Two methods are offered:

    ``asymptotic``
        Only available for the Hill estimator, whose limiting distribution is
        ``sqrt(k) (gamma_hat - gamma) -> N(0, gamma^2)``, giving the interval
        ``gamma_hat * (1 +/- z / sqrt(k))``. It assumes k is in the range where
        the estimator is unbiased, so it understates the true uncertainty when
        the threshold is chosen badly.

    ``bootstrap``
        Resamples the data with replacement and takes percentiles of the
        resulting estimates. Available for every estimator and free of
        distributional assumptions, at the cost of ``n_bootstrap`` refits. Note
        that resampling does not capture the bias from the choice of k, only
        the variance.

    Parameters
    ----------
    data : sequence of floats
        Sample values.
    k : int
        Number of top order statistics.
    estimator : str, optional
        One of ``hill``, ``generalized_hill``, ``moment`` or ``pickands``.
    level : float, optional
        Confidence level in (0, 1).
    method : str, optional
        ``asymptotic`` or ``bootstrap``.
    n_bootstrap : int, optional
        Number of bootstrap resamples.
    seed : int, optional
        Seed for reproducible bootstrap resampling.

    Returns
    -------
    dict
        ``gamma``, ``alpha``, ``lower``, ``upper``, ``level``, ``method``,
        ``estimator`` and ``k``. ``alpha`` is ``1/gamma``, or None when the
        estimate is not positive and the reciprocal is meaningless.

    Raises
    ------
    ValueError
        For an unknown estimator or method, or an out-of-range level.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(2000, seed=7)
    >>> result = tail_index_confidence_interval(data, k=200)
    >>> result["lower"] < result["gamma"] < result["upper"]
    True
    """
    if estimator not in _POINT_ESTIMATORS:
        raise ValueError(
            f"Unknown estimator {estimator!r}. "
            f"Available: {', '.join(sorted(_POINT_ESTIMATORS))}"
        )
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")

    point = _POINT_ESTIMATORS[estimator](data, k)

    if method == "asymptotic":
        if estimator != "hill":
            raise ValueError(
                "The asymptotic interval is only established for the Hill "
                f"estimator; use method='bootstrap' for {estimator!r}."
            )
        # sqrt(k) (gamma_hat - gamma) -> N(0, gamma^2)
        z = _phi_inverse(0.5 + level / 2.0)
        half_width = z * point / math.sqrt(k)
        lower, upper = point - half_width, point + half_width
    elif method == "bootstrap":
        lower, upper = _bootstrap_interval(
            data, k, _POINT_ESTIMATORS[estimator], level, n_bootstrap, seed
        )
    else:
        raise ValueError(f"Unknown method {method!r}. Available: asymptotic, bootstrap")

    return {
        "estimator": estimator,
        "k": k,
        "gamma": point,
        "alpha": 1.0 / point if point > 0.0 else None,
        "lower": lower,
        "upper": upper,
        "level": level,
        "method": method,
    }


def _bootstrap_interval(
    data: Sequence[float],
    k: int,
    estimate: Callable[[Sequence[float], int], float],
    level: float,
    n_bootstrap: int,
    seed: int | None,
) -> tuple[float, float]:
    """Percentile bootstrap interval for a tail index estimator.

    Resamples are drawn through the library's RNG wrapper so the interval is
    reproducible from a seed, like every other random operation here.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2.")

    rng = RNG(seed)
    n = len(data)
    values = []
    for _ in range(n_bootstrap):
        resample = [data[min(int(rng.uniform_0_1() * n), n - 1)] for _ in range(n)]
        try:
            values.append(estimate(resample, k))
        except (ValueError, ZeroDivisionError):
            # A resample can be degenerate in its upper tail; skip it rather
            # than abandoning the interval.
            continue

    if len(values) < 2:
        raise ValueError(
            "Bootstrap failed: too few resamples produced a usable estimate."
        )

    values.sort()
    tail = (1.0 - level) / 2.0
    last = len(values) - 1
    lo_index = max(0, min(last, round(tail * last)))
    hi_index = max(0, min(last, round((1.0 - tail) * last)))
    return values[lo_index], values[hi_index]
