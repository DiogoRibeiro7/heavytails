"""Estimators of the extreme-value index.

Two conventions are in circulation and they are reciprocals of each other:

* the **tail index** ``alpha``, from the regular variation form
  ``P(X > x) ~ L(x) * x**-alpha``, common in economics and network science;
* the **extreme-value index** ``xi``, also written ``gamma``, the convention of
  the extreme value theory literature, with ``gamma = 1 / alpha``.

**Every estimator in this module returns gamma, not alpha.** Larger gamma means
a heavier tail. Invert it to recover alpha:

    gamma = hill_estimator(data, k=100)
    alpha = 1 / gamma

The module name refers to the quantity being estimated, not to the
parameterisation of the return value. :func:`moment_estimator` is the one
exception in shape, returning the pair ``(gamma, alpha)`` for convenience, and
:func:`tail_index_confidence_interval` reports both.
"""

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


def smoothed_hill_estimator(data: Sequence[float], k: int, u: float = 2.0) -> float:
    """
    Resnick-Starica smoothed Hill estimator of the extreme-value index.

    The ordinary Hill estimate varies substantially with k, which is the
    practical difficulty the Hill plot exists to work around. This averages it
    over a range of k instead:

        gamma_hat(k, u) = 1/((u-1)k) * sum_{j=k+1}^{floor(u*k)} hill(j)

    The asymptotic variance falls from ``gamma**2`` to
    ``gamma**2 * 2*(u - 1 - log(u)) / (u - 1)**2``: about 0.61 times at
    ``u = 2`` and 0.45 times at ``u = 3``. The cost is bias, because a larger
    ``u`` averages over a wider range of k and so reaches further into the body
    of the distribution. Values between 2 and 3 are the usual compromise.

    Like the Hill estimator it assumes ``gamma > 0``; see
    :func:`generalized_hill_estimator` if the sign is in doubt.

    Parameters
    ----------
    data : sequence of floats
        Sample values. All must be positive.
    k : int
        Lower end of the averaging range.
    u : float, optional
        Smoothing parameter, strictly greater than 1. The average runs over
        ``j`` in ``(k, u*k]``, so ``u*k`` must be less than the sample size.

    Returns
    -------
    float
        The extreme-value index estimate gamma.

    Raises
    ------
    ValueError
        If ``u <= 1``, the averaging range is empty, or it runs past the end of
        the sample.

    References
    ----------
    Resnick, S., & Starica, C. (1997). Smoothing the Hill estimator.
    *Advances in Applied Probability*, 29(1), 271-293.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=11)
    >>> round(smoothed_hill_estimator(data, k=1000, u=2.0), 1)
    0.5
    """
    if u <= 1.0:
        raise ValueError("u must be strictly greater than 1.")

    x = sorted(data, reverse=True)
    n = len(x)
    upper = int(u * k)
    if k <= 1:
        raise ValueError("k must be greater than 1")
    if upper <= k:
        raise ValueError(
            f"the averaging range (k, u*k] is empty for k={k}, u={u}; increase u or k"
        )
    if upper >= n:
        raise ValueError(
            f"u*k = {upper} must be less than the sample size {n}; reduce k or u"
        )
    if x[-1] <= 0.0:
        raise ValueError("the smoothed Hill estimator requires positive data")

    # Prefix sums give every hill(j) in one pass:
    # hill(j) = (1/j) * sum_{i<j} log(x_i) - log(x_j)
    log_x = [math.log(v) for v in x]
    prefix = [0.0] * (upper + 1)
    for j in range(1, upper + 1):
        prefix[j] = prefix[j - 1] + log_x[j - 1]

    total = sum(prefix[j] / j - log_x[j] for j in range(k + 1, upper + 1))
    return total / (upper - k)


def smoothed_hill_variance_ratio(u: float) -> float:
    """
    Asymptotic variance of the smoothed Hill estimator relative to Hill.

    Returns ``2*(u - 1 - log(u)) / (u - 1)**2``, the factor by which
    :func:`smoothed_hill_estimator` reduces the asymptotic variance of the
    ordinary Hill estimator. It tends to 1 as ``u`` tends to 1 from above, and
    decreases as ``u`` grows.

    This describes variance only. Larger ``u`` also increases bias, so it is
    not a quantity to minimise blindly.

    Parameters
    ----------
    u : float
        Smoothing parameter, strictly greater than 1.

    Returns
    -------
    float
        The variance ratio, between 0 and 1.

    Examples
    --------
    >>> round(smoothed_hill_variance_ratio(2.0), 4)
    0.6137
    >>> round(smoothed_hill_variance_ratio(3.0), 4)
    0.4507
    """
    if u <= 1.0:
        raise ValueError("u must be strictly greater than 1.")
    return 2.0 * (u - 1.0 - math.log(u)) / (u - 1.0) ** 2


def _normalised_log_spacings(x_desc: list[float], k: int) -> list[float]:
    """Return ``Y_i = i * (log X_(i) - log X_(i+1))`` for ``i = 1..k``.

    Under an exact Pareto tail these are independent and exponentially
    distributed with mean gamma, by the Renyi representation of exponential
    order statistics. That is what makes the Hill estimator the sample mean of
    them, and what makes trimming the leading ones a well-behaved operation:
    the remainder is still an iid exponential sample.

    Args:
        x_desc: Sample sorted in decreasing order.
        k: Number of spacings to return.

    Returns:
        The k normalised log-spacings.
    """
    return [(i + 1) * (math.log(x_desc[i]) - math.log(x_desc[i + 1])) for i in range(k)]


def trimmed_hill_estimator(data: Sequence[float], k: int, r: int = 0) -> float:
    """
    Trimmed Hill estimator: Hill with the r largest observations discarded.

    The Hill estimator gives enormous leverage to the largest order statistics,
    which enter through unbounded logarithms of ratios, so a handful of
    contaminated observations is enough to destroy it. Replacing the three
    largest of ten thousand Pareto(2) draws with outliers moves the ordinary
    Hill estimate from 0.50 to 0.66; trimming five recovers 0.50.

    On clean data the cost is small. For the same sample, the standard
    deviation rises only from 0.0296 at ``r = 0`` to 0.0302 at ``r = 10``.

    ``r`` must exceed the number of contaminated observations. Trimming two
    when three are contaminated leaves the estimate essentially as bad as
    trimming none, because the third still enters through a spacing.

    Like the Hill estimator this assumes ``gamma > 0``; see
    :func:`generalized_hill_estimator` if the sign is in doubt.

    Parameters
    ----------
    data : sequence of floats
        Sample values. All must be positive.
    k : int
        Number of top order statistics to use, with ``1 < k < n``.
    r : int, optional
        Number of largest observations to discard, with ``0 <= r < k``.
        ``r = 0`` reproduces the ordinary Hill estimator exactly.

    Returns
    -------
    float
        The extreme-value index estimate gamma, equal to ``1 / alpha``.

    Raises
    ------
    ValueError
        If k or r is out of range, or the data is not positive.

    References
    ----------
    Bhattacharya, S., Kallitsis, M., & Stoev, S. (2019). Trimming the Hill
    estimator: robustness, optimality and adaptivity. arXiv:1705.03088.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = sorted(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1), reverse=True)
    >>> data[0] = 1e9  # one contaminated observation
    >>> round(trimmed_hill_estimator(data, k=300, r=5), 1)
    0.5
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    if not (0 <= r < k):
        raise ValueError(f"r must satisfy 0 <= r < k; got r={r}, k={k}")
    if x[k] <= 0.0:
        raise ValueError("the trimmed Hill estimator requires positive data")

    spacings = _normalised_log_spacings(x, k)
    return sum(spacings[r:]) / (k - r)


def trimmed_hill_plot(
    data: Sequence[float], k: int, max_trim: int | None = None
) -> list[tuple[int, float]]:
    """
    Trimmed Hill estimates across a range of r, for choosing the trimming level.

    Read it the way you read a Hill plot. The estimate typically moves sharply
    while r is below the number of contaminated observations and then flattens
    once they have all been discarded, so the elbow indicates how much
    contamination is present. A plot that is flat from ``r = 0`` suggests there
    is none.

    Parameters
    ----------
    data : sequence of floats
        Sample values.
    k : int
        Number of top order statistics to use.
    max_trim : int, optional
        Largest r to evaluate. Defaults to ``k // 10``, which is enough to
        reveal an elbow without spending the whole sample on trimming.

    Returns
    -------
    list of (int, float)
        ``(r, gamma_hat)`` pairs, ordered by r.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=1)
    >>> points = trimmed_hill_plot(data, k=250)
    >>> first_r, first_gamma = points[0]
    >>> first_r
    0
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    if max_trim is None:
        max_trim = max(1, k // 10)
    if not (0 < max_trim < k):
        raise ValueError(f"max_trim must satisfy 0 < max_trim < k; got {max_trim}")
    if x[k] <= 0.0:
        raise ValueError("the trimmed Hill estimator requires positive data")

    spacings = _normalised_log_spacings(x, k)
    total = sum(spacings)
    points = []
    for r in range(max_trim + 1):
        points.append((r, total / (k - r)))
        if r < max_trim:
            total -= spacings[r]
    return points


#: Estimators available to :func:`tail_index_confidence_interval`.
_POINT_ESTIMATORS: dict[str, Callable[..., float]] = {
    "hill": hill_estimator,
    "generalized_hill": generalized_hill_estimator,
    "smoothed_hill": smoothed_hill_estimator,
    "trimmed_hill": trimmed_hill_estimator,
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
    estimator_kwargs: dict[str, Any] | None = None,
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
    estimator_kwargs : dict, optional
        Extra keyword arguments for the estimator, such as ``{"r": 5}`` for the
        trimmed Hill estimator or ``{"u": 3.0}`` for the smoothed one. Without
        this the estimators run at their defaults, which for ``trimmed_hill``
        means no trimming at all and therefore no robustness.

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

    kwargs = dict(estimator_kwargs or {})

    def estimate(sample: Sequence[float], top: int) -> float:
        return _POINT_ESTIMATORS[estimator](sample, top, **kwargs)

    point = estimate(data, k)

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
        lower, upper = _bootstrap_interval(data, k, estimate, level, n_bootstrap, seed)
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
        "estimator_kwargs": kwargs,
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
