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
        Sample values. The top ``k + 1`` are read, and those must be
        positive; values below the threshold are never touched.
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
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
    >>> round(generalized_hill_estimator(data, k=2000), 1)
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
        Sample values. The top ``k + 1`` are read, and those must be
        positive; values below the threshold are never touched.
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
        Sample values. The top ``k + 1`` are read, and those must be
        positive; values below the threshold are never touched.
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


# Family-wise false-alarm rate for the deep-scan interlock in
# :func:`adaptive_trim_selection`. See there for why it is not `level`.
_INTERLOCK_LEVEL = 1e-4


def _spacing_p_value(spacings: list[float], j: int) -> float:
    """Exact probability that spacing ``j`` is as large as it is, by chance.

    Under a Pareto tail the normalised log-spacings are iid exponential, so
    ``Y_j`` is independent of the deeper ones and ``R = Y_j / mean(Y_{j+1..k})``
    has an exactly computable null distribution. Writing ``S`` for the sum of
    the ``m`` deeper spacings, ``Y_j / (Y_j + S)`` is ``Beta(1, m)``, so::

        P(R > t) = (m / (m + t))^m

    No asymptotics and no tabulated critical values: the p-value is uniform on
    (0, 1) under the null for any ``m``, which the test suite checks directly.
    """
    deeper = spacings[j + 1 :]
    m = len(deeper)
    mean = sum(deeper) / m
    if mean <= 0.0:  # pragma: no cover - needs k+1 identical order statistics
        return 0.0
    ratio = spacings[j] / mean
    return float((m / (m + ratio)) ** m)


def adaptive_trim_selection(
    data: Sequence[float],
    k: int,
    max_trim: int | None = None,
    level: float = 0.05,
) -> dict[str, Any]:
    """
    Choose the trimming parameter for the trimmed Hill estimator from the data.

    :func:`trimmed_hill_estimator` needs ``r`` to exceed the number of
    contaminated observations, and in practice nobody knows that number. This
    finds it.

    The normalised log-spacings are iid exponential under a Pareto tail, and
    contamination among the largest observations inflates one of them. The rule
    is to trim past the **deepest** anomalous spacing, not the first:

    Several outliers of similar size sit close together, so the gaps *between*
    them are small and only the gap below the last one is large. Stopping at
    the first ordinary-looking spacing would therefore find nothing at all when
    there is more than one outlier, and report the sample as clean.

    Every spacing from ``max_trim`` down to the first is tested at
    ``level / max_trim``, a Bonferroni correction for the scan, so on clean data
    the estimator over-trims with probability close to ``level`` -- measured at
    0.009, 0.052 and 0.094 for nominal 0.01, 0.05 and 0.10.

    Detection is not certain, and how likely it is depends on how extreme the
    outliers are. With three of them among 10,000 Pareto(2) draws and
    ``k = 300``, the trimming is found in 100% of samples at five times the true
    sample maximum, 95% at three times, 64% at twice and 46% at one and a half
    times. An outlier only half again the size of the largest genuine
    observation is not reliably distinguishable from the tail itself, and no
    procedure could make it so.

    Parameters
    ----------
    data : sequence of floats
        Sample values. The top ``k + 1`` are read and must be positive.
    k : int
        Number of top order statistics to use, with ``1 < k < n``.
    max_trim : int, optional
        Largest trimming considered. Defaults to ``k // 4``, which is generous:
        the scan is cheap and a limit that is too low is the one way this
        procedure fails badly.
    level : float, optional
        Family-wise probability of over-trimming clean data.

    Returns
    -------
    dict
        ``trim``: the chosen ``r``. ``gamma``: the trimmed Hill estimate at that
        ``r``. ``p_values``: the test for each spacing from 0 to ``max_trim-1``.
        ``saturated``: whether an anomaly was found *below* the scanned range,
        meaning ``max_trim`` is too small and ``trim`` is not to be trusted.
        ``deepest_anomaly``: where that anomaly was, or None.

    Raises
    ------
    ValueError
        If k, max_trim or level is out of range, or the data is not positive.

    References
    ----------
    Bhattacharya, S., Kallitsis, M., & Stoev, S. (2019). Trimming the Hill
    estimator: robustness, optimality and adaptivity. arXiv:1705.03088. The
    trimmed estimator is theirs; the selection rule here is a sequential exact
    test on the log-spacings rather than their procedure.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = sorted(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1), reverse=True)
    >>> for j in range(3):
    ...     data[j] = 1e6 * (j + 1)
    >>> result = adaptive_trim_selection(sorted(data, reverse=True), k=300)
    >>> result["trim"]
    3
    >>> round(result["gamma"], 3)
    0.479
    >>> clean = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1)
    >>> round(hill_estimator(clean, k=300), 3)  # what the outliers destroyed
    0.479
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    if max_trim is None:
        max_trim = max(1, k // 4)
    if not (0 < max_trim < k):
        raise ValueError(f"max_trim must satisfy 0 < max_trim < k; got {max_trim}")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    if x[k] <= 0.0:
        raise ValueError("the trimmed Hill estimator requires positive data")

    spacings = _normalised_log_spacings(x, k)

    # The trimming scan is a test at `level`, Bonferroni-corrected across the
    # spacings it examines.
    #
    # The deeper scan is not a test at `level` but a safety interlock, and is
    # set far stricter. A false alarm there turns a perfectly good estimate
    # into an exception, while contamination severe enough to matter produces
    # p-values around exp(-200) -- so the interlock loses nothing by demanding
    # overwhelming evidence, and at `level` it fired on one clean sample in
    # fifty.
    detect_to = max(max_trim, k // 2)
    deep_tests = detect_to - max_trim
    trim_threshold = level / max_trim
    deep_threshold = _INTERLOCK_LEVEL / deep_tests if deep_tests > 0 else 0.0

    p_values = [_spacing_p_value(spacings, j) for j in range(max_trim)]
    trim = 0
    for j in range(max_trim - 1, -1, -1):
        if p_values[j] < trim_threshold:
            trim = j + 1
            break

    # A gross outlier past max_trim leaves every scanned spacing looking
    # ordinary, because those are the gaps between outliers. The result is
    # "no contamination found" on a badly contaminated sample, which is the
    # one outcome worse than no estimate at all, so it is checked for. The
    # deepest such spacing is reported, because that is how far max_trim has
    # to reach: contamination shows up in several consecutive spacings, since
    # even a modest ratio between adjacent outliers is multiplied by its index.
    deepest = None
    for j in range(detect_to - 1, max_trim - 1, -1):
        if _spacing_p_value(spacings, j) < deep_threshold:
            deepest = j + 1
            break

    return {
        "trim": trim,
        "gamma": sum(spacings[trim:]) / (k - trim),
        "p_values": p_values,
        "saturated": deepest is not None,
        "deepest_anomaly": deepest,
    }


def adaptive_trimmed_hill_estimator(
    data: Sequence[float],
    k: int,
    max_trim: int | None = None,
    level: float = 0.05,
) -> float:
    """
    Trimmed Hill with the trimming chosen from the data.

    Fixed trimming forces a choice nobody can make well: too little leaves the
    contamination in, too much throws away good observations. This picks ``r``
    by testing the log-spacings, and on simulated contamination it picks the
    right number -- the median choice equals the number of planted outliers at
    1, 2, 3, 5 and 8 of them, and equals zero when there are none.

    What that buys, on 10,000 Pareto(2) draws with ``k = 300``:

    ============================  =========  ===========
    Sample                        Adaptive   Plain Hill
    ============================  =========  ===========
    clean                         0.5004     0.5007
    3 contaminated                0.5007     0.6004
    8 contaminated                0.5007     0.7971
    ============================  =========  ===========

    and on clean data it costs almost nothing: the standard deviation is 0.0295
    against 0.0292 for the plain estimator, a 1% loss. The robustness is close
    to free because trimming is applied only when the data asks for it.

    Parameters
    ----------
    data : sequence of floats
        Sample values. The top ``k + 1`` are read and must be positive.
    k : int
        Number of top order statistics to use, with ``1 < k < n``.
    max_trim : int, optional
        Largest trimming considered, defaulting to ``k // 4``.
    level : float, optional
        Family-wise probability of over-trimming clean data.

    Returns
    -------
    float
        The extreme-value index estimate gamma, equal to ``1 / alpha``.

    Raises
    ------
    ValueError
        If the arguments are out of range, or if contamination is found deeper
        than ``max_trim`` reaches. That case is an error rather than a number
        because the estimate would be indistinguishable from a clean one: with
        30 outliers and ``max_trim = 20`` the scan sees only the gaps between
        them, finds nothing, and reports 1.79 for a true 0.5.

    See Also
    --------
    adaptive_trim_selection : The chosen ``r`` and the tests behind it.
    trimmed_hill_estimator : Fixed trimming, when ``r`` is known.

    References
    ----------
    Bhattacharya, S., Kallitsis, M., & Stoev, S. (2019). Trimming the Hill
    estimator: robustness, optimality and adaptivity. arXiv:1705.03088.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = sorted(Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1), reverse=True)
    >>> for j in range(3):
    ...     data[j] = 1e6 * (j + 1)
    >>> round(adaptive_trimmed_hill_estimator(sorted(data, reverse=True), k=300), 3)
    0.479
    >>> round(hill_estimator(data, k=300), 3)  # the same sample, untrimmed
    0.581
    """
    result = adaptive_trim_selection(data, k, max_trim=max_trim, level=level)
    if result["saturated"]:
        raise ValueError(
            "Contamination reaches deeper than max_trim: a spacing at "
            f"{result['deepest_anomaly']} is anomalous but the scan stopped at "
            f"{max_trim if max_trim is not None else max(1, k // 4)}. The "
            "estimate would look like a clean sample rather than a wrong one, "
            f"so raise max_trim above {result['deepest_anomaly']}."
        )
    return float(result["gamma"])


def harmonic_moment_estimator(
    data: Sequence[float], k: int, beta: float = 1.0
) -> float:
    """
    Harmonic moment estimator of the extreme-value index.

    Where the Hill estimator averages ``log(X_(i) / u)`` for a threshold
    ``u = X_(k+1)``, this averages the powers of the reciprocal ratios
    ``R_i = u / X_(i)``, which lie in ``(0, 1]``. Under an exact Pareto tail
    those ratios are ``Beta(alpha, 1)`` distributed, so
    ``E[R**beta] = alpha / (alpha + beta)`` and

        alpha_hat = beta * H / (1 - H),   H = mean(R_i ** beta)

    The estimate returned is ``gamma = 1 / alpha_hat``.

    The point of the reciprocal form is bounded influence. Hill's contributions
    are unbounded above, so one sufficiently extreme observation moves the
    estimate arbitrarily far. Here a contaminated observation sent to infinity
    contributes ``R_i -> 0``, and its influence is bounded. Sending a single
    observation of ten thousand from ``1e2`` to ``1e30`` moves the Hill
    estimate from 0.502 to 0.631, and moves this one not at all.

    ``beta`` trades robustness against efficiency. Larger values weight the
    observations nearest the threshold more heavily and the extreme ones less,
    which is more robust and less efficient. As ``beta`` tends to zero the
    estimator tends to the Hill estimator.

    Like Hill this assumes ``gamma > 0``; see
    :func:`generalized_hill_estimator` if the sign is in doubt.

    Parameters
    ----------
    data : sequence of floats
        Sample values. The top ``k + 1`` are read, and those must be
        positive; values below the threshold are never touched.
    k : int
        Number of top order statistics to use, with ``1 < k < n``.
    beta : float, optional
        Robustness parameter, strictly positive. ``beta = 1`` is the t-Hill
        estimator, available separately as :func:`t_hill_estimator`.

    Returns
    -------
    float
        The extreme-value index estimate gamma, equal to ``1 / alpha``.

    Raises
    ------
    ValueError
        If k or beta is out of range, the data is not positive, or the sample
        is degenerate enough that ``H`` reaches 1.

    References
    ----------
    Beran, J., Schell, D., & Stehlik, M. (2014). The harmonic moment tail index
    estimator. *Annals of the Institute of Statistical Mathematics*, 66.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(50000, seed=11)
    >>> round(harmonic_moment_estimator(data, k=2500, beta=1.0), 1)
    0.5
    """
    if beta <= 0.0:
        raise ValueError("beta must be strictly positive.")

    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    threshold = x[k]
    if threshold <= 0.0:
        raise ValueError("the harmonic moment estimator requires positive data")

    h = sum((threshold / x[i]) ** beta for i in range(k)) / k
    if h >= 1.0:
        raise ValueError(
            "degenerate sample: the top k observations are not separated from "
            "the threshold, so the tail index is not identifiable"
        )

    alpha = beta * h / (1.0 - h)
    return float(1.0 / alpha)


def t_hill_estimator(data: Sequence[float], k: int) -> float:
    """
    t-Hill estimator of the extreme-value index.

    The harmonic moment estimator at ``beta = 1``, given its own name because
    the literature treats it as a distinct estimator with its own results. See
    :func:`harmonic_moment_estimator` for the derivation and for the meaning of
    ``beta``.

    It replaces Hill's unbounded logarithmic contributions with bounded
    reciprocal ratios, which is what makes it insensitive to how extreme a
    contaminated observation is.

    Parameters
    ----------
    data : sequence of floats
        Sample values. The top ``k + 1`` are read, and those must be
        positive; values below the threshold are never touched.
    k : int
        Number of top order statistics to use, with ``1 < k < n``.

    Returns
    -------
    float
        The extreme-value index estimate gamma, equal to ``1 / alpha``.

    References
    ----------
    Fabian, Z. (2001). Induced cores and their use in robust parametric
    estimation. *Communications in Statistics*.

    Jordanova, P., Stehlik, M., et al. (2016). Weak properties and robustness
    of t-Hill estimators. *Extremes*, 19(4).

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(50000, seed=11)
    >>> round(t_hill_estimator(data, k=2500), 1)
    0.5
    """
    return harmonic_moment_estimator(data, k, beta=1.0)


def _gpd_profile_log_likelihood(theta: float, exceedances: Sequence[float]) -> float:
    """Profile log-likelihood of the GPD at ``theta = xi / sigma``.

    Substituting ``theta`` reduces the two-parameter likelihood to one
    dimension (Grimshaw 1993): for fixed theta the profile maximum is
    ``xi = mean(log(1 + theta * y))`` and ``sigma = xi / theta``, at which the
    log-likelihood is ``n * (log|theta| - log|xi| - xi - 1)``.
    """
    if theta == 0.0:
        return -math.inf
    xi = sum(math.log1p(theta * y) for y in exceedances) / len(exceedances)
    if xi == 0.0:
        return -math.inf
    # sigma must be positive, which requires xi and theta to share a sign.
    if xi / theta <= 0.0:
        return -math.inf
    return len(exceedances) * (math.log(abs(theta)) - math.log(abs(xi)) - xi - 1.0)


def fit_generalized_pareto(
    exceedances: Sequence[float], scan: int = 200, iterations: int = 60
) -> dict[str, float]:
    """
    Fit a generalized Pareto distribution to exceedances by maximum likelihood.

    Uses the reduction of Grimshaw (1993), which turns the two-parameter
    problem into a one-dimensional search over ``theta = xi / sigma``. The
    profile is scanned coarsely to bracket the maximum and then refined by
    golden-section search, which needs no derivatives and no third-party
    optimiser.

    Valid for any sign of ``xi``, so unlike the Hill family this does not
    assume the tail is heavy.

    Parameters
    ----------
    exceedances : sequence of floats
        Amounts by which observations exceed a threshold. All must be strictly
        positive.
    scan : int, optional
        Number of points in the initial bracketing scan. The profile can be
        flat far from its maximum, so a scan is more reliable than starting the
        search from an arbitrary point.
    iterations : int, optional
        Golden-section refinement steps. Sixty reduces the bracket by a factor
        of about 1e-13, which is past the precision of the data.

    Returns
    -------
    dict
        ``xi`` (shape), ``sigma`` (scale), ``n`` and ``log_likelihood``.

    Raises
    ------
    ValueError
        If fewer than two exceedances are given, or any is not positive.

    References
    ----------
    Grimshaw, S. D. (1993). Computing maximum likelihood estimates for the
    generalized Pareto distribution. *Technometrics*, 35(2), 185-191.

    Examples
    --------
    >>> from heavytails import GeneralizedPareto
    >>> y = GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0).rvs(5000, seed=7)
    >>> round(fit_generalized_pareto(y)["xi"], 1)
    0.5
    """
    values = [float(y) for y in exceedances]
    if len(values) < 2:
        raise ValueError("need at least two exceedances to fit a GPD")
    if min(values) <= 0.0:
        raise ValueError("exceedances must be strictly positive")

    largest = max(values)
    ordered = sorted(values)
    median = ordered[len(ordered) // 2]

    # theta > -1/max(y) keeps 1 + theta*y positive for every observation.
    lo = -1.0 / largest + 1e-10

    # theta = xi/sigma, and sigma is a scale, so theta is of order 1/scale. The
    # median is the scale proxy rather than the mean, because the mean does not
    # exist for xi >= 1: at xi = 1.5 a bound of 20/mean puts the true theta of
    # 3.0 outside the search entirely. Both halves are spaced logarithmically
    # in |theta|, so the grid is equally fine near zero and near the boundary;
    # a linear negative grid is too coarse where the optimum sits close to
    # -1/max(y), which cost two decimal places at xi = -0.5.
    points = max(scan, 16)

    negative_high, negative_low = math.log(abs(lo)), math.log(abs(lo) * 1e-6)
    negatives = [
        -math.exp(negative_low + (negative_high - negative_low) * i / points)
        for i in range(points + 1)
    ]
    negatives.reverse()  # ascending, from just above lo to just below zero

    log_low, log_high = math.log(1e-6 / median), math.log(1e6 / median)
    positives = [
        math.exp(log_low + (log_high - log_low) * i / points) for i in range(points + 1)
    ]
    candidates: list[float] = negatives + positives

    best_theta = None
    best_value = -math.inf
    best_index = 0
    for index, theta in enumerate(candidates):
        value = _gpd_profile_log_likelihood(theta, values)
        if value > best_value:
            best_value, best_theta, best_index = value, theta, index

    if best_theta is None:
        raise ValueError(
            "could not bracket the likelihood maximum; the sample may be degenerate"
        )

    # Refine between the neighbours of the best grid point.
    a = candidates[max(best_index - 1, 0)]
    b = candidates[min(best_index + 1, len(candidates) - 1)]
    if a > b:
        a, b = b, a
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(iterations):
        c, d = b - golden * (b - a), a + golden * (b - a)
        if _gpd_profile_log_likelihood(c, values) > _gpd_profile_log_likelihood(
            d, values
        ):
            b = d
        else:
            a = c

    theta = 0.5 * (a + b)
    xi = sum(math.log1p(theta * y) for y in values) / len(values)
    return {
        "xi": float(xi),
        "sigma": float(xi / theta),
        "n": len(values),
        "log_likelihood": _gpd_profile_log_likelihood(theta, values),
    }


def gpd_mle_estimator(data: Sequence[float], k: int) -> float:
    """
    Peaks-over-threshold estimator: fit a GPD to the exceedances above X_(k+1).

    The parametric alternative to the semiparametric estimators in this module.
    Rather than averaging a functional of the upper order statistics, it fits a
    two-parameter model to the exceedances and estimates shape and scale
    jointly, which is what the Pickands-Balkema-de Haan theorem licenses.

    Valid for any sign of the index, and noticeably slower than the closed-form
    estimators because it optimises. That matters mainly for bootstrapping, so
    reduce ``n_bootstrap`` accordingly.

    Parameters
    ----------
    data : sequence of floats
        Sample values.
    k : int
        Number of exceedances to use, with ``1 < k < n``. The threshold is the
        ``(k+1)``-th largest observation.

    Returns
    -------
    float
        The extreme-value index estimate gamma, which is the GPD shape
        parameter.

    Raises
    ------
    ValueError
        If k is out of range or the exceedances are degenerate.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=3)
    >>> round(gpd_mle_estimator(data, k=2000), 1)
    0.5
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")

    threshold = x[k]
    exceedances = [x[i] - threshold for i in range(k) if x[i] > threshold]
    if len(exceedances) < 2:
        raise ValueError(
            "fewer than two observations exceed the threshold; the sample is "
            "tied at the top and the index is not identifiable"
        )
    return fit_generalized_pareto(exceedances)["xi"]


def recommended_rho_k(n: int) -> int:
    """Recommended number of order statistics for estimating rho.

    Estimating the second-order parameter needs far more of the sample than
    estimating gamma does, because it describes how the tail approaches its
    limit rather than the limit itself. The usual recommendation is
    ``min(n - 1, floor(2n / log log n))``, roughly 85% of the sample.

    Args:
        n: Sample size, at least 16 so that ``log log n`` is positive.

    Returns:
        The recommended k.

    Raises:
        ValueError: If n is below 16.
    """
    if n < 16:
        raise ValueError("need at least 16 observations to estimate rho")
    return min(n - 1, int(2.0 * n / math.log(math.log(n))))


def second_order_rho(
    data: Sequence[float], k: int | None = None, tau: float = 0.0
) -> float:
    """Estimate the second-order parameter rho of a regularly varying tail.

    Under second-order regular variation the tail behaves like
    ``C x**(-1/gamma) [1 + D x**(rho/gamma) + ...]`` with ``rho < 0``. The
    parameter controls how fast the tail approaches its Pareto limit, and so
    how quickly the Hill estimator's bias grows with k.

    Implements the estimator of Fraga Alves, Gomes and de Haan (2003), built
    from the first three moments of the log-excesses.

    Warning:
        This estimator is unstable, which is a property of the estimator
        rather than of this implementation. Sweeping k on a Frechet sample
        whose true rho is -1 gives estimates ranging from -0.07 to -20.5, the
        latter at a pole where the denominator crosses zero. Prefer supplying
        a known or assumed rho to :func:`bias_reduced_hill_estimator`, and
        plot the estimate against k before trusting any single value.

    Args:
        data: Sample values. The top ``k + 1`` are read and must be positive.
        k: Number of order statistics. Defaults to :func:`recommended_rho_k`,
            which is much larger than the k used for gamma.
        tau: Tuning parameter. ``0.0`` uses the logarithmic form, positive
            values the power form. Both are in the original paper, and the
            choice matters as much as k does.

    Returns:
        The estimate of rho, always negative.

    Raises:
        ValueError: If k is out of range, tau is negative, or the moments are
            degenerate.

    References:
        Fraga Alves, M. I., Gomes, M. I., & de Haan, L. (2003). A new class of
        semi-parametric estimators of the second order parameter.
        *Portugaliae Mathematica*, 60(2), 193-213.

    Examples:
        >>> from heavytails import Frechet
        >>> data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(20000, seed=11)
        >>> second_order_rho(data) < 0   # true value is -1
        True
    """
    if tau < 0.0:
        raise ValueError("tau must be non-negative")

    x = sorted(data, reverse=True)
    n = len(x)
    if k is None:
        k = recommended_rho_k(n)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    if x[k] <= 0.0:
        raise ValueError("rho estimation requires positive data")

    log_threshold = math.log(x[k])
    excess = [math.log(x[i]) - log_threshold for i in range(k)]
    m1 = sum(excess) / k
    m2 = sum(e * e for e in excess) / k
    m3 = sum(e * e * e for e in excess) / k
    if m1 <= 0.0 or m2 <= 0.0 or m3 <= 0.0:
        raise ValueError(
            "degenerate log-excess moments; the sample is tied in its upper tail"
        )

    if tau == 0.0:
        numerator = math.log(m1) - 0.5 * math.log(m2 / 2.0)
        denominator = 0.5 * math.log(m2 / 2.0) - math.log(m3 / 6.0) / 3.0
    else:
        first = m1**tau
        second = (m2 / 2.0) ** (tau / 2.0)
        third = (m3 / 6.0) ** (tau / 3.0)
        numerator = first - second
        denominator = second - third

    if denominator == 0.0:
        raise ValueError(
            "rho is not identifiable at this k: the denominator vanished. "
            "Try a different k or tau."
        )

    statistic = numerator / denominator
    if statistic == 3.0:
        raise ValueError("rho is not identifiable at this k: the estimator has a pole")
    return -abs(3.0 * (statistic - 1.0) / (statistic - 3.0))


def second_order_beta(data: Sequence[float], k: int, rho: float) -> float:
    """Estimate the second-order scale parameter beta, given rho.

    Implements the estimator of Gomes and Martins (2002), built from the
    normalised log-spacings weighted by powers of ``i / k``.

    Args:
        data: Sample values.
        k: Number of order statistics, matching the k used for gamma.
        rho: The second-order shape parameter, which must be negative.

    Returns:
        The estimate of beta.

    Raises:
        ValueError: If rho is not negative, k is out of range, or the
            estimator is degenerate.

    References:
        Gomes, M. I., & Martins, M. J. (2002). Asymptotically unbiased
        estimators of the tail index based on external estimation of the
        second order parameter. *Extremes*, 5(1), 5-31.
    """
    if rho >= 0.0:
        raise ValueError("rho must be negative")

    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")
    if x[k] <= 0.0:
        raise ValueError("beta estimation requires positive data")

    spacings = _normalised_log_spacings(x, k)

    def moment(power: float) -> tuple[float, float]:
        """Unweighted and spacing-weighted means of ``(i/k) ** -power``."""
        weights = [(i / k) ** (-power) for i in range(1, k + 1)]
        plain = sum(weights) / k
        weighted = sum(w * u for w, u in zip(weights, spacings, strict=True)) / k
        return plain, weighted

    d_rho, big_d_rho = moment(rho)
    _, big_d_zero = moment(0.0)
    _, big_d_two_rho = moment(2.0 * rho)

    denominator = d_rho * big_d_rho - big_d_two_rho
    if denominator == 0.0:
        raise ValueError("beta is not identifiable at this k and rho")
    return float(((k / n) ** rho) * (d_rho * big_d_zero - big_d_rho) / denominator)


def bias_reduced_hill_estimator(
    data: Sequence[float],
    k: int,
    rho: float | None = None,
    beta: float | None = None,
) -> float:
    """Bias-reduced Hill estimator of Caeiro, Gomes and Pestana (2005).

    The Hill estimator trades variance at small k against bias at large k, and
    that bias is systematic rather than random: it comes from the second-order
    behaviour of the tail, so it can be estimated and subtracted.

        gamma = hill(k) * (1 - beta / (1 - rho) * (n/k) ** rho)

    Measured over forty samples at ``n = 20000`` with rho supplied, the bias of
    the Hill estimator falls by a factor of four to twenty::

        case                k     Hill      corrected
        Frechet(2)          2000  0.0162    0.0046
        BurrXII(c=2, k=1)   2000  0.0298    0.0050
        BurrXII(c=1, k=2)   2000  0.1422    0.0081

    Note:
        **Supply rho where you can.** Estimating it works and the correction
        still helps, taking the worst case above from 0.1422 to 0.0354, but
        :func:`second_order_rho` is unstable and a poor estimate costs most of
        the benefit. Practitioners commonly fix rho at a canonical value such
        as -1 rather than estimate it.

    Args:
        data: Sample values. The top ``k + 1`` are read and must be positive.
        k: Number of top order statistics, with ``1 < k < n``.
        rho: Second-order shape parameter, which must be negative. Estimated
            with :func:`second_order_rho` when omitted.
        beta: Second-order scale parameter. Estimated with
            :func:`second_order_beta` when omitted.

    Returns:
        The extreme-value index estimate gamma, equal to ``1 / alpha``.

    Raises:
        ValueError: If k is out of range, rho is not negative, or either
            second-order parameter cannot be estimated.

    References:
        Caeiro, F., Gomes, M. I., & Pestana, D. (2005). Direct reduction of
        bias of the classical Hill estimator. *Revstat*, 3(2), 113-136.

    Examples:
        >>> from heavytails import Frechet
        >>> data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(20000, seed=1)
        >>> round(bias_reduced_hill_estimator(data, k=2000, rho=-1.0), 1)
        0.5
    """
    x = sorted(data, reverse=True)
    n = len(x)
    if not (1 < k < n):
        raise ValueError("k must be between 1 and n-1")

    if rho is None:
        rho = second_order_rho(x)
    if rho >= 0.0:
        raise ValueError("rho must be negative")
    if beta is None:
        beta = second_order_beta(x, k, rho)

    hill = hill_estimator(x, k)
    correction = 1.0 - (beta / (1.0 - rho)) * (n / k) ** rho
    return float(hill * correction)


#: Estimators available to :func:`tail_index_confidence_interval`.
_POINT_ESTIMATORS: dict[str, Callable[..., float]] = {
    "hill": hill_estimator,
    "generalized_hill": generalized_hill_estimator,
    "smoothed_hill": smoothed_hill_estimator,
    "trimmed_hill": trimmed_hill_estimator,
    "adaptive_trimmed_hill": adaptive_trimmed_hill_estimator,
    "harmonic_moment": harmonic_moment_estimator,
    "t_hill": t_hill_estimator,
    "gpd_mle": gpd_mle_estimator,
    "bias_reduced_hill": bias_reduced_hill_estimator,
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
