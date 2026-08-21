"""Threshold selection for peaks-over-threshold analysis.

Choosing the threshold is the decision that dominates a peaks-over-threshold
analysis, and no rule settles it. Too low and observations from the body of the
distribution contaminate the fit, biasing the result. Too high and few
exceedances remain, so the estimate is noisy. Everything here is a tool for
making that trade-off visible rather than a substitute for looking.

Two diagnostics and one automatic rule:

* :func:`mean_residual_life` -- the mean excess is linear in the threshold
  above a valid one, so the plot should straighten out.
* :func:`parameter_stability` -- the shape and the *modified* scale are
  constant above a valid threshold, so the plot should flatten.
* :func:`select_threshold` -- the lowest threshold at which a goodness-of-fit
  test does not reject the generalized Pareto model.

Read the first two before trusting the third.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from heavytails._special import _phi_inverse
from heavytails.heavy_tails import RNG
from heavytails.tail_index import fit_generalized_pareto

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "mean_residual_life",
    "parameter_stability",
    "return_level",
    "select_threshold",
]


def _default_thresholds(data: Sequence[float], count: int = 40) -> list[float]:
    """Candidate thresholds spanning the 70th to 99th percentile.

    Below the 70th percentile the generalized Pareto approximation has no
    reason to hold; above the 99th there are rarely enough exceedances to fit
    two parameters.
    """
    ordered = sorted(data)
    n = len(ordered)
    return [
        ordered[min(int(q * n), n - 1)]
        for q in (0.70 + (0.99 - 0.70) * i / (count - 1) for i in range(count))
    ]


def _excesses(data: Sequence[float], threshold: float) -> list[float]:
    """Amounts by which observations strictly exceed the threshold."""
    return [float(x) - threshold for x in data if x > threshold]


def mean_residual_life(
    data: Sequence[float],
    thresholds: Sequence[float] | None = None,
    level: float = 0.95,
) -> list[dict[str, Any]]:
    """
    Mean excess over a range of thresholds, with confidence intervals.

    If the generalized Pareto model holds above some threshold, the mean excess
    is **linear** in the threshold from there upwards. So the plot should be
    curved at low thresholds and straighten out once the model applies, and the
    point where it straightens is a candidate threshold.

    The right-hand end is always noisy, because it rests on a handful of
    observations. Judge linearity from the region where the intervals are still
    narrow.

    Args:
        data: Sample values.
        thresholds: Candidate thresholds. Defaults to forty spanning the 70th
            to 99th percentile.
        level: Confidence level for the interval around each mean excess.

    Returns:
        One dictionary per threshold with ``threshold``, ``mean_excess``,
        ``n_exceedances``, ``std_error``, ``lower`` and ``upper``. Thresholds
        with fewer than two exceedances are omitted.

    Raises:
        ValueError: If the sample is too small or ``level`` is invalid.

    Examples:
        >>> from heavytails import Pareto
        >>> data = Pareto(alpha=2.0, xm=1.0).rvs(10000, seed=1)
        >>> points = mean_residual_life(data)
        >>> points[0]["mean_excess"] > 0
        True
    """
    if len(data) < 10:
        raise ValueError("need at least 10 observations")
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")

    candidates = _default_thresholds(data) if thresholds is None else list(thresholds)
    z = _phi_inverse(0.5 + level / 2.0)

    points: list[dict[str, Any]] = []
    for threshold in candidates:
        excesses = _excesses(data, threshold)
        count = len(excesses)
        if count < 2:
            continue
        mean_excess = sum(excesses) / count
        variance = sum((e - mean_excess) ** 2 for e in excesses) / (count - 1)
        std_error = math.sqrt(variance / count)
        points.append(
            {
                "threshold": threshold,
                "mean_excess": mean_excess,
                "n_exceedances": count,
                "std_error": std_error,
                "lower": mean_excess - z * std_error,
                "upper": mean_excess + z * std_error,
            }
        )
    return points


def parameter_stability(
    data: Sequence[float],
    thresholds: Sequence[float] | None = None,
    min_exceedances: int = 30,
) -> list[dict[str, Any]]:
    """
    Fitted generalized Pareto parameters across a range of thresholds.

    Above a valid threshold the shape is the same at every higher threshold,
    and the scale grows linearly: ``sigma_u = sigma_0 + xi * (u - u_0)``. The
    **modified scale** ``sigma_u - xi * u`` removes that growth and is
    therefore constant. Both should flatten into a plateau above a valid
    threshold, and the start of the plateau is a candidate.

    The plot degrades badly at the top, where two parameters are being fitted
    to a few dozen points. That is not a defect in the fit; it is the variance
    half of the trade-off, and it is why this is read as a plot.

    Args:
        data: Sample values.
        thresholds: Candidate thresholds. Defaults to forty spanning the 70th
            to 99th percentile.
        min_exceedances: Skip thresholds leaving fewer exceedances than this.
            Fitting two parameters to fewer is not informative.

    Returns:
        One dictionary per threshold with ``threshold``, ``xi``, ``sigma``,
        ``modified_scale`` and ``n_exceedances``. Thresholds whose fit fails
        are omitted.

    Raises:
        ValueError: If the sample is too small.

    Examples:
        >>> from heavytails import Pareto
        >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        >>> points = parameter_stability(data)
        >>> abs(points[0]["xi"] - 0.5) < 0.1
        True
    """
    if len(data) < 10:
        raise ValueError("need at least 10 observations")

    candidates = _default_thresholds(data) if thresholds is None else list(thresholds)

    points: list[dict[str, Any]] = []
    for threshold in candidates:
        excesses = _excesses(data, threshold)
        if len(excesses) < min_exceedances:
            continue
        try:
            fit = fit_generalized_pareto(excesses)
        except ValueError:
            # A degenerate set of excesses at this threshold; the rest of the
            # sweep is still informative.
            continue
        points.append(
            {
                "threshold": threshold,
                "xi": fit["xi"],
                "sigma": fit["sigma"],
                "modified_scale": fit["sigma"] - fit["xi"] * threshold,
                "n_exceedances": len(excesses),
            }
        )
    return points


def select_threshold(
    data: Sequence[float],
    thresholds: Sequence[float] | None = None,
    alpha_level: float = 0.05,
    min_exceedances: int = 50,
) -> dict[str, Any]:
    """
    Choose the lowest threshold at which the generalized Pareto model fits.

    Walks the candidates upwards, fits the model to the excesses at each, and
    tests the fit with the Anderson-Darling statistic. The first threshold the
    test does not reject is returned, since a lower threshold keeps more data
    and so gives a less variable estimate.

    .. warning::

       The p-values are **conservative**, because the parameters are estimated
       from the same excesses being tested. A conservative test rejects less
       often than its nominal level, so this rule tends to select a threshold
       that is too *low*. Treat the answer as a starting point and check it
       against :func:`mean_residual_life` and :func:`parameter_stability`.

    Args:
        data: Sample values.
        thresholds: Candidates, tried in increasing order. Defaults to forty
            spanning the 70th to 99th percentile.
        alpha_level: Significance level for the goodness-of-fit test.
        min_exceedances: Skip thresholds leaving fewer exceedances than this.

    Returns:
        Dictionary with ``threshold``, ``xi``, ``sigma``, ``n_exceedances``,
        ``p_value``, ``exceedance_rate`` and ``candidates_tested``. When no
        candidate passes, ``threshold`` is None and ``candidates_tested``
        records every attempt, so the failure is inspectable.

    Raises:
        ValueError: If the sample is too small or ``alpha_level`` is invalid.

    Examples:
        >>> from heavytails import Pareto
        >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        >>> result = select_threshold(data)
        >>> result["threshold"] is not None
        True
    """
    from heavytails.validation import GoodnessOfFitTests  # noqa: PLC0415

    if len(data) < 20:
        raise ValueError("need at least 20 observations")
    if not (0.0 < alpha_level < 1.0):
        raise ValueError("alpha_level must be in (0,1).")

    candidates = _default_thresholds(data) if thresholds is None else list(thresholds)
    tests = GoodnessOfFitTests(alpha_level=alpha_level)
    attempted: list[dict[str, Any]] = []

    for threshold in sorted(candidates):
        excesses = _excesses(data, threshold)
        if len(excesses) < min_exceedances:
            continue
        try:
            fit = fit_generalized_pareto(excesses)
            result = tests.anderson_darling_test(
                excesses,
                "gpd",
                parameters_estimated=True,
                xi=fit["xi"],
                sigma=fit["sigma"],
                mu=0.0,
            )
        except (ValueError, OverflowError) as exc:
            attempted.append({"threshold": threshold, "error": str(exc)})
            continue

        attempted.append(
            {
                "threshold": threshold,
                "n_exceedances": len(excesses),
                "xi": fit["xi"],
                "p_value": result["p_value"],
                "rejected": result["reject"],
            }
        )
        if not result["reject"]:
            return {
                "threshold": threshold,
                "xi": fit["xi"],
                "sigma": fit["sigma"],
                "n_exceedances": len(excesses),
                "exceedance_rate": len(excesses) / len(data),
                "p_value": result["p_value"],
                "candidates_tested": attempted,
            }

    return {
        "threshold": None,
        "xi": None,
        "sigma": None,
        "n_exceedances": 0,
        "exceedance_rate": 0.0,
        "p_value": None,
        "candidates_tested": attempted,
    }


def return_level(
    data: Sequence[float],
    threshold: float,
    period: float,
    level: float = 0.95,
    n_bootstrap: int = 200,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Return level: the value exceeded once every ``period`` observations.

    Fits the generalized Pareto to the excesses above ``threshold`` and
    evaluates

        x_T = u + (sigma / xi) * ((T * rate) ** xi - 1)

    where ``rate`` is the observed exceedance rate. This is the calculation
    behind a "1-in-100-year" figure.

    Because ``x_T`` grows like ``T ** xi``, a small error in the shape becomes
    a large error in the return level, so the interval matters more than
    usual. It is obtained by resampling the data, refitting, and taking
    percentiles; that captures sampling variability but **not** the error from
    choosing the threshold, nor any doubt about whether the model applies at
    all.

    Measured coverage of a nominal 95% interval is about 0.88 for a Pareto tail
    at ``n = 20000``, falling to roughly 0.76 at ``n = 8000``. Treat the
    interval as a lower bound on the uncertainty rather than a calibrated
    statement, particularly on small samples.

    Args:
        data: Sample values.
        threshold: The threshold to fit above.
        period: Return period ``T``, in observations. Must exceed the
            reciprocal of the exceedance rate, or the level lies inside the
            body of the data where the model says nothing.
        level: Confidence level for the interval.
        n_bootstrap: Resamples for the interval.
        seed: Seed, for reproducibility.

    Returns:
        Dictionary with ``return_level``, ``lower``, ``upper``, ``period``,
        ``threshold``, ``xi``, ``sigma``, ``exceedance_rate`` and
        ``n_exceedances``.

    Raises:
        ValueError: If there are too few exceedances, the period is too short,
            or the arguments are out of range.

    Examples:
        >>> from heavytails import Pareto
        >>> data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        >>> result = return_level(data, threshold=10.0, period=1000, n_bootstrap=20)
        >>> result["lower"] < result["return_level"] < result["upper"]
        True
    """
    if not (0.0 < level < 1.0):
        raise ValueError("level must be in (0,1).")
    if period <= 1.0:
        raise ValueError("period must be greater than 1")
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")

    n = len(data)
    excesses = _excesses(data, threshold)
    if len(excesses) < 10:
        raise ValueError(
            f"only {len(excesses)} observations exceed {threshold}; "
            "lower the threshold or collect more data"
        )

    rate = len(excesses) / n
    if period * rate <= 1.0:
        raise ValueError(
            f"period={period} with an exceedance rate of {rate:.4g} gives a "
            "return level at or below the threshold, where the model says "
            "nothing; use a longer period"
        )

    def estimate(sample: Sequence[float]) -> float | None:
        rows = _excesses(sample, threshold)
        if len(rows) < 10:
            return None
        try:
            fit = fit_generalized_pareto(rows)
        except ValueError:
            return None
        xi, sigma = fit["xi"], fit["sigma"]
        observed_rate = len(rows) / len(sample)
        if observed_rate * period <= 1.0:
            return None
        if abs(xi) < 1e-12:
            return float(threshold + sigma * math.log(period * observed_rate))
        return float(threshold + (sigma / xi) * ((period * observed_rate) ** xi - 1.0))

    point = estimate(data)
    if point is None:  # pragma: no cover - guarded by the checks above
        raise ValueError("could not fit the generalized Pareto at this threshold")

    fit = fit_generalized_pareto(excesses)
    rng = RNG(seed)
    draws: list[float] = []
    for _ in range(n_bootstrap):
        resample = [data[min(int(rng.uniform_0_1() * n), n - 1)] for _ in range(n)]
        value = estimate(resample)
        if value is not None:
            draws.append(value)

    if len(draws) < 2:
        raise ValueError(
            "bootstrap failed: too few resamples produced a usable return level"
        )

    draws.sort()
    tail = (1.0 - level) / 2.0
    last = len(draws) - 1
    return {
        "return_level": point,
        "lower": draws[max(0, min(last, round(tail * last)))],
        "upper": draws[max(0, min(last, round((1.0 - tail) * last)))],
        "level": level,
        "period": period,
        "threshold": threshold,
        "xi": fit["xi"],
        "sigma": fit["sigma"],
        "exceedance_rate": rate,
        "n_exceedances": len(excesses),
    }
