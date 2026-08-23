"""Rendering for the diagnostics, using matplotlib.

The rest of the library returns coordinates rather than figures, which is what
keeps matplotlib out of the requirements. That is the right default, but it
means every user
writes the same twenty lines of matplotlib to look at a Hill plot, and the
documentation carries that boilerplate on several pages.

This module draws them. It needs the optional ``plot`` extra::

    pip install "heavytails[plot]"

Nothing here is imported by :mod:`heavytails.plotting`, which stays free of
third-party imports, so the library itself is unaffected whether matplotlib is
installed or not.

Every function takes an optional ``ax`` and returns the axes it drew on, so
plots compose into a larger figure::

    fig, axes = plt.subplots(1, 2)
    plot_tail(data, ax=axes[0])
    plot_hill(data, ax=axes[1])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from heavytails.plotting import qq_pareto, tail_loglog_plot
from heavytails.tail_index import hill_plot, trimmed_hill_plot
from heavytails.threshold import mean_residual_life, parameter_stability

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "plot_hill",
    "plot_mean_residual_life",
    "plot_parameter_stability",
    "plot_qq",
    "plot_tail",
    "plot_trimmed_hill",
]

# Enough to look smooth on a log axis without the cost mattering.
_REFERENCE_POINTS = 200

_INSTALL_HINT = (
    "Rendering the diagnostics requires matplotlib, which is an optional "
    "dependency.\nInstall it with:  pip install 'heavytails[plot]'\n"
    "The coordinate-returning functions in heavytails.plotting and "
    "heavytails.tail_index work without it."
)


def _axes(ax: Any) -> Any:
    """Return the given axes, or create one, raising helpfully without matplotlib."""
    if ax is not None:
        return ax
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ModuleNotFoundError as exc:  # pragma: no cover - needs matplotlib absent
        raise ModuleNotFoundError(_INSTALL_HINT) from exc
    return plt.subplots()[1]


def plot_tail(
    data: Sequence[float],
    ax: Any = None,
    fitted: Any = None,
    label: str = "empirical",
    **kwargs: Any,
) -> Any:
    """
    Log-log plot of the empirical survival function.

    A power-law tail is a straight line here, with slope ``-alpha``. This is
    the single most useful heavy-tail diagnostic, and the first thing to look
    at before fitting anything.

    Pass ``fitted`` to overlay a distribution's own survival curve on the same
    axes. That comparison is the one a goodness-of-fit statistic cannot make
    for you: it shows *where* the model and the data part company.

    Args:
        data: Sample values.
        ax: Axes to draw on. A new figure is created when omitted.
        fitted: Optional distribution whose survival curve is overlaid, drawn
            from a sample of the same size so the two are comparable.
        label: Legend label for the empirical curve.
        **kwargs: Passed to the empirical scatter call.

    Returns:
        The axes drawn on.

    Raises:
        ModuleNotFoundError: If matplotlib is not installed and no ``ax`` was
            given.

    Examples:
        >>> import matplotlib
        >>> matplotlib.use("Agg")
        >>> from heavytails import Pareto
        >>> from heavytails.viz import plot_tail
        >>> data = Pareto(alpha=2.0, xm=1.0).rvs(1000, seed=1)
        >>> ax = plot_tail(data)
        >>> ax.get_xlabel()
        'log x'
    """
    axes = _axes(ax)
    points = tail_loglog_plot(data)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    style = {"marker": ".", "markersize": 2, "linestyle": "none"}
    style.update(kwargs)
    axes.plot(xs, ys, label=label, **style)

    if fitted is not None:
        # The model's own survival function on a grid, not a sample drawn from
        # it. Drawing one put Monte Carlo noise into the reference curve, worst
        # in the far tail where its last points rested on a handful of draws:
        # against Pareto(alpha=2) at n=1000 the reference wandered up to 1.238
        # in log survival from the curve it was meant to represent -- a factor
        # of three, seed-dependent, and entirely in the region the plot exists
        # to show. A reader would have read it as misfit.
        grid = np.exp(np.linspace(min(xs), max(xs), _REFERENCE_POINTS))
        survival = np.asarray(fitted.sf(grid), dtype=float)
        # log(0) is not a point on the plot. A bounded model can put zero
        # survival inside the data's range, and those x are simply not drawn.
        visible = survival > 0.0
        axes.plot(
            np.log(grid[visible]),
            np.log(survival[visible]),
            linewidth=1.5,
            alpha=0.8,
            label=f"fitted {type(fitted).__name__}",
        )
        axes.legend()

    axes.set_xlabel("log x")
    axes.set_ylabel("log P(X > x)")
    axes.set_title("Log-log tail plot")
    return axes


def plot_qq(data: Sequence[float], ax: Any = None, **kwargs: Any) -> Any:
    """
    QQ plot of the sample against Pareto quantiles.

    Linear if the sample is Pareto-tailed. Departures show as systematic
    curvature: points above the line at the right mean a heavier tail than the
    reference, below means lighter.

    More sensitive than the log-log plot to what happens in the body of the
    distribution, so the two complement each other.

    Args:
        data: Sample values.
        ax: Axes to draw on. A new figure is created when omitted.
        **kwargs: Passed to the plotting call.

    Returns:
        The axes drawn on.
    """
    axes = _axes(ax)
    points = qq_pareto(data)
    style = {"marker": ".", "markersize": 2, "linestyle": "none"}
    style.update(kwargs)
    axes.plot([p[0] for p in points], [p[1] for p in points], **style)
    axes.set_xlabel("log(i/n)")
    axes.set_ylabel("log x")
    axes.set_title("Pareto QQ plot")
    return axes


def plot_hill(
    data: Sequence[float],
    ax: Any = None,
    ks: Sequence[int] | None = None,
    true_gamma: float | None = None,
    **kwargs: Any,
) -> Any:
    """
    Hill plot: the tail index estimate against the number of order statistics.

    Read the estimate off a stable plateau. Small ``k`` is noisy, large ``k``
    drifts as observations from the body enter. **If there is no plateau, the
    data does not support a tail index estimate**, and the plot is telling you
    that rather than failing to.

    Args:
        data: Sample values.
        ax: Axes to draw on. A new figure is created when omitted.
        ks: Values of k. Defaults to the logarithmic sweep of
            :func:`heavytails.tail_index.hill_plot`.
        true_gamma: Draws a horizontal reference line, useful on simulated
            data where the answer is known.
        **kwargs: Passed to the plotting call.

    Returns:
        The axes drawn on.
    """
    axes = _axes(ax)
    points = hill_plot(data, ks=ks)
    axes.plot([p[0] for p in points], [p[1] for p in points], **kwargs)
    if true_gamma is not None:
        axes.axhline(
            true_gamma, linestyle="--", linewidth=1, color="grey", label="true gamma"
        )
        axes.legend()
    axes.set_xscale("log")
    axes.set_xlabel("k (order statistics)")
    axes.set_ylabel("gamma")
    axes.set_title("Hill plot")
    return axes


def plot_trimmed_hill(
    data: Sequence[float],
    k: int,
    ax: Any = None,
    max_trim: int | None = None,
    **kwargs: Any,
) -> Any:
    """
    Trimmed Hill plot: the estimate against the number of observations trimmed.

    The estimate moves while ``r`` is below the number of contaminated
    observations and flattens once they are gone, so **the elbow says how much
    contamination is present**. A plot flat from ``r = 0`` says there is none.

    Args:
        data: Sample values.
        k: Number of top order statistics to use.
        ax: Axes to draw on. A new figure is created when omitted.
        max_trim: Largest ``r`` to evaluate.
        **kwargs: Passed to the plotting call.

    Returns:
        The axes drawn on.
    """
    axes = _axes(ax)
    points = trimmed_hill_plot(data, k, max_trim=max_trim)
    style = {"marker": "o", "markersize": 3}
    style.update(kwargs)
    axes.plot([p[0] for p in points], [p[1] for p in points], **style)
    axes.set_xlabel("r (observations trimmed)")
    axes.set_ylabel("gamma")
    axes.set_title(f"Trimmed Hill plot (k = {k})")
    return axes


def plot_mean_residual_life(
    data: Sequence[float],
    ax: Any = None,
    thresholds: Sequence[float] | None = None,
    level: float = 0.95,
    **kwargs: Any,
) -> Any:
    """
    Mean residual life plot, with a confidence band.

    Linear above a valid threshold, so look for where the curve straightens.
    The band widens towards the right as exceedances run out; judge linearity
    from the region where it is still narrow, not from the noisy end.

    Args:
        data: Sample values.
        ax: Axes to draw on. A new figure is created when omitted.
        thresholds: Candidate thresholds.
        level: Confidence level for the band.
        **kwargs: Passed to the plotting call.

    Returns:
        The axes drawn on.
    """
    axes = _axes(ax)
    points = mean_residual_life(data, thresholds=thresholds, level=level)
    us = [p["threshold"] for p in points]
    axes.plot(us, [p["mean_excess"] for p in points], **kwargs)
    axes.fill_between(
        us,
        [p["lower"] for p in points],
        [p["upper"] for p in points],
        alpha=0.2,
        label=f"{level:.0%} interval",
    )
    axes.legend()
    axes.set_xlabel("threshold u")
    axes.set_ylabel("mean excess")
    axes.set_title("Mean residual life plot")
    return axes


def plot_parameter_stability(
    data: Sequence[float],
    ax: Any = None,
    thresholds: Sequence[float] | None = None,
    parameter: str = "xi",
    **kwargs: Any,
) -> Any:
    """
    Parameter stability plot across candidate thresholds.

    Both the shape and the *modified* scale are constant above a valid
    threshold, so look for a plateau. The raw scale is not constant and grows
    with the threshold, which is why ``modified_scale`` is the more useful of
    the two to plot.

    The right-hand end degrades badly, where two parameters are fitted to a few
    dozen points. That is the variance half of the trade-off, not a defect.

    Args:
        data: Sample values.
        ax: Axes to draw on. A new figure is created when omitted.
        thresholds: Candidate thresholds.
        parameter: ``xi``, ``sigma`` or ``modified_scale``.
        **kwargs: Passed to the plotting call.

    Returns:
        The axes drawn on.

    Raises:
        ValueError: If ``parameter`` is not one of the three.
    """
    allowed = {"xi", "sigma", "modified_scale"}
    if parameter not in allowed:
        raise ValueError(
            f"Unknown parameter {parameter!r}. Available: {', '.join(sorted(allowed))}"
        )

    axes = _axes(ax)
    points = parameter_stability(data, thresholds=thresholds)
    style = {"marker": "o", "markersize": 3}
    style.update(kwargs)
    axes.plot([p["threshold"] for p in points], [p[parameter] for p in points], **style)
    axes.set_xlabel("threshold u")
    axes.set_ylabel(parameter)
    axes.set_title(f"Parameter stability: {parameter}")
    return axes
