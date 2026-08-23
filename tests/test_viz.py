"""Rendering of the diagnostics.

These tests assert structure rather than appearance: that each function draws
the data it says it draws, labels the axes, honours a supplied ``ax``, and
returns it so plots compose. Comparing pixels would be brittle and would not
catch the mistakes that matter, which are plotting the wrong series or
silently dropping points.

matplotlib is an optional extra, so the whole module skips without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip(
    "matplotlib", reason="matplotlib is the optional 'plot' extra"
)
matplotlib.use("Agg")  # No display in CI.
import matplotlib.pyplot as plt  # noqa: E402

from heavytails import Pareto  # noqa: E402
from heavytails.extra_distributions import GeneralizedPareto  # noqa: E402
import heavytails.plotting as primitives  # noqa: E402
from heavytails.plotting import qq_pareto, tail_loglog_plot  # noqa: E402
from heavytails.tail_index import hill_plot, trimmed_hill_plot  # noqa: E402
from heavytails.viz import (  # noqa: E402
    _INSTALL_HINT,
    plot_hill,
    plot_mean_residual_life,
    plot_parameter_stability,
    plot_qq,
    plot_tail,
    plot_trimmed_hill,
)

SWEEP = [1.9, 2.2, 2.8, 3.6, 5.0, 7.5, 12.0]


@pytest.fixture(scope="module")
def data() -> list[float]:
    return Pareto(alpha=2.0, xm=1.0).rvs(3000, seed=1)


@pytest.fixture(autouse=True)
def _close_figures():
    """Close figures after each test so the run does not leak them."""
    yield
    plt.close("all")


class TestPlotTail:
    def test_draws_every_point_from_the_primitive(self, data: list[float]) -> None:
        """The rendering must not quietly drop or resample the data."""
        ax = plot_tail(data)
        line = ax.get_lines()[0]
        assert len(line.get_xdata()) == len(tail_loglog_plot(data))

    def test_labels_the_axes(self, data: list[float]) -> None:
        ax = plot_tail(data)
        assert ax.get_xlabel() == "log x"
        assert ax.get_ylabel() == "log P(X > x)"
        assert ax.get_title()

    def test_overlays_a_fitted_distribution(self, data: list[float]) -> None:
        """The comparison a goodness-of-fit number cannot make for you."""
        ax = plot_tail(data, fitted=Pareto(alpha=2.0, xm=1.0))
        assert len(ax.get_lines()) == 2
        labels = [line.get_label() for line in ax.get_lines()]
        assert "empirical" in labels
        assert any("Pareto" in str(label) for label in labels)

    def test_without_a_fit_there_is_one_series(self, data: list[float]) -> None:
        assert len(plot_tail(data).get_lines()) == 1

    def test_the_overlay_is_the_model_curve_not_a_sample_from_it(
        self, data: list[float]
    ) -> None:
        """The reference must be the survival function, evaluated.

        It used to be an empirical curve of ``len(data)`` draws from the fitted
        model. That put Monte Carlo noise into the line the reader compares
        against, worst in the far tail where its last points rested on a
        handful of observations: for Pareto(alpha=2) at n=1000 the reference
        wandered up to 1.238 in log survival away from the curve it claimed to
        be -- a factor of three, in the region the plot exists to show, which
        reads as misfit and is not.
        """
        fitted = Pareto(alpha=2.0, xm=1.0)
        reference = plot_tail(data, fitted=fitted).get_lines()[1]
        x = np.asarray(reference.get_xdata(), dtype=float)
        y = np.asarray(reference.get_ydata(), dtype=float)

        expected = np.log(np.asarray(fitted.sf(np.exp(x)), dtype=float))
        np.testing.assert_allclose(y, expected, rtol=1e-13, atol=1e-15)

    def test_the_overlay_is_monotone_and_does_not_depend_on_a_seed(
        self, data: list[float]
    ) -> None:
        fitted = Pareto(alpha=2.0, xm=1.0)
        first = np.asarray(plot_tail(data, fitted=fitted).get_lines()[1].get_ydata())
        second = np.asarray(plot_tail(data, fitted=fitted).get_lines()[1].get_ydata())
        np.testing.assert_array_equal(first, second)
        assert np.all(np.diff(first) <= 1e-12), "a survival curve cannot rise"

    def test_a_bounded_model_drops_the_points_beyond_its_support(
        self, data: list[float]
    ) -> None:
        """``log(0)`` is not a point on the plot.

        A model with an upper endpoint inside the data's range has zero
        survival above it, and those x are simply not drawn rather than
        plotted at negative infinity.
        """
        bounded = GeneralizedPareto(xi=-0.5, sigma=1.0, mu=0.0)
        y = np.asarray(plot_tail(data, fitted=bounded).get_lines()[1].get_ydata())
        assert y.size > 0
        assert np.all(np.isfinite(y))

    def test_draws_on_a_supplied_axes(self, data: list[float]) -> None:
        _, ax = plt.subplots()
        returned = plot_tail(data, ax=ax)
        assert returned is ax
        assert len(ax.get_lines()) == 1

    def test_style_keywords_reach_matplotlib(self, data: list[float]) -> None:
        ax = plot_tail(data, color="red", markersize=5)
        line = ax.get_lines()[0]
        assert line.get_markersize() == 5


class TestPlotQQ:
    def test_draws_every_point(self, data: list[float]) -> None:
        ax = plot_qq(data)
        assert len(ax.get_lines()[0].get_xdata()) == len(qq_pareto(data))

    def test_labels_the_axes(self, data: list[float]) -> None:
        ax = plot_qq(data)
        assert ax.get_xlabel() == "log(i/n)"
        assert ax.get_ylabel() == "log x"

    def test_draws_on_a_supplied_axes(self, data: list[float]) -> None:
        _, ax = plt.subplots()
        assert plot_qq(data, ax=ax) is ax


class TestPlotHill:
    def test_draws_the_sweep(self, data: list[float]) -> None:
        ax = plot_hill(data)
        assert len(ax.get_lines()[0].get_xdata()) == len(hill_plot(data))

    def test_uses_a_logarithmic_x_axis(self, data: list[float]) -> None:
        """The interesting structure is at small k, so the sweep is log-spaced."""
        assert plot_hill(data).get_xscale() == "log"

    def test_reference_line_is_drawn_when_the_truth_is_known(
        self, data: list[float]
    ) -> None:
        ax = plot_hill(data, true_gamma=0.5)
        assert len(ax.get_lines()) == 2
        assert ax.get_legend() is not None

    def test_no_reference_line_by_default(self, data: list[float]) -> None:
        assert len(plot_hill(data).get_lines()) == 1

    def test_accepts_explicit_k_values(self, data: list[float]) -> None:
        ax = plot_hill(data, ks=[20, 50, 100])
        assert list(ax.get_lines()[0].get_xdata()) == [20, 50, 100]


class TestPlotTrimmedHill:
    def test_draws_the_trim_sweep(self, data: list[float]) -> None:
        ax = plot_trimmed_hill(data, k=200, max_trim=6)
        assert len(ax.get_lines()[0].get_xdata()) == len(
            trimmed_hill_plot(data, 200, max_trim=6)
        )

    def test_starts_at_no_trimming(self, data: list[float]) -> None:
        ax = plot_trimmed_hill(data, k=200, max_trim=6)
        assert ax.get_lines()[0].get_xdata()[0] == 0

    def test_title_records_the_k_used(self, data: list[float]) -> None:
        assert "200" in plot_trimmed_hill(data, k=200, max_trim=4).get_title()


class TestPlotMeanResidualLife:
    def test_draws_a_curve_and_a_band(self, data: list[float]) -> None:
        """The band is the point: judge linearity where it is narrow."""
        ax = plot_mean_residual_life(data, thresholds=SWEEP)
        assert len(ax.get_lines()) >= 1
        assert len(ax.collections) >= 1  # the fill_between band

    def test_labels_the_axes(self, data: list[float]) -> None:
        ax = plot_mean_residual_life(data, thresholds=SWEEP)
        assert ax.get_xlabel() == "threshold u"
        assert ax.get_ylabel() == "mean excess"

    def test_the_band_level_is_shown(self, data: list[float]) -> None:
        ax = plot_mean_residual_life(data, thresholds=SWEEP, level=0.9)
        assert "90%" in ax.get_legend().get_texts()[0].get_text()


class TestPlotParameterStability:
    @pytest.mark.parametrize("parameter", ["xi", "sigma", "modified_scale"])
    def test_draws_each_parameter(self, data: list[float], parameter: str) -> None:
        ax = plot_parameter_stability(data, thresholds=SWEEP, parameter=parameter)
        assert ax.get_ylabel() == parameter
        assert len(ax.get_lines()[0].get_xdata()) > 0

    def test_rejects_an_unknown_parameter(self, data: list[float]) -> None:
        with pytest.raises(ValueError, match="Available"):
            plot_parameter_stability(data, thresholds=SWEEP, parameter="nonsense")

    def test_draws_on_a_supplied_axes(self, data: list[float]) -> None:
        _, ax = plt.subplots()
        assert plot_parameter_stability(data, thresholds=SWEEP, ax=ax) is ax


class TestComposition:
    def test_plots_share_a_figure(self, data: list[float]) -> None:
        """Returning the axes is what makes a panel of diagnostics possible."""
        fig, axes = plt.subplots(2, 2)
        plot_tail(data, ax=axes[0][0])
        plot_qq(data, ax=axes[0][1])
        plot_hill(data, ax=axes[1][0])
        plot_mean_residual_life(data, thresholds=SWEEP, ax=axes[1][1])
        assert all(ax.get_title() for row in axes for ax in row)
        assert len(fig.get_axes()) == 4


class TestOptionalDependency:
    def test_the_library_imports_without_touching_matplotlib(self) -> None:
        """heavytails.plotting must stay free of third-party imports.

        The promise is that installing the library never requires matplotlib,
        so the module that returns coordinates cannot import it even
        indirectly.
        """
        source = Path(primitives.__file__).read_text(encoding="utf-8")
        assert "matplotlib" not in source

    def test_the_install_hint_names_the_extra(self) -> None:
        assert "heavytails[plot]" in _INSTALL_HINT
