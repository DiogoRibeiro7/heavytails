"""
heavytails
==========

Python library of heavy-tailed probability distributions.

This package implements continuous and discrete heavy-tailed distributions,
tail index estimators, and diagnostic utilities with NumPy-backed vectorized
evaluation.

Modules
--------
- heavytails.heavy_tails
    Core continuous heavy-tailed distributions (Pareto, Cauchy, Student-t, etc.)
- heavytails.extra_distributions
    Additional continuous families (GPD, Burr XII, Log-Logistic, Inverse-Gamma, Beta-Prime)
- heavytails.discrete
    Discrete heavy-tailed distributions (Zipf, Yule-Simon, Discrete Pareto)
- heavytails.tail_index
    Tail index estimators (Hill, Pickands, Moment)
- heavytails.plotting
    Diagnostic coordinates (log-log tails, QQ plots)
- heavytails.viz
    Rendering of those diagnostics, needing the optional ``plot`` extra
- heavytails.threshold
    Threshold selection for peaks-over-threshold analysis
- heavytails.risk
    Tail-risk metrics (VaR, expected shortfall)
- heavytails.actuarial
    Compound distributions, aggregate losses and reinsurance pricing
- heavytails.streaming
    Tail index estimation over a stream, in bounded memory
- heavytails.vectorized
    Evaluation over many points at once, using NumPy
- heavytails.multivariate
    Elliptical families and tail dependence for joint heavy tails
- heavytails.copula
    Dependence separated from the margins, and joint-tail diagnostics
- heavytails.timeseries
    GARCH, standardised residuals and the extremal index

Author: Diogo Ribeiro
License: MIT
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from ._special import ConvergenceError
from .discrete import DiscretePareto, YuleSimon, Zipf
from .extra_distributions import (
    BetaPrime,
    BurrXII,
    GeneralizedPareto,
    InverseGamma,
    LogLogistic,
)
from .heavy_tails import (
    Cauchy,
    Frechet,
    GEV_Frechet,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)
from .risk import (
    expected_shortfall,
    mean_exists,
    monte_carlo_tail_risk,
    tail_conditional_expectation,
    value_at_risk,
)
from .tail_index import (
    bias_reduced_hill_estimator,
    fit_generalized_pareto,
    generalized_hill_estimator,
    gpd_mle_estimator,
    harmonic_moment_estimator,
    hill_estimator,
    hill_plot,
    moment_estimator,
    orthogonalized_bias_reduced_hill_estimator,
    pickands_estimator,
    recommended_rho_k,
    second_order_beta,
    second_order_rho,
    smoothed_hill_estimator,
    smoothed_hill_variance_ratio,
    t_hill_estimator,
    tail_index_confidence_interval,
    threshold_averaged_orthogonalized_hill_estimator,
    threshold_averaged_orthogonalized_hill_selection,
    trimmed_hill_estimator,
    trimmed_hill_plot,
)
from .threshold import (
    mean_residual_life,
    parameter_stability,
    return_level,
    select_threshold,
)

try:
    __version__ = _version("heavytails")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "BetaPrime",
    "BurrXII",
    "Cauchy",
    "ConvergenceError",
    "DiscretePareto",
    "Frechet",
    "GEV_Frechet",
    "GeneralizedPareto",
    "InverseGamma",
    "LogLogistic",
    "LogNormal",
    "Pareto",
    "StudentT",
    "Weibull",
    "YuleSimon",
    "Zipf",
    "__version__",
    "bias_reduced_hill_estimator",
    "expected_shortfall",
    "fit_generalized_pareto",
    "generalized_hill_estimator",
    "gpd_mle_estimator",
    "harmonic_moment_estimator",
    "hill_estimator",
    "hill_plot",
    "mean_exists",
    "mean_residual_life",
    "moment_estimator",
    "monte_carlo_tail_risk",
    "orthogonalized_bias_reduced_hill_estimator",
    "parameter_stability",
    "pickands_estimator",
    "recommended_rho_k",
    "return_level",
    "second_order_beta",
    "second_order_rho",
    "select_threshold",
    "smoothed_hill_estimator",
    "smoothed_hill_variance_ratio",
    "t_hill_estimator",
    "tail_conditional_expectation",
    "tail_index_confidence_interval",
    "threshold_averaged_orthogonalized_hill_estimator",
    "threshold_averaged_orthogonalized_hill_selection",
    "trimmed_hill_estimator",
    "trimmed_hill_plot",
    "value_at_risk",
]
