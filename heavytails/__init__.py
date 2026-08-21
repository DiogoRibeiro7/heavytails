"""
heavytails
==========

Pure-Python library of heavy-tailed probability distributions.

This package implements continuous and discrete heavy-tailed distributions,
tail index estimators, and diagnostic utilities — entirely dependency-free,
using only the Python standard library.

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
    Diagnostic utilities (log-log tails, QQ plots)

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
from .tail_index import (
    generalized_hill_estimator,
    hill_estimator,
    hill_plot,
    moment_estimator,
    pickands_estimator,
    smoothed_hill_estimator,
    smoothed_hill_variance_ratio,
    tail_index_confidence_interval,
    trimmed_hill_estimator,
    trimmed_hill_plot,
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
    "generalized_hill_estimator",
    "hill_estimator",
    "hill_plot",
    "moment_estimator",
    "pickands_estimator",
    "smoothed_hill_estimator",
    "smoothed_hill_variance_ratio",
    "tail_index_confidence_interval",
    "trimmed_hill_estimator",
    "trimmed_hill_plot",
]
