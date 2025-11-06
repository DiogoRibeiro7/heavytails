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
    Discrete heavy-tailed distributions (Zipf, Yule–Simon, Discrete Pareto)
- heavytails.tail_index
    Tail index estimators (Hill, Pickands, Moment)
- heavytails.plotting
    Diagnostic utilities (log–log tails, QQ plots)

Author: Diogo Ribeiro
License: MIT
"""

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
from .tail_index import hill_estimator, moment_estimator, pickands_estimator

__all__ = [
    # Continuous
    "Pareto", "Cauchy", "StudentT", "LogNormal",
    "Weibull", "Frechet", "GEV_Frechet",
    "GeneralizedPareto", "BurrXII", "LogLogistic",
    "InverseGamma", "BetaPrime",

    # Discrete
    "Zipf", "YuleSimon", "DiscretePareto",

    # Tail estimators
    "hill_estimator", "pickands_estimator", "moment_estimator",
]
