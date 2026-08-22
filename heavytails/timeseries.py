"""Heavy tails that come from volatility clustering rather than from the shocks.

Financial returns are heavy tailed and they are volatility clustered, and the
second explains a large part of the first. Fitting a static distribution to raw
returns conflates the two, and the conflation is not small:

============================  ====================  ==================
Innovations                   Their tail index      Raw returns show
============================  ====================  ==================
Gaussian                      infinite              **4.50**
Student-t, 5 df               5                     **2.85**
============================  ====================  ==================

Measured on 60,000 simulated GARCH(1,1) returns with ``alpha = 0.10`` and
``beta = 0.88``. The first row is the one to sit with: the innovations have no
power-law tail whatsoever, and the returns they generate have a tail index of
4.5. Estimate a tail index on raw returns and a good part of what is measured
is the volatility process, not the shocks.

Which of the two you want depends on the question:

**The conditional tail** -- the innovation distribution -- answers "how large is
tomorrow's shock, given what volatility is now". Estimate it on the
standardised residuals.

**The unconditional tail** -- what the raw returns show -- answers "how large is
a return picked at random from the next decade". That is the right one for
long-horizon capital, and it is heavier.

There is a second consequence, and it breaks something more basic. Classical
extreme value theory assumes independent observations. Clustered extremes are
not independent, so a return period computed as though they were is wrong by a
factor: the **extremal index** measures it, and :func:`extremal_index`
estimates it. A value of 0.5 means extremes arrive in pairs, so a "hundred-year
event" happens half as often as its marginal probability suggests -- but when it
does, it brings another with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
import math
from typing import TYPE_CHECKING, Any

from heavytails.heavy_tails import RNG, ParameterError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "GARCH11",
    "decluster",
    "extremal_index",
    "fit_garch11",
]


@dataclass(frozen=True)
class GARCH11:
    """
    GARCH(1,1) with normal or Student-t innovations.

    The variance recursion is ``sigma2_t = omega + alpha r2_{t-1} + beta
    sigma2_{t-1}`` and the return is ``r_t = sigma_t z_t`` with ``z`` of unit
    variance. ``alpha + beta`` is the persistence: how much of today's
    volatility survives into tomorrow, and how slowly a shock decays.

    Args:
        omega: Constant in the variance recursion, positive.
        alpha: Weight on the last squared return, non-negative.
        beta: Weight on the last variance, non-negative.
        nu: Degrees of freedom of the innovations, above two, or None for
            normal innovations. The innovations are standardised to unit
            variance either way, so ``nu`` changes their shape and not their
            scale.

    Raises:
        ParameterError: If any parameter is out of range, or if
            ``alpha + beta >= 1``, where the unconditional variance does not
            exist and the process is not covariance stationary.

    Examples:
        >>> model = GARCH11(omega=1e-6, alpha=0.1, beta=0.85, nu=5.0)
        >>> round(model.persistence, 6)
        0.95
        >>> round(model.unconditional_variance * 1e6, 4)
        20.0
        >>> len(model.simulate(500, seed=1))
        500
    """

    omega: float
    alpha: float
    beta: float
    nu: float | None = None

    def __post_init__(self) -> None:
        if not (self.omega > 0.0) or not math.isfinite(self.omega):
            raise ParameterError("GARCH11 requires a finite omega > 0.")
        if self.alpha < 0.0 or self.beta < 0.0:
            raise ParameterError("alpha and beta must be non-negative.")
        if self.alpha + self.beta >= 1.0:
            raise ParameterError(
                f"alpha + beta must be below one for a stationary process; got "
                f"{self.alpha + self.beta:.6g}. Above one the unconditional "
                "variance does not exist."
            )
        if self.nu is not None and not (self.nu > 2.0):
            raise ParameterError(
                "nu must exceed two, below which the innovations have no "
                "variance to standardise."
            )

    @property
    def persistence(self) -> float:
        """``alpha + beta``: how much volatility survives to the next step."""
        return float(self.alpha + self.beta)

    @property
    def unconditional_variance(self) -> float:
        """``omega / (1 - alpha - beta)``, the long-run variance."""
        return float(self.omega / (1.0 - self.persistence))

    def _log_innovation_density(self, z: float) -> float:
        """Log density of the standardised innovation at ``z``."""
        if self.nu is None:
            return -0.5 * (math.log(2.0 * math.pi) + z * z)
        nu = self.nu
        return (
            math.lgamma(0.5 * (nu + 1.0))
            - math.lgamma(0.5 * nu)
            - 0.5 * math.log((nu - 2.0) * math.pi)
            - 0.5 * (nu + 1.0) * math.log1p(z * z / (nu - 2.0))
        )

    def _draw_innovation(self, rng: RNG) -> float:
        """One standardised innovation."""
        normal = rng.standard_normal()
        if self.nu is None:
            return normal
        mixing = rng.chisquare(self.nu) / self.nu
        raw = normal / math.sqrt(mixing)
        return raw / math.sqrt(self.nu / (self.nu - 2.0))

    def simulate(
        self, n: int, seed: int | None = None, burn_in: int = 1_000
    ) -> list[float]:
        """
        Simulate ``n`` returns.

        Args:
            n: Number of returns, positive.
            seed: Seed for reproducibility.
            burn_in: Steps discarded so the variance forgets its start. The
                default is generous; at a persistence of 0.99 a shock takes
                hundreds of steps to decay and too short a burn-in leaves the
                series still remembering its initial condition.

        Returns:
            The simulated returns.

        Raises:
            ValueError: If ``n`` is not positive or ``burn_in`` is negative.
        """
        if not isinstance(n, int) or n <= 0:
            raise ValueError("n must be a positive integer.")
        if burn_in < 0:
            raise ValueError("burn_in must not be negative.")
        rng = RNG(seed)
        variance = self.unconditional_variance
        out: list[float] = []
        for step in range(n + burn_in):
            value = math.sqrt(variance) * self._draw_innovation(rng)
            if step >= burn_in:
                out.append(value)
            variance = self.omega + self.alpha * value * value + self.beta * variance
        return out

    def conditional_variances(self, returns: Sequence[float]) -> list[float]:
        """
        The variance the model implies at each step, given the data.

        The recursion is deterministic once the returns are known, which is
        what makes a GARCH likelihood computable at all: the volatility is
        latent but not random given the past.

        Args:
            returns: The observed series.

        Returns:
            One variance per return, starting from the unconditional variance.

        Raises:
            ValueError: If ``returns`` is empty.
        """
        if not returns:
            raise ValueError("returns must not be empty.")
        variance = self.unconditional_variance
        variances = []
        for value in returns:
            variances.append(variance)
            variance = self.omega + self.alpha * value * value + self.beta * variance
        return variances

    def standardized_residuals(self, returns: Sequence[float]) -> list[float]:
        """
        ``r_t / sigma_t``: what is left once the volatility is divided out.

        **This is the series to estimate a tail index on.** The raw returns
        carry the innovation tail and the volatility dynamics together, and a
        Hill estimate of them measures a mixture of the two. Dividing by the
        fitted volatility leaves the innovations, whose tail is the one a
        conditional question is about.

        Args:
            returns: The observed series.

        Returns:
            The standardised residuals.
        """
        return [
            value / math.sqrt(variance)
            for value, variance in zip(
                returns, self.conditional_variances(returns), strict=True
            )
        ]

    def log_likelihood(self, returns: Sequence[float]) -> float:
        """
        Log likelihood of the series under this model.

        Args:
            returns: The observed series.

        Returns:
            The sum over steps of the innovation log density less half the log
            variance -- the Jacobian of dividing by the volatility.
        """
        total = 0.0
        for value, variance in zip(
            returns, self.conditional_variances(returns), strict=True
        ):
            total += self._log_innovation_density(
                value / math.sqrt(variance)
            ) - 0.5 * math.log(variance)
        return float(total)


def _nelder_mead(
    objective: Callable[[list[float]], float],
    start: list[float],
    max_iter: int = 2_000,
    tol: float = 1e-9,
) -> list[float]:
    """Minimise without derivatives.

    A GARCH likelihood has no closed-form gradient worth writing and only three
    or four parameters, which is exactly where a simplex method is the right
    tool: no derivatives to get wrong, and the dimension is far too small for
    its poor scaling to matter.
    """
    dim = len(start)
    simplex = [list(start)]
    for i in range(dim):
        point = list(start)
        point[i] += 0.5 if point[i] == 0.0 else 0.1 * abs(point[i]) + 0.1
        simplex.append(point)
    values = [objective(point) for point in simplex]

    for _ in range(max_iter):
        order = sorted(range(dim + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        if abs(values[-1] - values[0]) < tol * (abs(values[0]) + tol):
            break

        centroid = [sum(point[j] for point in simplex[:-1]) / dim for j in range(dim)]
        worst = simplex[-1]
        reflected = [centroid[j] + (centroid[j] - worst[j]) for j in range(dim)]
        reflected_value = objective(reflected)

        if reflected_value < values[0]:
            expanded = [
                centroid[j] + 2.0 * (centroid[j] - worst[j]) for j in range(dim)
            ]
            expanded_value = objective(expanded)
            if expanded_value < reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value < values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        else:
            contracted = [
                centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(dim)
            ]
            contracted_value = objective(contracted)
            if contracted_value < values[-1]:
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                best = simplex[0]
                simplex = [
                    [best[j] + 0.5 * (point[j] - best[j]) for j in range(dim)]
                    for point in simplex
                ]
                values = [objective(point) for point in simplex]
    return simplex[0]


def _unpack(parameters: list[float], student: bool) -> dict[str, float | None]:
    """Map unconstrained parameters onto a valid GARCH.

    The constraints -- positive omega, non-negative alpha and beta summing
    below one, nu above two -- are enforced by the parameterisation rather than
    by penalties, so the optimiser cannot wander into a region where the model
    does not exist and come back with a number.
    """
    omega = math.exp(parameters[0])
    persistence = 1.0 / (1.0 + math.exp(-parameters[1]))
    share = 1.0 / (1.0 + math.exp(-parameters[2]))
    alpha = persistence * share
    result: dict[str, float | None] = {
        "omega": omega,
        "alpha": alpha,
        "beta": persistence - alpha,
        "nu": None,
    }
    if student:
        result["nu"] = 2.0 + math.exp(parameters[3])
    return result


def fit_garch11(
    returns: Sequence[float],
    innovations: str = "t",
    max_iter: int = 2_000,
) -> dict[str, Any]:
    """
    Fit a GARCH(1,1) by maximum likelihood.

    Args:
        returns: The observed series, at least fifty observations.
        innovations: ``"t"`` for Student-t or ``"normal"``.
        max_iter: Maximum simplex iterations.

    Returns:
        ``model``, ``log_likelihood``, ``aic``, ``bic`` and ``n``.

    Raises:
        ValueError: If the series is too short, has no variation, or
            ``innovations`` is neither option.

    Examples:
        >>> truth = GARCH11(omega=2e-6, alpha=0.08, beta=0.90, nu=6.0)
        >>> fit = fit_garch11(truth.simulate(4000, seed=1))
        >>> 0.90 < fit["model"].persistence < 0.995
        True
    """
    if innovations not in {"t", "normal"}:
        raise ValueError(f"unknown innovations {innovations!r}. Available: t, normal")
    values = list(returns)
    if len(values) < 50:
        raise ValueError(f"at least 50 observations are needed; got {len(values)}.")
    variance = sum(value * value for value in values) / len(values)
    if variance <= 0.0:
        raise ValueError("the series has no variation to fit.")

    student = innovations == "t"
    start = [math.log(variance * 0.05), 2.2, -2.0]
    if student:
        start.append(math.log(4.0))

    def negative_log_likelihood(parameters: list[float]) -> float:
        try:
            model = GARCH11(**_unpack(parameters, student))  # type: ignore[arg-type]
            return -model.log_likelihood(values)
        except (ParameterError, ValueError, OverflowError):
            return math.inf

    best = _nelder_mead(negative_log_likelihood, start, max_iter=max_iter)
    model = GARCH11(**_unpack(best, student))  # type: ignore[arg-type]
    likelihood = model.log_likelihood(values)
    count = 4 if student else 3
    return {
        "model": model,
        "log_likelihood": likelihood,
        "aic": 2.0 * count - 2.0 * likelihood,
        "bic": count * math.log(len(values)) - 2.0 * likelihood,
        "n": len(values),
    }


def extremal_index(
    data: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    """
    Estimate the extremal index by the Ferro-Segers intervals method.

    The extremal index is the reciprocal of the mean cluster size, and it is
    what classical extreme value theory quietly assumes to be one. A value of
    0.5 means extremes arrive in pairs: the **rate** of independent episodes is
    half what counting exceedances suggests, so a "hundred-year event" happens
    half as often -- and brings a companion when it does.

    The intervals estimator works from the gaps between exceedances and needs
    **no declustering parameter**, which is its advantage over the runs method:
    a run length has to be chosen, the answer depends on the choice, and there
    is rarely a principled way to make it.

    Args:
        data: The observed series.
        threshold: Exceedances above this define the extremes.

    Returns:
        ``extremal_index``, ``n_exceedances``, ``mean_cluster_size`` and the
        ``branch`` of the estimator that applied.

    Raises:
        ValueError: If fewer than three exceedances are found, where there are
            at most two gaps and nothing to estimate from.

    Examples:
        Independent data has an extremal index of one, which the estimator
        recovers:

        >>> from heavytails import Pareto
        >>> sample = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=1)
        >>> result = extremal_index(sample, threshold=8.0)
        >>> result["extremal_index"] > 0.85
        True
    """
    positions = [i for i, value in enumerate(data) if value > threshold]
    if len(positions) < 3:
        raise ValueError(
            f"the threshold {threshold!r} leaves {len(positions)} exceedances; "
            "at least three are needed for two inter-exceedance times."
        )

    gaps = [float(later - earlier) for earlier, later in pairwise(positions)]
    count = len(positions)

    if max(gaps) > 2.0:
        # The bias-corrected branch, for data whose exceedances are not
        # adjacent. Subtracting one and two is the correction, and it is only
        # valid where the gaps can carry it.
        numerator = 2.0 * sum(gap - 1.0 for gap in gaps) ** 2
        denominator = (count - 1) * sum((gap - 1.0) * (gap - 2.0) for gap in gaps)
        branch = "bias-corrected"
    else:
        numerator = 2.0 * sum(gaps) ** 2
        denominator = (count - 1) * sum(gap * gap for gap in gaps)
        branch = "uncorrected"

    estimate = 1.0 if denominator <= 0.0 else min(1.0, numerator / denominator)

    return {
        "extremal_index": float(estimate),
        "n_exceedances": count,
        "mean_cluster_size": float(1.0 / estimate) if estimate > 0.0 else math.inf,
        "branch": branch,
    }


def decluster(
    data: Sequence[float],
    threshold: float,
    run_length: int = 1,
) -> dict[str, Any]:
    """
    Reduce clustered exceedances to one value per cluster.

    Exceedances separated by fewer than ``run_length`` non-exceedances are
    treated as one episode, and the largest is kept. That is what makes the
    remaining values approximately independent, which is what the classical
    peaks-over-threshold machinery in :mod:`heavytails.threshold` assumes.

    **The run length is a choice, and the answer depends on it.** That is the
    known weakness of the runs method, and the reason
    :func:`extremal_index` uses the intervals estimator instead. Comparing the
    cluster count here against ``extremal_index * n_exceedances`` is a cheap
    way to see whether the choice was reasonable.

    Args:
        data: The observed series.
        threshold: Exceedances above this define the extremes.
        run_length: Non-exceedances needed to separate two clusters.

    Returns:
        ``cluster_maxima``, ``cluster_sizes``, ``n_exceedances`` and
        ``n_clusters``.

    Raises:
        ValueError: If ``run_length`` is not positive.

    Examples:
        >>> series = [0.0, 5.0, 6.0, 0.0, 0.0, 0.0, 7.0, 0.0]
        >>> result = decluster(series, threshold=1.0, run_length=2)
        >>> result["cluster_maxima"]
        [6.0, 7.0]
        >>> result["n_exceedances"], result["n_clusters"]
        (3, 2)
    """
    if not isinstance(run_length, int) or run_length < 1:
        raise ValueError("run_length must be a positive integer.")

    positions = [i for i, value in enumerate(data) if value > threshold]
    if not positions:
        return {
            "cluster_maxima": [],
            "cluster_sizes": [],
            "n_exceedances": 0,
            "n_clusters": 0,
        }

    clusters: list[list[int]] = [[positions[0]]]
    for position in positions[1:]:
        if position - clusters[-1][-1] > run_length:
            clusters.append([position])
        else:
            clusters[-1].append(position)

    return {
        "cluster_maxima": [max(data[i] for i in cluster) for cluster in clusters],
        "cluster_sizes": [len(cluster) for cluster in clusters],
        "n_exceedances": len(positions),
        "n_clusters": len(clusters),
    }
