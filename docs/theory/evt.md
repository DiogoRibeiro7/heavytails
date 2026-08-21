# Extreme Value Theory

Extreme value theory (EVT) is the branch of probability that describes the
behaviour of the largest observations in a sample. Where the central limit
theorem tells you what happens to *averages*, EVT tells you what happens to
*maxima* — and the two answers are governed by different limit laws.

This page sets out the two classical frameworks, block maxima and peaks over
threshold, and shows which `heavytails` distributions implement each.

## Why a separate theory

Estimating a 99.9% quantile by taking the 99.9th percentile of a sample of 1000
observations means reading off the single largest value. That estimate has
enormous variance, and it can never exceed what you have already seen. EVT
replaces that dead end with a parametric model for the tail, fitted to the
observations that actually carry tail information.

The payoff is extrapolation: a fitted tail model can quote a 1-in-500-year loss
from 30 years of data, with an honest confidence interval.

## Block maxima and the Fisher–Tippett–Gnedenko theorem

Take independent, identically distributed $X_1, \dots, X_n$ and let
$M_n = \max(X_1, \dots, X_n)$. If there are norming constants $a_n > 0$ and
$b_n$ such that

$$
P\!\left(\frac{M_n - b_n}{a_n} \le x\right) \longrightarrow G(x)
$$

for a non-degenerate $G$, then $G$ must be the **generalized extreme value**
(GEV) distribution:

$$
G_\xi(x) = \exp\!\left\{-\left(1 + \xi x\right)^{-1/\xi}\right\},
\qquad 1 + \xi x > 0
$$

with the $\xi \to 0$ case read as $\exp(-e^{-x})$.

This is the Fisher–Tippett–Gnedenko theorem, and it is the reason EVT is
practical: whatever the parent distribution, the limit has just *one* shape
parameter.

### The three domains of attraction

The sign of $\xi$ splits every distribution into one of three families:

| $\xi$      | Limit law | Tail                            | Example parents             |
| ---------- | --------- | ------------------------------- | --------------------------- |
| $\xi > 0$  | Fréchet   | Power law, infinite upper bound | Pareto, Cauchy, Student-t   |
| $\xi = 0$  | Gumbel    | Exponential-ish, unbounded      | Normal, exponential, log-normal |
| $\xi < 0$  | Weibull   | Finite upper endpoint           | Uniform, beta               |

**`heavytails` is concerned with the Fréchet domain, $\xi > 0$.** The tail index
is $\alpha = 1/\xi$: larger $\xi$ means a heavier tail, and moments of order
$\ge \alpha$ do not exist.

Note the terminology trap: the *Weibull* limit law ($\xi < 0$) is a
short-tailed case, while the *Weibull distribution* with shape $k < 1$ is a
heavy-tailed parent that sits in the Gumbel domain. `heavytails` implements the
latter.

### In the library

```python
from heavytails import GEV_Frechet

# The Frechet branch of the GEV, parameterised by xi > 0.
gev = GEV_Frechet(xi=0.5, mu=0.0, sigma=1.0)
gev.ppf(0.99)
```

`xi=0.5` corresponds to a tail index $\alpha = 2$: the mean exists, the variance
does not.

## Peaks over threshold and the Pickands–Balkema–de Haan theorem

Block maxima discard almost all the data — one value per block. The peaks over
threshold (POT) approach keeps every observation that exceeds a high threshold
$u$, which is far more efficient.

The **Pickands–Balkema–de Haan theorem** says that as $u$ approaches the right
endpoint, the conditional excess distribution

$$
F_u(y) = P(X - u \le y \mid X > u)
$$

converges to the **generalized Pareto distribution** (GPD):

$$
H_{\xi,\sigma}(y) = 1 - \left(1 + \frac{\xi y}{\sigma}\right)^{-1/\xi},
\qquad y > 0
$$

The shape parameter $\xi$ is the *same* $\xi$ as in the GEV limit. Whichever
framework you use, you are estimating one number.

### In the library

```python
from heavytails import GeneralizedPareto

excesses = GeneralizedPareto(xi=0.5, sigma=1.0, mu=0.0)
excesses.sf(10.0)     # P(excess > 10)
excesses.ppf(0.99)    # 99% quantile of the excess distribution
```

### Choosing the threshold

Threshold selection is the bias–variance trade-off that dominates POT analysis:

- **Too low** — observations from the body of the distribution contaminate the
  fit, and the GPD approximation is biased.
- **Too high** — few exceedances remain and the estimate is noisy.

The standard tools are the mean residual life plot (linear above a valid
threshold) and the stability plot (parameter estimates flat above a valid
threshold). In practice, the log–log survival plot described in
[Tail Diagnostics](../guide/diagnostics.md) is a good first look: pick a
threshold where the plot becomes linear.

## Return levels

Once a tail model is fitted, the quantity practitioners actually want is the
**return level** — the value exceeded once every $T$ observations on average:

$$
x_T = u + \frac{\sigma}{\xi}\left[\left(\frac{T \, n_u}{n}\right)^{\xi} - 1\right]
$$

where $n_u$ is the number of exceedances of $u$ among $n$ observations. This is
the calculation behind a "1-in-100-year flood" or a regulatory value at risk.

Because $x_T$ grows like $T^{\xi}$, a small error in $\xi$ becomes a large error
in the return level. Reporting a confidence interval for $\xi$ is not optional.

## Practical consequences

- **Above $\alpha = 2$ the variance is infinite**, so sample standard deviations,
  and every method built on them, stop converging. See
  [Heavy-Tail Theory](heavy-tails.md).
- **Above $\alpha = 1$ the mean is infinite**, so sample averages wander without
  settling.
- **Aggregation does not tame the tail.** The sum of heavy-tailed variables is
  dominated by its largest term, which is why diversification arguments that
  assume finite variance fail here.

## Assumptions worth checking

The classical results above assume independent, identically distributed
observations. Real series often violate both:

- **Serial dependence** — clustered extremes inflate apparent exceedance rates.
  The extremal index quantifies the clustering and rescales return periods.
- **Non-stationarity** — trends and seasonality mean the tail itself moves, and
  the parameters must be modelled as functions of covariates.

`heavytails` implements the i.i.d. theory. Treat its output on dependent or
non-stationary data as a first approximation.

## References

- de Haan, L., & Ferreira, A. (2006). *Extreme Value Theory: An Introduction*.
  Springer.
- Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). *Modelling Extremal
  Events for Insurance and Finance*. Springer.
- Coles, S. (2001). *An Introduction to Statistical Modeling of Extreme Values*.
  Springer.
- Balkema, A. A., & de Haan, L. (1974). Residual life time at great age.
  *The Annals of Probability*, 2(5), 792–804.
- Pickands, J. (1975). Statistical inference using extreme order statistics.
  *The Annals of Statistics*, 3(1), 119–131.

## See also

- [Heavy-Tail Theory](heavy-tails.md) — regular variation and moment conditions
- [Tail Index Estimation Theory](tail-estimation.md) — estimating $\xi$
- [Tail Diagnostics](../guide/diagnostics.md) — checking the assumptions
