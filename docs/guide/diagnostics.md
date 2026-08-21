# Tail Diagnostics

Before fitting a distribution, it is worth asking whether the data is heavy-tailed
at all, and if so, roughly how heavy. Diagnostic plots answer that faster and more
honestly than a goodness-of-fit statistic, because they show *where* a model
disagrees with the data rather than only *how much*.

`heavytails.plotting` provides the two workhorse diagnostics. Both return plain
lists of `(x, y)` pairs, so the library keeps its promise of no runtime
dependencies — you plot the points with whatever you already use.

## Log–log survival plot

For a Pareto-type tail, the survival function decays as a power law:

$$
\bar{F}(x) = P(X > x) \sim C x^{-\alpha}, \qquad x \to \infty
$$

Taking logarithms turns that into a straight line:

$$
\log \bar{F}(x) = \log C - \alpha \log x
$$

So a heavy tail looks *linear* on log–log axes, with slope $-\alpha$. This is the
single most useful heavy-tail diagnostic.

```python
from heavytails import Pareto
from heavytails.plotting import tail_loglog_plot

samples = Pareto(alpha=1.5, xm=1.0).rvs(5000, seed=42)
points = tail_loglog_plot(samples)

# points is a list of (log x, log P(X > x)) pairs
points[:3]
```

`tail_loglog_plot` computes the empirical survival function
$\bar{F}_n(x_{(i)}) = (n - i)/n$ from the sorted sample and returns
$(\log x, \log \bar{F}_n(x))$ for every strictly positive observation. Values at
or below zero are dropped, since the logarithm is undefined there.

### Drawing it

`heavytails.viz` renders these directly, so you do not have to write the
matplotlib yourself. It needs the optional `plot` extra:

```bash
pip install "heavytails[plot]"
```

```python
from heavytails.viz import plot_tail

plot_tail(samples)
```

Every function takes an optional `ax` and returns it, so a panel of
diagnostics composes normally:

```python
import matplotlib.pyplot as plt
from heavytails.viz import plot_hill, plot_qq, plot_tail

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
plot_tail(samples, ax=axes[0])
plot_qq(samples, ax=axes[1])
plot_hill(samples, ax=axes[2], true_gamma=0.5)
```

The coordinate-returning functions stay available and dependency-free, so the
library itself never requires matplotlib.

### Reading the plot

| What you see                              | What it suggests                                     |
| ----------------------------------------- | ---------------------------------------------------- |
| A straight line over several decades       | A genuine power-law tail; the slope estimates $-\alpha$ |
| Curvature bending downwards                | A lighter tail than Pareto — log-normal or Weibull    |
| A straight body that bends only at the end | Finite-sample noise, not a change in the tail         |
| No linear region at all                    | The data is probably not heavy-tailed                 |

!!! warning "The last few points are noise"
    The extreme right of the plot rests on a handful of observations, so it
    scatters wildly. Judge linearity from the bulk of the tail, not from the
    final points, and never fit a line through the whole plot to estimate
    $\alpha$ — use the [tail index estimators](tail-estimation.md), which are
    designed for it.

## QQ plot against Pareto quantiles

A quantile–quantile plot compares the ordered sample against the quantiles a
reference distribution would produce. If the data follows the reference, the
points fall on a straight line.

```python
from heavytails.plotting import qq_pareto

qq_points = qq_pareto(samples)
```

`qq_pareto` returns $(\log(i/n), \log x_{(i)})$ pairs. Under a Pareto tail these
are linear, and departures show up as systematic curvature:

- **Points above the line at the right** — the sample has a heavier tail than the
  Pareto reference.
- **Points below the line at the right** — a lighter tail.
- **Curvature throughout** — the family is wrong, not merely mis-parameterised.

QQ plots are more sensitive than log–log plots to what happens in the body of the
distribution, so the two complement each other: use the log–log plot to judge the
tail, and the QQ plot to judge the fit as a whole.

## A practical workflow

1. Plot the log–log survival curve and look for a linear region.
2. If one exists, note where it starts — that is your candidate threshold.
3. Estimate the tail index over that region with a
   [Hill plot](tail-estimation.md#choosing-k).
4. [Fit candidate families](fitting.md) and compare them by AIC/BIC.
5. Return to the QQ plot to confirm the winner does not misbehave in the body.

Skipping step 1 is the most common way to end up confidently fitting a power law
to data that has none.

## Comparing against a fitted model

The diagnostics take any sequence of numbers, so you can run them on samples
drawn from a fitted model and overlay the two:

```python
from heavytails import Pareto
from heavytails.plotting import tail_loglog_plot

fitted = Pareto(alpha=1.53, xm=1.0)
model_points = tail_loglog_plot(fitted.rvs(5000, seed=1))
data_points = tail_loglog_plot(samples)
```

Plotting both on the same axes shows whether the fitted tail tracks the empirical
one, which is exactly the question a single goodness-of-fit number cannot answer.

## See also

- [Tail Index Estimation](tail-estimation.md) — turning the picture into a number
- [Parameter Fitting](fitting.md) — fitting and comparing families
- [Heavy-Tail Theory](../theory/heavy-tails.md) — why the log–log plot is linear
