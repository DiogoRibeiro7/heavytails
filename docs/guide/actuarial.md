# Aggregate Loss and Reinsurance

An insurance portfolio produces a random number of claims of random size. What
matters is the total:

$$S = X_1 + X_2 + \dots + X_N$$

`heavytails.actuarial` builds the distribution of $S$ from a frequency model and
a severity distribution, and prices the reinsurance written on top of it.

## A worked portfolio

Take a commercial property book: about ten claims a year, over-dispersed, with a
Pareto severity attaching at 10,000 and a tail index of 1.9. Policies carry a
25,000 deductible and a 500,000 limit.

```python
from heavytails import Pareto
from heavytails.actuarial import (
    LayeredSeverity, NegativeBinomial, PolicyTerms,
    compound_moments, panjer_recursion,
)

frequency = NegativeBinomial(r=4.0, beta=2.5)      # mean 10, variance 35
severity = Pareto(alpha=1.9, xm=10_000.0)          # mean 21,111
terms = PolicyTerms(deductible=25_000.0, limit=500_000.0)
```

### The severity has a mean; the portfolio has no variance

```python
compound_moments(frequency, severity)
# (211111.11, inf)
```

With `alpha = 1.9` the severity has a mean but no second moment, so **the
aggregate variance does not exist**. That is not a numerical inconvenience. Every
normal and translated-gamma approximation in the actuarial literature works by
matching two moments, and there is no second moment here to match. Reporting a
large finite number instead would let someone reach for one of them.

The policy limit changes this. A capped severity is bounded, so it has moments of
every order:

```python
layer = LayeredSeverity(severity, terms)
compound_moments(frequency, layer)[1]   # finite
```

Capping is what makes the arithmetic possible, and it is worth noticing that this
is the same reason catastrophe reinsurance is always written with a limit.

### Pricing the retained layer

```python
layer.cdf(0.0)   # 0.8246 -- most losses never reach the deductible
layer.mean()     # 4,556.43 expected payment per loss
frequency.mean() * layer.mean()   # 45,564.30 expected annual cost
```

The layer mean is a difference of limited expected values,
$E[X \wedge 525{,}000] - E[X \wedge 25{,}000]$, which stays exact even when the
underlying severity is heavy enough to make direct integration awkward.

### Per-loss or per-payment

`LayeredSeverity` has two bases, and choosing the wrong one is the classic error
in this calculation.

| Basis | Atom at zero | Pairs with |
| --- | --- | --- |
| `per-loss` | yes, $F(d)$ | the original frequency |
| `per-payment` | no | the frequency **thinned** by $S(d)$ |

Both describe the same portfolio and give the same aggregate distribution:

```python
per_payment = LayeredSeverity(severity, terms, basis="per-payment")
thinned = frequency.thin(layer.exceedance_probability)

panjer_recursion(frequency, layer, h=2_000.0, n=4_000)
panjer_recursion(thinned, per_payment, h=2_000.0, n=4_000)   # identical
```

Mixing them — a per-payment severity with an unthinned frequency — overstates the
expected aggregate by $1/S(d)$, here a factor of 5.7. It produces a number that
looks entirely plausible.

## Two routes to the aggregate distribution

### Panjer recursion

Exact given a discretised severity, and it returns the whole distribution rather
than a sample:

```python
aggregate = panjer_recursion(
    frequency, layer, h=2_000.0, n=4_000, method="mean-preserving"
)
aggregate.mean()                    # 45,564.30, matching the closed form
aggregate.value_at_risk(0.99)       # 418,000
aggregate.expected_shortfall(0.99)  # 527,068
```

Two things go wrong with it, and both are reported rather than returned:

**The recursion can start at zero.** It begins from $g_0 = P_N(f_0)$, which for a
Poisson is $e^{\lambda(f_0-1)}$. For a large expected claim count that underflows
to exactly zero, and since every later probability is a multiple of it, the whole
output becomes zeros — a result that is not a distribution and not obviously
wrong. `panjer_recursion` raises instead.

**The grid can be too short.** A heavy-tailed severity puts real mass beyond any
finite grid. `AggregateLoss` carries `truncated_mass` and `severity_tail_mass`,
and `ppf` returns `inf` above `1 - truncated_mass` rather than the last grid
point, which would look like an answer.

A policy limit removes the problem entirely: the capped severity fits inside the
grid, and `truncated_mass` here is $3 \times 10^{-16}$.

### Simulation

No grid to truncate and no $g_0$ to underflow, at the cost of slower convergence:

```python
from heavytails.actuarial import EmpiricalAggregate, simulate_aggregate_loss

sample = EmpiricalAggregate(
    simulate_aggregate_loss(frequency, layer, 100_000, seed=1)
)
sample.mean()                  # 45,623
sample.value_at_risk(0.95)     # 182,201  (Panjer: 182,000)
```

`EmpiricalAggregate` exposes the same interface, so the two routes are
interchangeable and both work with [`heavytails.risk`](../reference/risk.md).

For a severity with no mean at all — `alpha` below 1 — simulation is the only
route that runs.

## Reinsurance

### Per-risk excess of loss

The reinsurer pays $\min(\max(X - r, 0), \ell)$ on each individual claim, so the
expected annual cost is a difference of limited expected values. No grid, no
simulation:

```python
from heavytails.actuarial import excess_of_loss_premium

excess_of_loss_premium(frequency, severity, retention=100_000, limit=500_000)
# 11,199.23
excess_of_loss_premium(frequency, severity, retention=100_000, limit=None)
# 13,988.06
```

Layers are additive: pricing 5 to 20 and 20 to 50 gives exactly the same total as
pricing 5 to 50.

On a severity with no mean the unlimited layer costs `inf` and the limited one
does not, which is the whole reason those covers carry limits.

!!! warning "This is the expected loss cost, not a premium"
    It carries no loading for expenses, risk margin or cost of capital. For a
    heavy-tailed layer the risk margin is usually the larger part of the price,
    precisely because the variance that would ordinarily size it does not exist.

### Aggregate stop loss

The reinsurer pays whatever the *total* exceeds a retention:

```python
aggregate.stop_loss_premium(1_000_000.0)   # 2.45
```

Because the truncated mass sits entirely above the retention, the gridded value is
a lower bound. `stop_loss_premium` returns `inf` when what fell off the end would
move the answer by more than the tolerance, which defaults to 1%. A heavy tail
always truncates something, so the check is relative rather than absolute — an
absolute one would make the function useless on exactly the distributions it is
for.

## Choosing a frequency model

| Model | Dispersion | Use when |
| --- | --- | --- |
| `Binomial` | variance below mean | a closed group, at most one claim per policy |
| `Poisson` | variance equals mean | the default |
| `NegativeBinomial` | variance above mean | the claim rate is itself uncertain |

All three are the $(a,b,0)$ class, where $p_k/p_{k-1} = a + b/k$ — which is
exactly the property that makes Panjer's recursion possible, so the selection is
not a matter of convenience.

The choice matters more than it appears. A Poisson with the same mean as the
negative binomial above matches the expected aggregate to within a rounding error
while understating the 99th percentile, so checking the mean will not catch it.
Over-dispersion is the usual case: a gamma-mixed Poisson *is* negative binomial,
so uncertainty about the rate produces it automatically.

## Discretisation

Panjer needs the severity on an arithmetic grid, and `h` is the whole accuracy
story.

- `"mass"` assigns each point the probability of the interval around it. Simple,
  fast, and slightly biased in the mean.
- `"mean-preserving"` matches local moments, so the discretised mean is exact.
  Prefer it for pricing, where an error in the mean is an error in the premium.

Mean-preserving takes a *second difference* of limited expected values, which
amplifies any error in them by $1/h$ and can push a weight negative. Clipping a
material amount would leave a biased grid that still sums to something plausible,
so `discretise_severity` raises instead.
