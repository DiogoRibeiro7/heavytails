# Tail Index Estimation

Estimating the tail index from empirical data is fundamental to heavy-tail analysis. This guide covers the main estimation methods available in HeavyTails.

--------------------------------------------------------------------------------

## Overview

For heavy-tailed distributions with power-law tails:

$$ P(X > x) \sim Cx^{-\alpha} \quad \text{as } x \to \infty $$

the **tail index** $\alpha$ determines the heaviness of the tail. Estimating $\alpha$ from data allows you to:

- Assess the degree of tail heaviness
- Determine which moments exist
- Make inferences about extreme risks
- Choose appropriate parametric models

--------------------------------------------------------------------------------

## Available Estimators

Estimator    | Best For            | Pros                    | Cons
------------ | ------------------- | ----------------------- | ----------------
**Hill**     | Pareto-type tails   | Efficient, well-studied | Sensitive to $k$
**Pickands** | Extreme values      | Robust                  | Less efficient
**Moment**   | General heavy tails | Reduced bias            | Higher variance

--------------------------------------------------------------------------------

## Hill Estimator

The **Hill estimator** is the maximum likelihood estimator for the tail index under the Pareto-type assumption.

### Formula

$$ \hat{\gamma}_H = \frac{1}{k}\sum_{i=1}^k \ln\left(\frac{X_{(i)}}{X_{(k+1)}}\right) $$

where $X_{(1)} \geq X_{(2)} \geq \cdots \geq X_{(n)}$ are order statistics.

The tail index estimate is:

$$ \hat{\alpha} = \frac{1}{\hat{\gamma}_H} $$

### Usage

```python
from heavytails import Pareto
from heavytails.tail_index import hill_estimator

# Generate data from Pareto(α=2.5)
true_alpha = 2.5
pareto = Pareto(alpha=true_alpha, xm=1.0)
data = pareto.rvs(2000, seed=42)

# Estimate tail index using top k=200 order statistics
k = 200
gamma_hat = hill_estimator(data, k)
alpha_hat = 1.0 / gamma_hat

print(f"True α: {true_alpha:.2f}")
print(f"Estimated α: {alpha_hat:.2f}")
print(f"Relative error: {abs(alpha_hat - true_alpha)/true_alpha:.1%}")
```

**Output:**

```
True α: 2.50
Estimated α: 2.48
Relative error: 0.8%
```

### Choosing k

The number of upper order statistics $k$ is critical:

- **Too small $k$:** High variance (not enough data)
- **Too large $k$:** Bias (including non-tail observations)

**Rules of thumb:**

1. **Visual inspection:** Hill plot (see below)
2. **$k \approx \sqrt{n}$:** Simple heuristic
3. **$k = n/10$ to $n/4$:** Conservative range
4. **Cross-validation:** Minimize prediction error

### Hill Plot

Plot $\hat{\alpha}$ vs. $k$ to find stable region:

```python
from heavytails.tail_index import hill_estimator
import matplotlib.pyplot as plt

# Generate Hill estimates for different k
k_values = range(50, 500, 10)
alpha_estimates = []

for k in k_values:
    try:
        gamma = hill_estimator(data, k)
        alpha = 1.0 / gamma
        alpha_estimates.append(alpha)
    except:
        alpha_estimates.append(None)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, alpha_estimates, 'b-', linewidth=2)
plt.axhline(y=true_alpha, color='r', linestyle='--', label=f'True α={true_alpha}')
plt.xlabel('k (number of order statistics)', fontsize=12)
plt.ylabel('Hill estimate of α', fontsize=12)
plt.title('Hill Plot', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

Look for a **plateau** where the estimate is stable across different $k$ values.

--------------------------------------------------------------------------------

## Pickands Estimator

The **Pickands estimator** uses ratios of order statistics at different spacings.

### Formula

$$ \hat{\gamma}_P = \frac{1}{\ln m} \ln\left(\frac{X_{(k)} - X_{(2k)}}{X_{(mk)} - X_{(2mk)}}\right) $$

Typically $m=2$:

$$ \hat{\gamma}_P = \frac{1}{\ln 2} \ln\left(\frac{X_{(k)} - X_{(2k)}}{X_{(2k)} - X_{(4k)}}\right) $$

### Usage

```python
from heavytails.tail_index import pickands_estimator

# Pickands estimate
k = 200
m = 2
gamma_pickands = pickands_estimator(data, k, m)
alpha_pickands = 1.0 / gamma_pickands

print(f"Pickands estimate: α = {alpha_pickands:.2f}")
```

### Advantages

- **More robust** to departures from exact Pareto
- **Less sensitive** to choice of $k$
- **Theoretical guarantees** for broader distribution classes

### Disadvantages

- **Higher variance** than Hill estimator
- **Requires larger sample** (needs $4mk$ observations)

--------------------------------------------------------------------------------

## Moment Estimator

The **Dekkers-Einmahl-de Haan moment estimator** uses second-order tail behavior.

### Formula

$$ M_1^{(n)} = \frac{1}{k}\sum_{i=1}^k \ln\left(\frac{X_{(i)}}{X_{(k+1)}}\right) $$

$$ M_2^{(n)} = \frac{1}{k}\sum_{i=1}^k \left[\ln\left(\frac{X_{(i)}}{X_{(k+1)}}\right)\right]^2 $$

$$ \hat{\gamma}_M = M_1 + 1 - \frac{1}{2}\left(1 - \frac{M_1^2}{M_2}\right)^{-1} $$

### Usage

```python
from heavytails.tail_index import moment_estimator

# Moment estimator returns both γ and α
k = 200
gamma_moment, alpha_moment = moment_estimator(data, k)

print(f"Moment estimate: α = {alpha_moment:.2f}")
```

### Advantages

- **Reduced bias** compared to Hill
- **Better for second-order regular variation**
- **Asymptotic normality** under weaker conditions

### Disadvantages

- **Higher variance** than Hill in some cases
- **More complex formula**

--------------------------------------------------------------------------------

## Comparing Estimators

```python
from heavytails import Pareto
from heavytails.tail_index import hill_estimator, pickands_estimator, moment_estimator

# Generate data
true_alpha = 3.0
pareto = Pareto(alpha=true_alpha, xm=1.0)
data = pareto.rvs(5000, seed=42)

# Estimate with all three methods
k = 500

gamma_hill = hill_estimator(data, k)
alpha_hill = 1.0 / gamma_hill

gamma_pickands = pickands_estimator(data, k, m=2)
alpha_pickands = 1.0 / gamma_pickands

gamma_moment, alpha_moment = moment_estimator(data, k)

# Compare
print(f"True tail index: {true_alpha:.2f}")
print(f"Hill:            {alpha_hill:.2f} (error: {abs(alpha_hill-true_alpha):.3f})")
print(f"Pickands:        {alpha_pickands:.2f} (error: {abs(alpha_pickands-true_alpha):.3f})")
print(f"Moment:          {alpha_moment:.2f} (error: {abs(alpha_moment-true_alpha):.3f})")
```

**Typical Output:**

```
True tail index: 3.00
Hill:            2.98 (error: 0.020)
Pickands:        3.12 (error: 0.118)
Moment:          2.95 (error: 0.048)
```

--------------------------------------------------------------------------------

## Practical Guidelines

### Sample Size Requirements

Sample Size         | Recommended Approach
------------------- | -------------------------------------------------
$n < 100$           | Estimation unreliable; use parametric assumptions
$100 \leq n < 500$  | Use Hill with small $k$ ($\approx 20-50$)
$500 \leq n < 2000$ | Hill plot, choose stable $k$
$n \geq 2000$       | All estimators viable; compare results

### Workflow for Tail Index Estimation

1. **Visualize data**

  ```python
  import matplotlib.pyplot as plt
  import numpy as np

  # Log-log survival plot
  sorted_data = sorted(data, reverse=True)
  n = len(sorted_data)
  survival = [i/n for i in range(1, n+1)]

  plt.loglog(sorted_data[1:100], survival[1:100], 'o-')
  plt.xlabel('x')
  plt.ylabel('P(X > x)')
  plt.title('Log-Log Tail Plot')
  plt.grid(True, which='both', alpha=0.3)
  plt.show()
  ```

2. **Create Hill plot** (as shown above)

3. **Select $k$** based on stable region

4. **Estimate with multiple methods**

5. **Check sensitivity** to $k$ choice

6. **Report uncertainty** (bootstrap if needed)

### Example: Complete Analysis

```python
from heavytails.tail_index import hill_estimator, pickands_estimator, moment_estimator
import statistics

# Generate data (unknown distribution in practice)
from heavytails import Pareto
data = Pareto(alpha=2.5, xm=1.0).rvs(3000, seed=42)

# Explore different k values
k_range = [100, 200, 300, 400, 500]

results = []
for k in k_range:
    gamma_h = hill_estimator(data, k)
    gamma_p = pickands_estimator(data, k, m=2)
    gamma_m, alpha_m = moment_estimator(data, k)

    results.append({
        'k': k,
        'hill': 1.0 / gamma_h,
        'pickands': 1.0 / gamma_p,
        'moment': alpha_m
    })

# Display results
print("k\tHill\tPickands\tMoment")
for r in results:
    print(f"{r['k']}\t{r['hill']:.2f}\t{r['pickands']:.2f}\t\t{r['moment']:.2f}")

# Final estimate: median of Hill estimates
hill_estimates = [r['hill'] for r in results]
final_estimate = statistics.median(hill_estimates)
print(f"\nFinal estimate (median of Hill): α = {final_estimate:.2f}")
```

--------------------------------------------------------------------------------

## Confidence Intervals

### Asymptotic Confidence Interval (Hill)

Under regularity conditions:

$$ \hat{\alpha} \pm z_{\alpha/2} \cdot \frac{\hat{\alpha}}{\sqrt{k}} $$

where $z_{\alpha/2}$ is the standard normal quantile (e.g., 1.96 for 95%).

```python
import math

alpha_hat = 1.0 / hill_estimator(data, k=200)
k = 200

# 95% confidence interval
z = 1.96
se = alpha_hat / math.sqrt(k)
ci_lower = alpha_hat - z * se
ci_upper = alpha_hat + z * se

print(f"Estimate: {alpha_hat:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

### Bootstrap Confidence Interval

More robust approach:

```python
import random

def bootstrap_hill(data, k, n_bootstrap=1000, seed=42):
    """Bootstrap confidence interval for Hill estimator."""
    random.seed(seed)
    n = len(data)

    estimates = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = [data[random.randint(0, n-1)] for _ in range(n)]
        gamma = hill_estimator(bootstrap_sample, k)
        estimates.append(1.0 / gamma)

    # Percentile method
    estimates.sort()
    ci_lower = estimates[int(0.025 * n_bootstrap)]
    ci_upper = estimates[int(0.975 * n_bootstrap)]

    return ci_lower, ci_upper

ci_lower, ci_upper = bootstrap_hill(data, k=200)
print(f"Bootstrap 95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")
```

--------------------------------------------------------------------------------

## Special Cases

### Student-t Data

For Student-t($\nu$), the tail index is $\alpha = \nu$:

```python
from heavytails import StudentT
from heavytails.tail_index import hill_estimator

# Generate Student-t data
true_nu = 4.0
student = StudentT(nu=true_nu)
data = [abs(x) for x in student.rvs(2000, seed=42)]  # Take absolute values

# Estimate
gamma = hill_estimator(data, k=200)
alpha_hat = 1.0 / gamma

print(f"True ν (tail index): {true_nu:.2f}")
print(f"Estimated: {alpha_hat:.2f}")
```

### Mixed Data

If data comes from a mixture, estimate corresponds to the heaviest component:

```python
# Mixture: 90% Normal, 10% Pareto
import random
random.seed(42)

normal_data = [random.gauss(0, 1) for _ in range(900)]
pareto_data = Pareto(alpha=2.0, xm=1.0).rvs(100, seed=42)
mixed_data = normal_data + pareto_data
random.shuffle(mixed_data)

# Estimate captures Pareto component
gamma = hill_estimator([abs(x) for x in mixed_data], k=50)
alpha_hat = 1.0 / gamma
print(f"Estimated α: {alpha_hat:.2f}")  # Should be close to 2.0
```

--------------------------------------------------------------------------------

## Diagnostic Checks

### QQ-Plot Against Pareto

```python
import matplotlib.pyplot as plt
import math

# Estimate α
gamma = hill_estimator(data, k=200)
alpha_hat = 1.0 / gamma

# Theoretical quantiles (Pareto)
sorted_data = sorted(data, reverse=True)
n = len(sorted_data)
theoretical_quantiles = [(i/n)**(-1/alpha_hat) for i in range(1, n+1)]

# QQ plot
plt.figure(figsize=(8, 8))
plt.loglog(theoretical_quantiles[:200], sorted_data[:200], 'o', alpha=0.6)
plt.plot([min(theoretical_quantiles[:200]), max(theoretical_quantiles[:200])],
         [min(theoretical_quantiles[:200]), max(theoretical_quantiles[:200])],
         'r--', linewidth=2)
plt.xlabel('Theoretical Pareto Quantiles')
plt.ylabel('Sample Quantiles')
plt.title(f'QQ-Plot (α={alpha_hat:.2f})')
plt.grid(True, which='both', alpha=0.3)
plt.show()
```

If points follow the diagonal, Pareto model fits well.

--------------------------------------------------------------------------------

## Applications

### Financial Risk

```python
# Stock return losses
from heavytails.tail_index import hill_estimator

# Assuming 'returns' is your data
losses = [-r for r in returns if r < 0]  # Negative returns only

# Estimate tail index
gamma = hill_estimator(losses, k=100)
alpha_hat = 1.0 / gamma

print(f"Tail index of losses: {alpha_hat:.2f}")

if alpha_hat < 4:
    print("WARNING: Kurtosis may be infinite!")
if alpha_hat < 2:
    print("CRITICAL: Variance may be infinite!")
```

### Insurance Claims

```python
# Large claims analysis
claims = [...]  # Your claims data
large_claims = [c for c in claims if c > 10000]  # Threshold at $10,000

gamma = hill_estimator(large_claims, k=50)
alpha_hat = 1.0 / gamma

print(f"Tail index of large claims: {alpha_hat:.2f}")
```

--------------------------------------------------------------------------------

## References

1. **Hill, B. M. (1975)**. "A Simple General Approach to Inference About the Tail of a Distribution". _Annals of Statistics_, 3(5), 1163-1174.

2. **Pickands, J. (1975)**. "Statistical Inference Using Extreme Order Statistics". _Annals of Statistics_, 3(1), 119-131.

3. **Dekkers, A. L. M., Einmahl, J. H. J., & de Haan, L. (1989)**. "A Moment Estimator for the Index of an Extreme-Value Distribution". _Annals of Statistics_, 17(4), 1833-1855.

--------------------------------------------------------------------------------

## Next Steps

- **[Parameter Fitting](fitting.md)** - Fit parametric distributions
- **[Diagnostic Tools](diagnostics.md)** - Goodness-of-fit tests
- **[Examples](../examples/basic_usage.ipynb)** - Practical applications

## Choosing an estimator

`heavytails` provides four. They differ in the range of `gamma` they can
represent and in how efficiently they use the data.

| Estimator | Valid range | Efficiency | Use when |
| --- | --- | --- | --- |
| `smoothed_hill_estimator` | `gamma > 0` | Highest | Clean data, tail known to be heavy |
| `trimmed_hill_estimator` | `gamma > 0` | Near-Hill | Contamination, count roughly known |
| `adaptive_trimmed_hill_estimator` | `gamma > 0` | Near-Hill | Contamination of unknown *count* |
| `t_hill_estimator` | `gamma > 0` | Near-Hill | Contamination of unknown extremity |
| `harmonic_moment_estimator` | `gamma > 0` | Tunable | As above, with a robustness dial |
| `gpd_mle_estimator` | any `gamma` | Lower | Parametric peaks-over-threshold |
| `bias_reduced_hill_estimator` | `gamma > 0` | Near-Hill | The Hill plot has a visible slope |
| `orthogonalized_bias_reduced_hill_estimator` | `gamma > 0` | Lower | Second-order bias plus top contamination |
| `threshold_averaged_orthogonalized_hill_estimator` | `gamma > 0` | Lower | Bias correction with threshold uncertainty |
| `hill_estimator` | `gamma > 0` | High | The classical baseline |
| `generalized_hill_estimator` | any `gamma` | Near-Hill | You are not certain the tail is heavy |
| `moment_estimator` | any `gamma` | Near-Hill | As above; a useful cross-check |
| `pickands_estimator` | any `gamma` | Lowest | Cross-checking, not as a primary estimate |

`scripts/tail_index_study.py` measures this rather than asserting it. Root mean
squared error over 120 samples, using `k = n/20`:

| Scenario | n | hill | smoo u=2 | smoo u=3 | gen_hill | moment | pickands |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pareto(alpha=2), gamma=0.5 | 10000 | 0.023 | 0.018 | **0.015** | 0.050 | 0.050 | 0.083 |
| Uniform(0,1), gamma=-1 | 10000 | 1.026 | 1.040 | 1.054 | **0.053** | 0.108 | 0.081 |

Two things to read from that table.

Where the tail really is heavy, the **Hill family is the most efficient by a
factor of two or more**, and the smoothed variant beats plain Hill outright.

On the Uniform sample, whose upper endpoint is finite and whose `gamma` is
`-1`, **Hill is not merely inaccurate but structurally incapable**: it averages
log-excesses and can only ever return a positive number, so it reports about
`+0.026` for a true `-1`. No choice of `k` fixes that. The generalized Hill
estimator recovers `-0.99`.

The smoothed estimators inherit that limitation exactly, because they average
Hill estimates. Smoothing addresses variance in `k`, not the range of `gamma`.

If you do not already know the sign of `gamma`, do not start with Hill.

## Smoothing the Hill estimator

The ordinary Hill estimate varies substantially with `k`, which is the reason
the Hill plot exists. Resnick and Stărică (1997) average it over a range of `k`
instead:

$$\hat{\gamma}^{\text{smooHill}}_{k}(u) = \frac{1}{(u-1)k}\sum_{j=k+1}^{\lfloor uk \rfloor} \hat{\gamma}^{H}(j)$$
floor} \hat{\gamma}^{H}(j)$$

```python
from heavytails import Pareto, smoothed_hill_estimator, smoothed_hill_variance_ratio

data = Pareto(alpha=2.0, xm=1.0).rvs(20000, seed=11)
smoothed_hill_estimator(data, k=1000, u=2.0)

smoothed_hill_variance_ratio(2.0)   # 0.6137
smoothed_hill_variance_ratio(3.0)   # 0.4507
```

The asymptotic variance falls from `gamma**2` to
`gamma**2 * 2*(u - 1 - ln u) / (u - 1)**2`, a reduction of 39% at `u = 2` and
55% at `u = 3`. Measured over 250 samples at `n = 20000, k = 500`, the observed
ratio is 0.61 at `u = 2` against a predicted 0.6137.

`u` is a bias/variance dial, not a free win: a larger `u` averages over a wider
range of `k` and so reaches further into the body of the distribution. Values
between 2 and 3 are the usual compromise.

## Confidence intervals

A tail index without an interval is not usable. The estimate depends on a
choice of `k` that no rule fixes, and the sampling variability at realistic
sample sizes is large.

```python
from heavytails import Pareto, tail_index_confidence_interval

data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=7)

tail_index_confidence_interval(data, k=250)
# {'gamma': 0.49, 'alpha': 2.04, 'lower': 0.429, 'upper': 0.551, ...}
```

Two methods are available:

- `asymptotic` uses `sqrt(k)(gamma_hat - gamma) -> N(0, gamma^2)`, giving
  `gamma_hat * (1 +/- z/sqrt(k))`. It is only established for the Hill
  estimator, and requesting it for another raises rather than reporting a
  number with no basis.
- `bootstrap` resamples the data and takes percentiles. It works for every
  estimator and makes no distributional assumption, at the cost of refitting.

!!! warning "What the interval does not cover"
    Both methods capture **sampling variance only**. Neither captures the bias
    introduced by the choice of `k`. Measured coverage of a nominal 95%
    interval is about 92% at `n = 20000, k = 1000` for a Pareto tail, and it
    degrades further as `k` moves away from the plateau.

    An interval is not a substitute for looking at the Hill plot.

## The Hill plot

```python
from heavytails import Pareto, hill_plot

data = Pareto(alpha=2.0, xm=1.0).rvs(5000, seed=2)
points = hill_plot(data)   # [(k, gamma_hat), ...]
```

`hill_plot` sweeps `k` on a logarithmic grid, because the interesting structure
is at small `k`. Read the estimate off a stable plateau. If there is no
plateau, the data does not support a tail index estimate, and forcing one
produces a confident wrong answer.

## Which quantity is returned

Two conventions are in circulation and they are reciprocals of each other:

- the **tail index** `alpha`, from `P(X > x) ~ L(x) * x**-alpha`, common in
  economics and network science;
- the **extreme-value index** `xi`, also written `gamma`, the EVT convention,
  with `gamma = 1/alpha`.

**Every estimator here returns `gamma`.** Larger `gamma` means a heavier tail.
`tail_index_confidence_interval` reports both, and `moment_estimator` returns
the pair `(gamma, alpha)`.

This is worth checking before comparing against another implementation or a
published figure: a reciprocal-shaped discrepancy is almost always this.

## Contamination

Hill gives enormous leverage to the largest order statistics, because they
enter through unbounded logarithms of ratios. A handful of bad observations is
enough to destroy the estimate.

Replacing three observations out of ten thousand `Pareto(alpha=2)` draws with
outliers, at `k = 500`:

| Estimator | Mean over 120 samples | True |
| --- | --- | --- |
| `hill` | 0.597 | 0.5 |
| `smoothed_hill` (u=3) | 0.554 | 0.5 |
| `moment` | 1.015 | 0.5 |
| `generalized_hill` | 0.880 | 0.5 |
| **`trimmed_hill`** (r=5) | **0.503** | 0.5 |
| **`adaptive_trimmed_hill`** | **0.503** | 0.5 |
| `pickands` | 0.518 | 0.5 |

Note that `pickands` also survives, which is not usually why anyone reaches for
it: it uses only three order statistics, at positions `k`, `2k` and `4k`, so
contamination confined to the very top misses it entirely. That is robustness
by accident rather than by design, and it comes at the cost of by far the
highest variance of any estimator here.

### Trimming

```python
from heavytails import Pareto, trimmed_hill_estimator, trimmed_hill_plot

trimmed_hill_estimator(data, k=500, r=5)   # discard the 5 largest
```

`r = 0` reproduces the ordinary Hill estimator exactly.

!!! warning "r must exceed the number of contaminated observations"
    Trimming five when ten are contaminated is barely better than trimming
    none: the remaining bad values still enter through their spacings. In the
    table above, `trimmed_hill` with `r = 5` against **ten** outliers reports
    0.808 rather than 0.5.

    `adaptive_trimmed_hill_estimator` exists because of exactly this. Against
    the same ten outliers it reports 0.503, because it works out how many there
    are rather than being told.

On clean data trimming is close to free. Discarding ten observations from a
sample of ten thousand raises the standard deviation from 0.0296 to 0.0302, so
trimming a few by default is a cheap insurance policy.

### Choosing r from the data

Reading `r` off the plot works, and requires someone to look. The adaptive
variant works it out:

```python
from heavytails.tail_index import (
    adaptive_trim_selection, adaptive_trimmed_hill_estimator,
)

adaptive_trimmed_hill_estimator(data, k=300)     # picks r itself
adaptive_trim_selection(data, k=300)["trim"]     # and says what it picked
```

Under a Pareto tail the normalised log-spacings `Y_i = i(log X_(i) - log
X_(i+1))` are independent and exponential, so contamination among the largest
observations inflates one of them. Each spacing is tested against the mean of
the deeper ones, with an exactly computable null distribution:

$$P(R > t) = \left(\frac{m}{m+t}\right)^{m}, \qquad R = \frac{Y_j}{\overline{Y}_{j+1..k}}$$

No asymptotics and no tabulated critical values. Over 200 samples of 10,000
`Pareto(2)` draws with `k = 300`, the median `r` chosen equals the number of
outliers planted, at 0, 1, 2, 3, 5 and 8 of them.

!!! tip "Scan the deepest spacing first, not the first"
    Several outliers of similar size sit close together, so the gaps *between*
    them are small and only the gap *below* the last one is large. A rule that
    stopped at the first ordinary-looking spacing would report a badly
    contaminated sample as clean.

**Detection is a rate, not a certainty.** With three outliers among 10,000
draws it finds them in 100% of samples at five times the true sample maximum,
95% at three times, 64% at twice and 46% at one and a half times. An outlier
only half again the size of the largest genuine observation is not reliably
distinguishable from the tail itself.

**On clean data it costs almost nothing.** The standard deviation is 0.0295
against 0.0292 for the plain Hill estimator, because trimming is applied only
when the data asks for it. The `level` argument is the probability of
over-trimming a clean sample, and it means what it says: measured at 0.009,
0.052 and 0.094 for nominal 0.01, 0.05 and 0.10.

!!! danger "It refuses rather than guessing when the scan is too short"
    If contamination reaches deeper than `max_trim`, every scanned spacing is a
    gap *between* outliers and nothing looks anomalous. The estimator would
    report a badly wrong answer that is indistinguishable from a clean one --
    1.79 for a true 0.5, with 30 outliers and `max_trim = 20`. A separate
    interlock detects that case and raises instead, naming the limit to use.

### Choosing r

`trimmed_hill_plot` sweeps `r` the way the Hill plot sweeps `k`:

```python
for r, gamma in trimmed_hill_plot(data, k=300, max_trim=8):
    print(r, gamma)
```

On a sample with three contaminated observations the elbow is unmistakable:

```
r=0  gamma=0.6369
r=1  gamma=0.6367
r=2  gamma=0.6327
r=3  gamma=0.4789   <- contamination exhausted
r=4  gamma=0.4796
r=5  gamma=0.4801
```

The estimate moves while `r` is below the contamination count and flattens once
the bad values are gone, so the elbow tells you how many there were. A plot
that is flat from `r = 0` says there is no contamination to remove.

### Passing the tuning parameter to the interval helper

```python
tail_index_confidence_interval(
    data, k=300, estimator="trimmed_hill",
    method="bootstrap", estimator_kwargs={"r": 5},
)
```

Without `estimator_kwargs` the trimmed estimator runs at its default `r = 0`,
which is the ordinary Hill estimator and gives no robustness at all.

## Bounded influence

Trimming needs you to guess how many observations are contaminated. The
alternative is to bound each observation's influence so that no single one can
dominate however extreme it is.

Hill's contributions are `log(X_(i)/u)` for a threshold `u = X_(k+1)`, which
grow without limit. The harmonic moment family uses the reciprocal ratios
`R_i = u/X_(i)`, which lie in `(0, 1]`. A contaminated observation sent to
infinity contributes `R_i -> 0`, and its effect saturates.

```python
from heavytails import t_hill_estimator, harmonic_moment_estimator

t_hill_estimator(data, k=500)                      # beta = 1
harmonic_moment_estimator(data, k=500, beta=2.0)   # more robust, less efficient
```

Sending one observation out of ten thousand from `1e2` to `1e30`:

| Outlier | `hill` | `t_hill` | `harmonic` (beta=2) |
| --- | --- | --- | --- |
| 1e2 | 0.5016 | 0.5015 | 0.5018 |
| 1e12 | 0.5477 | 0.5017 | 0.5018 |
| 1e30 | 0.6306 | **0.5017** | **0.5018** |

The Hill estimate keeps degrading. The other two do not move at all.

### The derivation

With `R_i = u / X_(i)` for the top `k` observations, under an exact Pareto tail
`R ~ Beta(alpha, 1)`, so `E[R**beta] = alpha / (alpha + beta)`. Inverting:

$$\hat{\alpha} = \frac{\beta H}{1 - H}, \qquad H = \frac{1}{k}\sum_{i=1}^{k} R_i^{\beta}$$

`beta = 1` gives the **t-Hill** estimator. As `beta` tends to zero the estimator
tends to the **Hill** estimator, which is a useful check: at `beta = 0.001` the
two agree to five decimal places.

### Choosing beta

`beta` trades robustness against efficiency. Larger values weight observations
near the threshold more and extreme ones less. Deviation from the clean-sample
estimate, with twenty contaminated observations out of ten thousand:

| beta | 0.25 | 0.5 | 1.0 | 3.0 |
| --- | --- | --- | --- | --- |
| Shift under contamination | 0.124 | 0.047 | 0.013 | 0.000 |

Larger `beta` is monotonically more robust, at a cost in efficiency on clean
data. `beta = 1` to `2` is a reasonable default.

!!! note "Which robust estimator to reach for"
    Use `trimmed_hill_estimator` when you can see roughly how many observations
    are bad, for instance from `trimmed_hill_plot`. Use `t_hill_estimator` or
    `harmonic_moment_estimator` when you cannot, or when the contamination may
    be arbitrarily extreme. They address different problems: trimming removes
    bad values, bounded influence limits what any one of them can do.

## Peaks over threshold

Every estimator above is semiparametric: it averages some functional of the
upper order statistics and assumes nothing about their distribution beyond
regular variation. The parametric alternative fits a generalized Pareto
distribution to the exceedances, which is what the Pickands-Balkema-de Haan
theorem licenses, and estimates shape and scale jointly.

```python
from heavytails import gpd_mle_estimator, fit_generalized_pareto

gpd_mle_estimator(data, k=500)          # the shape parameter, i.e. gamma

threshold = sorted(data, reverse=True)[500]
excesses = [x - threshold for x in data if x > threshold]
fit_generalized_pareto(excesses)        # {'xi': ..., 'sigma': ..., ...}
```

Fitting is done by maximum likelihood, using the reduction of Grimshaw (1993):
substituting `theta = xi/sigma` turns the two-parameter problem into a
one-dimensional search, so no third-party optimiser is needed. The fit agrees
with `scipy.stats.genpareto.fit` to four decimal places for positive, near-zero
and negative shape.

It is a general-EVI estimator, so unlike the whole Hill family it handles a
bounded tail. On a Uniform(0,1) sample, where the true index is -1:

| Estimator | Mean | RMSE |
| --- | --- | --- |
| `hill` | +0.026 | 1.026 |
| `t_hill` | +0.026 | 1.026 |
| `gpd_mle` | **-1.056** | 0.078 |
| `generalized_hill` | -0.998 | 0.049 |

The cost is variance. On `Pareto(alpha=2)` at `n = 10000, k = 500` its RMSE is
0.072 against 0.023 for Hill, because it estimates two parameters where Hill
estimates one.

!!! note "It is much slower"
    A single fit at `k = 1000` takes about 25 ms, against microseconds for the
    closed-form estimators, because it optimises. That is irrelevant for one
    estimate and significant for bootstrapping, so reduce `n_bootstrap`
    accordingly.

## Bias correction

The Hill plot slopes rather than plateaus when the tail approaches its Pareto
limit slowly. That bias is systematic, not random, so it can be estimated and
subtracted. Under second-order regular variation

```
P(X > x) = C x**(-1/gamma) [1 + D x**(rho/gamma) + ...],   rho < 0
```

and the estimator of Caeiro, Gomes and Pestana (2005) removes the leading term:

```python
from heavytails import bias_reduced_hill_estimator

bias_reduced_hill_estimator(data, k=2000, rho=-1.0)
```

Bias over 30 samples at `n = 20000`, with `rho` supplied:

| Case | k | Hill | corrected | factor |
| --- | --- | --- | --- | --- |
| Fréchet(2) | 500 | 0.0064 | 0.0002 | 36x |
| Fréchet(2) | 2000 | 0.0162 | 0.0046 | 3.5x |
| BurrXII(c=2,k=1) | 2000 | 0.0298 | 0.0050 | 6x |
| BurrXII(c=1,k=2) | 2000 | 0.1422 | 0.0081 | **18x** |

The correction buys bias at some cost in variance, which is the usual trade:
on Fréchet(2) at `k = 500` the standard deviation roughly doubles.

### rho is the hard part

`second_order_rho` implements the Fraga Alves-Gomes-de Haan (2003) estimator.
**It is unstable, and that is a property of the estimator rather than of this
implementation.** Sweeping `k` on a Fréchet(2) sample of 100000, whose true
`rho` is -1:

| k/n | 0.02 | 0.05 | 0.10 | 0.20 | 0.40 | 0.60 |
| --- | --- | --- | --- | --- | --- | --- |
| `rho_hat` | -0.07 | -1.48 | **-20.48** | -1.69 | -1.05 | -1.09 |

The value at `k/n = 0.10` is a pole, where the estimator's denominator crosses
zero. Its published recommendation is a very large `k`, around 85% of the
sample, which `recommended_rho_k` implements.

!!! tip "Supply rho if you can"
    On the worst case above, supplying `rho` gives a bias of 0.0081 while
    estimating it gives 0.0354 — still four times better than not correcting,
    but most of the benefit is lost. Practitioners commonly fix `rho` at a
    canonical value such as -1 rather than estimate it, and that is usually the
    better choice.

`rho` is a property of the distribution, not of the sample size, so it is
sometimes known analytically: Fréchet has `rho = -1` for every shape parameter,
Student-t with `nu` degrees of freedom has `rho = -2/nu`, and
`BurrXII(c, k, s)` has `rho = -1/k`.

## Orthogonalized bias reduction

`bias_reduced_hill_estimator` estimates the leading bias and subtracts it after
the fact. `orthogonalized_bias_reduced_hill_estimator` instead builds the
cancellation into the log-spacing weights.

!!! note "Novelty"
    The weighted log-spacing formula is not presented here as a new estimator.
    Bias-cancelling weighted Hill and exponential-regression estimators are
    already part of the tail-index literature. In this package the local
    formula is a transparent building block for adaptive procedures over
    unknown contamination `r` and threshold `k`.

For normalised log-spacings

$$Z_j = j\{\log X_{(j)} - \log X_{(j+1)}\},$$

the second-order approximation is

$$E[Z_j] \approx \gamma + b\left(\frac{j}{k+1}\right)^{-\rho}.$$

The estimator returns the intercept from this one-covariate regression. In
weighted form,

$$\hat{\gamma} = \sum_j w_j Z_j,$$

where the weights satisfy

$$\sum_j w_j = 1, \qquad
\sum_j w_j\left(\frac{j}{k+1}\right)^{-\rho}=0.$$

The first identity targets `gamma`; the second removes the leading second-order
bias direction.

```python
from heavytails import Frechet, orthogonalized_bias_reduced_hill_estimator

data = Frechet(alpha=2.0, s=1.0, m=0.0).rvs(20000, seed=1)
orthogonalized_bias_reduced_hill_estimator(data, k=2000, rho=-1.0)
```

It can also reuse the adaptive trimming scan before fitting the intercept:

```python
orthogonalized_bias_reduced_hill_estimator(
    data, k=500, rho=-1.0, adaptive_trim=True
)
```

That combination is useful when the largest observations may be contaminated
and the Hill plot also shows second-order drift. It is not free: on an exact
Pareto tail the estimator has higher variance than Hill. Use it when the bias
or contamination is more damaging than that variance cost.

When `adaptive_trim=True`, the default trimming level is not a fixed 5% test.
It is a conservative sequence that decreases with sample size. That matters for
the oracle claim: a fixed false-alarm probability would not give
`P(r_hat = r) -> 1`.

### Threshold averaging

`threshold_averaged_orthogonalized_hill_estimator` adds a threshold-uncertainty
layer. It evaluates a logarithmic grid up to `k`, keeps thresholds whose
orthogonalized estimates remain statistically compatible, and averages the
stable set using the log-spacing covariance approximation.

```python
from heavytails import threshold_averaged_orthogonalized_hill_estimator

gamma = threshold_averaged_orthogonalized_hill_estimator(
    data,
    k=2000,
    min_k=500,
    grid_size=10,
    rho=-1.0,
    adaptive_trim=True,
)
```

The point estimator is cross-fitted by default. Threshold sets, `rho` and
aggregation weights are learned on one split of the sample and evaluated on the
other, then the roles are swapped. The trimming count is deliberately
re-estimated on the evaluation fold: a random split does not put the same number
of contaminated extremes in both halves. This is slower and a little noisier
than the full-sample diagnostic estimate, but it separates threshold selection
from the observations used for the final estimate without transferring an
absolute outlier count across folds.

For diagnostics, use the selection function:

```python
from heavytails import threshold_averaged_orthogonalized_hill_selection

details = threshold_averaged_orthogonalized_hill_selection(data, k=2000)
details["candidate_pairs"]          # [(r, k), ...]
details["stable_candidate_pairs"]   # admitted pairs
details["stable_thresholds"]
details["weights"]
details["local_estimates"]
details["variance_proxy"]           # sum(w_j**2) for each local candidate
```

The selector uses a compatibility cutoff that grows slowly with sample size.
The default aggregation is convex: weights are constrained to be non-negative
and to sum to one. Set `crossfit=False` on the point estimator when you want the
same full-sample value reported by the diagnostic selection function.

The research question is therefore not "is this weighted Hill formula new?" It
is whether the adaptive procedure can achieve near-oracle risk when both `r`
and `k` are unknown while second-order bias is present.

!!! warning "Current scope"
    The estimator adapts within a user-supplied threshold window
    `[min_k, k]`; it does not remove the need to choose a plausible envelope.
    A theorem would need deterministic sequences `min_k_n` and `max_k_n` and an
    assumption that the oracle threshold lies inside that range.

    Exact recovery of arbitrary contamination is also impossible when the
    outlier is statistically indistinguishable from a legitimate tail draw.
    The adaptive trimming theory needs a separation condition: the anomalous
    boundary spacing must be detectable at the vanishing test level.

    Finally, the default `rho=-1` should be read as a tuning choice. A first
    theorem should treat `rho` as known and bounded away from zero; plug-in
    estimation of `rho` is a separate problem.
