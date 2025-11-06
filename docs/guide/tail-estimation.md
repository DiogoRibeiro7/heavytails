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
