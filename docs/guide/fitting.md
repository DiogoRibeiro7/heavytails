# Parameter Fitting

This guide covers parameter estimation and distribution fitting for heavy-tailed distributions.

## Overview

Parameter fitting is the process of estimating distribution parameters from observed data. The heavytails library provides multiple approaches for parameter estimation.

## Current Status

⚠️ **Note**: Maximum Likelihood Estimation (MLE) and method of moments are planned features and not yet fully implemented. This guide covers available methods and workarounds.

## Available Fitting Methods

### Tail Index Estimation

For Pareto-type heavy tails, estimate the tail index directly:

```python
from heavytails import Pareto
from heavytails.tail_index import hill_estimator, pickands_estimator, moment_estimator

# Generate sample data from known distribution
true_pareto = Pareto(alpha=2.5, xm=1.0)
data = true_pareto.rvs(1000, seed=42)

# Estimate tail index using Hill estimator
k = 100  # number of upper order statistics to use
gamma_hill = hill_estimator(data, k)
alpha_hill = 1.0 / gamma_hill

print(f"True alpha: 2.5")
print(f"Estimated alpha (Hill): {alpha_hill:.4f}")

# Try different k values to check stability
k_values = [50, 100, 200, 300]
for k in k_values:
    gamma = hill_estimator(data, k)
    alpha_est = 1.0 / gamma
    print(f"k={k:3d}: alpha={alpha_est:.4f}")
```

### Pickands Estimator

Alternative tail index estimator:

```python
from heavytails import GeneralizedPareto
from heavytails.tail_index import pickands_estimator

# Generate GPD data
gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
data = gpd.rvs(1000, seed=42)

# Pickands estimator
k = 100
m = 2
gamma_pickands = pickands_estimator(data, k, m)

print(f"True xi (tail index): 0.3")
print(f"Estimated gamma: {gamma_pickands:.4f}")
```

### Moment Estimator

Dekkers-Einmahl-de Haan moment estimator:

```python
from heavytails.tail_index import moment_estimator

# Using Pareto data from earlier
gamma_moment, alpha_moment = moment_estimator(data, k=100)

print(f"Moment estimator:")
print(f"  Gamma: {gamma_moment:.4f}")
print(f"  Alpha: {alpha_moment:.4f}")
```

## Manual Parameter Fitting

### Quantile Matching

Fit parameters by matching theoretical and empirical quantiles:

```python
from heavytails import LogNormal
import math

# Generate data from known distribution
true_lognormal = LogNormal(mu=0.5, sigma=0.8)
data = true_lognormal.rvs(500, seed=42)

# Calculate empirical quantiles
sorted_data = sorted(data)
n = len(sorted_data)
empirical_median = sorted_data[n//2]
empirical_q25 = sorted_data[n//4]
empirical_q75 = sorted_data[3*n//4]

print("Empirical quantiles:")
print(f"  25th: {empirical_q25:.4f}")
print(f"  50th: {empirical_median:.4f}")
print(f"  75th: {empirical_q75:.4f}")

# Try to match parameters (simplified approach)
# For LogNormal: median = exp(mu)
mu_est = math.log(empirical_median)

# Estimate sigma from IQR
# IQR ≈ exp(mu) * (exp(1.35*sigma) - exp(-1.35*sigma))
# This is a rough approximation
iqr = empirical_q75 - empirical_q25
sigma_est = math.log(1 + iqr/empirical_median) / 1.35

fitted_lognormal = LogNormal(mu=mu_est, sigma=sigma_est)

print(f"\\nTrue parameters: mu=0.5, sigma=0.8")
print(f"Estimated parameters: mu={mu_est:.4f}, sigma={sigma_est:.4f}")

# Validate fit
fitted_median = fitted_lognormal.ppf(0.5)
print(f"\\nValidation:")
print(f"  Empirical median: {empirical_median:.4f}")
print(f"  Fitted median: {fitted_median:.4f}")
```

### Grid Search

Search over parameter space to minimize distance metric:

```python
from heavytails import Cauchy

# Generate Cauchy data
true_cauchy = Cauchy(x0=2.0, gamma=1.5)
data = true_cauchy.rvs(200, seed=42)

# Calculate empirical quartiles
sorted_data = sorted(data)
n = len(sorted_data)
q1 = sorted_data[n//4]
q2 = sorted_data[n//2]
q3 = sorted_data[3*n//4]

# Grid search over parameters
best_error = float('inf')
best_params = None

x0_range = [q2 + i*0.5 for i in range(-5, 6)]  # Around median
gamma_range = [0.5, 1.0, 1.5, 2.0, 2.5]

for x0_test in x0_range:
    for gamma_test in gamma_range:
        test_cauchy = Cauchy(x0=x0_test, gamma=gamma_test)
        # Compare quantiles
        error = (
            abs(test_cauchy.ppf(0.25) - q1) +
            abs(test_cauchy.ppf(0.50) - q2) +
            abs(test_cauchy.ppf(0.75) - q3)
        )
        if error < best_error:
            best_error = error
            best_params = (x0_test, gamma_test)

x0_fit, gamma_fit = best_params
print(f"True parameters: x0=2.0, gamma=1.5")
print(f"Fitted parameters: x0={x0_fit:.4f}, gamma={gamma_fit:.4f}")
print(f"Total error: {best_error:.4f}")
```

## Assessing Fit Quality

### Visual Diagnostics

Compare empirical and theoretical distributions:

```python
from heavytails import StudentT

# Fit Student-t to data
true_dist = StudentT(nu=4.0)
data = true_dist.rvs(500, seed=42)

# Assume we've estimated nu=4.5
fitted_dist = StudentT(nu=4.5)

# Compare CDFs at various points
test_points = sorted(data)[::50]  # Every 50th point

print("CDF comparison:")
print(f"{'x':>8s} {'Empirical':>10s} {'Fitted':>10s} {'Diff':>8s}")
for x in test_points:
    emp_cdf = sum(1 for d in data if d <= x) / len(data)
    theo_cdf = fitted_dist.cdf(x)
    diff = abs(emp_cdf - theo_cdf)
    print(f"{x:8.4f} {emp_cdf:10.6f} {theo_cdf:10.6f} {diff:8.6f}")
```

### Kolmogorov-Smirnov Statistic

Calculate maximum difference between empirical and theoretical CDF:

```python
def ks_statistic(data, distribution):
    """Calculate Kolmogorov-Smirnov test statistic."""
    sorted_data = sorted(data)
    n = len(sorted_data)
    max_diff = 0

    for i, x in enumerate(sorted_data):
        # Empirical CDF
        emp_cdf = (i + 1) / n
        # Theoretical CDF
        theo_cdf = distribution.cdf(x)
        # Track maximum difference
        diff = abs(emp_cdf - theo_cdf)
        if diff > max_diff:
            max_diff = diff

    return max_diff

# Example with Weibull
from heavytails import Weibull

data = Weibull(k=0.8, lam=1.2).rvs(300, seed=42)

# Test different parameter combinations
test_params = [
    (0.7, 1.2),
    (0.8, 1.2),
    (0.9, 1.2),
    (0.8, 1.0),
    (0.8, 1.4),
]

print("KS statistic for different parameters:")
for k_test, lam_test in test_params:
    test_dist = Weibull(k=k_test, lam=lam_test)
    ks = ks_statistic(data, test_dist)
    print(f"k={k_test:.1f}, lambda={lam_test:.1f}: KS={ks:.6f}")
```

## Practical Examples

### Fitting Financial Returns

```python
from heavytails import StudentT, Cauchy
from heavytails.tail_index import hill_estimator

# Simulate financial return data (in %)
# In practice, this would be real market data
true_returns = StudentT(nu=5.0)
returns_data = true_returns.rvs(1000, seed=42)

# Estimate tail index
k = int(len(returns_data) * 0.1)  # Use top 10%
gamma_hill = hill_estimator(returns_data, k)
print(f"Hill estimator gamma: {gamma_hill:.4f}")

# For Student-t: gamma = 1/nu, so nu ≈ 1/gamma
nu_estimated = 1.0 / gamma_hill
print(f"Estimated degrees of freedom: {nu_estimated:.2f}")
print(f"True degrees of freedom: 5.0")

# Validate fit
fitted_returns = StudentT(nu=nu_estimated)
print("\\nQuantile comparison:")
quantiles = [0.01, 0.05, 0.95, 0.99]
for q in quantiles:
    emp_q = sorted(returns_data)[int(q * len(returns_data))]
    theo_q = fitted_returns.ppf(q)
    print(f"  {q*100}%: Empirical={emp_q:.4f}, Theory={theo_q:.4f}")
```

### Fitting Insurance Claims

```python
from heavytails import Pareto
from heavytails.tail_index import hill_estimator

# Simulate insurance claim data
true_claims = Pareto(alpha=1.8, xm=1000)
claims_data = true_claims.rvs(500, seed=42)

# Estimate parameters
# 1. Minimum claim (xm) is empirical minimum
xm_est = min(claims_data)

# 2. Estimate alpha using Hill
k = 50  # Use top 50 claims
gamma_hill = hill_estimator(claims_data, k)
alpha_est = 1.0 / gamma_hill

print(f"True parameters: alpha=1.8, xm=1000")
print(f"Estimated: alpha={alpha_est:.4f}, xm={xm_est:.2f}")

# Create fitted distribution
fitted_claims = Pareto(alpha=alpha_est, xm=xm_est)

# Check fit on large claims
threshold = sorted(claims_data)[int(0.9 * len(claims_data))]
large_claims_empirical = [c for c in claims_data if c > threshold]
large_claims_count_emp = len(large_claims_empirical)
large_claims_prob_theo = fitted_claims.sf(threshold)
large_claims_count_theo = large_claims_prob_theo * len(claims_data)

print(f"\\nLarge claims (>{threshold:.0f}):")
print(f"  Empirical count: {large_claims_count_emp}")
print(f"  Theoretical expected: {large_claims_count_theo:.0f}")
```

### Fitting Network Data

```python
from heavytails import Zipf

# Simulate word frequency data (rank-frequency)
# In practice: word frequencies from text corpus
true_zipf = Zipf(s=1.5, kmax=1000)
frequency_data = true_zipf.rvs(500, seed=42)

# Estimate s parameter from rank-frequency relationship
# For Zipf: frequency ∝ rank^(-s)
# Use log-log regression on sorted frequencies

sorted_freq = sorted(frequency_data, reverse=True)
# Remove duplicates for cleaner regression
from collections import Counter
freq_counts = Counter(sorted_freq)
unique_ranks = list(range(1, len(freq_counts) + 1))
unique_freqs = list(freq_counts.keys())

# Simple log-log regression
import math
log_ranks = [math.log(r) for r in unique_ranks[:100]]  # Top 100
log_freqs = [math.log(f) for f in unique_freqs[:100]]

# Estimate slope (simplified OLS)
n = len(log_ranks)
mean_log_rank = sum(log_ranks) / n
mean_log_freq = sum(log_freqs) / n
numerator = sum((log_ranks[i] - mean_log_rank) * (log_freqs[i] - mean_log_freq)
                for i in range(n))
denominator = sum((log_ranks[i] - mean_log_rank)**2 for i in range(n))
s_estimated = -numerator / denominator  # Negative because inverse relationship

print(f"True s parameter: 1.5")
print(f"Estimated s: {s_estimated:.4f}")
```

## Best Practices

1. **Use multiple methods**: Different estimators have different properties
2. **Check stability**: Vary parameters (like k in Hill estimator) and check consistency
3. **Visual inspection**: Always plot data vs fitted distribution
4. **Bootstrap uncertainty**: Resample data to estimate parameter uncertainty
5. **Domain knowledge**: Use prior knowledge to constrain parameter search
6. **Sample size matters**: Tail estimation requires large samples (typically 1000+)

## Common Pitfalls

1. **Too few observations**: Tail index estimation needs sufficient tail data
2. **Choice of k**: Hill estimator sensitive to number of order statistics
3. **Finite sample bias**: All estimators have bias in finite samples
4. **Mixed distributions**: Real data may be mixture, not single distribution
5. **Outliers vs heavy tails**: Distinguish true tail behavior from contamination

## Future Features

The following features are planned for future releases:

- Maximum Likelihood Estimation (MLE) for all distributions
- Method of moments estimation
- Automated model selection (AIC, BIC)
- Bootstrap confidence intervals
- Goodness-of-fit tests
- Cross-validation for parameter selection

See [roadmap.py](https://github.com/diogoribeiro7/heavytails/blob/main/heavytails/roadmap.py) for details.

## Next Steps

- [Diagnostic Tools](diagnostics.md) - Validate distribution fits
- [Tail Index Estimation](tail-estimation.md) - Deep dive into tail estimators
- [Random Sampling](sampling.md) - Generate samples for validation
- [Working with PDFs and CDFs](pdf-cdf.md) - Understand probability functions
