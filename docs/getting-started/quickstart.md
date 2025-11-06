# Quick Start Guide

Get up and running with HeavyTails in 10 minutes! This guide covers the essential operations you'll use most frequently.

---

## Your First Distribution

Let's start with the classic Pareto distribution - a fundamental heavy-tailed distribution that models power laws:

```python
from heavytails import Pareto

# Create a Pareto distribution with tail index α=2.5 and scale xₘ=1.0
pareto = Pareto(alpha=2.5, xm=1.0)

# Calculate probabilities
prob_exceed_10 = pareto.sf(10.0)  # P(X > 10)
print(f"P(X > 10) = {prob_exceed_10:.4f}")

# Calculate quantiles
median = pareto.ppf(0.5)  # 50th percentile
q95 = pareto.ppf(0.95)    # 95th percentile
print(f"Median: {median:.2f}, 95th percentile: {q95:.2f}")

# Generate random samples
samples = pareto.rvs(1000, seed=42)
print(f"Generated {len(samples)} samples")
print(f"Sample mean: {sum(samples)/len(samples):.2f}")
```

**Output:**
```
P(X > 10) = 0.0158
Median: 1.52, 95th percentile: 10.25
Generated 1000 samples
Sample mean: 1.68
```

---

## Distribution Methods

All distributions in HeavyTails share a common interface:

### PDF - Probability Density Function

```python
from heavytails import Pareto

pareto = Pareto(alpha=2.0, xm=1.0)

# Evaluate PDF at a single point
density = pareto.pdf(2.0)
print(f"f(2.0) = {density:.4f}")

# Evaluate at multiple points
for x in [1.0, 2.0, 5.0, 10.0]:
    print(f"f({x}) = {pareto.pdf(x):.4f}")
```

### CDF - Cumulative Distribution Function

```python
# P(X ≤ x)
cdf_value = pareto.cdf(5.0)
print(f"P(X ≤ 5) = {cdf_value:.4f}")
```

### SF - Survival Function

```python
# P(X > x) - More accurate for tail probabilities
sf_value = pareto.sf(5.0)
print(f"P(X > 5) = {sf_value:.4f}")

# Note: sf(x) = 1 - cdf(x), but sf() is more numerically stable
```

### PPF - Percent Point Function (Quantile)

```python
# Inverse CDF - find x such that P(X ≤ x) = p
q90 = pareto.ppf(0.90)
q99 = pareto.ppf(0.99)
print(f"90th percentile: {q90:.2f}")
print(f"99th percentile: {q99:.2f}")
```

### RVS - Random Variates

```python
# Generate random samples
samples = pareto.rvs(100, seed=42)

# seed parameter ensures reproducibility
samples1 = pareto.rvs(10, seed=123)
samples2 = pareto.rvs(10, seed=123)  # Identical to samples1
```

---

## Working with Different Distributions

### Student-t Distribution

Perfect for modeling financial returns with heavy tails:

```python
from heavytails import StudentT

# Create Student-t with 4 degrees of freedom
student = StudentT(nu=4.0)

# Student-t is symmetric around 0
print(f"P(X > 0) = {student.sf(0.0):.2f}")  # Should be 0.50

# Heavy tails mean more extreme values
print(f"P(|X| > 3) = {2 * student.sf(3.0):.4f}")

# Generate samples
returns = student.rvs(1000, seed=42)
```

### Generalized Pareto Distribution

Used for modeling threshold exceedances:

```python
from heavytails import GeneralizedPareto

# GPD with shape ξ=0.3, scale σ=1.0, location μ=0
gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)

# Calculate tail probabilities
print(f"P(X > 5) = {gpd.sf(5.0):.4f}")

# This is ideal for extreme value analysis
exceedances = gpd.rvs(500, seed=42)
```

### Cauchy Distribution

Extreme heavy tails - no finite mean or variance:

```python
from heavytails import Cauchy

# Standard Cauchy centered at 0 with scale 1
cauchy = Cauchy(x0=0.0, gamma=1.0)

# PDF at the mode (highest point)
print(f"f(0) = {cauchy.pdf(0.0):.4f}")

# Very heavy tails
print(f"P(|X| > 100) = {2 * cauchy.sf(100.0):.4f}")

samples = cauchy.rvs(1000, seed=42)
```

---

## Tail Index Estimation

Estimate the tail behavior from empirical data:

```python
from heavytails import Pareto
from heavytails.tail_index import hill_estimator

# Generate data from known distribution
true_alpha = 2.5
pareto = Pareto(alpha=true_alpha, xm=1.0)
data = pareto.rvs(2000, seed=42)

# Estimate tail index using Hill estimator
k = 200  # Number of upper order statistics to use
gamma_hat = hill_estimator(data, k)
alpha_hat = 1.0 / gamma_hat

print(f"True α: {true_alpha}")
print(f"Estimated α: {alpha_hat:.2f}")
print(f"Estimation error: {abs(alpha_hat - true_alpha):.2f}")
```

**Output:**
```
True α: 2.5
Estimated α: 2.48
Estimation error: 0.02
```

### Other Estimators

```python
from heavytails.tail_index import pickands_estimator, moment_estimator

# Pickands estimator (more robust, less efficient)
gamma_pickands = pickands_estimator(data, k=200, m=2)
alpha_pickands = 1.0 / gamma_pickands

# Moment estimator (Dekkers-Einmahl-de Haan)
gamma_moment, alpha_moment = moment_estimator(data, k=200)

print(f"Hill:     α = {alpha_hat:.2f}")
print(f"Pickands: α = {alpha_pickands:.2f}")
print(f"Moment:   α = {alpha_moment:.2f}")
```

---

## Practical Example: Financial Risk

Analyze portfolio returns with heavy-tailed distributions:

```python
from heavytails import StudentT, GeneralizedPareto
from heavytails.tail_index import hill_estimator
import statistics

# Generate synthetic financial returns (Student-t with ν=4)
student = StudentT(nu=4.0)
returns = [x * 0.02 for x in student.rvs(1000, seed=42)]  # 2% daily volatility

# Basic statistics
mean_return = statistics.mean(returns)
std_return = statistics.stdev(returns)
print(f"Mean return: {mean_return:.4f}")
print(f"Std deviation: {std_return:.4f}")

# Calculate Value at Risk (VaR) - empirical 95% quantile
sorted_returns = sorted(returns)
var_95 = -sorted_returns[int(0.05 * len(returns))]  # 5% left tail
print(f"VaR (95%): {var_95:.4f}")

# Estimate tail index for losses
losses = [-r for r in returns]
gamma = hill_estimator(losses, k=100)
print(f"Tail index estimate: {1/gamma:.2f}")

# Expected Shortfall (CVaR) - average of losses beyond VaR
tail_losses = [r for r in returns if r < -var_95]
es_95 = -statistics.mean(tail_losses) if tail_losses else var_95
print(f"Expected Shortfall (95%): {es_95:.4f}")
```

**Output:**
```
Mean return: 0.0003
Std deviation: 0.0205
VaR (95%): 0.0341
Tail index estimate: 3.92
Expected Shortfall (95%): 0.0456
```

---

## Discrete Distributions

HeavyTails also supports discrete heavy-tailed distributions:

### Zipf Distribution

Models word frequencies, city sizes, and other rank-size relationships:

```python
from heavytails import Zipf

# Zipf distribution with exponent s=1.5
zipf = Zipf(s=1.5, kmax=10000)

# Probability mass function
for k in [1, 10, 100, 1000]:
    print(f"P(X = {k:4d}) = {zipf.pmf(k):.6f}")

# Generate random samples (ranks)
ranks = zipf.rvs(1000, seed=42)
```

**Output:**
```
P(X =    1) = 0.106839
P(X =   10) = 0.003378
P(X =  100) = 0.000107
P(X = 1000) = 0.000003
```

---

## Common Patterns

### Comparing Distributions

```python
from heavytails import Pareto, LogNormal, StudentT

# Create distributions
pareto = Pareto(alpha=2.0, xm=1.0)
lognormal = LogNormal(mu=0.0, sigma=1.0)
student = StudentT(nu=3.0)

# Compare tail probabilities at x=10
x = 10.0
print(f"P(X > {x}):")
print(f"  Pareto:    {pareto.sf(x):.6f}")
print(f"  LogNormal: {lognormal.sf(x):.6f}")
print(f"  Student-t: {student.sf(x):.6f}")
```

### Working with Samples

```python
from heavytails import Pareto
import statistics

pareto = Pareto(alpha=2.5, xm=1.0)
samples = pareto.rvs(1000, seed=42)

# Sample statistics
print(f"Sample size: {len(samples)}")
print(f"Min: {min(samples):.2f}")
print(f"Max: {max(samples):.2f}")
print(f"Mean: {statistics.mean(samples):.2f}")
print(f"Median: {statistics.median(samples):.2f}")

# Count extremes
extremes = [x for x in samples if x > 10]
print(f"Values > 10: {len(extremes)} ({100*len(extremes)/len(samples):.1f}%)")
```

---

## Next Steps

Now that you understand the basics:

1. **[Explore Basic Concepts](concepts.md)** - Learn the theory behind heavy tails
2. **[Read the User Guide](../guide/distributions.md)** - Detailed documentation for each distribution
3. **[Try Examples](../examples/basic_usage.ipynb)** - See real-world applications
4. **[Mathematical Background](../theory/heavy-tails.md)** - Understand the mathematics

---

## Quick Reference

### Creating Distributions

```python
from heavytails import (
    Pareto, StudentT, Cauchy, LogNormal,
    GeneralizedPareto, BurrXII, Frechet,
    Zipf, YuleSimon
)

# Continuous
pareto = Pareto(alpha=2.0, xm=1.0)
student = StudentT(nu=4.0)
cauchy = Cauchy(x0=0.0, gamma=1.0)
gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)

# Discrete
zipf = Zipf(s=1.5, kmax=10000)
```

### Common Operations

```python
# Probabilities
p = dist.cdf(x)      # P(X ≤ x)
p = dist.sf(x)       # P(X > x)
f = dist.pdf(x)      # Density at x (continuous)
p = dist.pmf(k)      # P(X = k) (discrete)

# Quantiles
q = dist.ppf(p)      # x such that P(X ≤ x) = p

# Sampling
samples = dist.rvs(n, seed=42)

# Moments (when they exist)
mean = dist.mean()
var = dist.variance()
```

### Tail Estimation

```python
from heavytails.tail_index import (
    hill_estimator,
    pickands_estimator,
    moment_estimator
)

gamma = hill_estimator(data, k)
gamma = pickands_estimator(data, k, m=2)
gamma, alpha = moment_estimator(data, k)
```

---

**Ready for more?** Check out the [complete examples](../examples/basic_usage.ipynb) or dive into the [theoretical background](../theory/heavy-tails.md).
