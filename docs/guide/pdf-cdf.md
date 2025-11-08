# Working with PDFs and CDFs

This guide covers how to work with Probability Density Functions (PDFs) and Cumulative Distribution Functions (CDFs) in the heavytails library.

## Overview

All distributions in heavytails provide four core methods:

- **PDF** (`pdf`): Probability Density Function
- **CDF** (`cdf`): Cumulative Distribution Function
- **SF** (`sf`): Survival Function (1 - CDF)
- **PPF** (`ppf`): Percent Point Function (inverse CDF/quantile function)

## Probability Density Function (PDF)

The PDF gives the relative likelihood of a continuous random variable taking on a specific value.

```python
from heavytails import Pareto

# Create a Pareto distribution
pareto = Pareto(alpha=2.5, xm=1.0)

# Evaluate PDF at a single point
density = pareto.pdf(2.0)
print(f"PDF at x=2.0: {density}")

# Evaluate PDF at multiple points
import math
x_values = [1.0, 2.0, 5.0, 10.0]
densities = [pareto.pdf(x) for x in x_values]
for x, p in zip(x_values, densities):
    print(f"PDF at x={x}: {p:.6f}")
```

### Key Properties

- PDF is always non-negative: `pdf(x) ≥ 0`
- For heavy-tailed distributions, PDF decays slowly for large x
- The integral of PDF over all values equals 1

## Cumulative Distribution Function (CDF)

The CDF gives the probability that a random variable is less than or equal to a given value.

```python
from heavytails import LogNormal

# Create a LogNormal distribution
lognormal = LogNormal(mu=0.0, sigma=1.0)

# Calculate probability P(X ≤ 2)
prob = lognormal.cdf(2.0)
print(f"P(X ≤ 2.0) = {prob:.4f}")

# Calculate probability in an interval P(a < X ≤ b)
a, b = 1.0, 3.0
prob_interval = lognormal.cdf(b) - lognormal.cdf(a)
print(f"P({a} < X ≤ {b}) = {prob_interval:.4f}")
```

### Key Properties

- CDF is monotonically increasing
- CDF ranges from 0 to 1
- CDF(−∞) = 0 and CDF(∞) = 1

## Survival Function (SF)

The survival function gives the probability that a random variable exceeds a given value: SF(x) = 1 - CDF(x)

```python
from heavytails import Cauchy

cauchy = Cauchy(x0=0.0, gamma=1.0)

# Calculate P(X > 5)
survival = cauchy.sf(5.0)
print(f"P(X > 5.0) = {survival:.6f}")

# For heavy tails, survival function decays slowly
x_values = [10, 50, 100, 500]
for x in x_values:
    print(f"P(X > {x}) = {cauchy.sf(x):.6e}")
```

### Why Use SF Instead of 1 - CDF?

For large values of x in heavy-tailed distributions, computing `1 - cdf(x)` can lose numerical precision. The `sf()` method provides more accurate results for tail probabilities.

## Percent Point Function (PPF)

The PPF is the inverse of the CDF - it returns the value x such that P(X ≤ x) = p.

```python
from heavytails import StudentT

# Create Student-t distribution with low degrees of freedom (heavy tails)
student_t = StudentT(nu=3.0)

# Find the median (50th percentile)
median = student_t.ppf(0.5)
print(f"Median: {median}")

# Find 95th percentile
p95 = student_t.ppf(0.95)
print(f"95th percentile: {p95}")

# Common quantiles
quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
values = [student_t.ppf(q) for q in quantiles]
for q, v in zip(quantiles, values):
    print(f"{q*100}th percentile: {v:.4f}")
```

### Relationship Between PPF and CDF

The PPF and CDF are inverse functions:

```python
from heavytails import Frechet

frechet = Frechet(alpha=2.0, s=1.0, m=0.0)

# Verify inverse relationship
x = 5.0
p = frechet.cdf(x)
x_recovered = frechet.ppf(p)
print(f"Original x: {x}")
print(f"After CDF then PPF: {x_recovered}")
print(f"Difference: {abs(x - x_recovered):.10f}")
```

## Working with Heavy Tails

Heavy-tailed distributions have special properties in their tails:

```python
from heavytails import GEV_Frechet

# GEV with heavy right tail (ξ > 0 is Fréchet type)
gev = GEV_Frechet(xi=0.3, mu=0.0, sigma=1.0)

# Tail probabilities decay more slowly than exponential
import math

print("Comparing tail decay:")
x_values = [5, 10, 20, 50]
for x in x_values:
    sf_val = gev.sf(x)
    exp_decay = math.exp(-x)  # Exponential decay for comparison
    print(f"x={x:2d}: SF={sf_val:.6e}, exp(-x)={exp_decay:.6e}")
```

## Practical Examples

### Risk Analysis

Calculate Value at Risk (VaR) and Expected Shortfall:

```python
from heavytails import BurrXII

# Model portfolio returns with heavy-tailed distribution
burr = BurrXII(c=1.5, k=2.0, s=0.1)

# Value at Risk at 95% confidence level
var_95 = burr.ppf(0.05)  # 5th percentile for losses
print(f"VaR(95%): {var_95:.4f}")

# Expected Shortfall (CVaR) approximation
# Average loss beyond VaR
n_samples = 10000
samples = burr.rvs(n_samples, seed=42)
losses_beyond_var = [s for s in samples if s < var_95]
es_95 = sum(losses_beyond_var) / len(losses_beyond_var) if losses_beyond_var else 0
print(f"Expected Shortfall(95%): {es_95:.4f}")
```

### Hypothesis Testing

Test if data comes from a specific distribution:

```python
from heavytails import Weibull

# Theoretical distribution (heavy-tailed Weibull with k < 1)
weibull = Weibull(k=0.8, lam=1.0)

# Generate sample data
sample_data = weibull.rvs(100, seed=123)

# Calculate empirical CDF at various points
test_points = [0.5, 1.0, 2.0, 5.0]
for x in test_points:
    theoretical_cdf = weibull.cdf(x)
    empirical_cdf = sum(1 for d in sample_data if d <= x) / len(sample_data)
    diff = abs(theoretical_cdf - empirical_cdf)
    print(f"x={x}: Theoretical CDF={theoretical_cdf:.4f}, "
          f"Empirical CDF={empirical_cdf:.4f}, Diff={diff:.4f}")
```

## Advanced Usage

### Handling Extreme Values

```python
from heavytails import GeneralizedPareto

# GPD for modeling exceedances over threshold
gpd = GeneralizedPareto(xi=0.2, sigma=1.0, mu=0.0)

# Be careful with extreme quantiles
try:
    # Very extreme quantile
    extreme = gpd.ppf(0.9999)
    print(f"99.99th percentile: {extreme}")

    # Verify it's numerically stable
    prob_check = gpd.cdf(extreme)
    print(f"Verification: CDF({extreme}) = {prob_check:.6f}")
except OverflowError:
    print("Quantile too extreme for numerical computation")
```

### Comparing Distributions

```python
from heavytails import Pareto, LogNormal, Cauchy

# Compare tail behavior of different distributions
distributions = {
    'Pareto': Pareto(alpha=2.0, xm=1.0),
    'LogNormal': LogNormal(mu=0.0, sigma=1.5),
    'Cauchy': Cauchy(x0=0.0, gamma=1.0)
}

print("\\nTail probability P(X > x):")
print(f"{'x':>6s} " + " ".join(f"{name:>12s}" for name in distributions.keys()))
for x in [5, 10, 20, 50, 100]:
    tail_probs = [dist.sf(x) for dist in distributions.values()]
    print(f"{x:6d} " + " ".join(f"{p:12.6e}" for p in tail_probs))
```

## Best Practices

1. **Use `sf()` for tail probabilities**: More numerically stable than `1 - cdf(x)`
2. **Check parameter validity**: Invalid parameters will raise `ParameterError`
3. **Be aware of support**: Each distribution has a specific support (domain)
4. **Test numerical stability**: For extreme parameters, verify results
5. **Use appropriate precision**: Heavy tails can have very small/large values

## Next Steps

- [Random Sampling](sampling.md) - Generate random samples
- [Parameter Fitting](fitting.md) - Estimate parameters from data
- [Tail Index Estimation](tail-estimation.md) - Estimate tail behavior
- [Diagnostic Tools](diagnostics.md) - Validate distribution fits
