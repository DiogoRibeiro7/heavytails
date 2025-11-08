# Random Sampling

This guide covers how to generate random samples from heavy-tailed distributions using the heavytails library.

## Overview

All distributions in heavytails implement the `rvs()` method for generating random variates. The library uses a deterministic random number generator for reproducibility.

## Basic Sampling

### Single Distribution

```python
from heavytails import Pareto

# Create a Pareto distribution
pareto = Pareto(alpha=2.5, xm=1.0)

# Generate 10 random samples
samples = pareto.rvs(10, seed=42)
print("Samples:", samples)

# Generate more samples with same seed for reproducibility
samples_again = pareto.rvs(10, seed=42)
assert samples == samples_again  # Same results!
```

### Multiple Distributions

```python
from heavytails import Cauchy, StudentT, LogNormal

distributions = {
    'Cauchy': Cauchy(x0=0.0, gamma=1.0),
    'Student-t': StudentT(nu=3.0),
    'LogNormal': LogNormal(mu=0.0, sigma=1.5)
}

# Generate samples from each
n_samples = 1000
samples_dict = {}
for name, dist in distributions.items():
    samples_dict[name] = dist.rvs(n_samples, seed=42)
    print(f"{name}: min={min(samples_dict[name]):.2f}, "
          f"max={max(samples_dict[name]):.2f}")
```

## Reproducibility

The `seed` parameter ensures reproducible results:

```python
from heavytails import Frechet

frechet = Frechet(alpha=2.0, s=1.0, m=0.0)

# Same seed = same samples
sample1 = frechet.rvs(5, seed=123)
sample2 = frechet.rvs(5, seed=123)
print("Sample 1:", sample1)
print("Sample 2:", sample2)
print("Equal:", sample1 == sample2)

# Different seed = different samples
sample3 = frechet.rvs(5, seed=456)
print("Sample 3:", sample3)
print("Different from 1:", sample1 != sample3)
```

## Large-Scale Sampling

Generate thousands or millions of samples efficiently:

```python
from heavytails import Weibull
import time

weibull = Weibull(k=0.8, lam=1.0)

# Time large sample generation
n_samples = 100_000
start = time.time()
samples = weibull.rvs(n_samples, seed=42)
elapsed = time.time() - start

print(f"Generated {n_samples:,} samples in {elapsed:.3f} seconds")
print(f"Rate: {n_samples/elapsed:,.0f} samples/second")
```

## Sample Statistics

Analyze generated samples:

```python
from heavytails import GEV_Frechet
import math

gev = GEV_Frechet(xi=0.3, mu=0.0, sigma=1.0)
samples = gev.rvs(10000, seed=42)

# Basic statistics
n = len(samples)
mean_sample = sum(samples) / n
variance_sample = sum((x - mean_sample)**2 for x in samples) / (n - 1)
std_sample = math.sqrt(variance_sample)

print(f"Sample size: {n}")
print(f"Sample mean: {mean_sample:.4f}")
print(f"Sample std: {std_sample:.4f}")
print(f"Sample min: {min(samples):.4f}")
print(f"Sample max: {max(samples):.4f}")

# Quantiles
sorted_samples = sorted(samples)
quantiles = [0.25, 0.50, 0.75, 0.95, 0.99]
for q in quantiles:
    idx = int(q * n)
    print(f"{q*100}th percentile: {sorted_samples[idx]:.4f}")
```

## Discrete Distributions

Sampling from discrete heavy-tailed distributions:

```python
from heavytails import Zipf, YuleSimon, DiscretePareto

# Zipf distribution (word frequencies, city sizes)
zipf = Zipf(s=2.0, kmax=1000)
zipf_samples = zipf.rvs(100, seed=42)
print("Zipf samples:", zipf_samples[:10])

# Yule-Simon distribution (species abundance)
yule = YuleSimon(rho=2.0)
yule_samples = yule.rvs(100, seed=42)
print("Yule-Simon samples:", yule_samples[:10])

# Discrete Pareto
discrete_pareto = DiscretePareto(alpha=2.5, k_min=1, k_max=1000)
dp_samples = discrete_pareto.rvs(100, seed=42)
print("Discrete Pareto samples:", dp_samples[:10])
```

## Monte Carlo Simulation

Use sampling for Monte Carlo estimation:

```python
from heavytails import BurrXII

burr = BurrXII(c=1.5, k=2.0, s=1.0)

# Estimate probability P(X > 5) using Monte Carlo
n_sim = 100_000
samples = burr.rvs(n_sim, seed=42)
monte_carlo_prob = sum(1 for x in samples if x > 5) / n_sim

# Compare with analytical result
analytical_prob = burr.sf(5.0)

print(f"Monte Carlo estimate: {monte_carlo_prob:.6f}")
print(f"Analytical result: {analytical_prob:.6f}")
print(f"Difference: {abs(monte_carlo_prob - analytical_prob):.6f}")
```

## Advanced Sampling Techniques

### Stratified Sampling

Sample from different regions of the distribution:

```python
from heavytails import LogLogistic

loglogistic = LogLogistic(alpha=2.0, beta=1.0)

# Sample from different quantile ranges
n_per_stratum = 250
strata = [(0.0, 0.25), (0.25, 0.75), (0.75, 1.0)]

stratified_samples = []
for lower, upper in strata:
    # Sample uniform probabilities in stratum
    import random
    random.seed(42)
    u_samples = [random.uniform(lower, upper) for _ in range(n_per_stratum)]
    # Convert to distribution samples using PPF
    stratum_samples = [loglogistic.ppf(u) for u in u_samples]
    stratified_samples.extend(stratum_samples)

print(f"Generated {len(stratified_samples)} stratified samples")
```

### Importance Sampling

Sample from tail regions more frequently:

```python
from heavytails import InverseGamma

inv_gamma = InverseGamma(alpha=2.0, beta=1.0)

# Focus sampling on tail (upper 10%)
n_samples = 1000
tail_threshold = inv_gamma.ppf(0.9)

# Generate samples and focus on tail
all_samples = inv_gamma.rvs(n_samples * 10, seed=42)
tail_samples = [x for x in all_samples if x > tail_threshold][:n_samples]

print(f"Tail threshold (90th percentile): {tail_threshold:.4f}")
print(f"Collected {len(tail_samples)} tail samples")
print(f"Mean of tail samples: {sum(tail_samples)/len(tail_samples):.4f}")
```

### Conditional Sampling

Sample from truncated distributions:

```python
from heavytails import BetaPrime

beta_prime = BetaPrime(alpha=2.0, beta=3.0)

# Sample X | X > 1 (condition on exceeding threshold)
threshold = 1.0
cdf_threshold = beta_prime.cdf(threshold)

# Generate conditional samples
n_samples = 1000
u_samples = []
import random
random.seed(42)
for _ in range(n_samples):
    # Sample from uniform(cdf(threshold), 1)
    u = random.uniform(cdf_threshold, 1.0)
    u_samples.append(u)

conditional_samples = [beta_prime.ppf(u) for u in u_samples]

print(f"All conditional samples > {threshold}: {all(x > threshold for x in conditional_samples)}")
print(f"Mean of conditional samples: {sum(conditional_samples)/len(conditional_samples):.4f}")
```

## Practical Applications

### Portfolio Simulation

Simulate portfolio returns with heavy tails:

```python
from heavytails import StudentT

# Student-t with low degrees of freedom for fat tails
returns = StudentT(nu=4.0)

# Simulate 1 year of daily returns (252 trading days)
n_days = 252
daily_returns = returns.rvs(n_days, seed=42)

# Calculate cumulative returns
cumulative_return = 1.0
for r in daily_returns:
    cumulative_return *= (1 + r/100)

print(f"Cumulative return: {(cumulative_return - 1) * 100:.2f}%")
print(f"Worst daily return: {min(daily_returns):.2f}%")
print(f"Best daily return: {max(daily_returns):.2f}%")
```

### Extreme Event Simulation

Simulate rare extreme events:

```python
from heavytails import GeneralizedPareto

# GPD for exceedances over high threshold
gpd = GeneralizedPareto(xi=0.25, sigma=10.0, mu=0.0)

# Simulate extreme losses
n_events = 1000
extreme_losses = gpd.rvs(n_events, seed=42)

# Analyze extreme events
threshold_percentiles = [0.90, 0.95, 0.99]
sorted_losses = sorted(extreme_losses, reverse=True)

print("Extreme loss percentiles:")
for p in threshold_percentiles:
    idx = int(p * n_events)
    print(f"{p*100}th percentile: ${sorted_losses[idx]:,.2f}")
```

### Network Traffic Simulation

Simulate heavy-tailed network traffic:

```python
from heavytails import Pareto

# Pareto for file size distribution (web traffic)
file_sizes = Pareto(alpha=1.5, xm=1024)  # xm = 1KB minimum

# Simulate 1000 file transfers
n_transfers = 1000
sizes = file_sizes.rvs(n_transfers, seed=42)

# Analyze traffic
total_data = sum(sizes)
avg_size = total_data / n_transfers

print(f"Total data transferred: {total_data/1024/1024:.2f} MB")
print(f"Average file size: {avg_size/1024:.2f} KB")
print(f"Largest file: {max(sizes)/1024:.2f} KB")

# Check heavy tail property: 20% of files account for 80% of traffic
sizes_sorted = sorted(sizes, reverse=True)
top_20_percent = sizes_sorted[:n_transfers//5]
print(f"Top 20% files account for {sum(top_20_percent)/total_data*100:.1f}% of traffic")
```

## Performance Considerations

### Batch Size

For very large samples, consider batch generation:

```python
from heavytails import Frechet

frechet = Frechet(alpha=2.0, s=1.0, m=0.0)

# Generate in batches to manage memory
total_needed = 1_000_000
batch_size = 100_000
all_samples = []

for i in range(0, total_needed, batch_size):
    batch = frechet.rvs(batch_size, seed=42+i)
    # Process batch immediately instead of storing all
    batch_mean = sum(batch) / len(batch)
    print(f"Batch {i//batch_size + 1}: mean = {batch_mean:.4f}")
    all_samples.extend(batch)

print(f"Generated {len(all_samples):,} total samples")
```

## Best Practices

1. **Always use seeds** for reproducibility in research and testing
2. **Validate samples** by checking basic statistics match theory
3. **Be aware of memory** when generating millions of samples
4. **Use appropriate n** for accurate Monte Carlo estimates
5. **Check for numerical issues** with extreme parameters

## Next Steps

- [Parameter Fitting](fitting.md) - Estimate distribution parameters from data
- [Tail Index Estimation](tail-estimation.md) - Estimate tail behavior
- [Diagnostic Tools](diagnostics.md) - Validate sample quality
- [Working with PDFs and CDFs](pdf-cdf.md) - Understand probability functions
