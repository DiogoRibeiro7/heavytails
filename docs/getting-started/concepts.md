# Basic Concepts

This guide introduces the fundamental concepts of heavy-tailed distributions, providing the mathematical and intuitive foundation for using the HeavyTails library.

--------------------------------------------------------------------------------

## What Makes a Distribution "Heavy-Tailed"?

A distribution is **heavy-tailed** if it has more probability mass in its tails compared to the exponential distribution. Intuitively, extreme events occur more frequently than in "light-tailed" distributions like the Normal or Exponential.

### Mathematical Definition

A random variable $X$ has a **heavy right tail** if:

$$ \lim_{x \to \infty} e^{\lambda x} P(X > x) = \infty \quad \text{for all } \lambda > 0 $$

This means the tail probability $P(X > x)$ decays slower than any exponential function $e^{-\lambda x}$.

### Intuitive Comparison

Consider the probability of an extreme event at $x = 10\sigma$ (10 standard deviations):

Distribution        | $P(X > 10\sigma)$  | Interpretation
------------------- | ------------------ | ----------------------
Normal              | $\approx 10^{-23}$ | Essentially impossible
Exponential         | $\approx 10^{-5}$  | Very rare
Pareto ($\alpha=2$) | $\approx 0.01$     | Happens regularly!

Heavy-tailed distributions assign significant probability to extreme outcomes that would be negligible under light-tailed assumptions.

--------------------------------------------------------------------------------

## Types of Heavy Tails

### Power-Law Tails (Pareto Type)

The strongest form of heavy tails, characterized by **regular variation**:

$$ P(X > x) \sim x^{-\alpha} L(x) \quad \text{as } x \to \infty $$

where $L(x)$ is a slowly varying function (roughly constant on log scale) and $\alpha > 0$ is the **tail index**.

**Key Property:** Scale invariance $$ \frac{P(X > tx)}{P(X > x)} \to t^{-\alpha} \quad \text{as } x \to \infty $$

**Examples:** Pareto, Student-t, Cauchy, Burr XII, Generalized Pareto (when $\xi > 0$)

**Applications:** Wealth distribution, earthquake magnitudes, city sizes, insurance claims

### Log-Heavy Tails

Slower decay than power laws but still heavier than exponential:

$$ P(X > x) \sim \frac{1}{x (\log x)^\beta} $$

**Examples:** Log-Normal (in the far tail)

**Applications:** Multiplicative growth processes, stock prices over long horizons

--------------------------------------------------------------------------------

## The Tail Index

For power-law tails, the **tail index** $\alpha$ determines how heavy the tail is:

### Interpretation

- **Smaller $\alpha$** → Heavier tails → More extreme events
- **Larger $\alpha$** → Lighter tails → Fewer extreme events

### Moment Existence

The tail index determines which moments exist:

$$ \mathbb{E}[X^p] < \infty \iff p < \alpha $$

Tail Index $\alpha$ | Existing Moments               | Example
------------------- | ------------------------------ | -------------------------------------------
$0 < \alpha \leq 1$ | No mean                        | Cauchy ($\alpha = 1$)
$1 < \alpha \leq 2$ | Mean exists, no variance       | Student-t with $\nu=3$ ($\alpha \approx 3$)
$2 < \alpha \leq 4$ | Variance exists, high kurtosis | Student-t with $\nu=5$ ($\alpha \approx 5$)
$\alpha > 4$        | Higher moments exist           | Less heavy tails

### Examples

```python
from heavytails import Pareto, StudentT, Cauchy

# Pareto with α=1.5: E[X] is infinite
pareto_heavy = Pareto(alpha=1.5, xm=1.0)
print(f"Mean (α=1.5): infinite")

# Pareto with α=2.5: E[X] exists, Var[X] is infinite
pareto_moderate = Pareto(alpha=2.5, xm=1.0)
print(f"Mean (α=2.5): {pareto_moderate.mean():.2f}")

# Cauchy: No finite moments
cauchy = Cauchy(x0=0.0, gamma=1.0)
print(f"Cauchy mean: undefined")
```

--------------------------------------------------------------------------------

## Why Heavy Tails Matter

### 1\. Real-World Phenomena

Many natural and social phenomena exhibit heavy tails:

- **Finance:** Stock market crashes, extreme returns
- **Insurance:** Catastrophic claims, natural disasters
- **Networks:** Internet traffic, degree distributions
- **Natural Disasters:** Earthquake magnitudes, flood levels
- **Social Systems:** Wealth inequality, city sizes, citations

### 2\. Breakdown of Classical Statistics

Heavy tails violate assumptions underlying many statistical methods:

#### Central Limit Theorem Failure

For $\alpha \leq 2$, sample means don't converge to a Normal distribution:

```python
from heavytails import Cauchy
import statistics

cauchy = Cauchy(x0=0.0, gamma=1.0)

# Average of 1000 Cauchy samples is still Cauchy!
sample_means = []
for _ in range(1000):
    samples = cauchy.rvs(1000, seed=None)
    sample_means.append(statistics.mean(samples))

# sample_means does NOT become Normal - remains heavy-tailed
```

#### Law of Large Numbers Failure

When the mean doesn't exist ($\alpha \leq 1$), sample averages don't converge:

```python
from heavytails import Pareto

# α=0.5: no mean exists
pareto = Pareto(alpha=0.5, xm=1.0)

# Sample mean is unstable
means = []
for n in [100, 1000, 10000]:
    samples = pareto.rvs(n, seed=42)
    means.append(statistics.mean(samples))

print(means)  # No convergence!
```

### 3\. Risk Underestimation

Using light-tailed models (e.g., Normal) when data is heavy-tailed leads to severe underestimation of tail risk:

```python
from heavytails import StudentT
import math

# True data generating process: Student-t(ν=4)
student = StudentT(nu=4.0)

# 99% VaR under Student-t
var_true = student.ppf(0.99)

# Misspecified 99% VaR assuming Normal
# For Normal(0,1): 99% quantile ≈ 2.33
var_normal = 2.33

print(f"True VaR (Student-t): {var_true:.2f}")
print(f"Misspecified VaR (Normal): {var_normal:.2f}")
print(f"Underestimation factor: {var_true/var_normal:.1f}x")
```

--------------------------------------------------------------------------------

## Regular Variation

A function $L(x)$ is **slowly varying** if:

$$ \lim_{x \to \infty} \frac{L(tx)}{L(x)} = 1 \quad \text{for all } t > 0 $$

A function $f(x)$ is **regularly varying** with index $\alpha$ if:

$$ f(x) = x^\alpha L(x) $$

where $L(x)$ is slowly varying.

### Examples

- **Slowly varying:** $\log x$, $\log \log x$, constants
- **Regularly varying (α=2):** $x^2$, $x^2 \log x$
- **Regularly varying (α=-1):** $1/x$, $1/(x \log x)$

### Why It Matters

For regularly varying tails:

1. **Scale invariance:** Relative tail probabilities depend only on ratios
2. **Asymptotic equivalence:** $P(X > x) \sim Cx^{-\alpha}$ for some constant $C$
3. **Estimation theory:** Hill estimator and other methods work

--------------------------------------------------------------------------------

## Extreme Value Theory

Heavy-tailed distributions are central to **extreme value theory** (EVT), which studies the behavior of maximum and minimum values.

### Fisher-Tippett-Gnedenko Theorem

The maximum of $n$ iid random variables converges (after normalization) to one of three types:

1. **Gumbel** ($\xi = 0$): Exponential-type tails (e.g., Normal, Exponential)
2. **Fréchet** ($\xi > 0$): Heavy tails (e.g., Pareto, Student-t)
3. **Weibull** ($\xi < 0$): Bounded support (e.g., Uniform)

**Connection:** $\xi = 1/\alpha$ for Pareto-type tails

### Generalized Extreme Value (GEV) Distribution

Unified family containing all three types:

$$ F(x) = \exp\left{-\left[1 + \xi \frac{x-\mu}{\sigma}\right]^{-1/\xi}\right} $$

- $\xi > 0$: Heavy-tailed (Fréchet domain)
- $\xi = 0$: Exponential-tailed (Gumbel domain)
- $\xi < 0$: Bounded (Weibull domain)

--------------------------------------------------------------------------------

## Peaks Over Threshold

An alternative EVT approach: model exceedances over a high threshold $u$.

### Balkema-de Haan-Pickands Theorem

Excesses $Y = X - u | X > u$ asymptotically follow a **Generalized Pareto Distribution** (GPD):

$$ F_u(y) = 1 - \left(1 + \xi \frac{y}{\sigma}\right)^{-1/\xi} $$

**Applications:**

- Financial risk: VaR and Expected Shortfall
- Insurance: Large claim modeling
- Hydrology: Flood frequency analysis

```python
from heavytails import GeneralizedPareto

# Model excesses over threshold u=10
gpd = GeneralizedPareto(xi=0.25, sigma=5.0, mu=10.0)

# Probability of exceeding u+20 given exceeding u
prob = gpd.sf(30.0)  # P(X > 30 | X > 10)
```

--------------------------------------------------------------------------------

## Tail Dependence

Heavy tails affect not just marginals but also dependence structures:

### Asymptotic Tail Dependence

For bivariate $(X, Y)$:

$$ \lambda_U = \lim_{u \to 1^-} P(Y > F_Y^{-1}(u) | X > F_X^{-1}(u)) $$

- $\lambda_U > 0$: Asymptotic tail dependence (extremes occur together)
- $\lambda_U = 0$: Asymptotic tail independence

**Example:** Normal copula has $\lambda_U = 0$, but Student-t copula has $\lambda_U > 0$

--------------------------------------------------------------------------------

## Summary

### Key Takeaways

1. **Heavy tails** → extreme events more likely than exponential
2. **Power-law tails** → $P(X > x) \sim x^{-\alpha}$
3. **Tail index $\alpha$** → determines moment existence and tail heaviness
4. **Classical statistics fail** → CLT, LLN break down
5. **Extreme value theory** → mathematical framework for extremes
6. **Real-world relevance** → finance, insurance, natural disasters

### When to Use Heavy-Tailed Distributions

Use heavy-tailed models when:

- Data exhibits frequent extreme values
- Sample kurtosis is high (> 6)
- QQ-plots show departure from Normal in tails
- Log-log tail plots are approximately linear
- Domain knowledge suggests power laws (wealth, networks, etc.)

### When NOT to Use Heavy-Tailed Distributions

Avoid heavy-tailed models when:

- Extremes are physically bounded
- Data is discrete counts with low variance
- Theoretical reasons suggest light tails
- Sample size is too small to assess tails

--------------------------------------------------------------------------------

## Visual Intuition

### Tail Comparison

Heavy vs. light tails on a log-log scale:

```python
from heavytails import Pareto, LogNormal
import math

# Create distributions
pareto = Pareto(alpha=2.0, xm=1.0)
lognormal = LogNormal(mu=0.0, sigma=1.0)

# Compare tail probabilities
for x in [10, 100, 1000, 10000]:
    p_pareto = pareto.sf(x)
    p_lognorm = lognormal.sf(x)
    print(f"P(X > {x:5d}): Pareto = {p_pareto:.2e}, LogNormal = {p_lognorm:.2e}")
```

On log-log scale:

- **Pareto:** Straight line with slope $-\alpha$
- **Log-Normal:** Curved downward (faster decay eventually)

--------------------------------------------------------------------------------

## Next Steps

- **[User Guide](../guide/distributions.md)** - Detailed distribution documentation
- **[Examples](../examples/basic_usage.ipynb)** - Practical applications
- **[Theory](../theory/heavy-tails.md)** - Rigorous mathematical treatment

--------------------------------------------------------------------------------

## Further Reading

### Books

- Embrechts, P., Klüppelberg, C., & Mikosch, T. (1997). _Modelling Extremal Events_
- Resnick, S. I. (2007). _Heavy-Tail Phenomena: Probabilistic and Statistical Modeling_
- Foss, S., Korshunov, D., & Zachary, S. (2013). _An Introduction to Heavy-Tailed and Subexponential Distributions_

### Papers

- Hill, B. M. (1975). "A Simple General Approach to Inference About the Tail of a Distribution"
- Pickands, J. (1975). "Statistical Inference Using Extreme Order Statistics"
- Beirlant, J., et al. (1996). "Tail Index Estimation and an Exponential Regression Model"
