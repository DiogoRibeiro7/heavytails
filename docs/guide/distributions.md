# Distributions Overview

HeavyTails provides a comprehensive collection of heavy-tailed probability distributions, covering continuous, discrete, and extreme value families.

--------------------------------------------------------------------------------

## Quick Reference Table

### Continuous Distributions

Distribution      | Parameters                                               | Support           | Tail Index           | Mean                                       | Variance
----------------- | -------------------------------------------------------- | ----------------- | -------------------- | ------------------------------------------ | -----------------------------
**Pareto**        | $\alpha > 0$, $x_m > 0$                                  | $[x_m, \infty)$   | $\alpha$             | $\frac{\alpha x_m}{\alpha-1}$ ($\alpha>1$) | Exists if $\alpha>2$
**Student-t**     | $\nu > 0$                                                | $\mathbb{R}$      | $\nu$                | $0$ ($\nu>1$)                              | $\frac{\nu}{\nu-2}$ ($\nu>2$)
**Cauchy**        | $x_0 \in \mathbb{R}$, $\gamma > 0$                       | $\mathbb{R}$      | $1$                  | Undefined                                  | Undefined
**Log-Normal**    | $\mu \in \mathbb{R}$, $\sigma > 0$                       | $(0, \infty)$     | $\infty$ (log-heavy) | $e^{\mu + \sigma^2/2}$                     | Exists
**GPD**           | $\xi \in \mathbb{R}$, $\sigma > 0$, $\mu \in \mathbb{R}$ | Varies with $\xi$ | $1/\xi$ ($\xi>0$)    | $\mu + \frac{\sigma}{1-\xi}$ ($\xi<1$)     | Exists if $\xi<1/2$
**Burr XII**      | $c > 0$, $k > 0$, $s > 0$                                | $(0, \infty)$     | $ck$                 | Exists if $c>1$                            | Exists if $c>2$
**Fréchet**       | $\alpha > 0$, $s > 0$, $m \in \mathbb{R}$                | $(m, \infty)$     | $\alpha$             | Exists if $\alpha>1$                       | Exists if $\alpha>2$
**GEV**           | $\xi > 0$, $\mu \in \mathbb{R}$, $\sigma > 0$            | Varies with $\xi$ | $1/\xi$              | Exists if $\xi<1$                          | Exists if $\xi<1/2$
**Log-Logistic**  | $\alpha > 0$, $\beta > 0$                                | $(0, \infty)$     | $\beta$              | Exists if $\beta>1$                        | Exists if $\beta>2$
**Inverse-Gamma** | $\alpha > 0$, $\beta > 0$                                | $(0, \infty)$     | $\alpha$             | $\frac{\beta}{\alpha-1}$ ($\alpha>1$)      | Exists if $\alpha>2$
**Beta-Prime**    | $\alpha > 0$, $\beta > 0$                                | $(0, \infty)$     | $\beta$              | $\frac{\alpha}{\beta-1}$ ($\beta>1$)       | Exists if $\beta>2$

### Discrete Distributions

Distribution        | Parameters            | Support                   | Tail Behavior               | Mean
------------------- | --------------------- | ------------------------- | --------------------------- | --------------------
**Zipf**            | $s > 1$, $k_{max}$    | ${1, 2, \ldots, k_{max}}$ | $P(X=k) \sim k^{-s}$        | Finite
**Yule-Simon**      | $\rho > 0$            | ${1, 2, 3, \ldots}$       | $P(X=k) \sim k^{-(\rho+1)}$ | Finite
**Discrete Pareto** | $\alpha > 0$, $L > 0$ | ${L, L+1, L+2, \ldots}$   | $P(X=k) \sim k^{-\alpha-1}$ | Exists if $\alpha>1$

--------------------------------------------------------------------------------

## Continuous Heavy-Tailed Distributions

### Pareto Distribution

The canonical power-law distribution.

**PDF:** $$ f(x) = \frac{\alpha x_m^\alpha}{x^{\alpha+1}}, \quad x \geq x_m $$

**CDF:** $$ F(x) = 1 - \left(\frac{x_m}{x}\right)^\alpha $$

**Usage:**

```python
from heavytails import Pareto

# Create Pareto with tail index α=2.5, scale xₘ=1.0
pareto = Pareto(alpha=2.5, xm=1.0)

# Properties
print(f"Mean: {pareto.mean():.2f}")                # 1.67
print(f"Variance: {pareto.variance():.2f}")        # 1.11
print(f"P(X > 10): {pareto.sf(10):.4f}")          # 0.0063

# Sampling
samples = pareto.rvs(1000, seed=42)
```

**Applications:**

- Wealth and income distribution
- City sizes and firm sizes
- Word frequencies
- Insurance claim sizes

**Parameter Constraints:**

- $\alpha > 0$ (tail index): Lower values → heavier tails
- $x_m > 0$ (scale): Minimum possible value

--------------------------------------------------------------------------------

### Student-t Distribution

Symmetric heavy-tailed distribution, fundamental in robust statistics.

**PDF:** $$ f(x) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}} $$

**Usage:**

```python
from heavytails import StudentT

# Create Student-t with ν=4 degrees of freedom
student = StudentT(nu=4.0)

# Symmetric distribution
print(f"P(X > 0): {student.sf(0):.2f}")           # 0.50
print(f"Median: {student.ppf(0.5):.2f}")          # 0.00

# Heavier tails than Normal
print(f"P(|X| > 3): {2*student.sf(3):.4f}")       # Much larger than Normal

samples = student.rvs(1000, seed=42)
```

**Applications:**

- Financial asset returns
- Robust regression
- Bayesian inference
- Hypothesis testing with outliers

**Parameter Constraints:**

- $\nu > 0$ (degrees of freedom): Lower $\nu$ → heavier tails
- $\nu = 1$: Cauchy distribution
- $\nu \to \infty$: Normal distribution

--------------------------------------------------------------------------------

### Cauchy Distribution

Extreme heavy tails with no finite moments.

**PDF:** $$ f(x) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]} $$

**Usage:**

```python
from heavytails import Cauchy

# Standard Cauchy
cauchy = Cauchy(x0=0.0, gamma=1.0)

# No finite mean!
print(f"Mode: {cauchy.ppf(0.5):.2f}")              # 0.00
print(f"Median: {cauchy.ppf(0.5):.2f}")            # 0.00
print(f"P(|X| > 100): {2*cauchy.sf(100):.4f}")     # Still significant!

samples = cauchy.rvs(1000, seed=42)
# Sample mean is unreliable!
```

**Applications:**

- Ratio of independent Normal variables
- Resonance in physics
- Anomaly detection
- Teaching examples of heavy tails

**Parameter Constraints:**

- $x_0 \in \mathbb{R}$ (location): Center/mode
- $\gamma > 0$ (scale): Spread parameter

--------------------------------------------------------------------------------

### Generalized Pareto Distribution (GPD)

The foundation of peaks-over-threshold extreme value modeling.

**PDF:** $$ f(x) = \frac{1}{\sigma}\left(1 + \xi\frac{x-\mu}{\sigma}\right)^{-1/\xi - 1} $$

**Usage:**

```python
from heavytails import GeneralizedPareto

# Heavy-tailed case (ξ > 0)
gpd = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)

# Threshold exceedances
threshold = 5.0
exceedance_prob = gpd.sf(threshold)
print(f"P(X > {threshold}): {exceedance_prob:.4f}")

# VaR calculation
var_99 = gpd.ppf(0.99)
print(f"99% VaR: {var_99:.2f}")

samples = gpd.rvs(1000, seed=42)
```

**Applications:**

- Financial risk management (VaR, ES)
- Insurance extreme claims
- Environmental extremes
- Reliability engineering

**Parameter Constraints:**

- $\xi \in \mathbb{R}$ (shape): $\xi > 0$ for heavy tails
- $\sigma > 0$ (scale)
- $\mu \in \mathbb{R}$ (location/threshold)

**Special Cases:**

- $\xi = 0$: Exponential distribution
- $\xi > 0$: Pareto-type (heavy tail)
- $\xi < 0$: Bounded support

--------------------------------------------------------------------------------

### Log-Normal Distribution

Heavy-tailed from multiplicative processes.

**PDF:** $$ f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right), \quad x > 0 $$

**Usage:**

```python
from heavytails import LogNormal

# Create Log-Normal
lognorm = LogNormal(mu=0.0, sigma=1.0)

# All moments exist
print(f"Mean: {lognorm.mean():.2f}")
print(f"Variance: {lognorm.variance():.2f}")

# Heavy right tail
print(f"P(X > 10): {lognorm.sf(10):.4f}")

samples = lognorm.rvs(1000, seed=42)
```

**Applications:**

- Stock prices and returns (multiplicative model)
- Income distribution
- Particle sizes
- Biological measurements

**Parameter Constraints:**

- $\mu \in \mathbb{R}$ (log-scale mean)
- $\sigma > 0$ (log-scale standard deviation)

--------------------------------------------------------------------------------

### Burr XII Distribution

Flexible heavy-tailed family with two shape parameters.

**PDF:** $$ f(x) = \frac{ck}{s}\left(\frac{x}{s}\right)^{c-1}\left[1 + \left(\frac{x}{s}\right)^c\right]^{-k-1}, \quad x > 0 $$

**Usage:**

```python
from heavytails import BurrXII

# Create Burr XII
burr = BurrXII(c=1.5, k=2.0, s=1.0)

# Flexible tail behavior
print(f"Tail index: {burr.c * burr.k:.2f}")

samples = burr.rvs(1000, seed=42)
```

**Applications:**

- Actuarial modeling
- Reliability analysis
- Income/wealth modeling
- Flexible parametric fitting

**Parameter Constraints:**

- $c > 0$ (first shape)
- $k > 0$ (second shape)
- $s > 0$ (scale)
- Tail index: $\alpha = ck$

--------------------------------------------------------------------------------

## Discrete Heavy-Tailed Distributions

### Zipf Distribution

Discrete power law - Zipf's law for ranks.

**PMF:** $$ P(X = k) = \frac{k^{-s}}{\zeta(s)}, \quad k = 1, 2, 3, \ldots $$

where $\zeta(s) = \sum_{n=1}^\infty n^{-s}$ is the Riemann zeta function.

**Usage:**

```python
from heavytails import Zipf

# Zipf with exponent s=1.5
zipf = Zipf(s=1.5, kmax=10000)

# Rank-frequency relationship
for rank in [1, 10, 100, 1000]:
    prob = zipf.pmf(rank)
    print(f"P(rank={rank}): {prob:.6f}")

samples = zipf.rvs(1000, seed=42)
```

**Applications:**

- Word frequencies (Zipf's law)
- City populations
- Website popularity
- Social network connections

**Parameter Constraints:**

- $s > 1$ (exponent): Higher $s$ → steeper decay
- $k_{max}$: Truncation for normalization

--------------------------------------------------------------------------------

### Yule-Simon Distribution

Preferential attachment and generative processes.

**PMF:** $$ P(X = k) = \rho B(k, \rho+1), \quad k = 1, 2, 3, \ldots $$

where $B(\cdot, \cdot)$ is the beta function.

**Usage:**

```python
from heavytails import YuleSimon

# Yule-Simon with ρ=2.0
yule = YuleSimon(rho=2.0)

# Tail behavior
print(f"P(X = 1): {yule.pmf(1):.4f}")
print(f"P(X = 100): {yule.pmf(100):.6f}")

samples = yule.rvs(1000, seed=42)
```

**Applications:**

- Citation networks
- Species abundance
- Internet topology
- Social network growth

**Parameter Constraints:**

- $\rho > 0$ (shape): Controls tail heaviness

--------------------------------------------------------------------------------

## Choosing the Right Distribution

### Decision Tree

```plaintext
Is your data discrete or continuous?
│
├─ Discrete
│   ├─ Rank/frequency data → Zipf
│   ├─ Growth/attachment process → Yule-Simon
│   └─ Count data with heavy tail → Discrete Pareto
│
└─ Continuous
    ├─ Are extremes the focus?
    │   ├─ Yes
    │   │   ├─ Block maxima → GEV / Fréchet
    │   │   └─ Threshold exceedances → GPD
    │   └─ No
    │       ├─ Symmetric tails?
    │       │   ├─ Yes
    │       │   │   ├─ Moderate heaviness → Student-t
    │       │   │   └─ Extreme heaviness → Cauchy
    │       │   └─ No (right tail only)
    │       │       ├─ Power law → Pareto
    │       │       ├─ Multiplicative → Log-Normal
    │       │       └─ Flexible fitting → Burr XII
```

### By Application Domain

Domain            | Recommended Distributions
----------------- | -------------------------
**Finance**       | Student-t, GPD, Cauchy
**Insurance**     | GPD, Burr XII, Pareto
**Networks**      | Zipf, Yule-Simon, Pareto
**Environmental** | GEV, GPD, Fréchet
**Reliability**   | Burr XII, Inverse-Gamma
**Economics**     | Pareto, Log-Normal

--------------------------------------------------------------------------------

## Common Operations

All distributions support a unified interface:

```python
from heavytails import Pareto  # Or any other distribution

dist = Pareto(alpha=2.0, xm=1.0)

# Probability functions
pdf_val = dist.pdf(5.0)          # Density
cdf_val = dist.cdf(5.0)          # P(X ≤ 5)
sf_val = dist.sf(5.0)            # P(X > 5)
quantile = dist.ppf(0.95)        # 95th percentile

# Sampling
samples = dist.rvs(1000, seed=42)

# Moments (when they exist)
mean = dist.mean()
variance = dist.variance()
```

--------------------------------------------------------------------------------

## Next Steps

- **[Working with PDFs and CDFs](pdf-cdf.md)** - Detailed probability calculations
- **[Random Sampling](sampling.md)** - Generation and reproducibility
- **[Tail Index Estimation](tail-estimation.md)** - Estimating tail heaviness from data
- **[Parameter Fitting](fitting.md)** - Fitting distributions to data
- **[Diagnostic Tools](diagnostics.md)** - Goodness-of-fit assessment

--------------------------------------------------------------------------------

## See Also

- **[API Reference](../reference/)** - Complete technical documentation
- **[Examples](../examples/basic_usage.ipynb)** - Practical applications
- **[Theory](../theory/heavy-tails.md)** - Mathematical foundations
