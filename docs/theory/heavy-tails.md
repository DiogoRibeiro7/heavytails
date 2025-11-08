# Heavy-Tail Theory

This section provides the mathematical foundations of heavy-tailed distributions, focusing on rigorous definitions, key theorems, and asymptotic properties.

---

## Formal Definitions

### Heavy-Tailed Distributions

A random variable $X$ (or its distribution $F$) is **heavy-tailed** if:

$$
\limsup_{x \to \infty} e^{\lambda x} \bar{F}(x) = \infty \quad \text{for all } \lambda > 0
$$

where $\bar{F}(x) = P(X > x) = 1 - F(x)$ is the survival function.

**Equivalently:** The tail is not dominated by any exponential distribution.

### Subexponential Distributions

A distribution $F$ on $[0, \infty)$ is **subexponential** (written $F \in \mathcal{S}$) if:

$$
\lim_{x \to \infty} \frac{\bar{F^{*2}}(x)}{\bar{F}(x)} = 2
$$

where $F^{*2}$ is the convolution of $F$ with itself.

**Interpretation:** The tail of a sum $X_1 + X_2$ is dominated by the maximum:

$$
P(X_1 + X_2 > x) \sim 2P(\max(X_1, X_2) > x) \sim 2P(X_1 > x)
$$

**Implication:** One large jump dominates the sum.

---

## Regular Variation

### Definition

A measurable function $L: (0, \infty) \to (0, \infty)$ is **slowly varying** (at infinity) if:

$$
\lim_{x \to \infty} \frac{L(tx)}{L(x)} = 1 \quad \text{for all } t > 0
$$

**Examples:**
- Constant functions: $L(x) = c$
- Logarithmic functions: $L(x) = \log x$, $(\log x)^\beta$, $\log \log x$
- Iterated logarithms

A function $f$ is **regularly varying** with index $\alpha \in \mathbb{R}$ (written $f \in RV_\alpha$) if:

$$
f(x) = x^\alpha L(x)
$$

where $L$ is slowly varying.

**Key Property:**
$$
\lim_{x \to \infty} \frac{f(tx)}{f(x)} = t^\alpha \quad \text{for all } t > 0
$$

### Karamata's Theorem

If $L$ is slowly varying and $\alpha > -1$, then:

$$
\int_a^x t^\alpha L(t) \, dt \sim \frac{x^{\alpha+1} L(x)}{\alpha+1} \quad \text{as } x \to \infty
$$

**Application:** Moments of regularly varying distributions.

---

## Pareto-Type Tails

### Definition

A distribution $F$ has a **Pareto-type tail** with index $\alpha > 0$ if:

$$
\bar{F}(x) = P(X > x) \sim x^{-\alpha} L(x) \quad \text{as } x \to \infty
$$

where $L$ is slowly varying.

**Equivalently:** $\bar{F} \in RV_{-\alpha}$.

### Moment Characterization

For Pareto-type tails with index $\alpha$:

$$
\mathbb{E}[X^p] < \infty \iff p < \alpha
$$

**Proof sketch:**
$$
\mathbb{E}[X^p] = \int_0^\infty px^{p-1} \bar{F}(x) \, dx
$$

For large $x$, $\bar{F}(x) \sim Cx^{-\alpha}$, so:

$$
\int_A^\infty px^{p-1} \cdot Cx^{-\alpha} \, dx = Cp \int_A^\infty x^{p-\alpha-1} \, dx
$$

This integral converges iff $p - \alpha - 1 < -1$, i.e., $p < \alpha$.

### Tail Equivalence

For Pareto-type tails:

$$
\bar{F}(x) \sim \frac{C}{x^\alpha} \quad \text{as } x \to \infty
$$

where $C = \alpha x_m^\alpha$ for Pareto($\alpha, x_m$).

**Examples:**
- **Pareto:** $\bar{F}(x) = (x_m/x)^\alpha$ for $x \geq x_m$
- **Student-t($\nu$):** $\bar{F}(x) \sim \frac{c_\nu}{x^\nu}$ as $x \to \infty$
- **Burr XII:** $\bar{F}(x) = [1 + (x/s)^c]^{-k} \sim (s/x)^{ck}$

---

## Domains of Attraction

### Extreme Value Theory Framework

Let $X_1, X_2, \ldots$ be iid with distribution $F$, and define:

$$
M_n = \max\{X_1, \ldots, X_n\}
$$

### Fisher-Tippett-Gnedenko Theorem

If there exist sequences $\{a_n > 0\}$ and $\{b_n\}$ such that:

$$
\frac{M_n - b_n}{a_n} \overset{d}{\to} G
$$

for some non-degenerate distribution $G$, then $G$ belongs to one of three types:

1. **Gumbel** ($\xi = 0$):
   $$
   G(x) = \exp\{-e^{-x}\}, \quad x \in \mathbb{R}
   $$

2. **Fréchet** ($\xi > 0$):
   $$
   G(x) = \begin{cases}
   0 & x \leq 0 \\
   \exp\{-x^{-1/\xi}\} & x > 0
   \end{cases}
   $$

3. **Weibull** ($\xi < 0$):
   $$
   G(x) = \begin{cases}
   \exp\{-(-x)^{-1/\xi}\} & x < 0 \\
   1 & x \geq 0
   \end{cases}
   $$

### Unified GEV Form

All three types can be written as:

$$
G(x) = \exp\left\{-\left[1 + \xi\frac{x-\mu}{\sigma}\right]_+^{-1/\xi}\right\}
$$

where $[z]_+ = \max(z, 0)$, $\sigma > 0$, and:
- $\xi > 0$: Fréchet (heavy tail)
- $\xi = 0$: Gumbel (exponential tail, limit as $\xi \to 0$)
- $\xi < 0$: Weibull (bounded support)

### Domain of Attraction Characterization

$F$ is in the **Fréchet domain** ($\xi > 0$) iff:

$$
\bar{F}(x) = x^{-1/\xi} L(x)
$$

where $L$ is slowly varying.

**Connection:** $\xi = 1/\alpha$ for Pareto-type tail index $\alpha$.

---

## Stable Distributions

### Definition

A random variable $X$ has a **stable distribution** if for any $n \geq 2$, there exist $a_n > 0$ and $b_n \in \mathbb{R}$ such that:

$$
X_1 + \cdots + X_n \overset{d}{=} a_n X + b_n
$$

where $X_1, \ldots, X_n \overset{iid}{\sim} X$.

### Characteristic Function

Stable distributions are characterized by:

$$
\phi(t) = \exp\left\{i\mu t - |\sigma t|^\alpha [1 - i\beta \text{sign}(t) \omega(t, \alpha)]\right\}
$$

where:
- $0 < \alpha \leq 2$: **stability parameter** (tail index)
- $-1 \leq \beta \leq 1$: skewness
- $\sigma > 0$: scale
- $\mu \in \mathbb{R}$: location

### Tail Behavior

For $\alpha < 2$, stable distributions have power-law tails:

$$
P(X > x) \sim C_+ x^{-\alpha}, \quad P(X < -x) \sim C_- x^{-\alpha}
$$

where $C_\pm$ depend on $\beta$.

### Special Cases

- $\alpha = 2$: **Normal** distribution (light tail)
- $\alpha = 1, \beta = 0$: **Cauchy** distribution
- $\alpha < 2, \beta = 1$: **Lévy** distribution (one-sided)

---

## Generalized Central Limit Theorem

### Classical CLT Failure

For heavy tails with $\alpha < 2$, the sample mean does NOT converge to a Normal distribution.

**Example (Cauchy):**
If $X_1, \ldots, X_n \overset{iid}{\sim} \text{Cauchy}$, then:

$$
\frac{X_1 + \cdots + X_n}{n} \overset{d}{=} X_1
$$

The average is still Cauchy!

### Generalized CLT

For $X_i \in$ domain of attraction of a stable law with $\alpha < 2$:

$$
\frac{S_n - b_n}{a_n} \overset{d}{\to} S_\alpha
$$

where $S_n = X_1 + \cdots + X_n$, $S_\alpha$ is a stable distribution, and:
- $a_n \sim n^{1/\alpha} L(n)$ (regular variation)
- $b_n = n\mathbb{E}[X]$ if $1 < \alpha < 2$; $b_n = 0$ if $\alpha \leq 1$

**Implication:** Convergence is slow; sample mean is unreliable.

---

## Second-Order Regular Variation

### Motivation

First-order theory: $\bar{F}(x) \sim x^{-\alpha} L(x)$

Second-order theory: How fast does $\bar{F}(x)$ converge to $Cx^{-\alpha}$?

### Definition

$\bar{F} \in 2RV_{\alpha, \rho}$ if there exists $\rho \leq 0$ and a function $A(x) \to 0$ such that:

$$
\lim_{x \to \infty} \frac{\frac{\bar{F}(tx)}{\bar{F}(x)} - t^{-\alpha}}{A(x)} = t^{-\alpha} \frac{t^\rho - 1}{\rho}
$$

for all $t > 0$.

**Interpretation:** $A(x)$ measures the rate of convergence to first-order asymptotics.

### Implications for Estimation

- **Hill estimator bias:** Asymptotic bias is $O(A(x_{n-k}))$
- **Bias-corrected estimators:** Use second-order structure
- **Optimal $k$ selection:** Balances bias and variance

---

## Tail Dependence and Copulas

### Tail Dependence Coefficient

For a bivariate distribution with marginals $F_1, F_2$ and copula $C$:

**Upper tail dependence coefficient:**
$$
\lambda_U = \lim_{u \to 1^-} P(U_2 > u \mid U_1 > u) = \lim_{u \to 1^-} \frac{1 - 2u + C(u, u)}{1 - u}
$$

where $U_i = F_i(X_i) \sim \text{Uniform}(0, 1)$.

### Examples

- **Gaussian copula:** $\lambda_U = 0$ (no tail dependence)
- **Student-t copula:** $\lambda_U > 0$ (tail dependence)
- **Clayton copula:** Lower tail dependence
- **Gumbel copula:** Upper tail dependence

**Financial implication:** Normal models underestimate joint extreme events.

---

## Sum of Heavy-Tailed Random Variables

### Subexponential Property

For iid subexponential $X_i$:

$$
P(X_1 + \cdots + X_n > x) \sim n P(X_1 > x) \quad \text{as } x \to \infty
$$

**Interpretation:** The tail of the sum is driven by the largest observation.

### Example

```python
from heavytails import Pareto
import statistics

# Pareto samples
pareto = Pareto(alpha=1.5, xm=1.0)
samples1 = pareto.rvs(1000, seed=42)
samples2 = pareto.rvs(1000, seed=43)

# Sum of two independent Pareto variables
sums = [x1 + x2 for x1, x2 in zip(samples1, samples2)]

# Tail is still heavy (not Normal!)
print(f"P(X1 > 100): {sum(1 for x in samples1 if x > 100)/1000:.4f}")
print(f"P(X1+X2 > 200): {sum(1 for s in sums if s > 200)/1000:.4f}")
# Approximately 2x the first probability
```

---

## Ruin Theory

### Classical Ruin Problem

Surplus process:
$$
U_t = u + ct - \sum_{i=1}^{N_t} X_i
$$

where:
- $u$: initial capital
- $c$: premium rate
- $N_t$: Poisson claim arrivals
- $X_i$: claim sizes

**Ruin probability:**
$$
\psi(u) = P(\text{ruin} | U_0 = u)
$$

### Heavy-Tailed Claims

For subexponential claim sizes:

$$
\psi(u) \sim \frac{\lambda \mathbb{E}[X^2]}{2c(\mathbb{E}[X])^2} \bar{F}_X(u) \quad \text{as } u \to \infty
$$

where $\lambda$ is claim arrival rate.

**Implication:** Ruin is driven by one large claim, not accumulation of small claims.

---

## Mathematical Results Summary

### Theorems

1. **Karamata's Theorem:** Integration of regularly varying functions
2. **Potter's Bounds:** Uniform bounds for slowly varying functions
3. **Representation Theorem:** $L(x) = c(x) \exp\{\int_1^x \frac{\epsilon(t)}{t} dt\}$ where $\epsilon(t) \to 0$
4. **Fisher-Tippett-Gnedenko:** Limiting distributions of maxima
5. **Balkema-de Haan-Pickands:** GPD limit for exceedances
6. **Breiman's Theorem:** Products and sums of heavy-tailed variables

### Key References

1. **Resnick, S. I. (2007).** *Heavy-Tail Phenomena*. Springer.
2. **Embrechts, P., Klüppelberg, C., Mikosch, T. (1997).** *Modelling Extremal Events*. Springer.
3. **Bingham, N. H., Goldie, C. M., Teugels, J. L. (1987).** *Regular Variation*. Cambridge.
4. **de Haan, L., Ferreira, A. (2006).** *Extreme Value Theory*. Springer.

---

## Next Steps

- **[Extreme Value Theory](evt.md)** - EVT framework in detail
- **[Tail Index Estimation Theory](tail-estimation.md)** - Estimator theory
- **[Examples](../examples/basic_usage.ipynb)** - Apply the theory
- **[API Reference](../reference/)** - Implementation details
