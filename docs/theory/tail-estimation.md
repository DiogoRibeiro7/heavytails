# Tail Index Estimation Theory

The tail index is the one number that determines how heavy a tail is: which
moments exist, how fast extreme quantiles grow, and whether averages converge at
all. This page covers the theory behind the three estimators in
`heavytails.tail_index`.

For the practical recipe, see
[Tail Index Estimation](../guide/tail-estimation.md); for the framework these
estimators sit in, see [Extreme Value Theory](evt.md).

## What is being estimated

A distribution has a regularly varying tail with index $\alpha > 0$ if

$$
\bar{F}(x) = P(X > x) = x^{-\alpha} L(x)
$$

where $L$ is slowly varying, meaning $L(tx)/L(x) \to 1$ for every $t > 0$. The
slowly varying part is the nuisance: it can be a constant, a logarithm, or
anything that eventually flattens out, and it is precisely what makes estimation
hard.

Two parameterisations are in use, and confusing them is the most common source of
error:

| Symbol   | Name                 | Relation           | Convention here      |
| -------- | -------------------- | ------------------ | -------------------- |
| $\alpha$ | Tail index           | $\alpha = 1/\gamma$ | Larger = lighter tail |
| $\gamma$ | Extreme-value index  | $\gamma = 1/\alpha$ | Larger = heavier tail |

!!! important "What the library returns"
    Every estimator in `heavytails.tail_index` returns $\gamma$, the
    extreme-value index. Invert it to obtain $\alpha$:

    ```python
    gamma = hill_estimator(data, k=100)
    alpha = 1 / gamma
    ```

    `moment_estimator` is the exception in shape only: it returns the pair
    `(gamma, alpha)` directly.

Moment existence follows immediately: $E[X^p] < \infty$ if and only if
$p < \alpha$. So $\alpha \le 1$ means no finite mean, and $\alpha \le 2$ means no
finite variance.

## The Hill estimator

The Hill estimator is the maximum likelihood estimator of $\gamma$ under an exact
Pareto tail. Sort the sample in decreasing order as
$X_{(1)} \ge X_{(2)} \ge \dots \ge X_{(n)}$ and take

$$
\hat{\gamma}^{\mathrm{Hill}}_{k} = \frac{1}{k} \sum_{i=1}^{k}
\log \frac{X_{(i)}}{X_{(k+1)}}
$$

It averages the log-excesses of the top $k$ order statistics over the $(k+1)$-th,
which acts as a random threshold.

```python
from heavytails import Pareto, hill_estimator

data = Pareto(alpha=1.5, xm=1.0).rvs(10_000, seed=42)
gamma = hill_estimator(data, k=100)
alpha = 1 / gamma          # ≈ 1.53 against a true 1.5
```

**Properties.** Consistent and asymptotically normal with variance
$\gamma^2 / k$ when $k \to \infty$ and $k/n \to 0$. It is the most efficient of
the three when the tail really is Pareto.

**Limitations.** It requires positive data, it is only valid for $\gamma > 0$,
and it is badly biased when the slowly varying part $L$ has not settled down by
the $k$-th order statistic. That bias is the reason for everything below.

## The Pickands estimator

Pickands' estimator uses three order statistics and needs no assumption that the
tail is exactly Pareto:

$$
\hat{\gamma}^{\mathrm{Pickands}}_{k} = \frac{1}{\log 2} \,
\log \frac{X_{(k)} - X_{(2k)}}{X_{(2k)} - X_{(4k)}}
$$

```python
from heavytails import pickands_estimator

gamma = pickands_estimator(data, k=100)
```

**Properties.** Consistent for *any* $\gamma \in \mathbb{R}$, including zero and
negative values, and location- and scale-invariant. That generality is its
selling point.

**Limitations.** Much higher variance than Hill, because it throws away all but
three order statistics, and it needs $4k \le n$. Use it as a robustness check on
a Hill estimate rather than as a primary estimator.

The `m` parameter in `pickands_estimator` generalises the ratio from 2 to
arbitrary spacing; `m=2` is the classical form.

## The moment estimator

The Dekkers–Einmahl–de Haan moment estimator combines the first two moments of
the log-excesses. With

$$
M^{(j)}_{k} = \frac{1}{k} \sum_{i=1}^{k}
\left(\log \frac{X_{(i)}}{X_{(k+1)}}\right)^{j}
$$

the estimator is

$$
\hat{\gamma}^{\mathrm{Mom}}_{k} = M^{(1)}_{k} + 1 -
\frac{1}{2}\left[1 - \frac{\left(M^{(1)}_{k}\right)^{2}}{M^{(2)}_{k}}\right]^{-1}
$$

The first term is exactly the Hill estimator; the correction extends validity to
$\gamma \le 0$.

```python
from heavytails import moment_estimator

gamma, alpha = moment_estimator(data, k=100)
```

**Properties.** Valid for all $\gamma \in \mathbb{R}$, with efficiency close to
Hill in the Fréchet domain. A good default when you are not certain the tail is
heavy.

## Choosing k

Every estimator above depends on $k$, and the choice is a bias–variance
trade-off:

- **Small $k$** — only the most extreme observations, so little bias but high
  variance. The estimate jumps around.
- **Large $k$** — observations from the body creep in, so low variance but
  systematic bias. The estimate drifts.

There is no universally correct $k$. The standard practice is the **Hill plot**:
compute $\hat{\gamma}_k$ for a range of $k$ and look for a stable plateau.

```python
from heavytails import hill_estimator

estimates = [(k, hill_estimator(data, k=k)) for k in range(20, 1000, 10)]
```

Plotted against $k$, a genuine power-law tail shows a flat region; read the
estimate off the middle of the plateau. If there is no plateau, the data does not
support a tail index estimate, and forcing one produces a confident wrong answer.

The rule of thumb $k \approx \min(n/10, 200)$, which the CLI uses as a default,
is a starting point for the plot, not a substitute for it.

### Why the plateau exists

Write the bias as a function of the second-order behaviour of $\bar{F}$. For a
tail of the form $\bar{F}(x) = C x^{-\alpha}(1 + D x^{-\beta} + o(x^{-\beta}))$,
the asymptotic mean squared error is minimised at

$$
k_{\mathrm{opt}} \propto n^{2\beta / (2\beta + \alpha)}
$$

The exponent $\beta$ governs how fast the slowly varying part converges, and it
is itself unknown — which is why automatic selection rules are fragile and the
plot remains the honest tool.

## Comparing the three

| Estimator | Valid range          | Efficiency | Data needed | Use when                                  |
| --------- | -------------------- | ---------- | ----------- | ----------------------------------------- |
| Hill      | $\gamma > 0$         | Highest    | $k < n$     | You are confident the tail is heavy       |
| Pickands  | Any $\gamma$         | Lowest     | $4k \le n$  | Cross-checking a Hill estimate            |
| Moment    | Any $\gamma$         | Near-Hill  | $k < n$     | You are unsure the tail is heavy at all   |

Agreement between all three over a common range of $k$ is strong evidence. Sharp
disagreement usually means the sample is too small, the threshold is too low, or
the tail is not regularly varying.

## What can go wrong

- **Serial dependence.** Clustered extremes reduce the effective sample size, so
  the nominal standard error $\gamma/\sqrt{k}$ is optimistic.
- **Mixtures.** A contaminated sample can look like a power law over a limited
  range; the Hill plot will show a slope rather than a plateau.
- **Truncation.** Real data has a largest possible value. A truncated Pareto
  bends downward at the extreme right of the log–log plot.
- **Too small a sample.** Below a few hundred observations, tail index estimates
  carry very wide intervals whatever the estimator.

## References

- Hill, B. M. (1975). A simple general approach to inference about the tail of a
  distribution. *The Annals of Statistics*, 3(5), 1163–1174.
  [doi:10.1214/aos/1176343247](https://doi.org/10.1214/aos/1176343247)
- Pickands, J. (1975). Statistical inference using extreme order statistics.
  *The Annals of Statistics*, 3(1), 119–131.
- Dekkers, A. L. M., Einmahl, J. H. J., & de Haan, L. (1989). A moment estimator
  for the index of an extreme-value distribution. *The Annals of Statistics*,
  17(4), 1833–1855.
- Resnick, S. I. (2007). *Heavy-Tail Phenomena: Probabilistic and Statistical
  Modeling*. Springer.

## See also

- [Tail Index Estimation](../guide/tail-estimation.md) — the practical guide
- [Tail Diagnostics](../guide/diagnostics.md) — checking for a power law first
- [Extreme Value Theory](evt.md) — where $\gamma$ comes from
