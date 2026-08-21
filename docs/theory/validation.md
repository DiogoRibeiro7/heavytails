# Validation Studies

A library that implements special functions from scratch has to earn trust. This
page describes how `heavytails` establishes that its densities, quantiles and
samplers are correct, and how you can re-run those checks yourself.

Three independent lines of evidence are used:

1. **Mathematical properties** that must hold for any distribution, checked
   symbolically or numerically without a reference implementation.
2. **Cross-validation against SciPy** for the families SciPy also implements.
3. **Property-based testing** over randomly generated parameters, which explores
   corners a fixed test suite would miss.

## Mathematical properties

These checks need no reference implementation, so they apply to every family
including the ones SciPy does not provide.

| Property                  | Statement                                   |
| ------------------------- | ------------------------------------------- |
| Non-negativity            | $f(x) \ge 0$ on the support                 |
| Normalisation             | $\int f(x)\,dx = 1$                         |
| Monotonicity              | $F$ is non-decreasing                       |
| Boundary behaviour        | $F \to 0$ at the lower endpoint, $\to 1$ at the upper |
| Survival complement       | $S(x) = 1 - F(x)$                           |
| Quantile inversion        | $F(F^{-1}(u)) = u$ for $u \in (0,1)$        |
| Moment existence          | $E[X^p]$ finite exactly when $p < \alpha$   |

`heavytails.validation` exposes these as runnable checks:

```python
from heavytails.validation import PropertyBasedTests

tests = PropertyBasedTests()
tests.test_pdf_nonnegativity("pareto")
tests.test_cdf_monotonicity("pareto")
tests.test_ppf_cdf_inverse("pareto")
```

Each returns a dictionary reporting whether the property held and, where it did
not, the worst case found.

The same checks are reachable from the command line:

```bash
heavytails validate pareto --params '{"alpha": 2.0, "xm": 1.0}'
```

### Why quantile inversion matters most

Several families have no closed-form quantile function, so `ppf` is computed by a
safeguarded Newton iteration. The round trip $F(F^{-1}(u)) = u$ is therefore the
single most informative check in the suite: it exercises the solver, the CDF, and
the bracketing logic at once, and it is where a numerical bug will surface first.

## Cross-validation against SciPy

Where SciPy implements the same family, the two are compared point by point
across the density, distribution and quantile functions.

```python
from heavytails.validation import NumericalValidation

validator = NumericalValidation(tolerance=1e-10)
result = validator.validate_against_scipy("pareto", {"alpha": 2.5, "xm": 1.0})
result["pass"], result["max_error"]
```

SciPy is an **optional** dependency, installed only in the `benchmarks` group.
When it is absent the comparison reports that it was skipped rather than failing,
so the check never turns into a hidden runtime dependency:

```bash
poetry install --with benchmarks
```

### Interpreting the tolerance

The default tolerance is a relative error of $10^{-10}$, close to the limit of
double precision for these expressions. Agreement at that level means the two
implementations differ only in floating-point rounding.

Larger discrepancies are expected, and acceptable, in two places:

- **Deep in the tail**, where $\bar{F}(x)$ underflows. Both implementations lose
  precision; the survival function is the numerically stable way to ask the
  question.
- **Near a parameter boundary**, such as $\xi \to 0$ in the GEV, where the
  expression is analytically removable but numerically delicate.

Where the two disagree, being closer to the analytic answer matters more than
being closer to SciPy. Discrepancies are resolved against the mathematics.

## Property-based testing

Fixed test cases check what the author thought to check. Property-based tests
generate parameters at random and assert the invariants above hold for all of
them, which is how boundary cases get found.

The test suite uses [Hypothesis](https://hypothesis.readthedocs.io/) for this:

```bash
poetry run pytest -m property
```

Failing cases are automatically shrunk to a minimal reproducer and recorded in
`.hypothesis/`, so a failure found once becomes a regression test.

## Goodness-of-fit tests

Validation above establishes that the implementations match the mathematics.
Goodness of fit asks the different question of whether a *model* matches your
*data*.

`heavytails.validation.GoodnessOfFitTests` provides two:

```python
from heavytails import Pareto
from heavytails.validation import GoodnessOfFitTests

data = Pareto(alpha=2.5, xm=1.0).rvs(500, seed=42)
tests = GoodnessOfFitTests()

tests.anderson_darling_test(data, "pareto", alpha=2.5, xm=1.0)
tests.kolmogorov_smirnov_test(data, "pareto", alpha=2.5, xm=1.0)
```

Each returns the statistic, a p-value, and a `reject` flag at the configured
significance level.

### Which one to use

**Prefer Anderson-Darling here.** Its statistic weights the tails of the
distribution, which is where heavy-tailed families differ from each other. The
Kolmogorov-Smirnov statistic is driven by the centre, which is exactly where
they agree, so it is least sensitive where you most need sensitivity.

$$
A^2 = -n - \frac{1}{n}\sum_{i=1}^{n}(2i-1)\left[\ln F(x_{(i)}) + \ln(1 - F(x_{(n+1-i)}))\right]
ight]
$$

$$
D = \max_i \max\left(\frac{i}{n} - F(x_{(i)}),\; F(x_{(i)}) - \frac{i-1}{n}\right)
ight)
$$

### Why this is not the same as AIC

AIC and BIC rank candidates against one another. The best of a set of poor
models still ranks first, and nothing in the ranking says whether it fits. A
goodness-of-fit test answers that separately, so
[`AutoFit.compare_distributions`](../reference/utilities.md) reports both.

### Estimated parameters

The p-values above assume the distribution was **fully specified** before
seeing the data. When the parameters were estimated from the same sample, the
fitted distribution is closer to the data than the null assumes, so the p-value
is conservative: the test rejects less often than its nominal level.

Passing `parameters_estimated=True` adds a `caveat` field saying so. For a
calibrated p-value in that case, use a parametric bootstrap: refit on many
samples drawn from the fitted model, and compare your statistic to that
distribution.

The asymptotic critical values used here are those for the fully specified
case: 1.933 at 10%, 2.492 at 5% and 3.857 at 1%. They are not the
D'Agostino-Stephens values quoted for the normality test with estimated mean
and variance, whose 5% critical value is 0.787 -- using those would reject a
correctly specified distribution about half the time.

## Numerical accuracy of the special functions

Two special functions are implemented from scratch, because depending on SciPy
would break the no-dependency promise:

| Function            | Method                                     | Used by                    |
| ------------------- | ------------------------------------------ | -------------------------- |
| Incomplete gamma    | Series expansion and continued fraction, switched on the argument | Inverse-Gamma |
| Incomplete beta     | Continued fraction with a symmetry transform | Student-t, Beta-Prime     |

Both are validated against SciPy over a grid spanning the ranges the
distributions actually use, and against known closed-form values at special
points. The switch between the series and the continued fraction is placed where
both converge well, so accuracy does not degrade at the crossover.

## Convergence and stability

Two further studies check behaviour that a single-point comparison cannot:

- **Convergence.** `convergence_validation` confirms that empirical moments and
  quantiles of large samples approach their theoretical values at the expected
  rate — and, importantly, that they *fail* to converge when the corresponding
  moment does not exist.
- **Parameter stability.** `parameter_stability_check` sweeps a family across its
  parameter range and looks for discontinuities, overflow and loss of precision.

```python
from heavytails.validation import parameter_stability_check

parameter_stability_check("pareto", alpha=2.0, xm=1.0)
```

## Reproducing the results

Everything on this page runs from a checkout:

```bash
poetry install --with dev,benchmarks
poetry run pytest                    # the full suite, including property tests
poetry run pytest -m property        # property-based tests only
```

Sampling is seeded through the library's deterministic RNG wrapper, so validation
runs are reproducible: the same seed gives the same sample and therefore the same
verdict.

## Known limitations

- Cross-validation only covers families SciPy implements. Burr XII,
  Log-Logistic and the discrete families rest on the property-based checks and
  closed-form special cases alone.
- Extreme parameter values, such as $\alpha < 10^{-2}$, are exercised but the
  achievable precision there is limited by double-precision arithmetic rather
  than by the algorithms.
- Validation establishes that the implementation matches the mathematics. It
  says nothing about whether a given family is the right model for your data —
  that is what [Tail Diagnostics](../guide/diagnostics.md) is for.

## See also

- [Testing](../development/testing.md) — how to run and extend the test suite
- [Heavy-Tail Theory](heavy-tails.md) — the properties being verified
- [API reference](../reference/utilities.md) — the validation helpers
