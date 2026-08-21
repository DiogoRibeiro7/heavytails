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

### Measured accuracy

`scripts/special_function_accuracy.py` sweeps the parameter ranges the
distributions actually use and compares against `mpmath` at 50 decimal digits,
which is far beyond double precision and so serves as exact. Reproduce it with:

```bash
poetry run python scripts/special_function_accuracy.py
```

| Function | Points | Worst relative error | Worst at |
| --- | --- | --- | --- |
| Regularized incomplete beta | 180 | 8.9e-13 | a=1000, b=0.5, x=0.5 |
| Regularized lower incomplete gamma | 90 | 5.4e-13 | a=1000, x=1000.5 |

Both are therefore accurate to roughly **12 significant digits** across the
ranges they are used in, against the 15 to 16 that double precision allows.

Accuracy does not degrade where the method changes. Measured on either side of
the switch point:

| Function | Just below | Just above |
| --- | --- | --- |
| Incomplete beta, at `x = (a+1)/(a+b+2)` | 7.3e-13 | 8.4e-12 |
| Incomplete gamma, at `x = a+1` | 1.1e-12 | 4.6e-14 |

`tests/test_special_accuracy.py` asserts these bounds, so a regression fails the
build rather than only changing a number on this page. `mpmath` is a
development dependency and is not needed to use the library.

### The bug this study found

The continued-fraction branch of the incomplete gamma was **wrong**, not merely
imprecise. Its Lentz recurrence started from `h = 1` instead of `h = 1/b₀` with
`b₀ = x + 1 - a`, and advanced `b` as `x + 2n - a` rather than `x + 1 - a + 2n`.
Dropping the leading term and shifting `b` by one produced plausible-looking
numbers that were badly wrong: `P(20, 21)` returned `0.0` against a true
`0.6157`.

That branch is reached whenever `x ≥ a + 1`, and `InverseGamma.cdf` evaluates
`P(α, β/x)`, so every `x` below roughly `β/(α+1)` was affected. Reported values
were wrong by factors of 2 to 17 in the lower tail.

This is the case for measuring rather than assuming. The function had passed
every property-based check the suite applied — it was monotone, it stayed in
[0, 1], it approached the right limits — because none of those properties
distinguishes a correct value from a consistently wrong one. Only comparison
against an independent high-precision reference did.

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
