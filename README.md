# heavytails

**A library of heavy-tailed probability distributions, vectorised over NumPy**

[![CI](https://github.com/DiogoRibeiro7/heavytails/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DiogoRibeiro7/heavytails/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/DiogoRibeiro7/heavytails/branch/main/graph/badge.svg)](https://codecov.io/gh/DiogoRibeiro7/heavytails)
[![PyPI](https://img.shields.io/pypi/v/heavytails.svg)](https://pypi.org/project/heavytails/)
[![Python versions](https://img.shields.io/pypi/pyversions/heavytails.svg)](https://pypi.org/project/heavytails/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22045594.svg)](https://doi.org/10.5281/zenodo.22045594)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-mkdocs--material-blue)](https://diogoribeiro7.github.io/heavytails)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/mypy-checked-blue)](https://mypy-lang.org/)

`heavytails` implements continuous and discrete heavy-tailed distributions, tail
index estimators, and diagnostic utilities with NumPy-backed vectorized
evaluation. Every density, quantile and sampler is derived from first
principles, so the implementation can be read, checked and taught rather than
taken on faith.

It targets research, teaching and simulation work in risk, finance, insurance and
extreme-value analysis.

---

## Features

- **NumPy-backed evaluation.** NumPy is the required runtime dependency, giving
  the distribution and estimator code efficient scalar and array evaluation.
- **Complete distribution interface.** PDF/PMF, CDF, survival function, quantile
  function and random sampling for every family, with survival functions computed
  directly so they stay accurate far into the tail where `1 - cdf(x)` has lost
  every significant digit.
- **Reproducible sampling** through a deterministic RNG wrapper.
- **Special functions from scratch** — incomplete gamma and incomplete beta —
  plus a safeguarded-Newton numeric PPF for families with no closed form.
- **Tail index estimation** with Hill-family, robust, bias-reduced,
  threshold-averaged and peaks-over-threshold estimators.
- **Parameter fitting** by maximum likelihood and method of moments, with
  AIC/BIC model comparison.
- **Diagnostics** for log–log tail plots and QQ plots.
- **A command-line interface** for sampling, fitting, comparison and
  benchmarking.
- **Typed throughout**, with a `py.typed` marker so downstream type checkers see
  the annotations.

---

## Installation

```bash
pip install heavytails
```

The command-line interface needs two extra packages; install it with the `cli`
extra:

```bash
pip install "heavytails[cli]"
```

To work on the library itself:

```bash
git clone https://github.com/DiogoRibeiro7/heavytails.git
cd heavytails
poetry install --with dev,docs
```

Requires Python 3.10 or newer.

---

## Quick start

```python
from heavytails import BurrXII, Pareto, hill_estimator

pareto = Pareto(alpha=1.5, xm=1.0)

pareto.pdf(2.0)        # density
pareto.cdf(2.0)        # distribution function
pareto.sf(10.0)        # survival function: P(X > 10)
pareto.ppf(0.99)       # 99th percentile
samples = pareto.rvs(10_000, seed=42)

# Recover the tail index from the sample. The estimators return the
# extreme-value index gamma = 1 / alpha, so invert it to read alpha back.
gamma = hill_estimator(samples, k=100)   # ≈ 0.65
alpha = 1 / gamma                        # ≈ 1.53, against a true 1.5

burr = BurrXII(c=1.2, k=2.5, s=3.0)
burr.ppf(0.95)
```

### Command line

```bash
heavytails list-distributions
heavytails sample pareto --params '{"alpha": 2.0, "xm": 1.0}' -n 1000 -o samples.txt
heavytails estimate-tail samples.txt --method hill
heavytails compare samples.txt
```

Run `heavytails --help` for the full command list.

---

## Available distributions

### Continuous

| Distribution            | Module                   | Heavy-tail regime |
| ----------------------- | ------------------------ | ----------------- |
| Pareto                  | `heavy_tails`            | always            |
| Cauchy                  | `heavy_tails`            | always            |
| Student-t               | `heavy_tails`            | small ν           |
| Log-Normal              | `heavy_tails`            | always            |
| Weibull                 | `heavy_tails`            | k < 1             |
| Fréchet                 | `heavy_tails`            | always            |
| GEV (Fréchet branch)    | `heavy_tails`            | ξ > 0             |
| Generalized Pareto      | `extra_distributions`    | ξ > 0             |
| Burr XII                | `extra_distributions`    | always            |
| Log-Logistic (Fisk)     | `extra_distributions`    | always            |
| Inverse-Gamma           | `extra_distributions`    | always            |
| Beta-Prime              | `extra_distributions`    | always            |

### Discrete

| Distribution     | Module     | Heavy-tail regime |
| ---------------- | ---------- | ----------------- |
| Zipf             | `discrete` | always            |
| Yule–Simon       | `discrete` | always            |
| Discrete Pareto  | `discrete` | always            |

Every continuous family provides `pdf`, `cdf`, `sf`, `ppf` and `rvs`; every
discrete family provides `pmf`, `cdf`, `ppf` and `rvs`.

### Estimation and diagnostics

| Module       | Contents                                              |
| ------------ | ----------------------------------------------------- |
| `tail_index` | Hill-family, robust, bias-reduced and POT estimators   |
| `plotting`   | Log–log tail plots and QQ plots                        |
| `utilities`  | Data I/O, automatic fitting and model comparison       |
| `validation` | Mathematical and numerical validation of the families  |
| `cli`        | Command-line entry point                               |

---

## Documentation

Full documentation, including the mathematical background, is at
**<https://diogoribeiro7.github.io/heavytails>**.

To build it locally:

```bash
make docs-serve
```

---

## Development

```bash
make install-dev   # install every dependency group
make hooks         # install the pre-commit hooks
make check         # everything CI runs: lint, format, types, tests, security
```

Individual targets are listed by `make help`. Contributions are welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md) for the branch flow, commit conventions and
review process, and [ROADMAP.md](ROADMAP.md) for what is planned next.

Notable changes are recorded in [CHANGELOG.md](CHANGELOG.md).

---

## License

MIT License © 2025 Diogo Ribeiro. See [LICENSE](LICENSE).

---

## Citation

If you use this package in research or teaching, please cite it. GitHub's
"Cite this repository" button reads [CITATION.cff](CITATION.cff), or use:

> Ribeiro, D. (2026). *heavytails: A Python Library for Heavy-Tailed
> Probability Distributions* (Version 0.4.0) [Computer software]. Zenodo.
> <https://doi.org/10.5281/zenodo.22045594>

**Which DOI to use.** [`10.5281/zenodo.22045594`](https://doi.org/10.5281/zenodo.22045594)
is the concept DOI: it always resolves to the most recent release, and citing
it means "this software, any version". Use it unless the exact version
matters. When reproducibility depends on the version you ran, cite that
version's own DOI instead -- for 0.4.0 that is
[`10.5281/zenodo.22062643`](https://doi.org/10.5281/zenodo.22062643). Every release
gets its own, listed on the Zenodo record.

```bibtex
@software{ribeiro_heavytails,
  author    = {Ribeiro, Diogo},
  title     = {heavytails: A Python Library for Heavy-Tailed
               Probability Distributions},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22045594},
  url       = {https://doi.org/10.5281/zenodo.22045594}
}
```

Shared citation metadata is maintained in `CITATION.cff`; Zenodo-specific
archive metadata is maintained in `.zenodo.json`. Both list the papers the
library implements, so citing a specific estimator is a matter of copying the
entry rather than tracking it down.
