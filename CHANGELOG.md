# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit tests in
  `heavytails.validation.GoodnessOfFitTests`, which previously raised
  `NotImplementedError`
  ([#301](https://github.com/DiogoRibeiro7/heavytails/issues/301)). Both are
  reported by `AutoFit.compare_distributions` and by the `heavytails compare`
  command, so a comparison now says whether the winning family fits, not only
  how it ranks.

### Fixed

- Quantile functions now behave the same way across every family
  ([#298](https://github.com/DiogoRibeiro7/heavytails/issues/298)). Previously
  the answer at the edges depended on which family you asked: `Pareto`,
  `Weibull`, `Frechet`, `GEV_Frechet`, `BurrXII` and `LogLogistic` raised
  `OverflowError` where the quantile simply exceeded the float range,
  `InverseGamma` raised a `ValueError` about failing to bracket the root,
  `LogNormal` returned `inf`, and `BetaPrime` returned a finite value. All of
  them now return `inf`.
- `DiscretePareto.ppf` accepted any float, including `0.0`, `1.5` and negatives,
  and silently returned the support bounds. It now raises `ValueError` like
  every other family.
- `_ppf_monotone` narrowed its bracket only on bisection fallbacks, never on
  accepted Newton steps, so a run of Newton steps could exhaust the iteration
  budget with the bracket as wide as it started. It now narrows on every
  iteration, which also fixed a non-monotonicity in `InverseGamma.ppf`.

### Added

- `ConvergenceError`, raised when a solver cannot reach its tolerance. It
  previously returned its best guess, which the caller could not distinguish
  from a converged answer.

- `LogNormal.ppf` raised `OverflowError` when the quantile exceeded the float
  range instead of returning `inf`. The median of `LogNormal(mu=1000)` is
  `exp(1000)`, which is genuinely not representable, so `inf` is the correct
  answer; raising broke parameter sweeps and made the failure look like a
  caller error. ([#296](https://github.com/DiogoRibeiro7/heavytails/issues/296))
- `LogNormal.sf` is computed with `math.erfc` rather than as `1 - cdf(x)`, which
  reached exactly zero by `x = 1e5` and carried no information beyond it. It now
  matches SciPy to about 1e-14 relative out to `x = 1e12`.

### Removed

- `roadmap.safe_lognormal_ppf`, a workaround that caught the overflow above. It
  is redundant now that `LogNormal.ppf` handles the case itself.

### Added

- Python 3.13 support, covered by the CI test matrix and declared in the package
  classifiers.
- `py.typed` marker, so the type annotations that ship with the package are
  visible to downstream type checkers. The `Typing :: Typed` classifier was
  previously advertised without one.
- `heavytails.__version__`, resolved from the installed distribution metadata.
- `--version` / `-V` flag on the `heavytails` command-line interface.
- `cli` installation extra (`pip install "heavytails[cli]"`). The console script
  depends on `typer` and `rich`, which were previously development-only
  dependencies, so the entry point was broken for anyone installing from PyPI.
- CodeQL analysis and dependency-review workflows.
- `.github/CODEOWNERS`, `.gitattributes` and `.zenodo.json`.
- `StudentT.cdf`, `StudentT.sf` and `StudentT.ppf`. The class previously offered
  only `pdf` and `rvs`, with a docstring stating that the CDF and PPF "require
  special functions not in stdlib" — but the regularized incomplete beta needed
  to write them was already implemented in `extra_distributions`. All three
  agree with SciPy to around 1e-14.
- `Cauchy.sf`, `Frechet.sf` and `GEV_Frechet.sf`, so every continuous family now
  provides the full interface the documentation advertises.
- `YuleSimon.sf` and `YuleSimon.ppf`. The survival function uses the closed form
  `P(X > k) = k * B(k, rho + 1)`, and the quantile function brackets and bisects
  rather than scanning linearly.
- `heavytails._special`, holding the shared numeric special functions so that
  both distribution modules can use them without an import cycle. The previous
  names remain importable from `heavytails.extra_distributions`.
- `_betaincinv_reg`, an inverse for the regularized incomplete beta. It solves in
  log-space with a symmetry reduction, which is what keeps extreme quantiles
  accurate.
- Documentation pages that existing pages already linked to but which had never
  been written: CLI reference, diagnostics guide, extreme value theory, tail
  index estimation theory, validation studies, architecture, benchmarking, code
  review, and an executable `basic_usage` notebook.

### Changed

- Migrated project metadata to the PEP 621 `[project]` table.
- Grouped Dependabot updates so routine bumps arrive as a few reviewable pull
  requests rather than one per package.
- Continuous integration now also runs on pull requests targeting `develop`,
  builds and metadata-checks the distributions before publishing, verifies that
  `poetry.lock` matches `pyproject.toml`, and builds the documentation with
  `--strict`.
- Security scanning fails the build on findings instead of uploading a report
  that no one reads. `safety` was replaced by `pip-audit`, which needs no
  account to run.
- Refreshed the locked dependency set, clearing 116 known vulnerabilities
  reported against the previously locked development and documentation
  toolchain.
- Pre-commit hooks are pinned to the same tool versions as the development
  dependencies, so local hooks and CI now agree.
- Replaced the `Makefile` targets, which measured coverage of `scripts/` rather
  than of the package.
- `mkdocstrings` is configured for Google-style docstrings, which is what the
  package actually uses. Under the previous `numpy` setting no `Args:` or
  `Returns:` section was parsed anywhere in the API reference.
- The documentation navigation lists every page. Five API reference entries all
  pointed at the same directory, and the `gen-files` script duplicated the
  hand-written reference pages while `literate-nav` looked for a `SUMMARY.md`
  that was never generated.

### Fixed

- `heavytails benchmark` raised `ZeroDivisionError` on platforms with a
  low-resolution wall clock, because a sub-millisecond timing measured exactly
  zero seconds. Timings now use `time.perf_counter()`.
- The same defect in the performance tests, which failed intermittently on
  Windows.
- Removed an invalid PyPI classifier (`Topic :: Scientific/Engineering ::
  Statistics`) that would have been rejected on upload.
- `scripts/pyproject_updater.py` depends on `tomlkit` and `packaging`, which
  were never declared and only happened to be installed transitively.
- Pinned Poetry 2.2.1 in every workflow. The previous 1.8.3 pin cannot read the
  version 2.1 lock file this repository uses.
- Removed a `preferred-citation` entry from `CITATION.cff` that pointed at an
  unpublished paper with a placeholder DOI, which citation tooling would have
  emitted as a real reference.
- `YuleSimon.pmf` raised `OverflowError` for k of about 170 and above, because
  it multiplied gamma functions that overflow individually even though their
  ratio is small. Since sampling called it in a loop, drawing from the tail
  crashed. It is now evaluated with `lgamma`.
- Nineteen documentation links pointed at pages that did not exist. The
  documentation job now builds with `--strict`, so a broken link fails CI.
- `convergence_validation` took a `_max_iter` parameter that was never used and
  documented it under a different name.
- Removed a stale `xfail` marker on the Student-t PPF convergence test. It was
  recording precision loss that the new incomplete beta inverse eliminates.

### Removed

- `tox.ini`, which ran the test suite against `scripts/` instead of the package
  and duplicated the CI matrix.
- `IMPROVEMENT_PLAN.md` from the repository root. It described gaps that have
  since been closed; forward-looking plans live in
  [ROADMAP.md](https://github.com/DiogoRibeiro7/heavytails/blob/main/ROADMAP.md).
- The `isort` development dependency and its configuration, superseded by
  Ruff's `I` rules.

## [0.1.0] - 2025-10-25

### Added

- Continuous heavy-tailed distributions implemented from first principles:
  Pareto, Cauchy, Student-t, Log-Normal, Weibull, Fréchet and GEV (ξ > 0).
- Additional continuous families: Generalized Pareto, Burr XII, Log-Logistic
  (Fisk), Inverse-Gamma and Beta-Prime.
- Discrete heavy-tailed distributions: Zipf, Yule–Simon and Discrete Pareto.
- Tail index estimators: Hill, Pickands and moment.
- Diagnostic plotting helpers for log–log tail and QQ plots.
- Deterministic RNG wrapper for reproducible sampling.
- Custom incomplete-gamma and incomplete-beta implementations, and a
  safeguarded-Newton numeric PPF solver for families without a closed form.
- `heavytails` command-line interface.
- Documentation site built with MkDocs Material.

[Unreleased]: https://github.com/DiogoRibeiro7/heavytails/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/DiogoRibeiro7/heavytails/releases/tag/v0.1.0
