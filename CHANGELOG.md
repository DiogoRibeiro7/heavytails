# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Runnable examples in the five core modules that had none
  ([#337](https://github.com/DiogoRibeiro7/heavytails/issues/337)): 86 across
  `heavy_tails`, `extra_distributions`, `discrete`, `plotting` and `_special`.
  Every value was computed rather than transcribed, and `tests/test_doctests.py`
  now guards all nine modules rather than the four that previously had any.

### Added

- `heavytails.vectorized`, evaluating `pdf`, `cdf`, `sf` and `ppf` over many
  points at once using NumPy when it is installed
  ([#308](https://github.com/DiogoRibeiro7/heavytails/issues/308)). Measured on
  the public call at 100,000 points, the 32 accelerated calls run 2.1x to 6.4x
  faster, median 3.9x. Eight families have kernels; LogNormal, StudentT,
  InverseGamma and BetaPrime cannot, because NumPy has neither the error
  function nor the incomplete beta and gamma, and `accelerated()` reports which
  is which rather than leaving a caller to guess. Without NumPy everything
  falls back to the loop. `scripts/vectorization_benchmark.py` produces the
  table.

### Fixed

- `GeneralizedPareto` returned negative probabilities below `mu` for every sign
  of `xi`: its validity check tested only `1 + xi z > 0`, which is the *upper*
  endpoint of a bounded distribution and is satisfied far below the support.
  `cdf(mu - 1)` returned -2.586 at `xi=0.4, mu=1`. Found by the new
  vectorisation tests, and now covered by a generic property over every family.
- `Weibull.pdf(0.0)` raised `ZeroDivisionError` for shape below one, where the
  density diverges. It returns `inf`.

- `heavytails.streaming`, tail index estimation over a stream without holding
  the sample ([#310](https://github.com/DiogoRibeiro7/heavytails/issues/310)).
  `TopK` maintains the largest values in `O(k)` memory with a min-heap;
  `StreamingTailIndex` builds the Hill and moment estimators on it and returns
  **bit-for-bit the batch result**, since both depend on the sample only
  through its top `k + 1` values. `WindowedTailIndex` does the same over the
  most recent observations, in `O(window)` memory, which is inherent rather
  than a gap: when the largest value in a window expires the new largest can
  be any of the survivors.

- `tests/reference_values.json` and `scripts/generate_reference_values.py`, a
  database of 209 values computed by mpmath at 50 decimal digits from the
  mathematical definitions, covering all twelve continuous families
  ([#311](https://github.com/DiogoRibeiro7/heavytails/issues/311)). Every
  point carries its condition number, so the tolerance follows from how well
  the quantity can be determined at that input rather than from a flat
  guess. Reading the table needs no mpmath and runs in about a second.
- `tests/test_distribution_properties.py`, applying the generic properties --
  quantile inversion, survival complementarity, monotonicity, non-negative
  density, support, reproducibility -- to every family from one registry
  rather than to four families by hand. Adds checks of the documented family
  relationships, of the tail index each family is supposed to have, and of
  moments existing exactly when the theory says.

### Fixed

Eight numerical defects, all found by the reference database on its first run
and all the same mistake: computing a small quantity by subtracting from one.

- `Cauchy.ppf` used `tan(pi(u - 1/2))`, whose argument sits next to `+-pi/2`
  where the tangent is arbitrarily steep. At `u = 1e-9` it returned
  -318309868.8 for a true -318309886.2, wrong in the eighth digit. Now uses
  the cotangent form, accurate to 1.9e-16.
- `_phi_inverse` refined its rational approximation using
  `0.5*(1 + erf(x/sqrt(2)))`, which cancels once `x` is a few units negative,
  so the correction was computed from noise. The relative error at `u = 1e-12`
  was 4.4e-07 -- worse than the unrefined approximation. Now 4.2e-16 across
  the range, which also improves every normal-quantile consumer, including
  `LogNormal.ppf` and the confidence intervals in `tail_index` and `threshold`.
- `GeneralizedPareto.ppf`, `BurrXII.ppf` and `Weibull.ppf` formed `1 - u`
  before taking a power or a logarithm, losing the lower tail to about seven
  digits. Now use `log1p`/`expm1`.
- `GeneralizedPareto.cdf`, `BurrXII.cdf`, `Weibull.cdf`, `Cauchy.cdf` and
  `LogNormal.cdf` computed a small probability as `1 - g(x)` with `g`
  approaching one. Each now takes the branch where its own value is the small
  quantity.

- `_gammainc_upper_reg`, the regularized upper incomplete gamma computed as
  itself rather than as `1 - P`
  ([#309](https://github.com/DiogoRibeiro7/heavytails/issues/309)). The
  subtraction cannot express a result below about 1e-16, so at `a=2, x=50` it
  returned exactly zero where the true value is 9.8e-21.
- `_gammaincinv_reg`, the inverse of the regularized incomplete gamma in both
  the lower and upper senses, so a caller can go through whichever of the two
  is its small quantity.

### Changed

- `InverseGamma.ppf` and `BetaPrime.ppf` invert the incomplete gamma and beta
  directly instead of bracketing and solving against their own distribution
  functions. Round-trip accuracy across the quantile range improves from
  2.2e-05 to 9.7e-15 and from 5.6e-04 to 3.5e-15 respectively.
- `_betaincinv_reg` starts from the small-`y` asymptote rather than bisecting
  the whole exponent range, cutting `StudentT.ppf` from 127 to 35 microseconds
  with identical accuracy.

### Fixed

- The safeguarded Newton iteration in `_betaincinv_reg` narrowed its bracket
  *after* computing the midpoint to fall back to, so on the first iteration --
  where the starting point is the midpoint by construction -- the fallback
  returned that same point and the no-progress check declared convergence.
  `I_y(50, 0.3) = 1e-3` came back with a relative error of 0.18. The fixed
  bisection to 1e-13 that preceded it had hidden this.
- `InverseGamma.cdf` computed `1 - P` and returned exactly zero throughout the
  lower tail; `cdf(0.02)` at `alpha=2, beta=1` is 9.8e-21, not 0.
- `BetaPrime.sf` computed `1 - cdf`, which is exactly zero above about `x=1e17`
  because `x/(x+s)` rounds to 1 there. It now uses the mirrored incomplete
  beta, whose argument `s/(x+s)` is computed rather than subtracted.

- `adaptive_trimmed_hill_estimator` and `adaptive_trim_selection`, choosing the
  trimming parameter for the trimmed Hill estimator from the data
  ([#321](https://github.com/DiogoRibeiro7/heavytails/issues/321)). Each
  normalised log-spacing is tested against the mean of the deeper ones, with an
  exactly computable null distribution, and the scan trims past the deepest
  anomaly. The median trimming chosen equals the number of planted outliers at
  0, 1, 2, 3, 5 and 8 of them, and on clean data the standard deviation is
  0.0295 against 0.0292 for the plain Hill estimator. Completes the eleven
  estimator benchmark suite; `scripts/tail_index_study.py` now runs twelve.
- `tests/test_doctests.py`, checking that the docstring examples in the
  numerical modules actually reproduce. The main suite does not collect
  doctests, so four had stopped working unnoticed.

### Fixed

- Four docstring examples reported values their code never produced. Two claimed
  a tail index estimate of 0.5 on samples too small for the estimator's own
  sampling variability, so the figure shown had been transcribed from what the
  estimator should give rather than measured; both produced 0.4.

- `heavytails.actuarial`, building the aggregate loss distribution from a
  frequency model and a severity, and pricing the reinsurance written on it
  ([#304](https://github.com/DiogoRibeiro7/heavytails/issues/304)). Frequency
  models `Poisson`, `NegativeBinomial` and `Binomial` -- the `(a,b,0)` class the
  Panjer recursion is defined on. `PolicyTerms` and `LayeredSeverity` apply
  deductibles, limits and coinsurance on either a per-loss or a per-payment
  basis. `panjer_recursion` gives the whole aggregate distribution;
  `simulate_aggregate_loss` and `EmpiricalAggregate` give the same interface
  from a sample, for the cases the recursion cannot reach. `compound_moments`
  reports the exact mean and variance, including `inf` when the severity is
  heavy enough that they do not exist. `limited_expected_value`,
  `excess_of_loss_premium` and `AggregateLoss.stop_loss_premium` price layers.
  Completes the actuarial item of roadmap Phase 4.

- `heavytails.viz`, rendering the diagnostics with matplotlib behind a new
  `plot` extra ([#302](https://github.com/DiogoRibeiro7/heavytails/issues/302)):
  `plot_tail`, `plot_qq`, `plot_hill`, `plot_trimmed_hill`,
  `plot_mean_residual_life` and `plot_parameter_stability`. Each takes an
  optional `ax` and returns it, so a panel of diagnostics composes normally,
  and `plot_tail` can overlay a fitted distribution against the empirical
  curve. `heavytails.plotting` keeps returning coordinates and stays free of
  third-party imports, so the library itself still never requires matplotlib.
  Install with `pip install "heavytails[plot]"`.

- `heavytails.threshold`, with `mean_residual_life`, `parameter_stability`,
  `select_threshold` and `return_level`
  ([#300](https://github.com/DiogoRibeiro7/heavytails/issues/300)). Choosing
  the threshold dominates a peaks-over-threshold analysis and no rule settles
  it, so the two diagnostics come first and the automatic rule is documented
  as a starting point. `select_threshold` uses a goodness-of-fit test whose
  p-values are conservative, which biases it towards thresholds that are too
  low. `return_level` reports a bootstrap interval whose measured coverage is
  about 0.88 against a nominal 0.95, falling to 0.76 on smaller samples,
  because it captures sampling variability but not the error in choosing the
  threshold.

- `heavytails.risk`, with `value_at_risk`, `expected_shortfall`,
  `tail_conditional_expectation`, `monte_carlo_tail_risk` and `mean_exists`
  ([#303](https://github.com/DiogoRibeiro7/heavytails/issues/303)). Expected
  shortfall has closed forms for the Pareto, log-normal, generalized Pareto and
  Weibull families and falls back to quadrature on the quantile function
  otherwise; the two paths share no code and agree to four decimal places.
  Expected shortfall returns `inf` whenever the distribution has no finite
  mean, rather than a large number that would look like a result, and the
  Monte Carlo estimator always reports standard errors.

## [0.3.0] - 2026-08-21

### Added

- `bias_reduced_hill_estimator`, with `second_order_rho`, `second_order_beta`
  and `recommended_rho_k`
  ([#329](https://github.com/DiogoRibeiro7/heavytails/issues/329)). The Hill
  estimator's bias at large `k` is systematic rather than random, so it can be
  estimated and subtracted; with `rho` supplied the measured bias falls by a
  factor of four to eighty. `second_order_rho` is documented as unstable,
  because it is: sweeping `k` on a sample whose true `rho` is -1 gives
  estimates from -0.07 to -20.5, the latter at a pole. Supplying `rho` is
  strongly preferred, and the correction still helps when it is estimated.

- `gpd_mle_estimator` and `fit_generalized_pareto`
  ([#327](https://github.com/DiogoRibeiro7/heavytails/issues/327)), the
  parametric peaks-over-threshold counterpart to the semiparametric estimators.
  Fitting uses the reduction of Grimshaw (1993), which turns the two-parameter
  likelihood into a one-dimensional search, so no third-party optimiser is
  needed; the fit matches `scipy.stats.genpareto.fit` to four decimal places
  for positive, near-zero and negative shape. Being a general-EVI estimator it
  handles a bounded tail, where the whole Hill family cannot.

- `harmonic_moment_estimator` and `t_hill_estimator`
  ([#325](https://github.com/DiogoRibeiro7/heavytails/issues/325)). Hill's
  contributions grow without limit, so one extreme observation moves the
  estimate arbitrarily far; these use the bounded reciprocal ratios
  `u / X_(i)` instead. Sending a single observation of ten thousand from `1e2`
  to `1e30` moves the Hill estimate from 0.502 to 0.631 and moves these not at
  all. `beta` trades robustness against efficiency, and the estimator tends to
  the Hill estimator as `beta` tends to zero.

- `scripts/tail_index_study.py` records the `heavytails` version, the git
  commit, the Python version and the run configuration in its JSON output, so
  a results file can be traced back to the code that produced it. The commit
  is recorded as well as the version because `importlib.metadata` reports what
  is installed, which lags a working tree after a version bump.

- `trimmed_hill_estimator` and `trimmed_hill_plot`, following Bhattacharya,
  Kallitsis and Stoev (2019)
  ([#323](https://github.com/DiogoRibeiro7/heavytails/issues/323)). Replacing
  three observations out of ten thousand with outliers moves the Hill estimate
  from 0.50 to 0.60; trimming five recovers 0.50. On clean data trimming ten
  raises the standard deviation only from 0.0296 to 0.0302.

- `estimator_kwargs` on `tail_index_confidence_interval`, so tuning parameters
  such as the trimming level `r` and the smoothing parameter `u` reach the
  estimator. Without it `trimmed_hill` silently ran at `r = 0`, which is the
  ordinary Hill estimator and gives no robustness at all.

- Contaminated scenarios in `scripts/tail_index_study.py`, since robustness is
  invisible on clean data.

### Changed

- The `tail_index` module docstring now states that every estimator returns the
  extreme-value index `gamma = 1/alpha`, not the tail index `alpha`
  ([#322](https://github.com/DiogoRibeiro7/heavytails/issues/322)). The module
  name refers to the quantity estimated, not the parameterisation returned, and
  the two conventions are reciprocals.

- `smoothed_hill_estimator`, the smoothed Hill estimator of Resnick and
  Stărică (1997), and `smoothed_hill_variance_ratio`, which reports the
  asymptotic variance reduction it achieves
  ([#319](https://github.com/DiogoRibeiro7/heavytails/issues/319)). Averaging
  the Hill estimate over `j` in `(k, u*k]` reduces its asymptotic variance by
  39% at `u = 2` and 55% at `u = 3`, and in the simulation study it has the
  lowest RMSE of any estimator here on a heavy tail. It inherits Hill's
  restriction to positive `gamma`.

- `generalized_hill_estimator`, the UH estimator of Beirlant, Vynckier and
  Teugels (1996)
  ([#299](https://github.com/DiogoRibeiro7/heavytails/issues/299)). Unlike the
  Hill estimator it is consistent for every extreme-value index, not only
  positive ones. On a Uniform(0,1) sample, whose index is -1, Hill can only
  ever return a positive number and reports about +0.026; the generalized Hill
  estimator recovers -0.99.

- `hill_plot`, which sweeps k on a logarithmic grid and returns the
  `(k, gamma)` series. The documentation already told readers to find a
  plateau; they now have something that produces one.

- `tail_index_confidence_interval`, with an asymptotic interval for the Hill
  estimator and a percentile bootstrap for all four. Requesting the asymptotic
  interval for an estimator that has no established closed form raises rather
  than reporting a number with no basis.

- `scripts/tail_index_study.py`, a simulation study reporting bias, standard
  deviation and RMSE for every estimator across known indices and sample sizes.
  Its results are summarised in the tail estimation guide.

- `_phi_inverse` moved from `heavy_tails` to `_special`, alongside the other
  numeric helpers, so `tail_index` can use it without importing the
  distribution module. It was never public.

## [0.2.0] - 2026-08-21

### Added

- `scripts/special_function_accuracy.py`, which sweeps both special functions
  against `mpmath` at 50 decimal digits and reports the worst relative error.
  Both are accurate to about 12 significant digits over the ranges the
  distributions use, with no degradation at the series/continued-fraction
  switch. `tests/test_special_accuracy.py` asserts those bounds.

- Kolmogorov-Smirnov and Anderson-Darling goodness-of-fit tests in
  `heavytails.validation.GoodnessOfFitTests`, which previously raised
  `NotImplementedError`
  ([#301](https://github.com/DiogoRibeiro7/heavytails/issues/301)). Both are
  reported by `AutoFit.compare_distributions` and by the `heavytails compare`
  command, so a comparison now says whether the winning family fits, not only
  how it ranks.

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

- The continued-fraction branch of the regularized lower incomplete gamma was
  wrong, not merely imprecise: its Lentz recurrence was missing its leading
  term and its `b` was shifted by one, so `P(20, 21)` returned `0.0` against a
  true `0.6157`
  ([#297](https://github.com/DiogoRibeiro7/heavytails/issues/297)). That branch
  is reached whenever `x >= a + 1`, and `InverseGamma.cdf` evaluates
  `P(alpha, beta/x)`, so its reported probabilities were wrong by factors of 2
  to 17 in the lower tail. Found by comparing against `mpmath`; the
  property-based checks could not have caught it, because a consistently wrong
  value is still monotone and still lies in [0, 1].

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

- `roadmap.improved_incomplete_beta`, a placeholder that delegated to
  `_betainc_reg` unchanged. The accuracy work it stood in for is now done and
  measured.

- `roadmap.safe_lognormal_ppf`, a workaround that caught the overflow above. It
  is redundant now that `LogNormal.ppf` handles the case itself.

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

[Unreleased]: https://github.com/DiogoRibeiro7/heavytails/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/DiogoRibeiro7/heavytails/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/DiogoRibeiro7/heavytails/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/DiogoRibeiro7/heavytails/releases/tag/v0.1.0
