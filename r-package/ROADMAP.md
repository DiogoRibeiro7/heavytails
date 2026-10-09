# heavytailsR roadmap

The R edition develops native R implementations of the statistical functionality in the [Python project](https://github.com/DiogoRibeiro7/heavytails), not wrappers. Use focused issues and pull requests targeting `main`.

## Phase 1: package foundation
- [x] Installable package directory, DESCRIPTION, NAMESPACE and manual pages.
- [x] Pareto type I density/CDF/quantile/random generation with validation.
- [x] Hill extreme-value index (gamma = 1/alpha).
- [x] Pareto VaR and expected shortfall, with correct infinite-mean behaviour.
- [x] Unit tests and separate cross-platform R CMD check workflow.
- [ ] Confirm package name availability with pak::pkg_name_check().
- [ ] Validate the built source tarball with R CMD check --as-cran.

## Phase 2: distributions and inference
- [ ] Generalized Pareto distribution with a continuous zero-shape limit.
- [ ] Peaks-over-threshold inference, threshold selection and return levels.
- [ ] Further heavy-tailed families (Burr XII, log-logistic, Fréchet, Student-t).
- [ ] Tail-index estimators (moment, Pickands, trimmed and bias-corrected Hill).
- [ ] Parameter fitting with diagnostics and statistical uncertainty.

## Phase 3: applications and CRAN quality
- [ ] Expanded risk models and actuarial calculations.
- [ ] Copulas, dependence and time-series extremes.
- [ ] Vignettes and independent numerical reference fixtures.
- [ ] Performance and stress testing for extreme tails.
- [ ] CRAN incoming checks, documentation audit and source tarball release.

## Engineering principles

All estimators must document their parameterisation, domain and assumptions. Test support boundaries, numerical tails and invalid types. Avoid Python runtime dependencies. Do not claim CRAN readiness before successful build and checks on the final tarball.
