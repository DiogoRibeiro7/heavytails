# heavytails Roadmap

This roadmap outlines the planned development phases for the **heavytails** project, focusing on expanding its scope from a pure heavy-tailed distribution library to a comprehensive framework for tail-risk modeling, simulation, and diagnostics.

---

## 🧭 Phase 1 — Core Implementation ✅ *(Completed)*

**Goal:** Build a solid mathematical foundation of continuous heavy-tailed distributions.

* [x] Implement Pareto, Cauchy, Student-t, LogNormal, Weibull (k<1), Fréchet, GEV (ξ>0)
* [x] Implement deterministic RNG wrapper and helper utilities
* [x] Add closed-form PDF, CDF, SF, and PPF methods for all distributions
* [x] Add `extra_distributions.py` with GPD, Burr XII, LogLogistic, Inverse-Gamma, and BetaPrime
* [x] Add repository structure, Poetry packaging, and full README

---

## 🧩 Phase 2 — Expansion & Validation *(In Progress)*

**Goal:** Extend functionality beyond continuous distributions, add validation layers, and prepare for publication.

### Implemented

* [x] Discrete heavy-tailed distributions (Zipf, Yule–Simon, Discrete Pareto)
* [x] Tail index estimators (Hill, Pickands, Moment)
* [x] Plotting utilities (log–log tail plots, QQ plots)
* [x] Unit test suite for continuous and discrete families
* [x] CI pipeline integration with GitHub Actions

### Next Steps

* [ ] Validate numerical stability of incomplete beta/gamma implementations
* [ ] Extend test coverage with edge cases and numerical comparisons
* [ ] Add benchmarks for sampling performance and asymptotic accuracy

---

## 📈 Phase 3 — Analytical Tools *(Upcoming)*

**Goal:** Move from modeling to inference and diagnostics.

* [x] Implement additional tail-index estimators (Generalized Hill, Resnick–Stărică smoothed Hill)
* [x] Add bias-correction and variance estimation tools
* [x] Develop tail QQ and Hill plot visual diagnostics (optional matplotlib support)
* [x] Implement EVT-based threshold selection and excess fitting

---

## 🧠 Phase 4 — Simulation & Applications *(Planned)*

**Goal:** Provide applied modules for risk analysis and extreme-event simulation.

* [x] Monte Carlo simulation utilities for heavy-tail risk estimation
* [x] Tail-risk metrics (VaR, ES, tail conditional expectation)
* [x] Actuarial layer: aggregate-loss models and compound distributions
* [ ] Integration with log-based or empirical tail fitting (e.g., datasets)

---

## 📚 Phase 5 — Documentation & Dissemination *(Planned)*

**Goal:** Make `heavytails` reproducible, documented, and publishable.

* [ ] Full API documentation using MkDocs or Sphinx
* [ ] Add theoretical appendix (mathematical definitions, tail proofs)
* [ ] Write and release technical report / whitepaper
* [ ] Submit paper to *The R Journal* or *Journal of Open Source Software* (JOSS)

---

## 🧩 Long-Term Vision

* Extend library to **multivariate heavy-tailed models** (Elliptical, Student-t Copulas)
* Integrate **time-series tail modeling** (ARCH/GARCH, stable innovations)
* Develop **tail simulation kernels** in Rust or Fortran for performance
* Expose a unified Python API for both distribution modeling and tail inference

---

## 💡 Considered, Not Committed

These shipped as empty signatures raising `NotImplementedError` and were removed
in #312. A module of empty signatures is indistinguishable from a module of
working code until you call it, so the ideas live here instead, where they
promise nothing.

**Plausible for a statistics library, unscheduled.** Regime-switching models;
survival analysis with heavy-tailed hazards; Bayesian parameter estimation and
MCMC; spatial processes and kriging; distribution classification by machine
learning; GPU-accelerated sampling; vine copulas; tail geometry via convex hulls
and Voronoi diagrams; interoperability shims for NumPy and scikit-learn.

**Infrastructure worth building.** A distribution registry, to replace the
string-keyed lookups the CLI and `AutoFit` both rely on — this is the one piece
of #312 kept as work rather than dropped. Alongside it, data-quality assessment
and configuration management.

**Deliberately out of scope.** Web scraping for financial data, an interactive
tutorial system, a plugin loader, unit conversion, a citation manager, a web
service. These belong in a project that uses this library, not in a
distributions library.

---

**Maintainer:** Diogo Ribeiro
**License:** MIT
**Repository:** [https://github.com/DiogoRibeiro7/heavytails](https://github.com/DiogoRibeiro7/heavytails)
