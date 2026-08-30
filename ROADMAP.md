# heavytails Roadmap

What is built, and what is worth building next.

This file was previously organised into phases, which stopped describing the
project some releases ago: it listed the documentation site, the benchmarks,
the distribution registry, the multivariate models, the copulas and the GARCH
work as future plans while all of them were shipping. A roadmap that
understates what exists misleads anyone deciding whether to depend on the
library, so it is now written against the current tree.

--------------------------------------------------------------------------------

## Shipped

**Distributions.** Continuous heavy-tailed families and the discrete ones
(Zipf, Yule–Simon, Discrete Pareto), with array-aware evaluation throughout and
quantile functions that search rather than approximate where no closed form
exists. `heavytails.registry` maps a name to a family, so callers can work
generically.

**Tail-index estimation.** Hill, Pickands and moment estimators, the
generalized Hill and the Resnick–Stărică smoothed Hill, bias correction and
variance estimation, harmonic-moment (t-Hill) estimators, and adaptive trimming
driven by a conservative spacing scan.

**Peaks over threshold.** Mean residual life and parameter-stability
diagnostics, threshold selection, generalized Pareto fitting and return levels.

**Risk and actuarial.** Value at risk, expected shortfall, tail conditional
expectation, Monte Carlo tail risk; Poisson, negative-binomial and binomial
frequency models, policy terms, layered severity and limited expected values.

**Dependent extremes.** Elliptical models including the multivariate normal and
Student-t with fitting, the tail dependence coefficient; Gaussian, Student-t,
Gumbel and Galambos copulas with empirical tail dependence; GARCH fitting, the
extremal index and declustering.

**Streaming.** Top-k maintenance and tail-index estimation over streams and
sliding windows, for data that does not fit in memory.

**Infrastructure.** A documentation site built under `--strict`, sampling and
vectorisation benchmarks, accuracy tests for the incomplete beta and gamma
implementations against independent references, and a test suite in the
thousands run across Python 3.10–3.13 on Linux, macOS and Windows.

**Release lifecycle.** `make release-check` verifies that the five files
carrying the release identity agree before anything is tagged, and
`release-preflight` adds that the tag is still free. `make verify-release`
asks the other question afterwards --- whether the tag, the GitHub release,
PyPI and the Zenodo archive all describe the same version --- because a
release can be tagged and published and still never reach PyPI, which is how
0.6.0 was lost. The replication archive is checked as an artifact rather than
as source: its manifest verifies, and the scripts it ships still run from the
layout they are deposited in.

--------------------------------------------------------------------------------

## Next

Ordered by how much each would add for someone doing applied extreme value
work. Each is absent today, not partially present.

**1. A block-maxima workflow.** `GEV_Frechet` exists as a distribution, but
nothing extracts block maxima, fits a GEV to them, or produces return levels
and confidence intervals from that fit. Block maxima is one of the two standard
EVT routes and the library currently offers only the other one. This is the
largest gap between what the package contains and what a practitioner expects.

**2. Unified fit and result objects.** Estimators return bare numbers or
tuples, so the threshold that produced an estimate, its standard error and its
diagnostics travel separately from the estimate itself, or not at all. A common
result type would let return levels, plots and comparisons take a fit rather
than a scattering of arguments.

**3. Inference for multivariate extremal dependence.** The models and the
coefficients are implemented; the inference around them is not. Estimating
dependence with uncertainty, and testing asymptotic independence, is what turns
those models into something a paper can rest on.

**4. Return levels that know how the threshold was chosen.** Intervals
currently condition on the threshold as if it were given. It was estimated, and
ignoring that understates the uncertainty in exactly the quantity most often
reported.

**5. Worked examples on real data.** The documentation demonstrates the API on
simulated data. Reproducible analyses of public datasets would show the
diagnostics deciding something, which is the part that is hard to learn from a
reference page.

**6. Rare-event simulation.** Importance sampling for tail probabilities beyond
the range where naive Monte Carlo returns anything but zero.

### Release and replication tooling

Two gaps with a known cost, both found by review rather than by CI.

**Archive-versus-manifest verification.** `verify_release` establishes that a
version is public --- tag, release, PyPI, a Zenodo DOI that resolves --- but
not that the archive at that DOI *contains* the replication package the
manuscript cites. Those came apart once: a 31-file archive was cited by a
33-file package, and every existing check passed while it was true. The
comparison that caught it is a download and a manifest diff, and it is
currently done by hand.

**A cross-platform reproduction path.** `REPRODUCE.md` asks for `gunzip`,
`sha256sum`, `diff` and a shell loop, while the environment record the same
archive ships says Windows. A small Python verifier doing the decompression,
the JSON comparison and the checksums would make the documented environment
and the documented workflow the same environment.

--------------------------------------------------------------------------------

## Research

The repository carries the replication package for *Sparse Contamination in
Tail-Index Estimation: Detectability, Negligibility, and Risk* under
`research/sparse_contamination/replication_package/`. It is a study of when a
handful of contaminated order statistics can be detected, when they can be
ignored, and when they change a risk number --- using this library's spacing
scan and harmonic-moment estimators.

The package is archived with each release, so the version DOI of the release
that carries it is what the manuscript cites. That keeps one object serving two
lifecycles, and the seams show: the archive changes whenever the software
around it does, and the manuscript bundled inside an archive cannot cite that
archive's own DOI, because the DOI does not exist until the deposit is made. A
dedicated deposit with a reserved DOI would separate them; the current
arrangement is a deliberate choice to avoid a second record, not an oversight.

The frozen results record what produced them --- `heavytails` 0.5.0 at commit
`755e6ad` --- independently of whatever version the library has since reached.

--------------------------------------------------------------------------------

## Considered, Not Committed

These shipped as empty signatures raising `NotImplementedError` and were removed
in #312. A module of empty signatures is indistinguishable from a module of
working code until you call it, so the ideas live here instead, where they
promise nothing.

**Plausible for a statistics library, unscheduled.** Regime-switching models;
survival analysis with heavy-tailed hazards; Bayesian parameter estimation and
MCMC; spatial processes and kriging; distribution classification by machine
learning; vine copulas.

**Deliberately out of scope.** Native extensions in Rust or Fortran. The
vectorised NumPy paths carry the performance-critical work, and a second
toolchain would cost more in build and packaging complexity than the remaining
speedup is worth.
