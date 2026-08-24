# Evaluating an adaptive threshold selector

This is the protocol the hard-prefix selector was eventually judged by, written
down so the next candidate is judged the same way. It exists because the first
few rounds of that work judged the selector by the wrong things, and each wrong
thing looked reasonable at the time.

## The principle

> **Adaptation should be evaluated by excess risk relative to a non-adaptive
> benchmark, not by the apparent correctness of its intermediate decisions.**

The hard-prefix selector is the worked example. It *does* detect second-order
bias: at n = 10,000 it accepts the full threshold grid on 95% of exact Pareto
samples and 62% of Burr samples with ρ = -1/4, and when it truncates it
truncates hard. The detection is real, survives calibration, and is not an
artefact of the tuning — running exact Pareto at each ρ separates the two, and
the law accounts for 0.312 of the gap against 0.015 for the tuning.

It still made the estimator worse. Statistical discrimination did not convert
into decision-theoretic improvement:

    discrimination  =/=>  lower risk

That is the finding, and it is more useful than the selector would have been.

## The admission test

In this order. Each stage can only be read if the one before it passed,
because otherwise the comparison confounds size with power, or power with risk.

**1. Null-size calibration, separately for each ρ.**
The null distribution of the compatibility statistic moves with the ρ the
orthogonalized weights are built from, so a cutoff shared across scenarios
compares sizes as well as laws. Calibrate on exact Pareto:

    minimise   c
    subject to P_Pareto,ρ(acceptable behaviour | c) ≥ 1 - α

The smallest qualifying cutoff, not the closest to the target. `1 - α` is a
lower bound, not a symmetric objective, and among correctly sized rules the
tightest retains the most power — which is the candidate's best case, and
therefore the honest one to test.

**2. Held-out discrimination.**
Calibrate and evaluate on disjoint seed ranges. A cutoff chosen and judged on
the same Monte Carlo samples is chosen partly on their noise.

**3. Paired risk comparison.**
Cutoffs evaluated on the same seeds give paired losses. The standard error of
either RMSE says nothing about their difference, and the estimates are highly
correlated, so the marginal uncertainty overstates it badly. Compute the
per-replication loss difference with a bootstrap interval.

**4. The no-selection benchmark.**
Compare against taking every threshold. A candidate that discriminates
beautifully and does not beat this has not earned its variance.

## What does not count as evidence

- **High acceptance under the null.** Under exact Pareto the best threshold
  *is* the largest one, so a correctly sized rule should accept nearly
  everything. That is type-I calibration, not a loss of selectivity. An earlier
  draft of this work read it the other way round.
- **Recovery rates, or prettier threshold choices.** These are the intermediate
  decisions the principle above declines to be judged by.
- **Marginal risk differences.** Without pairing, an effect the size of the one
  that matters here is invisible inside the noise of either arm.
- **A single sample size.** The ambiguous cell at n = 10,000 became
  significantly harmful at n = 50,000.

## The apparatus

| stage | where |
| --- | --- |
| null-size calibration by ρ | `selector_scale.py:_calibrate`, `selector_diagnostics.py:_selection_rate` |
| production cross-fit path | `selector_diagnostics.py:_trace_crossfit`, held to the estimator by a parity test |
| discrimination | `selector_power.py`, run-level and fold-level stable fractions |
| paired risk | `selector_closure.py:_paired_difference` |
| no-selection benchmark | `selector_closure.py:NO_SELECTION` |

Results carry a `provenance` block naming the commit, the interpreter and the
NumPy version. Superseded results are archived rather than overwritten, under
`archive/`, with the reason each was superseded.

## The open question

The empirical lesson turns into a theoretical one. The question is not

> does the selector identify threshold instability?

but

> **when is acting on that instability worth its variance cost?**

Which wants a risk property rather than a consistency property. Something of
the shape

    R(gamma_hat over the selected set) - inf over K of R(gamma_hat over K) <= eps_n

or, more directly comparable to what is measured here, an excess-risk statement
against the full-grid aggregate. A selector with a bound of that kind would be
worth building even if it detected instability less often than this one does.
