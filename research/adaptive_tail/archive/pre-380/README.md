# Results generated before #380

These are historical diagnostics, not results for the current research design.
They are kept because the history is useful; they are here so that they do not
look canonical.

Every file in this directory was produced before #380 changed two things that
move the numbers:

**The trim envelope.** `_admissible_max_trim` took `min_k // 2 - 1`. It now
takes `_crossfit_min_threshold(n, min_k) - 2`, the smallest threshold any
cross-fit fold actually reaches. At `n = 1000` with `k_min = 10` the fold
minimum is 5, so the envelope is **3 where it used to be 4**. Runs at that
sample size therefore searched a trim range the current code does not.

**The decomposition split.** `cross_fitted_adaptive` now uses `seed=None`, the
deterministic split the production estimator and `oracle_experiment.py` both
use, rather than an independently randomised one. The randomised variant is
retained separately as `cross_fitted_adaptive_randomized`, so the two are
comparable instead of conflated.

| file | commit | why superseded |
| --- | --- | --- |
| `clean_pareto_decomposition_fractions.json` | `eb5b9ee5` | trim envelope, split |
| `clean_pareto_decomposition_intermediate.json` | `eb5b9ee5` | trim envelope, split |
| `oracle_intermediate_reduced_results.json` | `eb5b9ee5` | trim envelope |
| `pilot_results.json` | `125bd8b5` | trim envelope |

`pilot_results.json` is here for the same reason as the rest, and it is worth
saying why explicitly: it was produced by `oracle_experiment.py`, and #380
changed `_admissible_max_trim` in that file too. It predates the change by
provenance stamp. Its 240 cells remain a real run of a real experiment -- the
`adaptive_failure_rate` of 0 across all of them is still a fact about the code
at `125bd8b5` -- but the envelope it searched is not the current one.

Each file carries its own `provenance` block with the commit that produced it,
so none of this has to be taken on trust.
