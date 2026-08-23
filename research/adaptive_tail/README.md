# Adaptive Tail Estimator Research Track

This directory is for the methodological question behind the adaptive
threshold-averaged orthogonalized Hill estimator. The library implementation is
usable, but the paper claim needs separate evidence:

1. Define the second-order tail class
   `F_2RV(gamma, rho, A)` and a contamination class `C(r_n, Delta_n)`.
2. Prove a uniform local expansion for the trimmed orthogonalized estimator
   `gamma_hat_BR(r, k)`.
3. Prove adaptive trimming consistency under second-order regular variation:
   false trimming probability tends to zero on clean data, and the planted
   count is recovered under a detectable boundary-spacing separation.
4. Prove concentration over the candidate `k` grid.
5. Derive, rather than choose heuristically, the compatibility penalty used by
   the threshold selector.
6. Prove or refute an oracle inequality for the adaptive estimator.

The empirical target is the oracle risk ratio

```text
Q_n = R(gamma_hat_adaptive) / min_{(r,k) in A_n} R(gamma_hat_BR(r,k)).
```

The generic benchmark in `scripts/tail_index_study.py` is not designed to
measure this. It compares named estimators at a single external `k`. The
experiment here varies the contamination count, contamination strength,
second-order parameter and threshold envelope, then compares the adaptive
estimator against an empirical oracle over `(r, k)`.

The oracle is selected out of sample: half the Monte Carlo replications choose
the best `(r, k)`, the other half evaluates it, and then the roles are swapped.
The candidate `k` values are the exact logarithmic grid used by the adaptive
estimator, not the raw envelope fractions.

Two threshold envelopes are supported:

- `--k-grid-mode fractions` preserves the frozen pilot grid derived from fixed
  fractions of `n`.
- `--k-grid-mode intermediate` uses a logarithmic grid from
  `n^intermediate_min_power` to `n^intermediate_max_power`, with defaults
  `n^(1/3)` to `n^(2/3)`. This is the theory-oriented envelope because
  `k -> infinity` and `k/n -> 0`.

For any threshold envelope, the adaptive estimator and the local oracle use the
same admissible trim range:

```text
h_n = min(max_trim, k_min_crossfit - 2).
```

Here `k_min_crossfit` is the smallest threshold reached by the production
two-fold cross-fit scaling rule. This keeps every fold-level local candidate
inside the admissible `r < k - 1` region.

Rows where `contamination_count > h_n` are labeled with
`contamination_supported: false`; they are outside the declared trimming
envelope and should not be used as evidence for recovery claims.

## Initial Experiment

Run a quick smoke check:

```bash
python research/adaptive_tail/oracle_experiment.py --trials 2 --sample-sizes 500 --scenarios pareto,burr_rho_half
```

Run a pilot grid:

```bash
python research/adaptive_tail/oracle_experiment.py --trials 200 --json oracle-results.json
```

Run the same design on the intermediate threshold envelope:

```bash
python research/adaptive_tail/oracle_experiment.py \
  --trials 200 \
  --k-grid-mode intermediate \
  --json oracle-intermediate-results.json
```

The output reports:

- `adaptive_rmse`: unconditional RMSE of the adaptive estimator, present only
  when every replication succeeds.
- `adaptive_rmse_success`: RMSE conditional on estimator success.
- `adaptive_failure_rate`: fraction of replications where the adaptive
  estimator refused to produce a value.
- `oracle_rmse`: out-of-sample empirical oracle RMSE over the adaptive
  estimator's exact `(r, k)` grid.
- `risk_ratio`: `adaptive_mse / oracle_mse`, reported only when the adaptive
  estimator has no failures.
- `contamination_supported`: whether the planted contamination count is inside
  the shared adaptive/oracle trim envelope.
- `risk_ratio_bootstrap`: bootstrap uncertainty for the risk ratio. Each
  bootstrap draw resamples Monte Carlo replications and reruns the oracle
  select/evaluate split, so the interval includes oracle-selection variability
  rather than conditioning on the originally selected oracle pairs.
- `delta`: contamination strength. Clean cells with `contamination_count == 0`
  are run once with `delta: null`, because contamination strength is irrelevant
  when no observations are replaced.
- `trim_recovery_vanishing`: probability, with a Wilson interval, that the
  adaptive trimming rule recovers the planted contamination count at the
  largest candidate threshold using the same vanishing level as the estimator.
- `trim_recovery_fixed_005`: the same diagnostic under the fixed 5% level.

JSON output is structured as:

```text
{
  "provenance": {...},
  "configuration": {...},
  "results": [...]
}
```

The provenance block records `heavytails_version`, `version_source`,
`git_commit`, `python_version` and `numpy_version`, so frozen result artifacts
can be traced to the working tree and numerical runtime that produced them.

For exact Pareto scenarios, `rho_true` is `null` and `rho_used` records the
orthogonalization tuning value. This avoids treating an exact Pareto tail as if
it had an identified second-order parameter.

The script is intentionally explicit rather than optimized. It is a research
artifact for deciding whether the estimator deserves a theorem, not a public
API.

## Clean Pareto Decomposition

The clean Pareto pilot showed that the adaptive estimator can sit noticeably
above the local oracle even when the model is exact. Before scaling the oracle
experiment, decompose that penalty on the same simulated samples:

```bash
python research/adaptive_tail/clean_pareto_decomposition.py \
  --trials 200 \
  --json clean-pareto-decomposition.json
```

This report compares:

- `best_local_oracle_oos`: out-of-sample empirical local oracle over fixed
  `(r, k)` candidates, using the same two-fold Monte Carlo select/evaluate
  rotation as the oracle experiment. This is the denominator for the reported
  decomposition ratios.
- `best_local_oracle_in_sample`: in-sample empirical MSE minimum over fixed
  local `(r, k)` candidates, reported only as a winner's-curse diagnostic.
- `full_sample_selected_local`: the final stable local estimator selected by
  the adaptive threshold rule, without threshold aggregation.
- `full_sample_adaptive_aggregation`: the full-sample adaptive aggregate.
- `cross_fitted_adaptive`: the production cross-fitted adaptive estimator,
  using the estimator's default deterministic split.
- `cross_fitted_adaptive_randomized`: a secondary diagnostic using independent
  split seeds.

The primary cross-fit row matches `oracle_experiment.py`. The randomized
secondary row uses data seeds `s` and split seeds `1_000_000_000 + s`.

Use `--k-grid-mode intermediate` on this script to inspect the same
decomposition under the theory-oriented threshold envelope.

## Selector Diagnostics

The intermediate-grid results show that clean-Pareto excess risk is dominated
by threshold compatibility selection. Before running more contamination grids,
trace and calibrate that selector under exact Pareto:

```bash
python research/adaptive_tail/selector_diagnostics.py \
  --n 10000 \
  --k-grid-mode intermediate \
  --json selector-diagnostics.json
```

This report calibrates candidate compatibility cutoffs on calibration seeds,
evaluates the selected cutoff on held-out seeds, and records cross-fit fold
traces with training thresholds, per-threshold trims, stable sets, evaluation
trims, weights and failure stages.
