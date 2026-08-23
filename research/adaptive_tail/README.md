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

## Selector calibration result (n = 10,000, clean Pareto)

**No compatibility cutoff reaches 95% joint acceptance while the compatibility
test is still testing anything.** Retuning the constant cannot fix this.

The calibrated quantity is the one production depends on: both cross-fit folds
succeed *and* both reach their own scaled top threshold, with every trial in
the denominator. `selector_diagnostics.py` writes
`selector_calibration_n10000.json`; the cutoff sweep below is
`selector_cutoff_sweep_n10000.json`.

| critical | joint | per-fold | per-fold² | stable set / 10 |
| --- | --- | --- | --- | --- |
| 1.0 | 0.010 | 0.077 | 0.006 | 3.9 |
| 2.0 | 0.530 | 0.738 | 0.545 | 8.4 |
| 3.0 | 0.820 | 0.910 | 0.828 | 9.4 |
| 3.5 | 0.905 | 0.953 | 0.907 | 9.7 |
| 4.0 | 0.940 | 0.970 | 0.941 | 9.8 |
| 5.0 | 0.990 | 0.995 | 0.990 | 10.0 |
| 6.0 | 1.000 | 1.000 | 1.000 | **10.0** |

Three things follow.

**Joint acceptance is the square of the per-fold rate, everywhere.** 0.910² =
0.828 against a measured 0.820; 0.970² = 0.941 against 0.940. The two fold
selectors are effectively independent, so cross-fitting *squares* the
probability of a premature stop. Reaching joint 0.95 requires per-fold 0.975 —
a far stricter demand than the full-sample calibration in #380 was making, and
the reason that calibration would have chosen a cutoff that does not deliver.

**The cutoffs that reach the target have stopped selecting.** At `critical = 4`
the stable set averages 9.8 of 10 thresholds; by `critical = 6` it is the whole
grid on every trial. A compatibility test that accepts everything is not a test.
So there is no constant that is both large enough to pass and small enough to
mean something.

**The failure-conditioning correction did not bite here, and would elsewhere.**
Fold failure rate is 0.000 at every cutoff on clean Pareto, so joint and
conditional differ only through the joint requirement. The distinction matters
under contamination, where failures are not rare.

Across five independent seed ranges at `critical = 3`, joint acceptance runs
0.755 to 0.890 — so single-range figures are worth about ±0.05, and the
calibration-to-holdout drop in `selector_calibration_n10000.json` is seed-range
variation rather than selection bias.

### What this argues for

Redesigning the rule, not its constant. The current rule requires a *hard stable
prefix*: every threshold from the smallest up to the one being accepted must
pass. One marginal threshold anywhere in the prefix truncates the whole set, and
cross-fitting gives that failure two chances per estimate. Candidates worth
considering: a soft or weighted stable set, a rule that tolerates isolated
rejections inside the prefix, or accepting on the largest stable *interval*
rather than the prefix.
