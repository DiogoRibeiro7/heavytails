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

## Selector calibration and power (n = 10,000)

**The original compatibility cutoff is severely under-calibrated for the
cross-fit geometry: `c` of roughly 4 to 5 is needed for nominal clean-Pareto
joint acceptance, against the heuristic `sqrt(2 log M)` of about 2.15 at
M = 10. The calibrated selector has genuine discrimination, but at n = 10,000
that discrimination does not improve squared-error risk** — in three of four
scenarios it measurably worsens it, and in the fourth it is indistinguishable
from taking every threshold.

Artifacts: `selector_calibration_n10000.json`,
`selector_cutoff_sweep_n10000.json`, `selector_power_n10000.json`,
`selector_power_noselection_n10000.json` and `selector_closure_n10000.json`
(matched null sizes across ρ, and paired loss differences).

### The null curve

The calibrated quantity is the one production depends on: both cross-fit folds
succeed *and* both reach their own scaled top threshold, every trial in the
denominator.

| critical | joint | per-fold | per-fold² | stable set / 10 |
| --- | --- | --- | --- | --- |
| 2.0 | 0.530 | 0.738 | 0.545 | 8.4 |
| 3.0 | 0.820 | 0.910 | 0.828 | 9.4 |
| 4.0 | 0.940 | 0.970 | 0.941 | 9.8 |
| 5.0 | 0.990 | 0.995 | 0.990 | 10.0 |

Joint acceptance is the square of the per-fold rate throughout. That is what
theory predicts and is best read as a validation check: the split is
value-independent and the folds hold disjoint independent observations, so
`P(A1 ∩ A2) ≈ p²` on clean IID data. It does still impose a real cost —
joint 0.95 needs per-fold 0.975 — and it is the reason the full-sample
calibration was measuring an easier quantity. It should **not** be assumed to
carry over to contaminated samples, where contamination planted before the
split is shared between the folds dependently.

Across five independent seed ranges at `c = 3`, joint acceptance runs 0.755 to
0.890, so single-range figures at 200 trials are worth about ±0.05.

### High null acceptance is not a loss of selectivity

An earlier version of this section claimed that the cutoffs reaching the target
"have stopped selecting", on the grounds that the stable set then averages 9.8
of 10 thresholds. That inference is wrong. Under exact Pareto the best threshold
**is** the largest one, so a correctly sized test *should* accept essentially
the whole grid. Near-total null acceptance is type-I calibration. Only the power
curve can say whether the rule still discriminates, and it was not measured.

### The power curve

Measured at r = 0, 200 trials per cell. `p10 frac` is the tenth percentile of
`K_final / k_max`, where `K_final` is the top of the stable set.

| scenario | c | full-grid | p10 frac | rmse | bias |
| --- | --- | --- | --- | --- | --- |
| pareto | 4.0 | 0.950 | 1.000 | 0.0626 | -0.003 |
| hall ρ=-1/2 | 4.0 | 0.940 | 1.000 | 0.0781 | -0.008 |
| burr ρ=-1/2 | 4.0 | 0.875 | 1.000 | 0.0928 | -0.011 |
| **burr ρ=-1/4** | 4.0 | **0.615** | **0.362** | 0.2198 | -0.112 |
| pareto | 5.0 | 0.995 | 1.000 | 0.0485 | 0.000 |
| burr ρ=-1/4 | 5.0 | 0.765 | 0.711 | 0.2059 | -0.117 |

**The rule has power.** At `c = 4`, Pareto accepts the full grid 95% of the
time and Burr with ρ = -1/4 only 62%. Taking the shallower of the two folds per
replication, the bottom decile of Burr runs cuts the grid to **0.259** of its
span while Pareto's stays at 1. (The `p10 frac` column above pools the folds
and reads 0.362; the run-level figure is the one to quote about a
*replication*.) That is the intended behaviour, and it survives calibration.

**And it is the law, not the tuning.** Each scenario carries its own
`rho_used`, so Pareto at ρ = -1 against Burr at ρ = -1/4 moves the law and the
weights together. Running exact Pareto at each ρ separates them: at `c = 4`,
null acceptance is 0.943 at ρ = -1, 0.912 at ρ = -1/2 and 0.927 at ρ = -1/4. So
of the gap down to Burr's 0.615, the tuning accounts for **0.015** and the law
for **0.312**.

### But the power does not buy accuracy

Against no compatibility selection at all (`c = 100`, the whole grid always
accepted):

| scenario | c=4 | c=5 | c=6 | no selection |
| --- | --- | --- | --- | --- |
| pareto | 0.0626 | 0.0485 | 0.0485 | **0.0474** |
| hall ρ=-1/2 | 0.0781 | 0.0712 | 0.0679 | **0.0679** |
| burr ρ=-1/2 | 0.0928 | 0.0828 | 0.0791 | **0.0789** |
| burr ρ=-1/4 | 0.2198 | **0.2059** | 0.2078 | 0.2080 |

For Pareto, Hall and Burr ρ = -1/2, RMSE falls towards the no-selection
benchmark as the cutoff is relaxed. Burr ρ = -1/4 is **not** monotone: it has a
shallow minimum at `c = 5`, 0.2059 against 0.2080 with no selection.

That 0.002 cannot be judged against the standard error of an RMSE, because
every cutoff is evaluated on the same seeds and the comparison is paired. The
right quantity is the per-replication loss difference, and with 400 trials and
a paired bootstrap it comes out sharper than the marginal reading suggested:

| scenario | c | ΔMSE vs no selection | 95% CI | |
| --- | --- | --- | --- | --- |
| pareto | 4 | +0.00107 | [+0.00039, +0.00189] | selection **hurts** |
| pareto | 5 | +0.00055 | [+0.00002, +0.00131] | hurts |
| hall ρ=-1/2 | 4 | +0.00107 | [+0.00040, +0.00186] | hurts |
| hall ρ=-1/2 | 5 | +0.00055 | [+0.00013, +0.00110] | hurts |
| burr ρ=-1/2 | 4 | +0.00176 | [+0.00075, +0.00284] | hurts |
| burr ρ=-1/2 | 5 | +0.00082 | [+0.00023, +0.00152] | hurts |
| burr ρ=-1/4 | 4 | -0.00113 | [-0.00536, +0.00328] | indistinguishable |
| burr ρ=-1/4 | 5 | -0.00056 | [-0.00386, +0.00301] | indistinguishable |

In three of the four scenarios the interval excludes zero on the wrong side:
selection is measurably *worse* than taking every threshold. In the fourth --
the alternative it detects best -- the point estimate favours selection and the
interval contains zero comfortably.

Bias tells the same story. Cutting the grid short on Burr ρ = -1/4 at `c = 4`
removes about 0.004 of bias (-0.112 against -0.117 at `c = 5`) and costs enough
variance to raise RMSE by 0.014.

So the rejections the rule makes are correct rejections — it really is finding
the biased configurations — and they still do not pay for themselves. The same
ordering holds under contamination: at r = 1, Δ = 10 on Pareto, RMSE runs
0.147, 0.112, 0.107 across `c` = 4, 4.5, 5.

### What this does and does not establish

Established: the heuristic constant is far too aggressive under cross-fitting;
a calibrated constant exists; and at n = 10,000 the calibrated rule retains
genuine discrimination against second-order bias.

Not established: that threshold-compatibility selection is worthless in
general. This is one sample size, one estimator, four scenarios, and squared
error of the threshold-averaged estimate as the criterion. A rule that
discriminates without improving squared-error risk may still be worth having
under a different loss, and larger n or stronger second-order bias may change
the balance.

What it does argue is that **no redesign should be judged by null acceptance or
by discrimination alone.** A soft or tolerant prefix will look better on both
while doing nothing for the estimate. Any candidate should be compared at
matched clean-Pareto size, against no selection, on the error of the estimator
it feeds.

## The same closure at n = 50,000

**The balance does not tip. It moves further the other way.** At n = 10,000 the
decisive cell — Burr ρ = -1/4, the slowest second-order decay here and the
alternative the rule detects best — was indistinguishable from taking every
threshold, with the point estimate mildly favouring selection. At n = 50,000 it
is measurably worse, and so is every other scenario.

Artifact: `selector_scale_n50000.json`. Design held fixed from the n = 10,000
closure: the cutoff is calibrated per ρ on exact Pareto to 95% joint
acceptance, each scenario is run at the cutoff calibrated for *its* ρ, and the
comparison against no selection is paired by seed with a bootstrap interval.
Calibration and evaluation use disjoint seed ranges.

### Calibration tightens with n, as it should

| ρ | c at n = 50,000 | clean-Pareto acceptance |
| --- | --- | --- |
| -1 | 3.5 | 0.960 |
| -1/2 | 4.5 | 0.963 |
| -1/4 | 4.5 | 0.953 |

Against roughly 4 to 5 at n = 10,000. The null distribution concentrates, so a
correctly sized rule can afford to be tighter.

### Paired ΔMSE against taking every threshold

| scenario | c | ΔMSE | 95% CI | |
| --- | --- | --- | --- | --- |
| pareto | 3.5 | +0.00026 | [+0.00006, +0.00053] | selection hurts |
| hall ρ=-1/2 | 4.5 | +0.00023 | [+0.00002, +0.00052] | selection hurts |
| burr ρ=-1/2 | 4.5 | +0.00046 | [+0.00014, +0.00085] | selection hurts |
| **burr ρ=-1/4** | 4.5 | **+0.00161** | **[+0.00036, +0.00299]** | **selection hurts** |

All four intervals exclude zero on the wrong side. At n = 10,000 three did and
the fourth contained zero; at n = 50,000 the fourth has joined them.

### Why it goes this way

The rule still has power: Burr ρ = -1/4 accepts the full grid on 0.858 of
samples against 0.953 under the matched null, and when it does truncate it
truncates hard — the tenth percentile of the run-level stable fraction is
0.298. The discrimination is real and survives calibration at both sample
sizes.

What does not survive is the balance. Truncating buys almost no bias: -0.0800
against -0.0829 with no selection, about 0.003. It costs RMSE 0.1211 → 0.1276,
about 0.0065. The reduction in bias from truncating stays too small relative to
the variance introduced by estimating from fewer tail observations.

Both components can shrink with n, and this says nothing about which shrinks
faster — an earlier draft claimed the variance cost grows with n, which these
data do not show. What they do show is that the balance is against truncation
at both sample sizes tested, and that the ambiguous case at n = 10,000 lands on
the harmful side at n = 50,000.

### Status

Frozen under squared-error loss. Two sample sizes, four scenarios, matched
null sizes, paired intervals: at n = 10,000 selection is worse in three
scenarios and indistinguishable in the fourth; at n = 50,000 it is worse in all
four. A future selector should be measured against taking every threshold, on
the error of the estimator it feeds, before its rejection behaviour is
discussed at all.

This does not say threshold-compatibility selection is worthless under a
different loss, at a different grid, or against second-order decay slower than
ρ = -1/4. It does say that tuning this rule further, under squared error, on
these alternatives, is not where the value is.

### Why the smallest qualifying cutoff

Calibration is a constrained minimisation:

    minimise   c
    subject to P_Pareto,ρ(joint full acceptance | c) ≥ 0.95

Larger `c` is more permissive, so the smallest qualifying cutoff is the rule
with the most remaining power subject to correct null size. That matters for
reading the result: the selector was given its best reasonable chance —
**maximum selectivity subject to correct null calibration** — and still lost
under squared error in all four scenarios at n = 50,000. A looser qualifying
cutoff would have weakened selection for nothing and made the negative
conclusion easier to reach.

The rule is not "closest to 0.95 from either side". 95% is a lower bound on
clean-null acceptance, not a symmetric target, so 0.947 does not qualify by
being nearer to it than 0.960 is. Comparing a mis-sized rule against no
selection would confound size with risk.

## Status of this line: frozen

Hard-prefix compatibility selection is frozen under squared-error loss. The
record is two sample sizes, null calibration matched by ρ, demonstrated power,
paired risk comparisons against taking every threshold, and the slowest tested
second-order regime moving from inconclusive to significantly harmful.

**The next work here is not another experiment.** Not n = 100,000, not another
cutoff grid, not a tolerant-prefix variant. The question has been answered for
this selector, and the consequence is methodological: an adaptive threshold
rule needs an oracle or risk argument, not merely consistency at detecting
threshold instability. Detecting instability correctly and improving the
estimate are different properties, and this line of work is a worked example of
the first without the second.

A future selector earns attention by beating the `c = 100` benchmark under the
estimator's actual loss. Until it does, its rejection behaviour is not evidence
for it.

The protocol that judgement rests on is written up separately in
[evaluating-a-selector.md](evaluating-a-selector.md): the order of the stages,
what does not count as evidence, which script implements each stage, and the
risk property a future selector would need. It is worth reading before
proposing one, because several of the wrong ways to judge a selector looked
reasonable while this work was doing them.
