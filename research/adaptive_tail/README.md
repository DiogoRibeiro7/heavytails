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
second-order parameter and threshold grid, then compares the adaptive estimator
against the empirical oracle over `(r, k)`.

## Initial Experiment

Run a quick smoke check:

```bash
python research/adaptive_tail/oracle_experiment.py --trials 2 --sample-sizes 500 --scenarios pareto,burr_rho_half
```

Run a larger exploratory grid:

```bash
python research/adaptive_tail/oracle_experiment.py --trials 200 --json oracle-results.json
```

The output reports:

- `adaptive_rmse`: RMSE of the adaptive estimator.
- `oracle_rmse`: best empirical RMSE over the supplied `(r, k)` grid.
- `risk_ratio`: `adaptive_rmse**2 / oracle_rmse**2`.
- `trim_recovery`: probability that the adaptive trimming rule recovers the
  planted contamination count at the largest candidate threshold.

The script is intentionally explicit rather than optimized. It is a research
artifact for deciding whether the estimator deserves a theorem, not a public
API.
