"""Does selection start paying at a larger sample size?

At n = 10,000 the calibrated compatibility rule discriminates but does not
improve squared-error risk: against taking every threshold it is measurably
worse for Pareto, Hall and Burr rho = -1/2, and indistinguishable for Burr
rho = -1/4 (#389).

There is one obvious reason that might change with n. The variance cost of
truncating the threshold grid shrinks as the sample grows; the second-order
bias it avoids does not. So the balance could tip, and Burr rho = -1/4 -- the
slowest second-order decay here, and the alternative the rule detects best --
is where it would tip first.

The design is the one #389 arrived at, held fixed:

* the cutoff is calibrated **per rho**, on exact Pareto, to nominal joint
  acceptance. The null distribution moves with rho, so a single cutoff shared
  across scenarios would compare sizes as well as laws.
* each scenario is then run at the cutoff calibrated for *its* rho.
* the comparison against no selection is **paired** by seed, with a bootstrap
  interval on the per-replication loss difference.

Calibration and evaluation use disjoint seed ranges, so the cutoff is not
chosen and judged on the same Monte Carlo samples.
"""

from __future__ import annotations

# ruff: noqa: E402
# The sys.path bootstrap has to run before the imports that depend on it.
import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.adaptive_tail.oracle_experiment import SCENARIOS
from research.adaptive_tail.selector_closure import NO_SELECTION, _paired_difference
from research.adaptive_tail.selector_diagnostics import (
    _admissible_max_trim,
    _provenance,
    _selection_rate,
    _thresholds_for_mode,
)
from research.adaptive_tail.selector_power import _cell


def _calibrate(
    *,
    n: int,
    k_grid: list[int],
    max_trim: int,
    rho: float,
    critical_grid: list[float],
    target: float,
    trials: int,
    seed_start: int,
) -> dict[str, Any]:
    """Smallest cutoff on the grid reaching ``target`` clean-Pareto acceptance.

    Smallest rather than largest: among cutoffs that are correctly sized, the
    tightest is the one that retains the most power, and the question is
    whether a correctly sized rule helps at all.
    """
    curve = []
    for critical in critical_grid:
        row = _selection_rate(
            n=n,
            k_grid=k_grid,
            max_trim=max_trim,
            rho=rho,
            critical=critical,
            trials=trials,
            seed_start=seed_start,
        )
        row.pop("outcomes", None)
        curve.append(row)

    qualifying = [r for r in curve if r["joint_acceptance_rate"] >= target]
    met = bool(qualifying)
    chosen = (
        qualifying[0] if met else max(curve, key=lambda r: r["joint_acceptance_rate"])
    )
    return {
        "rho": rho,
        "target": target,
        "target_met": met,
        "calibrated_critical": chosen["critical"],
        "calibrated_acceptance": chosen["joint_acceptance_rate"],
        "curve": curve,
    }


def build_report(
    *,
    n: int,
    scenarios: list[str],
    critical_grid: list[float],
    target: float,
    calibration_trials: int,
    evaluation_trials: int,
    calibration_seed_start: int,
    evaluation_seed_start: int,
    max_trim: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    k_grid = _thresholds_for_mode(
        n,
        k_grid_mode="intermediate",
        k_fractions=[0.02, 0.05, 0.10],
        intermediate_grid_size=10,
        intermediate_min_power=1.0 / 3.0,
        intermediate_max_power=2.0 / 3.0,
    )
    admissible = _admissible_max_trim(n, k_grid[0], max_trim)

    rhos = sorted({SCENARIOS[key].rho_used for key in scenarios})
    calibration = {
        rho: _calibrate(
            n=n,
            k_grid=k_grid,
            max_trim=admissible,
            rho=rho,
            critical_grid=critical_grid,
            target=target,
            trials=calibration_trials,
            seed_start=calibration_seed_start,
        )
        for rho in rhos
    }

    shared = {
        "n": n,
        "k_grid": k_grid,
        "max_trim": admissible,
        "contamination": 0,
        "delta": 0.0,
        "trials": evaluation_trials,
        "seed_start": evaluation_seed_start,
    }

    comparisons = []
    for key in scenarios:
        rho = SCENARIOS[key].rho_used
        critical = calibration[rho]["calibrated_critical"]
        selected = _cell(scenario_key=key, critical=critical, **shared)
        everything = _cell(scenario_key=key, critical=NO_SELECTION, **shared)
        comparisons.append(
            {
                "scenario": key,
                "rho_used": rho,
                "calibrated_critical": critical,
                "calibrated_null_acceptance": calibration[rho]["calibrated_acceptance"],
                "full_grid_acceptance": selected["joint_full_acceptance_rate"],
                "run_stable_fraction": selected["run_stable_fraction"],
                "rmse_selected": selected["rmse"],
                "rmse_no_selection": everything["rmse"],
                "bias_selected": selected["bias"],
                "bias_no_selection": everything["bias"],
                "paired_vs_no_selection": _paired_difference(
                    selected["replications"],
                    everything["replications"],
                    draws=bootstrap_draws,
                    seed=23,
                ),
            }
        )

    return {
        "provenance": _provenance(),
        "purpose": (
            "Whether the calibrated compatibility rule begins to pay under "
            "squared-error loss at a larger sample size, where the variance "
            "cost of truncating the grid falls but the second-order bias it "
            "avoids does not."
        ),
        "configuration": {
            "n": n,
            "k_grid": k_grid,
            "admissible_max_trim": admissible,
            "scenarios": scenarios,
            "critical_grid": critical_grid,
            "target_acceptance": target,
            "calibration_trials": calibration_trials,
            "evaluation_trials": evaluation_trials,
            "calibration_seed_start": calibration_seed_start,
            "evaluation_seed_start": evaluation_seed_start,
            "bootstrap_draws": bootstrap_draws,
            "no_selection_critical": NO_SELECTION,
            "crossfit_seed": "None, matching the production estimator default",
        },
        "calibration_by_rho": list(calibration.values()),
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50_000)
    parser.add_argument(
        "--scenarios", default="pareto,hall_rho_half,burr_rho_half,burr_rho_quarter"
    )
    parser.add_argument("--critical-grid", default="3,3.5,4,4.5,5,5.5,6")
    parser.add_argument("--target", type=float, default=0.95)
    parser.add_argument("--calibration-trials", type=int, default=300)
    parser.add_argument("--evaluation-trials", type=int, default=400)
    parser.add_argument("--calibration-seed-start", type=int, default=110_000)
    parser.add_argument("--evaluation-seed-start", type=int, default=120_000)
    parser.add_argument("--max-trim", type=int, default=8)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    report = build_report(
        n=args.n,
        scenarios=args.scenarios.split(","),
        critical_grid=[float(v) for v in args.critical_grid.split(",")],
        target=args.target,
        calibration_trials=args.calibration_trials,
        evaluation_trials=args.evaluation_trials,
        calibration_seed_start=args.calibration_seed_start,
        evaluation_seed_start=args.evaluation_seed_start,
        max_trim=args.max_trim,
        bootstrap_draws=args.bootstrap_draws,
    )

    if args.json:
        args.json.write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
        )

    print(f"n = {report['configuration']['n']}, clean Pareto calibration by rho:")
    for row in report["calibration_by_rho"]:
        note = "" if row["target_met"] else "  (target not met)"
        print(
            f"  rho {row['rho']:>6.2f}  c = {row['calibrated_critical']:>4.1f}  "
            f"acceptance {row['calibrated_acceptance']:.3f}{note}"
        )

    print("\nCalibrated selector against taking every threshold, paired:")
    print(
        f"  {'scenario':<18} {'c':>5} {'accept':>7} {'dMSE':>10} {'95% CI':>24} {'verdict':>14}"
    )
    for row in report["comparisons"]:
        d = row["paired_vs_no_selection"]
        if d is None:
            continue
        interval = f"[{d['bootstrap_lower']:+.5f}, {d['bootstrap_upper']:+.5f}]"
        verdict = (
            "selection helps"
            if d["favours_selection"]
            else ("indistinct" if d["interval_contains_zero"] else "selection hurts")
        )
        print(
            f"  {row['scenario']:<18} {row['calibrated_critical']:>5.1f} "
            f"{row['full_grid_acceptance']:>7.3f} "
            f"{d['mean_mse_difference']:>+10.5f} {interval:>24} {verdict:>14}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
