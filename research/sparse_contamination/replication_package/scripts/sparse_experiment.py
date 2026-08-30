"""Exact-Pareto sparse-contamination experiment for the private note.

The simulation uses the Renyi spacing representation directly.  This keeps the
experiment aligned with the theorem: clean normalized spacings are iid
Exp(mean=gamma), and equal-factor contamination adds S=r log(Delta) to the
boundary spacing.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from _provenance import base_provenance  # noqa: E402


DEFAULT_GAMMA = 0.5
DEFAULT_N_VALUES = (1_000, 10_000, 50_000)
DEFAULT_R_VALUES = (1, 3, 5, 10)
DEFAULT_BETAS = (0.5, 1.0, 2.0)
SIGNAL_RULES = (
    ("0", "zero", 0.0),
    ("0.5I", "I", 0.5),
    ("1.5I", "I", 1.5),
    ("0.5D", "D", 0.5),
    ("1.5D", "D", 1.5),
    ("0.5R", "R", 0.5),
    ("1.5R", "R", 1.5),
)


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _alpha(n: int) -> float:
    return min(0.05, 1.0 / math.log(n) ** 2)


def _interlock_level(n: int) -> float:
    return min(1e-4, 1.0 / n**2)


def _scales(n: int, r: int, gamma: float, h: int) -> dict[str, float]:
    k = int(math.floor(math.sqrt(n)))
    alpha = _alpha(n)
    return {
        "I": gamma * math.sqrt(r * k / (k - r)),
        "D": gamma * math.log(h / alpha),
        "R": gamma * math.sqrt(k),
    }


def _signal_grid(
    n: int,
    r: int,
    gamma: float,
    h: int,
    tolerance: float,
) -> list[dict[str, float | str]]:
    scales = _scales(n, r, gamma, h)
    points: list[dict[str, float | str]] = []
    for label, scale_name, factor in SIGNAL_RULES:
        signal = 0.0 if scale_name == "zero" else factor * scales[scale_name]
        duplicate = None
        for point in points:
            other = float(point["signal"])
            bound = tolerance * max(1.0, abs(signal), abs(other))
            if abs(signal - other) <= bound:
                duplicate = point
                break
        if duplicate is None:
            points.append({"label": label, "signal": signal})
        else:
            duplicate["label"] = f"{duplicate['label']}|{label}"
    return points


def _spacing_p_values(spacings: np.ndarray, h: int) -> np.ndarray:
    trials, k = spacings.shape
    p_values = np.empty((trials, h), dtype=float)
    suffix = np.cumsum(spacings[:, ::-1], axis=1)[:, ::-1]
    for j in range(h):
        m = k - j - 1
        deeper_mean = suffix[:, j + 1] / m
        ratio = spacings[:, j] / deeper_mean
        p_values[:, j] = (m / (m + ratio)) ** m
    return p_values


def _adaptive_selection(
    spacings: np.ndarray,
    n: int,
    h: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    trials, k = spacings.shape
    p_values = _spacing_p_values(spacings, h)
    rejected = p_values < (alpha / h)
    trims = np.zeros(trials, dtype=int)
    any_rejected = np.any(rejected, axis=1)
    if np.any(any_rejected):
        trims[any_rejected] = h - np.argmax(rejected[any_rejected, ::-1], axis=1)

    detect_to = max(h, k // 2)
    failures = np.zeros(trials, dtype=bool)
    deep_tests = detect_to - h
    if deep_tests > 0:
        deep_p_values = _spacing_p_values(spacings, detect_to)[:, h:detect_to]
        failures = np.any(
            deep_p_values < (_interlock_level(n) / deep_tests),
            axis=1,
        )
    return trims, failures


def _trimmed_from_spacings(spacings: np.ndarray, trims: np.ndarray) -> np.ndarray:
    trials, k = spacings.shape
    suffix = np.cumsum(spacings[:, ::-1], axis=1)[:, ::-1]
    return suffix[np.arange(trials), trims] / (k - trims)


def _harmonic_from_spacings(spacings: np.ndarray, beta: float) -> np.ndarray:
    _, k = spacings.shape
    weights = np.arange(1, k + 1, dtype=float)
    log_excess = np.cumsum((spacings / weights)[:, ::-1], axis=1)[:, ::-1]
    h_beta = np.mean(np.exp(-beta * log_excess), axis=1)
    return (1.0 - h_beta) / (beta * h_beta)


def _summarize_estimator(
    estimates: np.ndarray,
    truth: float,
    hill_dirty_loss: np.ndarray,
) -> dict[str, float | int]:
    usable = np.isfinite(estimates)
    usable_estimates = estimates[usable]
    losses = (usable_estimates - truth) ** 2
    if usable_estimates.size == 0:
        return {
            "usable_trials": 0,
            "failure_rate": 1.0,
            "mean": math.nan,
            "bias": math.nan,
            "variance": math.nan,
            "mse": math.nan,
            "rmse": math.nan,
            "mse_se": math.nan,
            "benefit_vs_contaminated_hill": math.nan,
            "benefit_vs_contaminated_hill_se": math.nan,
        }

    benefit = hill_dirty_loss[usable] - losses
    return {
        "usable_trials": int(usable_estimates.size),
        "failure_rate": float(1.0 - usable_estimates.size / estimates.size),
        "mean": float(np.mean(usable_estimates)),
        "bias": float(np.mean(usable_estimates - truth)),
        "variance": float(np.var(usable_estimates, ddof=1))
        if usable_estimates.size > 1
        else math.nan,
        "mse": float(np.mean(losses)),
        "rmse": float(math.sqrt(float(np.mean(losses)))),
        "mse_se": float(np.std(losses, ddof=1) / math.sqrt(losses.size))
        if losses.size > 1
        else math.nan,
        "benefit_vs_contaminated_hill": float(np.mean(benefit)),
        "benefit_vs_contaminated_hill_se": float(
            np.std(benefit, ddof=1) / math.sqrt(benefit.size)
        )
        if benefit.size > 1
        else math.nan,
    }


def _cell(
    rng: np.random.Generator,
    n: int,
    r: int,
    signal_label: str,
    signal: float,
    trials: int,
    gamma: float,
    h: int,
    betas: tuple[float, ...],
    include_replicates: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    k = int(math.floor(math.sqrt(n)))
    alpha = _alpha(n)
    scales = _scales(n, r, gamma, h)
    effective_r = 0 if signal == 0.0 else r
    delta = 1.0 if effective_r == 0 else math.exp(signal / effective_r)

    clean = rng.exponential(scale=gamma, size=(trials, k))
    contaminated = clean.copy()
    if effective_r > 0:
        contaminated[:, effective_r - 1] += signal

    hill_clean = np.mean(clean, axis=1)
    hill_dirty = np.mean(contaminated, axis=1)
    oracle_trim = (
        hill_dirty
        if effective_r == 0
        else np.mean(contaminated[:, effective_r:], axis=1)
    )
    trims, adaptive_failures = _adaptive_selection(contaminated, n, h, alpha)
    adaptive_trim = _trimmed_from_spacings(contaminated, trims)
    adaptive_trim = np.where(adaptive_failures, np.nan, adaptive_trim)

    estimates: dict[str, np.ndarray] = {
        "hill": hill_dirty,
        "oracle_trimmed_hill": oracle_trim,
        "adaptive_trimmed_hill": adaptive_trim,
    }
    for beta in betas:
        estimates[f"harmonic_moment_beta_{beta:g}"] = _harmonic_from_spacings(
            contaminated, beta
        )

    clean_loss = (hill_clean - gamma) ** 2
    hill_dirty_loss = (hill_dirty - gamma) ** 2
    contamination_cost = hill_dirty_loss - clean_loss

    base = {
        "n": n,
        "k": k,
        "h": h,
        "alpha": alpha,
        "gamma": gamma,
        "nominal_r": r,
        "effective_r": effective_r,
        "signal_label": signal_label,
        "signal": signal,
        "delta": delta,
        "I": scales["I"],
        "D": scales["D"],
        "R": scales["R"],
        "signal_over_I": signal / scales["I"],
        "signal_over_D": signal / scales["D"],
        "signal_over_R": signal / scales["R"],
        "trials": trials,
        "trim_recovery_rate": float(np.mean(trims == effective_r)),
        "detection_rate": float(np.mean(trims > 0)),
        "overtrim_rate": float(np.mean(trims > effective_r)),
        "undertrim_rate": float(np.mean(trims < effective_r)),
        "adaptive_interlock_failure_rate": float(np.mean(adaptive_failures)),
        "contamination_cost": float(np.mean(contamination_cost)),
        "contamination_cost_se": float(
            np.std(contamination_cost, ddof=1) / math.sqrt(trials)
        ),
    }

    summary_rows = []
    for estimator, values in estimates.items():
        row = {**base, "estimator": estimator}
        row.update(_summarize_estimator(values, gamma, hill_dirty_loss))
        summary_rows.append(row)

    replicate_rows = []
    if include_replicates:
        for trial in range(trials):
            row = {
                **base,
                "trial": trial,
                "trim": int(trims[trial]),
                "adaptive_interlock_failure": bool(adaptive_failures[trial]),
                "hill_clean": float(hill_clean[trial]),
                "hill_clean_loss": float(clean_loss[trial]),
            }
            for estimator, values in estimates.items():
                value = float(values[trial]) if np.isfinite(values[trial]) else math.nan
                row[estimator] = value
                row[f"{estimator}_loss"] = (
                    float((value - gamma) ** 2) if math.isfinite(value) else math.nan
                )
            replicate_rows.append(row)
    return summary_rows, replicate_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    rng = np.random.default_rng(args.seed)
    summary_rows: list[dict[str, Any]] = []
    replicate_handle = None
    replicate_writer = None

    if args.replicates_csv is not None:
        args.replicates_csv.parent.mkdir(parents=True, exist_ok=True)
        replicate_handle = args.replicates_csv.open("w", newline="", encoding="utf-8")

    try:
        for n in args.n_values:
            for r in args.r_values:
                for point in _signal_grid(n, r, args.gamma, args.h, args.tolerance):
                    cell_summary, cell_replicates = _cell(
                        rng=rng,
                        n=n,
                        r=r,
                        signal_label=str(point["label"]),
                        signal=float(point["signal"]),
                        trials=args.trials,
                        gamma=args.gamma,
                        h=args.h,
                        betas=args.betas,
                        include_replicates=replicate_handle is not None,
                    )
                    summary_rows.extend(cell_summary)
                    if replicate_handle is not None:
                        if replicate_writer is None:
                            replicate_writer = csv.DictWriter(
                                replicate_handle,
                                fieldnames=list(cell_replicates[0]),
                            )
                            replicate_writer.writeheader()
                        replicate_writer.writerows(cell_replicates)
        if replicate_handle is not None:
            replicate_handle.flush()
    finally:
        if replicate_handle is not None:
            replicate_handle.close()

    _write_csv(args.summary_csv, summary_rows)

    report = {
        "provenance": base_provenance(REPO_ROOT),
        "configuration": {
            "trials": args.trials,
            "seed": args.seed,
            "gamma": args.gamma,
            "n_values": list(args.n_values),
            "r_values": list(args.r_values),
            "h": args.h,
            "betas": list(args.betas),
            "signal_rules": [
                {"label": label, "scale": scale, "factor": factor}
                for label, scale, factor in SIGNAL_RULES
            ],
            "duplicate_signal_tolerance": args.tolerance,
            "paired_replicates": True,
        },
        "outputs": {
            "summary_csv": str(args.summary_csv),
            "replicates_csv": str(args.replicates_csv)
            if args.replicates_csv is not None
            else None,
        },
        "rows": len(summary_rows),
        "seconds": time.time() - started,
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--n-values", type=_parse_ints, default=DEFAULT_N_VALUES)
    parser.add_argument("--r-values", type=_parse_ints, default=DEFAULT_R_VALUES)
    parser.add_argument("--h", type=int, default=10)
    parser.add_argument("--betas", type=_parse_floats, default=DEFAULT_BETAS)
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("research/sparse_contamination/primary_summary.csv"),
    )
    parser.add_argument("--replicates-csv", type=Path)
    parser.add_argument(
        "--json",
        type=Path,
        default=Path("research/sparse_contamination/primary_report.json"),
    )
    args = parser.parse_args()

    if args.trials <= 1:
        raise ValueError("--trials must exceed 1")
    if args.gamma <= 0.0:
        raise ValueError("--gamma must be positive")
    if args.h <= 0:
        raise ValueError("--h must be positive")
    if any(beta <= 0.0 for beta in args.betas):
        raise ValueError("--betas must all be positive")

    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
