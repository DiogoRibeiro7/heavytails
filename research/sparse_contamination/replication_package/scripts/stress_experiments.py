"""Post-specified stress experiments for the sparse-contamination manuscript.

These runs are explicitly *not* part of the frozen primary design.  They were
specified after the primary results were seen, in response to three questions
about how far the three-scale picture depends on the choices that produced it:

1. Detection transition and level regime.  The primary grid tests only
   ``0.5D`` and ``1.5D``, and the asymptotic separation
   ``I^T << D_n`` relies on a vanishing family-wise level
   ``alpha_n = min(0.05, 1/log^2 n)``.  Study 1 maps detection on a fine grid in
   ``S/D_n`` under both that regime and a fixed ``alpha = 0.05``.

2. Unequal contamination factors.  Under equal factors only the boundary
   spacing moves, so "r contaminated extremes" collapses to a single-spike
   detection problem.  Study 2 holds the Hill bias fixed and spreads the same
   total signal across the top ``r`` spacings, which is the comparison Lemma 2
   implies but the primary design does not run.

3. Second-order tails.  The primary run samples exact Pareto, where the
   normalized spacings are exactly iid exponential.  Study 3 samples a Burr
   law, whose spacings are only approximately exponential, and asks whether the
   normalized scales still organize the outcome once ordinary Hill carries
   threshold bias.

The primary artifacts are never read or written here.

Usage::

    python stress_experiments.py
    python stress_experiments.py --trials 1000    # quick pass
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from sparse_experiment import (  # frozen driver: reuse its exact machinery
    _adaptive_selection,
    _harmonic_from_spacings,
    _trimmed_from_spacings,
)

HERE = Path(__file__).resolve().parent
GAMMA = 0.5
H_SCAN = 10
BETAS = (0.5, 1.0, 2.0)
ESTIMATORS = ("hill", "oracle_trimmed_hill", "adaptive_trimmed_hill")


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def alpha_vanishing(n: int) -> float:
    """The level the frozen design uses."""
    return min(0.05, 1.0 / math.log(n) ** 2)


def alpha_fixed(_n: int) -> float:
    """A conventional fixed family-wise level, for the regime comparison."""
    return 0.05


def scales(n: int, r: int, alpha: float) -> dict[str, float]:
    """The three scales at a given level; only D depends on alpha."""
    k = math.isqrt(n)
    return {
        "I": GAMMA * math.sqrt(r * k / (k - r)),
        "D": GAMMA * math.log(H_SCAN / alpha),
        "R": GAMMA * math.sqrt(k),
    }


def pareto_spacings(rng: np.random.Generator, trials: int, k: int) -> np.ndarray:
    """Exact-Pareto normalized spacings: iid Exp(gamma), as in the frozen run."""
    return rng.exponential(GAMMA, size=(trials, k))


def order_stat_spacings(
    rng: np.random.Generator,
    trials: int,
    n: int,
    k: int,
    qbar,
) -> np.ndarray:
    """Normalized log-spacings from the top k+1 order statistics of any tail.

    ``qbar(p)`` is the survival quantile, so ``Fbar(qbar(p)) = p``.  With
    ``Gamma_i`` the partial sums of iid Exp(1) and ``T ~ Gamma(n+1)``, the i-th
    largest observation is ``qbar(Gamma_i / T)``; only ``k+1`` of them are
    needed, so the cost does not grow with ``n``.
    """
    gaps = rng.exponential(1.0, size=(trials, k + 1))
    partial = np.cumsum(gaps, axis=1)
    rest = rng.gamma(shape=n - k, scale=1.0, size=(trials, 1))
    p = partial / (partial[:, -1:] + rest)
    log_x = np.log(qbar(p))
    j = np.arange(1, k + 1, dtype=float)
    return j * (log_x[:, :-1] - log_x[:, 1:])


def burr_qbar(tau: float):
    """Survival quantile of the Burr law with tail index GAMMA.

    ``Fbar(x) = (1 + x^tau)^(-lam)`` with ``lam = 1/(tau*GAMMA)`` has tail index
    ``GAMMA`` and relative second-order term of order ``x^(-tau)``, which
    corresponds to second-order parameter ``rho = -tau*GAMMA``.  Larger ``tau``
    is therefore closer to exact Pareto.
    """
    lam = 1.0 / (tau * GAMMA)
    return lambda p: (p ** (-1.0 / lam) - 1.0) ** (1.0 / tau)


def pareto_qbar(p):
    """Survival quantile of the exact Pareto law, for validating the generator."""
    return p**-GAMMA


def shift_equal(k: int, r: int, signal: float) -> np.ndarray:
    """Equal-factor contamination: the whole signal lands on spacing r."""
    shift = np.zeros(k)
    shift[r - 1] = signal
    return shift


def shift_spread(k: int, r: int, signal: float) -> np.ndarray:
    """Unequal factors with the same Hill bias, spread over the top r spacings.

    Taking ``log Delta_j = c (r - j + 1)`` gives spacing shifts ``j c`` for
    every ``j <= r`` by Lemma 2, and the total shift, hence the Hill bias, is
    the same ``signal`` as the equal-factor case.
    """
    shift = np.zeros(k)
    c = 2.0 * signal / (r * (r + 1))
    shift[:r] = c * np.arange(1, r + 1, dtype=float)
    return shift


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    clean: np.ndarray,
    shift: np.ndarray,
    *,
    n: int,
    r_true: int,
    alpha: float,
) -> dict[str, Any]:
    """Score one cell, mirroring the frozen run's estimator set and contrasts."""
    dirty = clean + shift
    trims, _ = _adaptive_selection(dirty, n, H_SCAN, alpha)
    k = clean.shape[1]

    values = {
        "hill": dirty.mean(axis=1),
        "oracle_trimmed_hill": _trimmed_from_spacings(
            dirty, np.full(clean.shape[0], r_true)
        ),
        "adaptive_trimmed_hill": _trimmed_from_spacings(dirty, trims),
    }
    for beta in BETAS:
        values[f"harmonic_moment_beta_{beta:g}"] = _harmonic_from_spacings(dirty, beta)

    hill_loss = (values["hill"] - GAMMA) ** 2
    clean_loss = (clean.mean(axis=1) - GAMMA) ** 2
    row: dict[str, Any] = {
        "n": n,
        "k": k,
        "alpha": alpha,
        "r": r_true,
        "trials": clean.shape[0],
        "detection_rate": float(np.mean(trims > 0)),
        "trim_recovery_rate": float(np.mean(trims == r_true)),
        "contamination_cost": float(np.mean(hill_loss - clean_loss)),
    }
    for name, est in values.items():
        loss = (est - GAMMA) ** 2
        benefit = hill_loss - loss
        row[f"{name}_mse"] = float(np.mean(loss))
        row[f"{name}_benefit"] = float(np.mean(benefit))
        row[f"{name}_benefit_se"] = float(
            np.std(benefit, ddof=1) / math.sqrt(benefit.size)
        )
    return row


# --------------------------------------------------------------------------- #
# studies
# --------------------------------------------------------------------------- #
def study_detection_transition(rng, trials: int) -> list[dict[str, Any]]:
    """Fine grid in S/D_n, under a vanishing and a fixed family-wise level."""
    rows = []
    ratios = [0.4, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0]
    for n in (10_000, 50_000):
        k = math.isqrt(n)
        for regime, alpha_fn in (
            ("vanishing", alpha_vanishing),
            ("fixed", alpha_fixed),
        ):
            alpha = alpha_fn(n)
            for r in (1, 5):
                sc = scales(n, r, alpha)
                for ratio in ratios:
                    signal = ratio * sc["D"]
                    clean = pareto_spacings(rng, trials, k)
                    row = evaluate(
                        clean, shift_equal(k, r, signal), n=n, r_true=r, alpha=alpha
                    )
                    row.update(
                        study="detection_transition",
                        level_regime=regime,
                        signal=signal,
                        signal_over_D=signal / sc["D"],
                        signal_over_I=signal / sc["I"],
                        signal_over_R=signal / sc["R"],
                    )
                    rows.append(row)
    return rows


def study_unequal_factors(rng, trials: int) -> list[dict[str, Any]]:
    """Equal against spread contamination at matched Hill bias."""
    rows = []
    for n in (10_000, 50_000):
        k = math.isqrt(n)
        alpha = alpha_vanishing(n)
        for r in (3, 5, 10):
            sc = scales(n, r, alpha)
            for ratio in (0.5, 1.0, 1.5, 2.5):
                signal = ratio * sc["D"]
                for profile, builder in (
                    ("equal", shift_equal),
                    ("spread", shift_spread),
                ):
                    clean = pareto_spacings(rng, trials, k)
                    row = evaluate(
                        clean, builder(k, r, signal), n=n, r_true=r, alpha=alpha
                    )
                    row.update(
                        study="unequal_factors",
                        profile=profile,
                        signal=signal,
                        signal_over_D=signal / sc["D"],
                        signal_over_I=signal / sc["I"],
                        signal_over_R=signal / sc["R"],
                    )
                    rows.append(row)
    return rows


def study_second_order(rng, trials: int) -> list[dict[str, Any]]:
    """Burr tails, where the spacings are only approximately exponential."""
    rows = []
    for n in (10_000, 50_000):
        k = math.isqrt(n)
        alpha = alpha_vanishing(n)
        for tau in (2.0, 1.0, 0.5):  # rho = -tau*GAMMA: -1 (mild) to -0.25 (severe)
            for r in (1, 5):
                sc = scales(n, r, alpha)
                for ratio in (0.0, 0.5, 1.5):
                    signal = ratio * sc["D"]
                    clean = order_stat_spacings(rng, trials, n, k, burr_qbar(tau))
                    row = evaluate(
                        clean, shift_equal(k, r, signal), n=n, r_true=r, alpha=alpha
                    )
                    row.update(
                        study="second_order",
                        tau=tau,
                        signal=signal,
                        signal_over_D=signal / sc["D"],
                        signal_over_I=signal / sc["I"],
                        signal_over_R=signal / sc["R"],
                    )
                    rows.append(row)
    return rows


def self_check(rng, trials: int) -> dict[str, float]:
    """Validate the order-statistic generator and locate the Burr ladder.

    Feeding it the exact-Pareto survival quantile must return spacings that are
    iid Exp(gamma); this checks the construction itself.  The Burr rows then
    show how far each tau sits from that reference, since larger tau means a
    faster-decaying second-order term.
    """
    n = 10_000
    k = math.isqrt(n)
    out: dict[str, float] = {"target_mean": GAMMA, "target_var": GAMMA**2}
    z = order_stat_spacings(rng, trials, n, k, pareto_qbar)
    out["pareto_mean"] = float(z.mean())
    out["pareto_var"] = float(z.var())
    for tau in (8.0, 4.0, 2.0, 1.0):
        z = order_stat_spacings(rng, trials, n, k, burr_qbar(tau))
        out[f"burr_tau{tau:g}_mean"] = float(z.mean())
        out[f"burr_tau{tau:g}_hill_bias"] = float(z.mean() - GAMMA)
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    """Run the three stress studies and write their artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--outdir", type=Path, default=HERE)
    args = parser.parse_args()

    started = time.time()
    rng = np.random.default_rng(args.seed)
    checks = self_check(rng, args.trials)
    rows: list[dict[str, Any]] = []
    for study in (
        study_detection_transition,
        study_unequal_factors,
        study_second_order,
    ):
        rows.extend(study(rng, args.trials))

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = args.outdir / "stress_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "configuration": {
            "trials": args.trials,
            "seed": args.seed,
            "gamma": GAMMA,
            "h": H_SCAN,
            "betas": list(BETAS),
            "post_specified": True,
        },
        "self_check": checks,
        "rows": len(rows),
        "seconds": time.time() - started,
    }
    (args.outdir / "stress_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"wrote {path} ({len(rows)} rows)")
    print(
        "self-check (Pareto path must match Exp(gamma); Burr bias grows as tau falls):"
    )
    for key, value in checks.items():
        print(f"    {key:26s} {value:+.4f}")
    print(f"elapsed {report['seconds']:.0f}s")


if __name__ == "__main__":
    main()
