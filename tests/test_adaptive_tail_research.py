"""Research-only oracle experiment guards."""

from __future__ import annotations

import json

from research.adaptive_tail import oracle_experiment as experiment


def test_oracle_grid_matches_the_adaptive_log_grid() -> None:
    assert experiment._thresholds_from_fractions(1000, [0.02, 0.05, 0.10]) == [
        20,
        45,
        100,
    ]


def test_oracle_squared_errors_are_reordered_by_replication_index() -> None:
    folds = [
        experiment.FoldEvaluation(
            selected_pair=(0, 30),
            selection_mse=1.0,
            evaluation_indices=[2, 3],
            evaluation_squared_errors=[(2, 20.0), (3, 30.0)],
        ),
        experiment.FoldEvaluation(
            selected_pair=(1, 30),
            selection_mse=1.0,
            evaluation_indices=[0, 1],
            evaluation_squared_errors=[(0, 0.0), (1, 10.0)],
        ),
    ]

    assert experiment._oracle_squared_by_index(folds, trials=4) == [
        0.0,
        10.0,
        20.0,
        30.0,
    ]


def test_report_contains_provenance_configuration_and_jsonable_results() -> None:
    report = experiment.build_report(
        trials=2,
        sample_sizes=[300],
        scenario_keys=["pareto"],
        contamination_counts=[0],
        deltas=[2.0],
        k_fractions=[0.05, 0.10],
        max_trim=4,
        bootstrap_draws=2,
    )

    assert set(report) == {"provenance", "configuration", "results"}
    assert report["configuration"]["oracle"] == (
        "two-fold Monte Carlo select/evaluate rotation"
    )
    assert report["provenance"]["python_version"]

    row = report["results"][0]
    assert row["rho_true"] is None
    assert row["rho_used"] == -1.0
    assert row["k_grid"] == [15, 30]
    assert row["oracle_pairs"]
    assert "trim_recovery_vanishing" in row
    assert "trim_recovery_fixed_005" in row

    json.dumps(report, allow_nan=False)


def test_adaptive_failure_invalidates_the_primary_risk_ratio(
    monkeypatch,
) -> None:
    calls = 0

    def sometimes_fails(*args, **kwargs) -> float:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic failure")
        return 0.5

    monkeypatch.setattr(
        experiment,
        "threshold_averaged_orthogonalized_hill_estimator",
        sometimes_fails,
    )

    row = experiment._evaluate_cell(
        experiment.SCENARIOS["pareto"],
        n=300,
        contamination_count=0,
        delta=2.0,
        trials=2,
        k_fractions=[0.05, 0.10],
        max_trim=4,
        bootstrap_draws=0,
    )

    assert row["adaptive_failure_rate"] == 0.5
    assert row["adaptive_rmse_success"] == 0.0
    assert row["adaptive_rmse"] is None
    assert row["risk_ratio"] is None
