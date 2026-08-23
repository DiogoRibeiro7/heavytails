"""Research-only oracle experiment guards."""

from __future__ import annotations

import json
import statistics

from research.adaptive_tail import clean_pareto_decomposition as decomposition
from research.adaptive_tail import oracle_experiment as experiment


def test_oracle_grid_matches_the_adaptive_log_grid() -> None:
    assert experiment._thresholds_from_fractions(1000, [0.02, 0.05, 0.10]) == [
        20,
        45,
        100,
    ]


def test_intermediate_grid_uses_vanishing_threshold_fractions() -> None:
    small = experiment._thresholds_from_intermediate_powers(
        1000, grid_size=4, min_power=1.0 / 3.0, max_power=2.0 / 3.0
    )
    large = experiment._thresholds_from_intermediate_powers(
        1_000_000, grid_size=4, min_power=1.0 / 3.0, max_power=2.0 / 3.0
    )

    assert small[0] == 10
    assert small[-1] == 100
    assert small == sorted(small)
    assert large[-1] / 1_000_000 < small[-1] / 1000


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


def test_bootstrap_summary_reports_statistic_standard_deviation() -> None:
    ratios = [1.0, 2.0, 4.0, 8.0]

    summary = experiment._bootstrap_summary(ratios)

    assert summary["se"] == statistics.stdev(ratios)
    assert summary["se"] != experiment._standard_error(ratios)


def test_clean_cells_do_not_repeat_the_delta_sweep() -> None:
    report = experiment.build_report(
        trials=2,
        sample_sizes=[300],
        scenario_keys=["pareto"],
        contamination_counts=[0, 2],
        deltas=[1.5, 2.0],
        k_fractions=[0.05, 0.10],
        max_trim=4,
        bootstrap_draws=0,
    )

    assert [
        (row["contamination_count"], row["delta"]) for row in report["results"]
    ] == [
        (0, None),
        (2, 1.5),
        (2, 2.0),
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
    assert report["provenance"]["heavytails_version"]
    assert report["provenance"]["heavytails_version_source"] in {
        "pyproject.toml",
        "installed distribution metadata",
    }
    assert report["provenance"]["version_source"] in {
        "pyproject.toml",
        "installed distribution metadata",
    }
    assert report["provenance"]["numpy_version"]

    row = report["results"][0]
    assert row["rho_true"] is None
    assert row["rho_used"] == -1.0
    assert row["k_grid"] == [15, 30]
    assert row["oracle_pairs"]
    assert "trim_recovery_vanishing" in row
    assert "trim_recovery_fixed_005" in row

    json.dumps(report, allow_nan=False)


def test_report_can_use_the_intermediate_grid() -> None:
    report = experiment.build_report(
        trials=2,
        sample_sizes=[1000],
        scenario_keys=["pareto"],
        contamination_counts=[0],
        deltas=[2.0],
        k_fractions=[0.05, 0.10],
        k_grid_mode="intermediate",
        intermediate_grid_size=4,
        intermediate_min_power=1.0 / 3.0,
        intermediate_max_power=2.0 / 3.0,
        max_trim=4,
        bootstrap_draws=0,
    )

    row = report["results"][0]
    assert report["configuration"]["k_grid_mode"] == "intermediate"
    assert row["k_grid_mode"] == "intermediate"
    assert row["k_grid"] == experiment._thresholds_from_intermediate_powers(
        1000,
        grid_size=4,
        min_power=1.0 / 3.0,
        max_power=2.0 / 3.0,
    )


def test_intermediate_oracle_and_adaptive_share_the_trim_envelope() -> None:
    row = experiment._evaluate_cell(
        experiment.SCENARIOS["pareto"],
        n=1000,
        contamination_count=5,
        delta=2.0,
        trials=2,
        k_fractions=[0.05, 0.10],
        k_grid_mode="intermediate",
        intermediate_grid_size=4,
        intermediate_min_power=1.0 / 3.0,
        intermediate_max_power=2.0 / 3.0,
        max_trim=8,
        bootstrap_draws=0,
    )

    assert row["k_grid"][0] == 10
    assert row["admissible_max_trim"] == 4
    assert row["adaptive_max_trim"] == 4
    assert row["r_grid"] == [0, 1, 2, 3, 4]
    assert not row["contamination_supported"]
    assert all(pair is None or pair[0] <= 4 for pair in row["oracle_pairs"])


def test_clean_pareto_decomposition_reports_the_four_layers() -> None:
    report = decomposition.build_report(
        trials=3,
        sample_sizes=[300],
        k_fractions=[0.05, 0.10],
        max_trim=4,
    )

    assert report["configuration"]["target"].startswith("clean-Pareto decomposition")
    row = report["results"][0]
    assert set(row["methods"]) == {
        "best_local_oracle_oos",
        "best_local_oracle_in_sample",
        "full_sample_selected_local",
        "full_sample_adaptive_aggregation",
        "cross_fitted_adaptive",
    }
    assert row["methods"]["best_local_oracle_oos"]["oracle_pairs"]
    assert row["methods"]["best_local_oracle_oos"]["ratio_to_best_local_oos"] == 1.0
    assert row["methods"]["best_local_oracle_in_sample"]["selected_pair"]
    assert "ratio_to_best_local_oos" in row["methods"]["cross_fitted_adaptive"]
    assert report["configuration"]["split_seed_offset"] == 1_000_000_000
    assert row["full_sample_selected_local_pair_counts"]
    assert row["selected_trim_frequency"]
    assert row["stable_set_size_mean"] is not None
    assert "stable_trim_frequency_within_stable_thresholds" in row

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
