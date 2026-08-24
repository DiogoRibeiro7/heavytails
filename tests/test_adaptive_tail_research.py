"""Research-only oracle experiment guards."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import pytest
from research.adaptive_tail import clean_pareto_decomposition as decomposition
from research.adaptive_tail import oracle_experiment as experiment
from research.adaptive_tail import (
    selector_closure,
    selector_diagnostics,
    selector_power,
    selector_scale,
)
from research.adaptive_tail.oracle_experiment import SCENARIOS
from scripts._provenance import _git_commit

from heavytails import Pareto
from heavytails.tail_index import (
    threshold_averaged_orthogonalized_hill_estimator,
)


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
    assert row["crossfit_min_k"] == 5
    assert row["admissible_max_trim"] == 3
    assert row["adaptive_max_trim"] == 3
    assert row["r_grid"] == [0, 1, 2, 3]
    assert not row["contamination_supported"]
    assert all(pair is None or pair[0] <= 3 for pair in row["oracle_pairs"])


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
        "cross_fitted_adaptive_randomized",
    }
    assert row["methods"]["best_local_oracle_oos"]["oracle_pairs"]
    assert row["methods"]["best_local_oracle_oos"]["ratio_to_best_local_oos"] == 1.0
    assert row["methods"]["best_local_oracle_in_sample"]["selected_pair"]
    assert "ratio_to_best_local_oos" in row["methods"]["cross_fitted_adaptive"]
    assert (
        "ratio_to_best_local_oos" in row["methods"]["cross_fitted_adaptive_randomized"]
    )
    assert report["configuration"]["production_crossfit_split_seed"].startswith("None")
    assert report["configuration"]["split_seed_offset"] == 1_000_000_000
    assert row["full_sample_selected_local_pair_counts"]
    assert row["selected_trim_frequency"]
    assert row["stable_set_size_mean"] is not None
    assert "stable_trim_frequency_within_stable_thresholds" in row

    json.dumps(report, allow_nan=False)


def test_selector_diagnostics_trace_and_calibration_are_jsonable() -> None:
    report = selector_diagnostics.build_report(
        n=300,
        k_grid_mode="intermediate",
        k_fractions=[0.05, 0.10],
        intermediate_grid_size=4,
        intermediate_min_power=1.0 / 3.0,
        intermediate_max_power=2.0 / 3.0,
        max_trim=8,
        rho=-1.0,
        target_acceptance=0.5,
        calibration_trials=2,
        holdout_trials=2,
        calibration_seed_start=100,
        holdout_seed_start=200,
        critical_grid=[1.0, 2.0],
        trace_count=1,
    )

    assert report["configuration"]["crossfit_min_k"] == 4
    assert report["configuration"]["admissible_max_trim"] == 2
    assert report["selected_critical"]["critical"] in {1.0, 2.0}
    assert report["holdout"]["trials"] == 2
    assert report["traces"]
    assert set(report["traces"][0]) == {
        "data_seed",
        "default_critical",
        "calibrated_critical",
    }

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


def test_the_trace_reproduces_the_production_estimate() -> None:
    """The instrument must agree with what it is instrumenting.

    ``_trace_crossfit`` walks the cross-fit path itself so it can record the
    thresholds, trims, stable sets and weights the production estimator throws
    away. That is a second implementation of the same procedure, which is the
    arrangement this repository has been bitten by three times, so it is held
    to the original rather than trusted to stay in step.
    """
    data = Pareto(alpha=2.0, xm=1.0).rvs(600, seed=4)
    kwargs = {
        "k": 60,
        "min_k": 10,
        "grid_size": 6,
        "rho": -1.0,
        "max_trim": 3,
        "critical": 2.0,
    }

    traced = selector_diagnostics._trace_crossfit(data, seed=None, **kwargs)
    produced = threshold_averaged_orthogonalized_hill_estimator(
        data,
        kwargs["k"],
        min_k=kwargs["min_k"],
        grid_size=kwargs["grid_size"],
        rho=kwargs["rho"],
        adaptive_trim=True,
        max_trim=kwargs["max_trim"],
        critical=kwargs["critical"],
        crossfit=True,
        seed=None,
    )

    assert traced["gamma"] is not None
    assert traced["gamma"] == pytest.approx(produced, rel=1e-12, abs=0.0)


def test_the_trace_records_a_failure_where_production_raises() -> None:
    """When the estimator cannot produce a value, the trace says so.

    It must not quietly return a different number, which is the failure mode
    that would make the diagnostic worse than useless: a trace that disagrees
    with production exactly where production is in trouble.
    """
    data = Pareto(alpha=2.0, xm=1.0).rvs(40, seed=9)
    kwargs = {
        "k": 30,
        "min_k": 25,
        "grid_size": 4,
        "rho": -1.0,
        "max_trim": 12,
        "critical": 0.05,
    }

    traced = selector_diagnostics._trace_crossfit(data, seed=None, **kwargs)
    try:
        produced = threshold_averaged_orthogonalized_hill_estimator(
            data,
            kwargs["k"],
            min_k=kwargs["min_k"],
            grid_size=kwargs["grid_size"],
            rho=kwargs["rho"],
            adaptive_trim=True,
            max_trim=kwargs["max_trim"],
            critical=kwargs["critical"],
            crossfit=True,
            seed=None,
        )
    except ValueError:
        assert traced["gamma"] is None
        assert traced["failure_rate"] > 0.0
        assert any(fold["stage"] != "success" for fold in traced["folds"])
    else:
        # Production managed it, so the trace must have too, and must agree.
        assert traced["gamma"] == pytest.approx(produced, rel=1e-12, abs=0.0)


def test_calibration_counts_every_trial_in_the_denominator() -> None:
    """The target is joint over both folds succeeding and both accepting.

    Reported conditionally -- hits over successes -- a cutoff that fails a
    tenth of the time and accepts on the rest reads as 100%. It is usable nine
    times in ten, and this study treats a failure as invalidating the estimate
    everywhere else.
    """
    row = selector_diagnostics._selection_rate(
        n=400,
        k_grid=[10, 20, 40],
        max_trim=3,
        rho=-1.0,
        critical=2.0,
        trials=8,
        seed_start=31_000,
    )

    assert 0.0 <= row["joint_acceptance_rate"] <= 1.0
    # The joint event cannot be commoner than both folds merely succeeding.
    assert row["joint_acceptance_rate"] <= row["both_folds_succeeded_rate"]
    # And the conditional diagnostic can only be at least as flattering.
    conditional = row["fold_acceptance_rate_given_success"]
    if conditional is not None and row["fold_failure_rate"] > 0.0:
        assert conditional >= row["joint_acceptance_rate"]


class TestProvenanceInALinkedWorktree:
    """A linked worktree keeps its own HEAD and shares refs with its origin.

    Two result files were written with ``"git_commit": null`` from an ordinary
    checkout because of this: the helper resolved the worktree's git directory,
    found HEAD naming ``refs/heads/...``, looked for that ref beside it, and
    gave up. The ref lives in the repository named by ``commondir``.
    """

    def _worktree(self, tmp_path, ref: str, sha: str, *, packed: bool):
        common = tmp_path / "main" / ".git"
        (common / "refs" / "heads").mkdir(parents=True)
        linked = common / "worktrees" / "wt"
        linked.mkdir(parents=True)
        (linked / "HEAD").write_text(f"ref: {ref}\n", encoding="utf-8")
        # Relative, exactly as git writes it.
        (linked / "commondir").write_text("../..\n", encoding="utf-8")
        if packed:
            (common / "packed-refs").write_text(
                f"# pack-refs with: peeled\n{sha} {ref}\n", encoding="utf-8"
            )
        else:
            (common / ref).write_text(f"{sha}\n", encoding="utf-8")
        root = tmp_path / "checkout"
        root.mkdir()
        (root / ".git").write_text(f"gitdir: {linked}\n", encoding="utf-8")
        return root

    def test_the_loose_ref_is_found_in_the_common_directory(self, tmp_path) -> None:
        sha = "a" * 40
        root = self._worktree(tmp_path, "refs/heads/topic", sha, packed=False)
        assert _git_commit(root) == sha

    def test_the_packed_ref_is_found_in_the_common_directory(self, tmp_path) -> None:
        sha = "b" * 40
        root = self._worktree(tmp_path, "refs/heads/topic", sha, packed=True)
        assert _git_commit(root) == sha

    def test_a_detached_head_still_reports_its_commit(self, tmp_path) -> None:
        root = tmp_path / "checkout"
        root.mkdir()
        git_dir = tmp_path / "gitdir"
        git_dir.mkdir()
        sha = "c" * 40
        (git_dir / "HEAD").write_text(f"{sha}\n", encoding="utf-8")
        (root / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
        assert _git_commit(root) == sha

    def test_this_very_checkout_reports_a_commit(self) -> None:
        """Whether or not the suite is being run from a linked worktree."""
        commit = _git_commit(Path(__file__).resolve().parents[1])
        assert commit is not None, "provenance would be written as null"
        assert len(commit) == 40
        assert all(c in "0123456789abcdef" for c in commit)


def test_selector_power_report_is_jsonable_and_separates_null_from_alternative() -> (
    None
):
    """The power study must record the diagnostic the calibration cannot.

    Null acceptance alone says nothing about whether the rule discriminates,
    because under exact Pareto the best threshold is the largest one and a
    correctly sized test should accept the whole grid. What distinguishes a
    working rule from a disabled one is the stable-set fraction under an
    alternative, so the report has to carry it.
    """
    report = selector_power.build_report(
        n=400,
        critical_grid=[4.0],
        scenarios=["pareto", "burr_rho_quarter"],
        contaminations=[0],
        deltas=[2.0],
        trials=4,
        seed_start=81_000,
        max_trim=3,
    )
    json.dumps(report, allow_nan=False)

    assert report["provenance"]["git_commit"] is not None
    assert len(report["cells"]) == 2
    for cell in report["cells"]:
        assert 0.0 <= cell["joint_full_acceptance_rate"] <= 1.0
        fraction = cell["stable_fraction"]
        assert fraction is not None
        # A fraction of the fold's own top threshold, so bounded by one.
        assert 0.0 <= fraction["p10"] <= fraction["median"] <= 1.0
        assert cell["rmse"] is None or cell["rmse"] >= 0.0


def test_contamination_is_planted_before_the_split() -> None:
    """On the largest observations, so the folds share it dependently.

    That is why the clean-data identity joint = per-fold squared must not be
    carried over to contaminated samples.
    """
    sample = [1.0, 5.0, 2.0, 4.0, 3.0]
    contaminated = selector_power._contaminate(sample, 2, 10.0)
    assert sorted(contaminated, reverse=True)[:2] == [50.0, 40.0]
    assert sorted(contaminated)[:3] == [1.0, 2.0, 3.0]
    assert selector_power._contaminate(sample, 0, 10.0) == sample


def test_the_paired_difference_is_paired_by_seed() -> None:
    """A cutoff comparison run on the same seeds is a paired comparison.

    Judging it against the standard error of either RMSE ignores that the two
    estimates are highly correlated, and overstates the uncertainty of their
    difference by a large factor. Replications where either side failed are
    dropped from both, so the difference is over a common set.
    """
    treatment = [
        {"seed": 1, "squared_error": 0.10},
        {"seed": 2, "squared_error": 0.20},
        {"seed": 3, "squared_error": None},
        {"seed": 4, "squared_error": 0.40},
    ]
    reference = [
        {"seed": 1, "squared_error": 0.05},
        {"seed": 2, "squared_error": 0.30},
        {"seed": 3, "squared_error": 0.10},
        {"seed": 5, "squared_error": 0.90},
    ]

    result = selector_closure._paired_difference(
        treatment, reference, draws=200, seed=3
    )
    assert result is not None
    # Seed 3 dropped (treatment failed), seed 4 and 5 unmatched.
    assert result["paired_replications"] == 2
    assert result["mean_mse_difference"] == pytest.approx(
        ((0.10 - 0.05) + (0.20 - 0.30)) / 2
    )
    assert result["bootstrap_lower"] <= result["mean_mse_difference"]
    assert result["mean_mse_difference"] <= result["bootstrap_upper"]


def test_a_cutoff_compared_with_itself_shows_no_difference() -> None:
    """The reference against itself must be exactly zero, interval included."""
    rows = [{"seed": s, "squared_error": 0.1 * s} for s in range(1, 12)]
    result = selector_closure._paired_difference(rows, rows, draws=200, seed=5)
    assert result is not None
    assert result["mean_mse_difference"] == 0.0
    assert result["bootstrap_lower"] == 0.0
    assert result["bootstrap_upper"] == 0.0
    assert result["interval_contains_zero"]
    assert not result["favours_selection"]


def test_the_rho_used_for_the_weights_can_be_overridden() -> None:
    """Null size has to be matchable on the tuning, not just the law.

    Each scenario carries its own ``rho_used``, so comparing Pareto against
    Burr moves the law and the orthogonalized weights together. Separating
    them is what makes the power claim about the law.
    """
    default = selector_power._cell(
        scenario_key="pareto",
        n=400,
        k_grid=[10, 20, 40],
        max_trim=3,
        critical=4.0,
        contamination=0,
        delta=0.0,
        trials=3,
        seed_start=82_000,
    )
    overridden = selector_power._cell(
        scenario_key="pareto",
        n=400,
        k_grid=[10, 20, 40],
        max_trim=3,
        critical=4.0,
        contamination=0,
        delta=0.0,
        trials=3,
        seed_start=82_000,
        rho_used=-0.25,
    )
    assert default["rho_used"] == -1.0
    assert default["rho_used_is_scenario_default"]
    assert overridden["rho_used"] == -0.25
    assert not overridden["rho_used_is_scenario_default"]


class TestPerRhoCalibration:
    """The cutoff is chosen per rho, and which qualifying one matters.

    The null distribution of the compatibility statistic moves with the rho the
    orthogonalized weights are built from, so a single cutoff shared across
    scenarios would compare sizes as well as laws -- which is the confound
    #389 had to separate out after the fact.
    """

    def _curve(self, rates: dict[float, float]) -> list[dict[str, float]]:
        return [
            {"critical": c, "joint_acceptance_rate": rate} for c, rate in rates.items()
        ]

    def test_the_smallest_qualifying_cutoff_is_chosen(self, monkeypatch) -> None:
        """Among correctly sized rules the tightest keeps the most power.

        Taking a looser one would bias the experiment towards finding no
        effect, by making the selector do less.
        """
        rates = {3.0: 0.80, 4.0: 0.93, 5.0: 0.96, 6.0: 0.99}
        calls = iter(self._curve(rates))
        monkeypatch.setattr(
            selector_scale, "_selection_rate", lambda **kwargs: dict(next(calls))
        )

        result = selector_scale._calibrate(
            n=1000,
            k_grid=[10, 20],
            max_trim=3,
            rho=-1.0,
            critical_grid=[3.0, 4.0, 5.0, 6.0],
            target=0.95,
            trials=1,
            seed_start=0,
        )
        assert result["target_met"]
        assert result["calibrated_critical"] == 5.0
        assert result["calibrated_acceptance"] == 0.96

    def test_when_nothing_qualifies_the_best_is_reported_and_flagged(
        self, monkeypatch
    ) -> None:
        rates = {3.0: 0.60, 4.0: 0.88, 5.0: 0.91, 6.0: 0.90}
        calls = iter(self._curve(rates))
        monkeypatch.setattr(
            selector_scale, "_selection_rate", lambda **kwargs: dict(next(calls))
        )

        result = selector_scale._calibrate(
            n=1000,
            k_grid=[10, 20],
            max_trim=3,
            rho=-1.0,
            critical_grid=[3.0, 4.0, 5.0, 6.0],
            target=0.95,
            trials=1,
            seed_start=0,
        )
        assert not result["target_met"]
        # The best available, not the largest searched.
        assert result["calibrated_critical"] == 5.0
        assert result["calibrated_acceptance"] == 0.91

    def test_each_rho_is_calibrated_separately(self) -> None:
        """Scenarios sharing a rho share a cutoff; differing rhos do not."""
        rhos = {
            key: SCENARIOS[key].rho_used
            for key in ("pareto", "hall_rho_half", "burr_rho_half", "burr_rho_quarter")
        }
        assert rhos["hall_rho_half"] == rhos["burr_rho_half"]
        assert rhos["pareto"] != rhos["burr_rho_quarter"]
        assert len(set(rhos.values())) == 3
