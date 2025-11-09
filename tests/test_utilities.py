"""Tests for utilities.py module."""

import json
from pathlib import Path
import random
import tempfile

import pytest

from heavytails import LogNormal, Pareto
from heavytails.utilities import AutoFit, DataIO, ParameterValidator, StatisticalSummary


class TestDataIO:
    """Test data I/O utilities."""

    def test_csv_write_and_read_basic(self):
        """Test basic CSV write and read operations."""
        data = [1.5, 2.3, 3.7, 4.2, 5.8]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"

            # Write
            DataIO.write_csv(data, filepath)
            assert filepath.exists()

            # Read
            read_data = DataIO.read_csv(filepath)
            assert len(read_data) == len(data)
            assert all(
                abs(a - b) < 1e-10 for a, b in zip(read_data, data, strict=False)
            )

    def test_csv_write_with_metadata(self):
        """Test CSV writing with metadata."""
        data = [1.0, 2.0, 3.0]
        metadata = {"distribution": "pareto", "alpha": 2.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_meta.csv"
            DataIO.write_csv(data, filepath, metadata=metadata)

            # Check file exists and contains metadata
            with filepath.open(encoding="utf-8") as f:
                content = f.read()
                assert "# Metadata:" in content
                assert "distribution" in content

    def test_csv_read_with_column_name(self):
        """Test CSV reading with specific column name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_cols.csv"

            # Write CSV with multiple columns manually
            with filepath.open("w", encoding="utf-8") as f:
                f.write("name,value,other\n")
                f.write("a,1.5,x\n")
                f.write("b,2.3,y\n")
                f.write("c,3.7,z\n")

            # Read specific column
            data = DataIO.read_csv(filepath, column="value")
            assert len(data) == 3
            assert abs(data[0] - 1.5) < 1e-10
            assert abs(data[1] - 2.3) < 1e-10
            assert abs(data[2] - 3.7) < 1e-10

    def test_csv_read_auto_detect(self):
        """Test CSV reading with automatic column detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_auto.csv"

            # Write CSV with header
            with filepath.open("w", encoding="utf-8") as f:
                f.write("values\n")
                f.write("1.5\n2.3\n3.7\n")

            data = DataIO.read_csv(filepath)
            assert len(data) == 3
            assert abs(data[0] - 1.5) < 1e-10

    def test_csv_read_nonexistent_file(self):
        """Test that reading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            DataIO.read_csv("nonexistent.csv")

    def test_csv_write_empty_data(self):
        """Test that writing empty data raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.csv"
            with pytest.raises(ValueError, match="Cannot write empty data"):
                DataIO.write_csv([], filepath)

    def test_csv_read_only_comments(self):
        """Test CSV with only comment lines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "comments.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("# This is a comment\n")
                f.write("# Another comment\n")

            with pytest.raises(ValueError, match="No data lines found in CSV"):
                DataIO.read_csv(filepath)

    def test_csv_read_column_not_found(self):
        """Test CSV reading with non-existent column."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_nocol.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("col1,col2\n")
                f.write("1.5,2.3\n")

            with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
                DataIO.read_csv(filepath, column="nonexistent")

    def test_csv_read_non_numeric_values_in_column(self):
        """Test CSV with non-numeric values mixed in."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_mixed.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("value\n")
                f.write("1.5\n")
                f.write("text\n")
                f.write("2.3\n")
                f.write("\n")  # Empty row
                f.write("3.7\n")

            data = DataIO.read_csv(filepath, column="value")
            # Should skip the non-numeric and empty values
            assert len(data) == 3
            assert abs(data[0] - 1.5) < 1e-10
            assert abs(data[1] - 2.3) < 1e-10
            assert abs(data[2] - 3.7) < 1e-10

    def test_csv_read_empty_with_header(self):
        """Test CSV with header but no data rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "header_only.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("value\n")

            with pytest.raises(ValueError, match="CSV file is empty"):
                DataIO.read_csv(filepath)

    def test_csv_read_no_numerical_columns(self):
        """Test CSV with no numerical columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_numbers.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("name,text\n")
                f.write("alice,hello\n")
                f.write("bob,world\n")

            with pytest.raises(ValueError, match="No numerical columns found"):
                DataIO.read_csv(filepath)

    def test_csv_read_no_header_with_data(self):
        """Test CSV without header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_header.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("1.5,2.0\n")
                f.write("2.3,3.0\n")
                f.write("3.7,4.0\n")

            data = DataIO.read_csv(filepath)
            assert len(data) == 3
            assert abs(data[0] - 1.5) < 1e-10

    def test_csv_read_no_header_with_empty_values(self):
        """Test CSV without header with empty values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty_vals.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write(",1.5\n")  # Empty first value
                f.write("2.3,\n")  # Empty second value
                f.write(",,3.7\n")  # Multiple empty values

            data = DataIO.read_csv(filepath)
            # Takes first numeric value from each row, skipping empty values
            assert len(data) >= 1

    def test_csv_read_no_header_empty_rows(self):
        """Test CSV without header with empty rows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty_rows.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("1.5\n")
                f.write("\n")  # Empty row
                f.write("2.3\n")
                f.write("\n")  # Another empty row

            data = DataIO.read_csv(filepath)
            assert len(data) == 2

    def test_csv_read_no_numerical_data(self):
        """Test CSV with no numerical data at all."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "no_nums.csv"
            with filepath.open("w", encoding="utf-8") as f:
                f.write("text1,text2\n")
                f.write("hello,world\n")

            with pytest.raises(ValueError, match="No numerical columns found"):
                DataIO.read_csv(filepath)

    def test_json_write_and_read_basic(self):
        """Test basic JSON write and read operations."""
        data = [1.5, 2.3, 3.7, 4.2, 5.8]
        metadata = {"distribution": "pareto", "alpha": 2.5}

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"

            # Write
            DataIO.write_json(data, filepath, metadata=metadata)
            assert filepath.exists()

            # Read
            result = DataIO.read_json(filepath)
            assert "data" in result
            assert "metadata" in result
            assert len(result["data"]) == len(data)
            assert result["metadata"]["distribution"] == "pareto"

    def test_json_read_invalid_format(self):
        """Test that reading invalid JSON raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "invalid.json"

            # Write invalid JSON (no "data" key)
            with filepath.open("w", encoding="utf-8") as f:
                json.dump({"other": [1, 2, 3]}, f)

            with pytest.raises(ValueError, match="must contain a 'data' key"):
                DataIO.read_json(filepath)

    def test_json_write_empty_data(self):
        """Test that writing empty data raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "empty.json"
            with pytest.raises(ValueError, match="Cannot write empty data"):
                DataIO.write_json([], filepath)

    def test_json_read_nonexistent_file(self):
        """Test JSON read with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            DataIO.read_json("nonexistent.json")

    def test_json_read_not_dict(self):
        """Test JSON read with non-dict content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "notdict.json"
            with filepath.open("w", encoding="utf-8") as f:
                json.dump([1, 2, 3], f)  # List instead of dict

            with pytest.raises(TypeError, match="must contain a dictionary"):
                DataIO.read_json(filepath)

    def test_json_read_data_not_list(self):
        """Test JSON read with 'data' not being a list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "notlist.json"
            with filepath.open("w", encoding="utf-8") as f:
                json.dump({"data": "not a list"}, f)

            with pytest.raises(TypeError, match="'data' must be a list"):
                DataIO.read_json(filepath)

    def test_json_read_non_numeric_data(self):
        """Test JSON read with non-numeric data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "nonnumeric.json"
            with filepath.open("w", encoding="utf-8") as f:
                json.dump({"data": [1.5, "text", 2.3]}, f)

            with pytest.raises(ValueError, match="must contain numeric values"):
                DataIO.read_json(filepath)


class TestAutoFit:
    """Test automatic parameter fitting."""

    def test_fit_pareto_distribution(self):
        """Test fitting Pareto distribution."""
        # Generate known Pareto data
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        fitter = AutoFit()
        result = fitter.fit_distribution(data, "pareto")

        assert "distribution" in result
        assert result["distribution"] == "pareto"
        assert "parameters" in result
        assert "alpha" in result["parameters"]
        assert "xm" in result["parameters"]

        # Check parameters are reasonable
        assert abs(result["parameters"]["alpha"] - 2.5) < 0.5

    def test_fit_lognormal_distribution(self):
        """Test fitting LogNormal distribution."""
        dist = LogNormal(mu=0.5, sigma=1.0)
        data = dist.rvs(500, seed=42)

        fitter = AutoFit()
        result = fitter.fit_distribution(data, "lognormal")

        assert result["distribution"] == "lognormal"
        assert "mu" in result["parameters"]
        assert "sigma" in result["parameters"]

    def test_fit_auto_selection(self):
        """Test automatic distribution selection."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        fitter = AutoFit()
        result = fitter.fit_distribution(data, "auto")

        assert "distribution" in result
        assert "parameters" in result
        assert "AIC" in result
        assert "BIC" in result
        assert "rank_AIC" in result
        assert "all_candidates" in result

        # For Pareto data, Pareto should rank well
        assert result["distribution"] in ["pareto", "lognormal", "cauchy", "weibull"]

    def test_fit_invalid_distribution(self):
        """Test that invalid distribution raises error."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        fitter = AutoFit()

        with pytest.raises(ValueError, match="not supported"):
            fitter.fit_distribution(data, "invalid_dist")

    def test_compare_distributions(self):
        """Test comparing multiple distributions."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        fitter = AutoFit()
        results = fitter.compare_distributions(data, ["pareto", "lognormal"])

        assert "pareto" in results
        assert "lognormal" in results

        # Check structure
        for dist_name in results:
            assert "params" in results[dist_name]
            assert "AIC" in results[dist_name]
            assert "BIC" in results[dist_name]
            if results[dist_name]["log_likelihood"] != float("-inf"):
                assert "rank_AIC" in results[dist_name]
                assert "rank_BIC" in results[dist_name]

        # Pareto should rank best for Pareto data
        if results["pareto"]["log_likelihood"] != float("-inf"):
            assert results["pareto"]["rank_AIC"] == 1

    def test_compare_distributions_default(self):
        """Test comparing distributions with default list."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        fitter = AutoFit()
        results = fitter.compare_distributions(data)

        # Should use default common distributions
        assert len(results) >= 2
        assert any(d in results for d in ["pareto", "lognormal", "cauchy", "weibull"])


class TestParameterValidator:
    """Test parameter validation."""

    def test_validate_pareto_valid(self):
        """Test valid Pareto parameters."""
        # Should not raise
        ParameterValidator.validate_pareto(2.5, 1.0)
        ParameterValidator.validate_pareto(0.5, 0.1)
        ParameterValidator.validate_pareto(5.0, 100.0)

    def test_validate_pareto_invalid_alpha(self):
        """Test invalid Pareto alpha."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ParameterValidator.validate_pareto(-1.0, 1.0)

        with pytest.raises(ValueError, match="alpha must be positive"):
            ParameterValidator.validate_pareto(0.0, 1.0)

    def test_validate_pareto_invalid_xm(self):
        """Test invalid Pareto xm."""
        with pytest.raises(ValueError, match=r"xm .* must be positive"):
            ParameterValidator.validate_pareto(2.5, -1.0)

        with pytest.raises(ValueError, match=r"xm .* must be positive"):
            ParameterValidator.validate_pareto(2.5, 0.0)

    def test_validate_pareto_extreme_values_warning(self):
        """Test warnings for extreme parameter values."""
        with pytest.warns(UserWarning, match="Very large alpha"):
            ParameterValidator.validate_pareto(150.0, 1.0)

        with pytest.warns(UserWarning, match="Very small alpha"):
            ParameterValidator.validate_pareto(0.05, 1.0)

    def test_validate_lognormal_valid(self):
        """Test valid LogNormal parameters."""
        ParameterValidator.validate_lognormal(0.0, 1.0)
        ParameterValidator.validate_lognormal(1.5, 0.5)

    def test_validate_lognormal_invalid_sigma(self):
        """Test invalid LogNormal sigma."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            ParameterValidator.validate_lognormal(0.0, -1.0)

        with pytest.raises(ValueError, match="sigma must be positive"):
            ParameterValidator.validate_lognormal(0.0, 0.0)

    def test_validate_cauchy_valid(self):
        """Test valid Cauchy parameters."""
        ParameterValidator.validate_cauchy(0.0, 1.0)
        ParameterValidator.validate_cauchy(5.0, 2.5)

    def test_validate_cauchy_invalid_gamma(self):
        """Test invalid Cauchy gamma."""
        with pytest.raises(ValueError, match=r"gamma .* must be positive"):
            ParameterValidator.validate_cauchy(0.0, -1.0)

    def test_suggest_parameters_pareto(self):
        """Test parameter suggestions for Pareto."""
        validator = ParameterValidator()
        ranges = validator.suggest_parameters("pareto")

        assert "alpha" in ranges
        assert "xm" in ranges
        assert isinstance(ranges["alpha"], tuple)
        assert len(ranges["alpha"]) == 2

    def test_suggest_parameters_with_data(self):
        """Test data-driven parameter suggestions."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
        validator = ParameterValidator()
        ranges = validator.suggest_parameters("pareto", data)

        assert "alpha" in ranges
        assert "xm" in ranges

        # xm should be suggested near minimum
        assert ranges["xm"][0] < min(data)


class TestStatisticalSummary:
    """Test statistical summary calculations."""

    def test_basic_stats_simple_data(self):
        """Test basic statistics on simple data."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = StatisticalSummary(data)
        stats = summary.basic_stats()

        assert stats["n"] == 5
        assert abs(stats["mean"] - 3.0) < 1e-10
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
        assert "std" in stats
        assert "variance" in stats
        assert "skewness" in stats
        assert "kurtosis" in stats

    def test_basic_stats_quantiles(self):
        """Test quantile calculations."""
        data = list(range(1, 101))  # 1 to 100
        summary = StatisticalSummary(data)
        stats = summary.basic_stats()

        assert "q_0.25" in stats
        assert "q_0.5" in stats
        assert "q_0.75" in stats
        assert "q_0.95" in stats

        # Check approximate values
        assert abs(stats["q_0.5"] - 50.5) < 1.0  # Median
        assert abs(stats["q_0.25"] - 25.75) < 2.0  # Q1
        assert abs(stats["q_0.75"] - 75.25) < 2.0  # Q3

    def test_tail_statistics_pareto_data(self):
        """Test tail statistics on Pareto-generated data."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        summary = StatisticalSummary(data)
        tail_stats = summary.tail_statistics()

        assert "hill_estimates" in tail_stats
        assert "hill_mean" in tail_stats
        assert "tail_ratio_95_50" in tail_stats
        assert "tail_ratio_99_50" in tail_stats
        assert "heavy_tail_indicator" in tail_stats
        assert "coefficient_of_variation" in tail_stats

        # Tail ratio should be higher than light-tailed distributions
        assert tail_stats["tail_ratio_95_50"] > 2.0

        # Hill estimate should be reasonable (gamma = 1/alpha ≈ 0.4 for alpha=2.5)
        if tail_stats["hill_mean"] is not None:
            assert 0.2 < tail_stats["hill_mean"] < 0.8

    def test_tail_statistics_normal_like_data(self):
        """Test tail statistics on normal-like data."""
        # Create data with light tails
        random.seed(42)
        data = [random.gauss(0, 1) for _ in range(500)]

        summary = StatisticalSummary(data)
        tail_stats = summary.tail_statistics()

        # Tail ratio should exist and be calculable
        assert tail_stats["tail_ratio_95_50"] > 0
        # For normal data with potential negative values, ratio can be large
        # Just verify it's computable, not specific value

    def test_diagnostic_summary_pareto_data(self):
        """Test diagnostic summary on Pareto data."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        summary = StatisticalSummary(data)
        diagnostics = summary.diagnostic_summary()

        assert "likely_heavy_tailed" in diagnostics
        assert "confidence" in diagnostics
        assert "diagnostics" in diagnostics
        assert "recommended_distributions" in diagnostics

        # Should identify as heavy-tailed
        assert diagnostics["likely_heavy_tailed"] is True
        assert diagnostics["confidence"] in ["high", "medium", "low"]
        assert len(diagnostics["diagnostics"]) > 0

    def test_diagnostic_summary_recommendations(self):
        """Test distribution recommendations."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        summary = StatisticalSummary(data)
        diagnostics = summary.diagnostic_summary()

        # Should recommend heavy-tailed distributions
        recommended = diagnostics["recommended_distributions"]
        assert isinstance(recommended, list)
        assert len(recommended) > 0
        # Should include at least one heavy-tailed distribution
        assert any(d in recommended for d in ["Pareto", "LogNormal", "Cauchy"])

    def test_empty_data_error(self):
        """Test that empty data raises error."""
        with pytest.raises(ValueError, match="Data cannot be empty"):
            StatisticalSummary([])

    def test_non_finite_data_error(self):
        """Test that non-finite data raises error."""
        data = [1.0, 2.0, float("nan"), 3.0]
        with pytest.raises(ValueError, match="non-finite values"):
            StatisticalSummary(data)

    def test_quantile_boundary_cases(self):
        """Test quantile calculation at boundaries."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        summary = StatisticalSummary(data)

        # Test 0 and 1 quantiles
        assert summary._quantile(0.0) == 1.0
        assert summary._quantile(1.0) == 5.0

    def test_quantile_invalid_value(self):
        """Test that invalid quantile raises error."""
        data = [1.0, 2.0, 3.0]
        summary = StatisticalSummary(data)

        with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
            summary._quantile(-0.1)

        with pytest.raises(ValueError, match="Quantile must be between 0 and 1"):
            summary._quantile(1.5)


class TestIntegration:
    """Integration tests for utilities."""

    def test_full_workflow_csv_to_analysis(self):
        """Test complete workflow: generate data, save to CSV, load, analyze."""
        # 1. Generate data
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "data.csv"

            # 2. Save to CSV
            DataIO.write_csv(data, filepath, metadata={"source": "Pareto(2.5, 1.0)"})

            # 3. Load from CSV
            loaded_data = DataIO.read_csv(filepath)

            # 4. Analyze
            summary = StatisticalSummary(loaded_data)
            diagnostics = summary.diagnostic_summary()

            # 5. Fit distribution
            fitter = AutoFit()
            result = fitter.fit_distribution(loaded_data, "pareto")

            # Verify complete workflow
            assert len(loaded_data) == len(data)
            assert diagnostics["likely_heavy_tailed"] is True
            assert result["distribution"] == "pareto"
            assert abs(result["parameters"]["alpha"] - 2.5) < 0.5

    def test_full_workflow_json_to_analysis(self):
        """Test workflow with JSON format."""
        dist = Pareto(alpha=2.5, xm=1.0)
        data = dist.rvs(500, seed=42)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "data.json"

            # Save with metadata
            metadata = {"distribution": "Pareto", "alpha": 2.5, "xm": 1.0}
            DataIO.write_json(data, filepath, metadata=metadata)

            # Load and verify metadata preserved
            loaded = DataIO.read_json(filepath)
            assert loaded["metadata"]["distribution"] == "Pareto"

            # Analyze
            summary = StatisticalSummary(loaded["data"])
            tail_stats = summary.tail_statistics()

            # Verify statistics are calculable
            assert "heavy_tail_indicator" in tail_stats
            assert "tail_ratio_95_50" in tail_stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
