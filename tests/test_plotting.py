"""Tests for plotting utilities."""

import math

from heavytails.plotting import qq_pareto, tail_loglog_plot


class TestTailLogLogPlot:
    """Tests for tail_loglog_plot function."""

    def test_basic_functionality(self) -> None:
        """Test basic tail log-log plot generation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = tail_loglog_plot(data)

        # Should return list of tuples
        assert isinstance(result, list)
        assert all(isinstance(point, tuple) for point in result)
        assert all(len(point) == 2 for point in result)

    def test_positive_values_only(self) -> None:
        """Test that only positive values are included."""
        data = [-1.0, 0.0, 1.0, 2.0, 3.0]
        result = tail_loglog_plot(data)

        # Should only include positive values (1.0, 2.0, 3.0)
        assert len(result) == 3

    def test_sorted_output(self) -> None:
        """Test that output is based on sorted data."""
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        result = tail_loglog_plot(data)

        # X values should be in ascending order (log scale)
        x_values = [x for x, _ in result]
        assert x_values == sorted(x_values)

    def test_log_scale_values(self) -> None:
        """Test that values are correctly log-transformed."""
        data = [1.0, 2.0, 3.0]
        result = tail_loglog_plot(data)

        # Check that x-values are log-transformed
        for i, (log_x, _log_survival) in enumerate(result):
            expected_log_x = math.log(sorted(data)[i])
            assert math.isclose(log_x, expected_log_x, rel_tol=1e-9)

    def test_survival_probabilities(self) -> None:
        """Test that survival probabilities are correct."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = tail_loglog_plot(data)
        n = len(data)

        # Check survival probabilities
        for i, (_, log_survival) in enumerate(result):
            expected_survival = (n - i) / n
            expected_log_survival = math.log(expected_survival)
            assert math.isclose(log_survival, expected_log_survival, rel_tol=1e-9)

    def test_empty_data(self) -> None:
        """Test with empty data."""
        data: list[float] = []
        result = tail_loglog_plot(data)
        assert result == []

    def test_single_value(self) -> None:
        """Test with single value."""
        data = [5.0]
        result = tail_loglog_plot(data)
        assert len(result) == 1

    def test_all_zeros(self) -> None:
        """Test with all zero values."""
        data = [0.0, 0.0, 0.0]
        result = tail_loglog_plot(data)
        # Should exclude all zeros
        assert result == []

    def test_mixed_positive_zero(self) -> None:
        """Test with mix of positive and zero values."""
        data = [0.0, 1.0, 0.0, 2.0, 0.0]
        result = tail_loglog_plot(data)
        # Should only include positive values
        assert len(result) == 2


class TestQQPareto:
    """Tests for qq_pareto function."""

    def test_basic_functionality(self) -> None:
        """Test basic QQ plot generation."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = qq_pareto(data)

        # Should return list of tuples
        assert isinstance(result, list)
        assert all(isinstance(point, tuple) for point in result)
        assert all(len(point) == 2 for point in result)

    def test_output_length(self) -> None:
        """Test that output has correct length."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = qq_pareto(data)
        # Should have n-1 points
        assert len(result) == len(data) - 1

    def test_sorted_output(self) -> None:
        """Test that output is based on sorted data."""
        data = [5.0, 1.0, 3.0, 2.0, 4.0]
        result = qq_pareto(data)

        # Check that we get the same result as sorted data
        sorted_data = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_result = qq_pareto(sorted_data)

        for (x1, y1), (x2, y2) in zip(result, expected_result, strict=False):
            assert math.isclose(x1, x2, rel_tol=1e-9)
            assert math.isclose(y1, y2, rel_tol=1e-9)

    def test_log_transformed_values(self) -> None:
        """Test that values are correctly log-transformed."""
        data = [1.0, 2.0, 3.0, 4.0]
        result = qq_pareto(data)
        n = len(data)
        sorted_data = sorted(data)

        # Check log transformations
        for i, (log_quantile, log_value) in enumerate(result):
            expected_log_quantile = math.log((i + 1) / n)
            expected_log_value = math.log(sorted_data[i])
            assert math.isclose(log_quantile, expected_log_quantile, rel_tol=1e-9)
            assert math.isclose(log_value, expected_log_value, rel_tol=1e-9)

    def test_empty_data(self) -> None:
        """Test with empty data."""
        data: list[float] = []
        result = qq_pareto(data)
        assert result == []

    def test_single_value(self) -> None:
        """Test with single value."""
        data = [5.0]
        result = qq_pareto(data)
        assert result == []

    def test_two_values(self) -> None:
        """Test with two values."""
        data = [1.0, 2.0]
        result = qq_pareto(data)
        assert len(result) == 1

    def test_positive_values(self) -> None:
        """Test with various positive values."""
        data = [1.0, 10.0, 100.0, 1000.0]
        result = qq_pareto(data)

        # All results should be finite
        for log_q, log_val in result:
            assert math.isfinite(log_q)
            assert math.isfinite(log_val)

    def test_negative_quantiles(self) -> None:
        """Test that theoretical quantiles are negative (log of values < 1)."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = qq_pareto(data)

        # Since i/n < 1, log(i/n) should be negative
        for log_q, _ in result:
            assert log_q < 0
