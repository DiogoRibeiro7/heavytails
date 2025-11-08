"""
Utility functions and helper classes for HeavyTails library.

This module contains various utility functions for data I/O, parameter estimation,
and statistical analysis.
"""

import csv
import json
import math
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any


class DataIO:
    """
    Data import/export utilities for various file formats.

    Supports CSV and JSON formats for data and metadata storage.
    Provides robust error handling and automatic column detection.
    """

    @staticmethod
    def read_csv(filepath: Path | str, column: str | None = None) -> list[float]:
        """
        Read numerical data from CSV file.

        Args:
            filepath: Path to CSV file
            column: Column name to read (if None, auto-detects first numerical column)

        Returns:
            List of numerical data values

        Raises:
            ValueError: If file cannot be read or no numerical data found
            FileNotFoundError: If file does not exist

        Examples:
            >>> # Create sample CSV
            >>> import tempfile
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            ...     f.write('value\\n1.5\\n2.3\\n3.7\\n')
            ...     temp_path = f.name
            >>> data = DataIO.read_csv(temp_path)
            >>> len(data)
            3
            >>> import os
            >>> os.unlink(temp_path)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                # Filter out comment lines
                non_comment_lines = [
                    line for line in f if not line.strip().startswith("#")
                ]

            if not non_comment_lines:
                raise ValueError("No data lines found in CSV (only comments)")

            # Create StringIO from non-comment lines
            csv_content = "".join(non_comment_lines)
            csv_file = StringIO(csv_content)

            # Detect if first line is header
            first_line = non_comment_lines[0].strip()
            first_value = first_line.split(",")[0].strip()
            try:
                float(first_value)
                has_header = False
            except ValueError:
                has_header = True

            # Reset StringIO
            csv_file.seek(0)

            try:
                if has_header and column:
                    # Read specific column by name
                    reader = csv.DictReader(csv_file)
                    data = []
                    for row in reader:
                        if column not in row:
                            raise ValueError(f"Column '{column}' not found in CSV")
                        value_str = row[column].strip()
                        if value_str:  # Skip empty strings
                            try:
                                data.append(float(value_str))
                            except ValueError:
                                # Skip non-numeric values
                                continue
                elif has_header and not column:
                    # Auto-detect first numerical column
                    reader = csv.DictReader(csv_file)
                    first_row = next(reader, None)
                    if first_row is None:
                        raise ValueError("CSV file is empty")

                    # Find first numerical column
                    num_column = None
                    for col_name, value in first_row.items():
                        # Skip if value is not a string (e.g., None key with list value)
                        if not isinstance(value, str):
                            continue
                        try:
                            float(value)
                            num_column = col_name
                            break
                        except ValueError:
                            continue

                    if num_column is None:
                        raise ValueError("No numerical columns found in CSV")

                    # Read the data
                    csv_file.seek(0)
                    reader = csv.DictReader(csv_file)
                    data = []
                    for row in reader:
                        value_str = row[num_column].strip()
                        if value_str:
                            try:
                                data.append(float(value_str))
                            except ValueError:
                                continue
                else:
                    # No header, read all rows and take first numerical value
                    reader = csv.reader(csv_file)
                    data = []
                    for row in reader:
                        # Skip empty rows
                        if not row:
                            continue

                        for value in row:
                            if not value:  # Skip empty strings
                                continue
                            try:
                                data.append(float(value.strip()))
                                break  # Take first number from each row
                            except ValueError:
                                continue

                if not data:
                    raise ValueError("No numerical data found in CSV file")

                return data

            except csv.Error as e:
                raise ValueError(f"Error parsing CSV file: {e}") from e

        except Exception as e:
            # Re-raise ValueError as is, wrap others
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error reading CSV file: {e}") from e

    @staticmethod
    def write_csv(
        data: list[float],
        filepath: Path | str,
        metadata: dict[str, Any] | None = None,
        column_name: str = "value",
    ) -> None:
        """
        Write numerical data to CSV file.

        Args:
            data: List of numerical values
            filepath: Path to output CSV file
            metadata: Optional metadata to include as header comments
            column_name: Name for the data column (default: "value")

        Raises:
            ValueError: If data is empty

        Examples:
            >>> import tempfile
            >>> import os
            >>> data = [1.5, 2.3, 3.7]
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            ...     temp_path = f.name
            >>> DataIO.write_csv(data, temp_path)
            >>> os.path.exists(temp_path)
            True
            >>> os.unlink(temp_path)
        """
        if not data:
            raise ValueError("Cannot write empty data to CSV")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                # Write metadata as comments if provided
                if metadata:
                    f.write("# Metadata:\n")
                    for key, value in metadata.items():
                        f.write(f"# {key}: {value}\n")
                    f.write("#\n")

                # Write header and data
                writer = csv.writer(f)
                writer.writerow([column_name])
                for value in data:
                    writer.writerow([value])

        except OSError as e:
            raise ValueError(f"Error writing CSV file: {e}") from e

    @staticmethod
    def read_json(filepath: Path | str) -> dict[str, Any]:
        """
        Read data and metadata from JSON file.

        Expected JSON structure:
        {
            "data": [1.5, 2.3, 3.7, ...],
            "metadata": {
                "distribution": "pareto",
                "parameters": {"alpha": 2.5, "xm": 1.0}
            }
        }

        Args:
            filepath: Path to JSON file

        Returns:
            Dictionary with "data" and optional "metadata" keys

        Raises:
            ValueError: If file cannot be read or has invalid format
            FileNotFoundError: If file does not exist

        Examples:
            >>> import tempfile
            >>> import json
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ...     json.dump({"data": [1.5, 2.3], "metadata": {"dist": "test"}}, f)
            ...     temp_path = f.name
            >>> result = DataIO.read_json(temp_path)
            >>> len(result["data"])
            2
            >>> import os
            >>> os.unlink(temp_path)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"JSON file not found: {filepath}")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = json.load(f)

            if not isinstance(content, dict):
                raise ValueError("JSON file must contain a dictionary")

            if "data" not in content:
                raise ValueError("JSON file must contain a 'data' key")

            # Validate data is list of numbers
            data = content["data"]
            if not isinstance(data, list):
                raise ValueError("'data' must be a list")

            # Convert to floats
            try:
                data = [float(x) for x in data]
            except (ValueError, TypeError) as e:
                raise ValueError(f"'data' must contain numeric values: {e}") from e

            content["data"] = data
            return content

        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing JSON file: {e}") from e

    @staticmethod
    def write_json(
        data: list[float], filepath: Path | str, metadata: dict[str, Any] | None = None
    ) -> None:
        """
        Write data and metadata to JSON file.

        Args:
            data: List of numerical values
            filepath: Path to output JSON file
            metadata: Optional metadata dictionary

        Raises:
            ValueError: If data is empty

        Examples:
            >>> import tempfile
            >>> import os
            >>> data = [1.5, 2.3, 3.7]
            >>> with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            ...     temp_path = f.name
            >>> DataIO.write_json(data, temp_path, {"distribution": "test"})
            >>> os.path.exists(temp_path)
            True
            >>> os.unlink(temp_path)
        """
        if not data:
            raise ValueError("Cannot write empty data to JSON")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        content = {"data": data}
        if metadata:
            content["metadata"] = metadata

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)
        except OSError as e:
            raise ValueError(f"Error writing JSON file: {e}") from e


class AutoFit:
    """
    Automatic parameter estimation for heavy-tailed distributions.

    Provides MLE-based parameter estimation and model selection using AIC/BIC.
    Integrates with roadmap.py implementations.
    """

    def __init__(self) -> None:
        """Initialize AutoFit with list of supported distributions."""
        self.available_distributions = [
            "pareto",
            "lognormal",
            "cauchy",
            "studentt",
            "weibull",
            "frechet",
            "generalizedpareto",
            "burrxii",
            "loglogistic",
            "inversegamma",
            "betaprime",
        ]

    def fit_distribution(
        self, data: list[float], distribution: str = "auto"
    ) -> dict[str, Any]:
        """
        Fit distribution parameters to data using MLE.

        Args:
            data: Sample data to fit
            distribution: Distribution name or "auto" for automatic selection

        Returns:
            Dictionary with fitted parameters and fit quality metrics

        Examples:
            >>> from heavytails import Pareto
            >>> dist = Pareto(alpha=2.5, xm=1.0)
            >>> data = dist.rvs(500, seed=42)
            >>> fitter = AutoFit()
            >>> result = fitter.fit_distribution(data, "pareto")
            >>> "parameters" in result
            True
            >>> abs(result["parameters"]["alpha"] - 2.5) < 0.5
            True
        """
        from heavytails.roadmap import fit_mle, model_comparison  # noqa: PLC0415

        if distribution == "auto":
            # Automatic distribution selection
            return self._auto_select_distribution(data)
        else:
            # Fit specific distribution
            if distribution.lower() not in self.available_distributions:
                raise ValueError(
                    f"Distribution '{distribution}' not supported. "
                    f"Available: {', '.join(self.available_distributions)}"
                )

            try:
                params = fit_mle(data, distribution)
                return {
                    "distribution": distribution,
                    "parameters": params,
                    "method": "MLE",
                    "sample_size": len(data),
                }
            except Exception as e:
                raise ValueError(f"Failed to fit {distribution}: {e}") from e

    def compare_distributions(
        self, data: list[float], distributions: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Compare multiple distribution fits and rank by quality.

        Args:
            data: Sample data
            distributions: List of distribution names (if None, uses common ones)

        Returns:
            Dictionary of fit results for each distribution, with rankings

        Examples:
            >>> from heavytails import Pareto
            >>> dist = Pareto(alpha=2.5, xm=1.0)
            >>> data = dist.rvs(500, seed=42)
            >>> fitter = AutoFit()
            >>> results = fitter.compare_distributions(data, ["pareto", "lognormal"])
            >>> "pareto" in results
            True
            >>> results["pareto"]["rank_AIC"] == 1  # Pareto should rank best
            True
        """
        from heavytails.roadmap import model_comparison  # noqa: PLC0415

        if distributions is None:
            # Use common heavy-tailed distributions
            distributions = ["pareto", "lognormal", "cauchy", "weibull"]

        # Validate distribution names
        for dist in distributions:
            if dist.lower() not in self.available_distributions:
                raise ValueError(
                    f"Distribution '{dist}' not supported. "
                    f"Available: {', '.join(self.available_distributions)}"
                )

        try:
            return model_comparison(data, distributions)
        except Exception as e:
            raise ValueError(f"Distribution comparison failed: {e}") from e

    def _auto_select_distribution(self, data: list[float]) -> dict[str, Any]:
        """
        Automatically select best-fitting distribution.

        Uses AIC to select the best distribution from common candidates.
        """
        from heavytails.roadmap import model_comparison  # noqa: PLC0415

        # Try common heavy-tailed distributions
        candidates = ["pareto", "lognormal", "cauchy", "weibull"]

        results = model_comparison(data, candidates)

        # Find distribution with lowest AIC (valid fits only)
        valid_results = {
            k: v for k, v in results.items() if v["log_likelihood"] != float("-inf")
        }

        if not valid_results:
            raise ValueError("Failed to fit any distribution to the data")

        best_dist = min(valid_results.keys(), key=lambda d: valid_results[d]["AIC"])

        return {
            "distribution": best_dist,
            "parameters": results[best_dist]["params"],
            "method": "MLE",
            "sample_size": len(data),
            "AIC": results[best_dist]["AIC"],
            "BIC": results[best_dist]["BIC"],
            "rank_AIC": results[best_dist]["rank_AIC"],
            "rank_BIC": results[best_dist]["rank_BIC"],
            "all_candidates": results,
        }


class ParameterValidator:
    """
    Enhanced parameter validation with informative error messages.

    Provides detailed parameter validation with helpful suggestions
    and typical parameter ranges for all distributions.
    """

    @staticmethod
    def validate_pareto(alpha: float, xm: float) -> None:
        """
        Validate Pareto parameters with detailed feedback.

        Args:
            alpha: Shape parameter (tail index)
            xm: Scale parameter (minimum value)

        Raises:
            ValueError: If parameters are invalid, with helpful suggestions
        """
        if alpha <= 0:
            raise ValueError(
                f"Pareto alpha must be positive, got {alpha}.\n"
                f"Interpretation:\n"
                f"  - alpha < 1: Infinite mean (very heavy tail)\n"
                f"  - alpha ∈ [1,2): Finite mean, infinite variance\n"
                f"  - alpha >= 2: Finite mean and variance\n"
                f"Typical range: 0.5 to 5.0\n"
                f"Financial data: often 1.5 to 3.0"
            )

        if xm <= 0:
            raise ValueError(
                f"Pareto xm (minimum value) must be positive, got {xm}.\n"
                f"Suggestion: xm should be the minimum possible value.\n"
                f"For data analysis, try xm = min(data) or slightly less.\n"
                f"Typical range: 0.1 to 1000 depending on application"
            )

        # Warnings for extreme values
        if alpha > 100:
            warnings.warn(
                "Very large alpha (>100) may cause numerical precision issues",
                stacklevel=2,
            )
        if alpha < 0.1:
            warnings.warn(
                "Very small alpha (<0.1) creates extremely heavy tails", stacklevel=2
            )

    @staticmethod
    def validate_lognormal(mu: float, sigma: float) -> None:
        """Validate LogNormal parameters."""
        if sigma <= 0:
            raise ValueError(
                f"LogNormal sigma must be positive, got {sigma}.\n"
                f"Interpretation: sigma controls tail heaviness\n"
                f"  - sigma < 0.5: Light tail\n"
                f"  - sigma ∈ [0.5, 1.5]: Moderate tail\n"
                f"  - sigma > 1.5: Heavy tail\n"
                f"Typical range: 0.1 to 3.0"
            )

        if sigma > 10:
            warnings.warn(
                "Very large sigma (>10) may cause numerical overflow", stacklevel=2
            )
        if abs(mu) > 100:
            warnings.warn(
                "Large |mu| (>100) may cause numerical issues", stacklevel=2
            )

    @staticmethod
    def validate_cauchy(x0: float, gamma: float) -> None:
        """Validate Cauchy parameters."""
        if gamma <= 0:
            raise ValueError(
                f"Cauchy gamma (scale) must be positive, got {gamma}.\n"
                f"Interpretation: gamma controls spread\n"
                f"  - Smaller gamma: more peaked distribution\n"
                f"  - Larger gamma: flatter distribution\n"
                f"Typical range: 0.1 to 10.0"
            )

        if gamma > 1000:
            warnings.warn(
                "Very large gamma (>1000) may cause numerical issues", stacklevel=2
            )

    @staticmethod
    def suggest_parameters(
        distribution: str, data: list[float] | None = None
    ) -> dict[str, tuple[float, float]]:
        """
        Suggest reasonable parameter ranges based on data or defaults.

        Args:
            distribution: Distribution name
            data: Optional data for data-driven suggestions

        Returns:
            Dictionary mapping parameter names to (min, max) ranges

        Examples:
            >>> validator = ParameterValidator()
            >>> ranges = validator.suggest_parameters("pareto")
            >>> "alpha" in ranges
            True
        """
        dist_lower = distribution.lower()

        if data is not None:
            # Data-driven suggestions
            data_min = min(data)
            data_max = max(data)
            data_mean = sum(data) / len(data)
            data_std = math.sqrt(
                sum((x - data_mean) ** 2 for x in data) / (len(data) - 1)
            )

            if dist_lower == "pareto":
                # Suggest xm based on minimum, alpha from data properties
                return {
                    "alpha": (0.5, 5.0),
                    "xm": (max(0.01, data_min * 0.9), data_min),
                }
            elif dist_lower == "lognormal":
                log_data = [math.log(x) for x in data if x > 0]
                if log_data:
                    log_mean = sum(log_data) / len(log_data)
                    log_std = math.sqrt(
                        sum((x - log_mean) ** 2 for x in log_data) / (len(log_data) - 1)
                    )
                    return {
                        "mu": (log_mean - log_std, log_mean + log_std),
                        "sigma": (max(0.1, log_std * 0.5), log_std * 2.0),
                    }
            elif dist_lower == "cauchy":
                return {
                    "x0": (data_mean - data_std, data_mean + data_std),
                    "gamma": (data_std * 0.1, data_std * 2.0),
                }

        # Default ranges (no data)
        default_ranges = {
            "pareto": {"alpha": (0.5, 5.0), "xm": (0.1, 10.0)},
            "lognormal": {"mu": (-2.0, 2.0), "sigma": (0.1, 3.0)},
            "cauchy": {"x0": (-10.0, 10.0), "gamma": (0.1, 10.0)},
            "weibull": {"k": (0.5, 5.0), "lam": (0.5, 10.0)},
            "studentt": {"nu": (2.1, 30.0)},
        }

        return default_ranges.get(dist_lower, {})


class StatisticalSummary:
    """
    Comprehensive statistical summary for heavy-tailed data.

    Provides descriptive statistics, tail-specific measures,
    and diagnostics for heavy-tail analysis.
    """

    def __init__(self, data: list[float]):
        """
        Initialize with data.

        Args:
            data: List of numerical values

        Raises:
            ValueError: If data is empty or contains non-finite values
        """
        if not data:
            raise ValueError("Data cannot be empty")

        if any(not math.isfinite(x) for x in data):
            raise ValueError("Data contains non-finite values")

        self.data = sorted(data)
        self.n = len(data)

    def basic_stats(self) -> dict[str, float]:
        """
        Calculate comprehensive basic statistics.

        Returns:
            Dictionary with mean, std, variance, skewness, kurtosis, quantiles

        Examples:
            >>> data = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0]
            >>> summary = StatisticalSummary(data)
            >>> stats = summary.basic_stats()
            >>> "mean" in stats
            True
            >>> stats["n"]
            7
        """
        data = self.data
        n = self.n

        # Basic moments
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(variance)

        # Skewness and kurtosis
        if std > 0:
            m3 = sum((x - mean) ** 3 for x in data) / n
            m4 = sum((x - mean) ** 4 for x in data) / n
            skewness = m3 / (std**3)
            kurtosis = m4 / (variance**2) if variance > 0 else 0.0
        else:
            skewness = 0.0
            kurtosis = 0.0

        # Quantiles
        quantiles = self._calculate_quantiles([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

        # Range and IQR
        data_range = data[-1] - data[0]
        iqr = quantiles["q_0.75"] - quantiles["q_0.25"]

        return {
            "n": n,
            "mean": mean,
            "std": std,
            "variance": variance,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "excess_kurtosis": kurtosis - 3.0,
            "min": data[0],
            "max": data[-1],
            "range": data_range,
            "iqr": iqr,
            **quantiles,
        }

    def tail_statistics(self) -> dict[str, Any]:
        """
        Calculate heavy-tail specific statistics.

        Returns:
            Dictionary with Hill estimates, tail ratios, and tail indicators

        Examples:
            >>> from heavytails import Pareto
            >>> dist = Pareto(alpha=2.5, xm=1.0)
            >>> data = dist.rvs(500, seed=42)
            >>> summary = StatisticalSummary(data)
            >>> tail_stats = summary.tail_statistics()
            >>> "tail_ratio_95_50" in tail_stats
            True
        """
        from heavytails.tail_index import hill_estimator  # noqa: PLC0415

        # Hill estimates for different k values
        hill_estimates = []
        k_values = []

        for k_fraction in [0.05, 0.1, 0.2]:
            k = int(self.n * k_fraction)
            if 5 < k < self.n // 2:
                try:
                    gamma = hill_estimator(self.data, k)
                    hill_estimates.append(gamma)
                    k_values.append(k)
                except (ValueError, ZeroDivisionError):
                    pass

        # Tail ratios (indicators of tail heaviness)
        q99 = self._quantile(0.99)
        q95 = self._quantile(0.95)
        q90 = self._quantile(0.90)
        q50 = self._quantile(0.50)

        tail_ratio_99_50 = q99 / q50 if q50 > 0 else float("inf")
        tail_ratio_95_50 = q95 / q50 if q50 > 0 else float("inf")
        tail_ratio_99_95 = q99 / q95 if q95 > 0 else float("inf")

        # Heavy-tail indicator (heuristic: tail ratio > 5 suggests heavy tails)
        heavy_tail_indicator = tail_ratio_95_50 > 5.0

        # Coefficient of variation (high CV often indicates heavy tails)
        stats = self.basic_stats()
        cv = stats["std"] / stats["mean"] if stats["mean"] > 0 else float("inf")

        return {
            "hill_estimates": hill_estimates,
            "hill_k_values": k_values,
            "hill_mean": (
                sum(hill_estimates) / len(hill_estimates) if hill_estimates else None
            ),
            "hill_std": (
                math.sqrt(
                    sum(
                        (x - sum(hill_estimates) / len(hill_estimates)) ** 2
                        for x in hill_estimates
                    )
                    / len(hill_estimates)
                )
                if len(hill_estimates) > 1
                else None
            ),
            "tail_ratio_99_50": tail_ratio_99_50,
            "tail_ratio_95_50": tail_ratio_95_50,
            "tail_ratio_99_95": tail_ratio_99_95,
            "coefficient_of_variation": cv,
            "heavy_tail_indicator": heavy_tail_indicator,
            "excess_kurtosis": stats["excess_kurtosis"],
        }

    def diagnostic_summary(self) -> dict[str, Any]:
        """
        Provide comprehensive heavy-tail diagnostics.

        Returns:
            Dictionary with diagnostics and recommendations

        Examples:
            >>> from heavytails import Pareto
            >>> dist = Pareto(alpha=2.5, xm=1.0)
            >>> data = dist.rvs(500, seed=42)
            >>> summary = StatisticalSummary(data)
            >>> diagnostics = summary.diagnostic_summary()
            >>> "likely_heavy_tailed" in diagnostics
            True
        """
        basic = self.basic_stats()
        tail = self.tail_statistics()

        # Diagnostic criteria for heavy tails
        diagnostics = []

        # 1. High tail ratio
        if tail["tail_ratio_95_50"] > 5.0:
            diagnostics.append("High tail ratio (95th/50th > 5) suggests heavy tails")

        # 2. Positive excess kurtosis
        if basic["excess_kurtosis"] > 3.0:
            diagnostics.append(
                f"High excess kurtosis ({basic['excess_kurtosis']:.2f}) indicates heavy tails"
            )

        # 3. Positive skewness
        if basic["skewness"] > 1.0:
            diagnostics.append(
                f"High skewness ({basic['skewness']:.2f}) suggests right-skewed heavy tail"
            )

        # 4. High coefficient of variation
        if tail["coefficient_of_variation"] > 2.0:
            diagnostics.append(
                f"High CV ({tail['coefficient_of_variation']:.2f}) indicates high variability"
            )

        # Overall assessment
        n_indicators = len(diagnostics)
        if n_indicators >= 3:
            likely_heavy_tailed = True
            confidence = "high"
        elif n_indicators >= 2:
            likely_heavy_tailed = True
            confidence = "medium"
        elif n_indicators >= 1:
            likely_heavy_tailed = True
            confidence = "low"
        else:
            likely_heavy_tailed = False
            confidence = "none"

        # Recommended distributions
        if likely_heavy_tailed:
            if tail["hill_mean"] is not None and tail["hill_mean"] < 0.5:
                recommended = ["Pareto", "Cauchy"]
            elif tail["hill_mean"] is not None and tail["hill_mean"] < 1.0:
                recommended = ["Pareto", "LogNormal", "Weibull"]
            else:
                recommended = ["LogNormal", "Weibull", "Pareto"]
        else:
            recommended = ["Normal", "LogNormal", "Weibull"]

        return {
            "likely_heavy_tailed": likely_heavy_tailed,
            "confidence": confidence,
            "n_indicators": n_indicators,
            "diagnostics": diagnostics,
            "recommended_distributions": recommended,
            "sample_size_adequate": self.n >= 200,
            "basic_stats": basic,
            "tail_stats": tail,
        }

    def _quantile(self, q: float) -> float:
        """Calculate quantile using linear interpolation."""
        if not 0 <= q <= 1:
            raise ValueError("Quantile must be between 0 and 1")

        if q == 0:
            return self.data[0]
        if q == 1:
            return self.data[-1]

        # Linear interpolation
        idx = q * (self.n - 1)
        lower_idx = int(idx)
        upper_idx = min(lower_idx + 1, self.n - 1)
        fraction = idx - lower_idx

        return self.data[lower_idx] * (1 - fraction) + self.data[upper_idx] * fraction

    def _calculate_quantiles(self, quantiles: list[float]) -> dict[str, float]:
        """Calculate multiple quantiles."""
        return {f"q_{q}": self._quantile(q) for q in quantiles}


# TODO: Add configuration management system
class ConfigurationManager:
    """
    Configuration management for HeavyTails library settings.

    Should manage:
    - Default tolerances for numerical algorithms
    - Random number generator settings
    - Plotting preferences
    - Performance optimization flags
    - Caching behavior
    """

    def __init__(self, config_file: Path | None = None) -> None:
        # TODO: Implement configuration loading and management
        self.config: dict[str, Any] = {}
        self.config_file = config_file or Path.home() / ".heavytails" / "config.json"

    def load_config(self) -> dict[str, Any]:
        # TODO: Load configuration from file
        # LABELS: configuration, file-io
        """Load configuration from file."""
        raise NotImplementedError("Configuration loading not implemented")

    def save_config(self, config: dict[str, Any]) -> None:
        # TODO: Save configuration to file
        # LABELS: configuration, file-io
        """Save configuration to file."""
        raise NotImplementedError("Configuration saving not implemented")


# TODO: Implement data quality assessment tools
class DataQualityAssessment:
    """
    Data quality assessment for statistical analysis.

    Should check for:
    - Missing values and outliers
    - Data type consistency
    - Sufficient sample size
    - Independence assumptions
    - Stationarity (for time series)
    """

    def __init__(self, data: list[float]):
        self.data = data

    def assess_quality(self) -> dict[str, Any]:
        # TODO: Comprehensive data quality assessment
        # LABELS: data-quality, assessment
        """Assess data quality for statistical analysis."""
        raise NotImplementedError("Data quality assessment not implemented")

    def detect_outliers(self, method: str = "iqr") -> list[int]:
        # TODO: Detect outliers using various methods
        # LABELS: outlier-detection, data-cleaning
        """Detect outliers in the data."""
        raise NotImplementedError("Outlier detection not implemented")

    def suggest_preprocessing(self) -> list[str]:
        # TODO: Suggest data preprocessing steps
        # LABELS: data-preprocessing, recommendations
        """Suggest preprocessing steps."""
        raise NotImplementedError("Preprocessing suggestions not implemented")


# HACK: Using simple string-based distribution names - need better type system
@dataclass
class DistributionMetadata:
    """
    Metadata container for distribution information.

    Should replace string-based distribution identification
    with proper type system that includes:
    - Distribution family information
    - Parameter constraints
    - Mathematical properties
    - Implementation details
    """

    name: str
    family: str
    parameters: dict[str, dict[str, Any]]
    properties: dict[str, Any]

    def __post_init__(self) -> None:
        # TODO: Implement proper distribution type system
        # LABELS: type-system, metadata
        pass

    def validate_parameters(self, **params: Any) -> bool:
        # TODO: Validate parameters against constraints
        # LABELS: parameter-validation, metadata
        """Validate parameters against distribution constraints."""
        raise NotImplementedError("Metadata-based validation not implemented")


# TODO: Add web scraping utilities for financial data
class FinancialDataScraper:
    """
    Web scraping utilities for financial and economic data.

    Data sources to support:
    - Yahoo Finance for stock data
    - FRED for economic indicators
    - Central bank websites
    - Financial news sentiment

    Should be used for testing and examples, not production.
    """

    def __init__(self) -> None:
        # TODO: Implement web scraping capabilities
        self.session = None

    def get_stock_returns(
        self, symbol: str, start_date: str, end_date: str
    ) -> list[float]:
        # TODO: Scrape stock return data
        # LABELS: web-scraping, finance, stock-data
        """Get stock return data from web sources."""
        raise NotImplementedError("Stock data scraping not implemented")

    def get_economic_indicators(self, indicator: str) -> list[float]:
        # TODO: Scrape economic indicator data
        # LABELS: web-scraping, economics, indicators
        """Get economic indicator data."""
        raise NotImplementedError("Economic data scraping not implemented")


# TODO: Implement citation and bibliography utilities
class CitationManager:
    """
    Citation and bibliography management for research use.

    Should provide:
    - Automatic citation generation
    - Bibliography formatting
    - DOI lookup and validation
    - Reference database management
    """

    def __init__(self) -> None:
        # TODO: Implement citation management
        self.references: dict[str, Any] = {}

    def generate_citation(self, distribution: str, format: str = "bibtex") -> str:
        # TODO: Generate citation for distribution implementation
        # LABELS: citations, bibliography
        """Generate citation for distribution usage."""
        raise NotImplementedError("Citation generation not implemented")

    def export_bibliography(self, format: str = "bibtex") -> str:
        # TODO: Export complete bibliography
        # LABELS: bibliography, export
        """Export bibliography in specified format."""
        raise NotImplementedError("Bibliography export not implemented")


# NOTE: Consider adding interactive tutorials and examples
class InteractiveTutorials:
    """
    Interactive tutorials for learning heavy-tailed distributions.

    Should provide:
    - Step-by-step guided examples
    - Interactive parameter exploration
    - Visualization of distribution properties
    - Quiz and assessment features
    - Progress tracking
    """

    def __init__(self) -> None:
        # TODO: Implement interactive tutorial system
        self.tutorials: dict[str, Any] = {}

    def start_tutorial(self, topic: str) -> dict[str, Any]:
        # TODO: Start interactive tutorial session
        # LABELS: education, tutorials
        """Start an interactive tutorial."""
        raise NotImplementedError("Interactive tutorials not implemented")

    def generate_exercises(self, difficulty: str = "beginner") -> list[dict[str, Any]]:
        # TODO: Generate practice exercises
        # LABELS: education, exercises
        """Generate practice exercises."""
        raise NotImplementedError("Exercise generation not implemented")


# TODO: Implement plugin system for extensions
class PluginManager:
    """
    Plugin system for extending HeavyTails functionality.

    Should allow:
    - Third-party distribution implementations
    - Custom estimation methods
    - Additional plotting backends
    - Domain-specific extensions
    """

    def __init__(self) -> None:
        # TODO: Implement plugin discovery and loading
        self.plugins: dict[str, Any] = {}

    def load_plugin(self, plugin_name: str) -> Any:
        # TODO: Load and register plugin
        # LABELS: plugins, extensibility
        """Load a plugin module."""
        raise NotImplementedError("Plugin loading not implemented")

    def list_plugins(self) -> list[str]:
        # TODO: List available plugins
        # LABELS: plugins, discovery
        """List available plugins."""
        raise NotImplementedError("Plugin listing not implemented")


# TODO: Add unit conversion utilities for different scales
class UnitConverter:
    """
    Unit conversion utilities for different measurement scales.

    Useful for:
    - Converting between log and linear scales
    - Financial data (returns vs prices)
    - Time scale conversions
    - Probability scale transformations
    """

    @staticmethod
    def log_returns_to_prices(
        log_returns: list[float], initial_price: float = 1.0
    ) -> list[float]:
        # TODO: Convert log returns to price series
        # LABELS: finance, conversion, log-returns
        """Convert log returns to price series."""
        raise NotImplementedError("Log return conversion not implemented")

    @staticmethod
    def scale_transform(
        data: list[float], from_scale: str, to_scale: str
    ) -> list[float]:
        # TODO: Transform data between different scales
        # LABELS: transformation, scaling
        """Transform data between scales."""
        raise NotImplementedError("Scale transformation not implemented")


if __name__ == "__main__":
    print("Utilities module loaded.")
    print("Contains TODO items for improving library usability and functionality.")
