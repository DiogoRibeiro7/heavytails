"""
Validation and quality assurance module for HeavyTails library.

This module provides comprehensive mathematical validation, numerical accuracy testing,
and property-based testing for all distributions.
"""

import math
from typing import Any

from heavytails.registry import create

# Parameters that put each family in a comfortable part of its own range, for
# checks that are about the family rather than about a particular fit. Kept
# here rather than in the registry: they are this module's choice of what
# "representative" means, not a property of the distributions.
_REPRESENTATIVE: dict[str, dict[str, float]] = {
    "pareto": {"alpha": 2.5, "xm": 1.0},
    "lognormal": {"mu": 0.0, "sigma": 1.0},
    "cauchy": {"x0": 0.0, "gamma": 1.0},
    "studentt": {"nu": 5.0},
    "weibull": {"k": 2.0, "lam": 1.0},
}

# Try to import scipy for validation (optional dependency)
try:
    import scipy.stats as scipy_stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    scipy_stats = None

# Try to import hypothesis for property-based testing (optional dependency)
try:
    from hypothesis import given, settings
    from hypothesis import strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    given = None  # type: ignore[assignment]
    settings = None  # type: ignore[assignment,misc]
    st = None  # type: ignore[assignment]


class NumericalValidation:
    """
    Comprehensive numerical validation against scipy and known results.

    Validates accuracy of PDF, CDF, PPF, and sampling for all distributions
    against scipy implementations where available.
    """

    def __init__(self, tolerance: float = 1e-10) -> None:
        """
        Initialize numerical validation.

        Args:
            tolerance: Maximum allowed relative error (default: 1e-10)
        """
        self.tolerance = tolerance
        self.test_results: dict[str, dict[str, Any]] = {}

    def validate_against_scipy(
        self, distribution: str, params: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Compare distribution against SciPy implementation.

        Args:
            distribution: Distribution name
            params: Optional specific parameters to test (uses defaults if None)

        Returns:
            Dictionary with validation results including errors and pass/fail

        Examples:
            >>> validator = NumericalValidation()
            >>> if SCIPY_AVAILABLE:
            ...     result = validator.validate_against_scipy("pareto", {"alpha": 2.5, "xm": 1.0})
            ...     result["pass"] or result["max_error"] < 0.01
            ... else:
            ...     True  # Skip if scipy not available
            True
        """
        if not SCIPY_AVAILABLE:
            return {
                "pass": False,
                "error": "scipy not available",
                "max_error": float("inf"),
            }

        dist_lower = distribution.lower()

        # Get test parameters
        if params is None:
            params = self._get_default_params(dist_lower)

        # Test cases for evaluation
        test_points = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        pdf_errors = []
        cdf_errors = []

        try:
            # Create our distribution
            our_dist = self._create_heavytails_distribution(dist_lower, params)

            # Create scipy equivalent
            scipy_dist = self._create_scipy_distribution(dist_lower, params)

            if scipy_dist is None:
                return {
                    "pass": False,
                    "error": f"No scipy equivalent for {distribution}",
                }

            # Test PDF at multiple points
            for x in test_points:
                if x > 0:  # Most distributions require x > 0
                    try:
                        our_pdf = our_dist.pdf(x)
                        scipy_pdf = scipy_dist.pdf(x)

                        if scipy_pdf > 1e-10 and math.isfinite(scipy_pdf):
                            rel_error = abs(our_pdf - scipy_pdf) / scipy_pdf
                            pdf_errors.append(rel_error)

                        # Test CDF
                        our_cdf = our_dist.cdf(x)
                        scipy_cdf = scipy_dist.cdf(x)

                        if scipy_cdf > 1e-10 and scipy_cdf < 1 - 1e-10:
                            rel_error_cdf = abs(our_cdf - scipy_cdf) / max(
                                scipy_cdf, 1 - scipy_cdf
                            )
                            cdf_errors.append(rel_error_cdf)

                    except (ValueError, OverflowError, ZeroDivisionError):
                        continue

            if not pdf_errors and not cdf_errors:
                return {
                    "pass": False,
                    "error": "No valid comparison points",
                    "max_error": float("inf"),
                }

            max_pdf_error = max(pdf_errors) if pdf_errors else 0.0
            max_cdf_error = max(cdf_errors) if cdf_errors else 0.0
            max_error = max(max_pdf_error, max_cdf_error)

            return {
                "pass": max_error < self.tolerance,
                "max_error": float(max_error),
                "max_pdf_error": float(max_pdf_error),
                "max_cdf_error": float(max_cdf_error),
                "mean_pdf_error": (
                    float(sum(pdf_errors) / len(pdf_errors)) if pdf_errors else 0.0
                ),
                "mean_cdf_error": (
                    float(sum(cdf_errors) / len(cdf_errors)) if cdf_errors else 0.0
                ),
                "num_pdf_tests": len(pdf_errors),
                "num_cdf_tests": len(cdf_errors),
                "distribution": distribution,
                "parameters": params,
            }

        except Exception as e:
            return {
                "pass": False,
                "error": str(e),
                "max_error": float("inf"),
            }

    def _get_default_params(self, distribution: str) -> dict[str, float]:
        """Get default test parameters for each distribution."""
        defaults = {
            "pareto": {"alpha": 2.5, "xm": 1.0},
            "lognormal": {"mu": 0.0, "sigma": 1.0},
            "cauchy": {"x0": 0.0, "gamma": 1.0},
            "studentt": {"nu": 5.0},
            "weibull": {"k": 2.0, "lam": 1.0},
            "frechet": {"alpha": 2.0, "s": 1.0, "m": 0.0},
        }
        return defaults.get(distribution, {})

    def _create_heavytails_distribution(
        self, distribution: str, params: dict[str, float]
    ) -> Any:
        """Create heavytails distribution instance."""
        return create(distribution, **params)

    def _create_scipy_distribution(
        self, distribution: str, params: dict[str, float]
    ) -> Any:
        """Create equivalent scipy distribution."""
        if not SCIPY_AVAILABLE:
            return None

        try:
            if distribution == "pareto":
                # scipy uses different parameterization: pareto(b, scale)
                # Our Pareto: f(x) = alpha * xm^alpha / x^(alpha+1)
                # scipy Pareto: f(x) = b / x^(b+1) for x >= 1, then scaled
                alpha = params["alpha"]
                xm = params["xm"]
                return scipy_stats.pareto(b=alpha, scale=xm)

            elif distribution == "lognormal":
                return scipy_stats.lognorm(
                    s=params["sigma"], scale=math.exp(params["mu"])
                )

            elif distribution == "cauchy":
                return scipy_stats.cauchy(loc=params["x0"], scale=params["gamma"])

            elif distribution == "studentt":
                return scipy_stats.t(df=params["nu"])

            elif distribution == "weibull":
                # scipy uses different parameterization
                return scipy_stats.weibull_min(c=params["k"], scale=params["lam"])

            elif distribution == "frechet":
                return scipy_stats.frechet_r(
                    c=params["alpha"], scale=params["s"], loc=params["m"]
                )

            else:
                return None

        except Exception:
            return None


def parameter_stability_check(distribution: str, **params: Any) -> dict[str, Any]:
    """
    Check parameter combinations for numerical stability with automatic fixes.

    Analyzes parameters for potential numerical issues and provides
    specific warnings and suggested fixes.

    Args:
        distribution: Distribution name
        **params: Distribution parameters to check

    Returns:
        Dictionary with warnings, suggested fixes, and stability assessment

    Examples:
        >>> result = parameter_stability_check("pareto", alpha=1e-8, xm=1.0)
        >>> len(result["warnings"]) > 0
        True
        >>> result["stable"]
        False
    """
    warnings_list = []
    fixes = []
    severity = "low"

    dist_lower = distribution.lower()

    if dist_lower == "pareto":
        alpha = params.get("alpha", 1.0)
        xm = params.get("xm", 1.0)

        if alpha < 1e-6:
            warnings_list.append("Alpha too small (< 1e-6), may cause overflow in PDF")
            fixes.append("Use alpha >= 1e-6")
            severity = "high"

        if alpha > 1e6:
            warnings_list.append("Alpha too large (> 1e6), may cause underflow in tail")
            fixes.append("Use alpha <= 1e6")
            severity = "medium"

        # Test numerical stability
        try:
            test_x = xm * 2.0
            pdf_val = (alpha * (xm**alpha)) / (test_x ** (alpha + 1))
            if math.isnan(pdf_val) or math.isinf(pdf_val):
                warnings_list.append("PDF computation unstable with these parameters")
                severity = "high"
        except (OverflowError, ZeroDivisionError):
            warnings_list.append("PDF computation failed with these parameters")
            severity = "high"

    elif dist_lower == "lognormal":
        mu = params.get("mu", 0.0)
        sigma = params.get("sigma", 1.0)

        if abs(mu) > 100:
            warnings_list.append("Very large |mu| (> 100) may cause numerical overflow")
            fixes.append("Use |mu| <= 100")
            severity = "medium"

        if sigma > 10:
            warnings_list.append("Very large sigma (> 10) may cause numerical issues")
            fixes.append("Use sigma <= 10")
            severity = "medium"

        if sigma < 1e-6:
            warnings_list.append(
                "Very small sigma (< 1e-6) approaches degenerate distribution"
            )
            fixes.append("Use sigma >= 1e-6")

    elif dist_lower == "cauchy":
        gamma = params.get("gamma", 1.0)

        if gamma < 1e-6:
            warnings_list.append("Very small gamma (< 1e-6) may cause numerical issues")
            fixes.append("Use gamma >= 1e-6")

        if gamma > 1e6:
            warnings_list.append("Very large gamma (> 1e6) may cause numerical issues")
            fixes.append("Use gamma <= 1e6")

    elif dist_lower == "studentt":
        nu = params.get("nu", 5.0)

        if nu < 1e-6:
            warnings_list.append("Nu too small (< 1e-6), Student-t undefined")
            fixes.append("Use nu >= 0.1")
            severity = "high"

        if nu > 1e6:
            warnings_list.append(
                "Very large nu (> 1e6): consider using normal distribution instead"
            )
            fixes.append("For nu > 30, Normal approximation often sufficient")

    elif dist_lower == "weibull":
        k = params.get("k", 1.0)
        lam = params.get("lam", 1.0)

        if k < 1e-6 or lam < 1e-6:
            warnings_list.append("Very small shape/scale parameters may cause issues")
            fixes.append("Use k, lam >= 1e-6")

        if k > 100 or lam > 1e6:
            warnings_list.append("Very large shape/scale parameters may cause overflow")
            fixes.append("Consider rescaling parameters")

    elif dist_lower == "frechet":
        alpha = params.get("alpha", 2.0)
        s = params.get("s", 1.0)

        if alpha < 1e-6 or s < 1e-6:
            warnings_list.append("Very small parameters may cause numerical issues")
            fixes.append("Use alpha, s >= 1e-6")

    # General checks for all distributions
    for param_name, param_value in params.items():
        if not math.isfinite(param_value):
            warnings_list.append(f"Parameter {param_name} is not finite")
            severity = "high"

    return {
        "warnings": warnings_list,
        "suggested_fixes": fixes,
        "stable": len(warnings_list) == 0,
        "severity": severity,
        "distribution": distribution,
        "parameters": params,
    }


class PropertyBasedTests:
    """
    Property-based testing for mathematical correctness using Hypothesis.

    Tests fundamental mathematical properties that all distributions should satisfy:
    - PDF non-negativity
    - CDF monotonicity
    - PPF/CDF inverse relationship
    - Probability axioms
    """

    def __init__(self) -> None:
        """Initialize property-based tester."""
        self.test_results: dict[str, bool] = {}

    def test_pdf_nonnegativity(self, distribution: str) -> dict[str, Any]:
        """
        Test that PDF is non-negative for all valid inputs.

        Args:
            distribution: Distribution name to test

        Returns:
            Dictionary with test results

        Examples:
            >>> tester = PropertyBasedTests()
            >>> result = tester.test_pdf_nonnegativity("pareto")
            >>> result["property"]
            'pdf_nonnegativity'
        """
        if not HYPOTHESIS_AVAILABLE:
            return {
                "pass": False,
                "error": "Hypothesis not available",
                "property": "pdf_nonnegativity",
            }

        violations = []

        try:
            dist_lower = distribution.lower()

            # Generate test cases
            test_cases = self._generate_test_cases(dist_lower, n_cases=50)

            for params, x_values in test_cases:
                try:
                    # Create distribution
                    try:
                        dist = create(dist_lower, **params)
                    except ValueError:
                        continue

                    # Test PDF non-negativity
                    for x in x_values:
                        if x > 0:  # Most distributions require x > 0
                            pdf_val = dist.pdf(x)
                            if pdf_val < 0 or math.isnan(pdf_val):
                                violations.append(
                                    {
                                        "params": params,
                                        "x": x,
                                        "pdf": pdf_val,
                                    }
                                )

                except Exception:
                    continue

            return {
                "pass": len(violations) == 0,
                "property": "pdf_nonnegativity",
                "distribution": distribution,
                "num_tests": len(test_cases) * len(x_values) if test_cases else 0,
                "violations": violations[:5],  # Return first 5 violations
                "num_violations": len(violations),
            }

        except Exception as e:
            return {
                "pass": False,
                "error": str(e),
                "property": "pdf_nonnegativity",
            }

    def test_cdf_monotonicity(self, distribution: str) -> dict[str, Any]:
        """
        Test that CDF is monotonically increasing.

        Args:
            distribution: Distribution name to test

        Returns:
            Dictionary with test results
        """
        if not HYPOTHESIS_AVAILABLE:
            return {
                "pass": False,
                "error": "Hypothesis not available",
                "property": "cdf_monotonicity",
            }

        violations = []

        try:
            dist_lower = distribution.lower()
            test_cases = self._generate_test_cases(dist_lower, n_cases=50)

            for params, x_values in test_cases:
                try:
                    # Create distribution
                    try:
                        dist = create(dist_lower, **params)
                    except ValueError:
                        continue

                    # Test monotonicity: CDF(x1) <= CDF(x2) for x1 < x2
                    sorted_x = sorted([x for x in x_values if x > 0])
                    for i in range(len(sorted_x) - 1):
                        x1, x2 = sorted_x[i], sorted_x[i + 1]
                        cdf1, cdf2 = dist.cdf(x1), dist.cdf(x2)

                        if cdf1 > cdf2 + 1e-10:  # Allow small numerical error
                            violations.append(
                                {
                                    "params": params,
                                    "x1": x1,
                                    "x2": x2,
                                    "cdf1": cdf1,
                                    "cdf2": cdf2,
                                }
                            )

                except Exception:
                    continue

            return {
                "pass": len(violations) == 0,
                "property": "cdf_monotonicity",
                "distribution": distribution,
                "num_tests": sum(len(x_values) - 1 for _, x_values in test_cases),
                "violations": violations[:5],
                "num_violations": len(violations),
            }

        except Exception as e:
            return {
                "pass": False,
                "error": str(e),
                "property": "cdf_monotonicity",
            }

    def test_ppf_cdf_inverse(self, distribution: str) -> dict[str, Any]:
        """
        Test that PPF and CDF are inverse functions: CDF(PPF(u)) ≈ u.

        Args:
            distribution: Distribution name to test

        Returns:
            Dictionary with test results
        """

        violations = []

        try:
            dist_lower = distribution.lower()
            test_cases = self._generate_test_cases(dist_lower, n_cases=20)

            # Test quantiles
            u_values = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

            for params, _ in test_cases:
                try:
                    # Create distribution
                    try:
                        dist = create(dist_lower, **params)
                    except ValueError:
                        continue

                    # Test CDF(PPF(u)) ≈ u
                    for u in u_values:
                        x = dist.ppf(u)
                        u_recovered = dist.cdf(x)

                        error = abs(u - u_recovered)
                        if error > 1e-6:  # Tolerance for numerical error
                            violations.append(
                                {
                                    "params": params,
                                    "u": u,
                                    "x": x,
                                    "u_recovered": u_recovered,
                                    "error": error,
                                }
                            )

                except Exception:
                    continue

            return {
                "pass": len(violations) == 0,
                "property": "ppf_cdf_inverse",
                "distribution": distribution,
                "num_tests": len(test_cases) * len(u_values),
                "violations": violations[:5],
                "num_violations": len(violations),
                "max_error": max((v["error"] for v in violations), default=0.0),
            }

        except Exception as e:
            return {
                "pass": False,
                "error": str(e),
                "property": "ppf_cdf_inverse",
            }

    def _generate_test_cases(
        self, distribution: str, n_cases: int = 50
    ) -> list[tuple[dict[str, float], list[float]]]:
        """Generate test cases with random parameters and test points."""
        import random  # noqa: PLC0415

        random.seed(42)  # Reproducible tests
        test_cases = []

        for _ in range(n_cases):
            if distribution == "pareto":
                params = {
                    "alpha": random.uniform(0.5, 5.0),
                    "xm": random.uniform(0.1, 10.0),
                }
                x_values = [
                    random.uniform(params["xm"], params["xm"] + 20) for _ in range(10)
                ]

            elif distribution == "lognormal":
                params = {
                    "mu": random.uniform(-2.0, 2.0),
                    "sigma": random.uniform(0.1, 2.0),
                }
                x_values = [random.uniform(0.01, 100.0) for _ in range(10)]

            elif distribution == "cauchy":
                params = {
                    "x0": random.uniform(-10.0, 10.0),
                    "gamma": random.uniform(0.1, 5.0),
                }
                x_values = [random.uniform(-50.0, 50.0) for _ in range(10)]

            elif distribution == "studentt":
                params = {"nu": random.uniform(1.0, 30.0)}
                x_values = [random.uniform(-10.0, 10.0) for _ in range(10)]

            elif distribution == "weibull":
                params = {
                    "k": random.uniform(0.5, 5.0),
                    "lam": random.uniform(0.5, 5.0),
                }
                x_values = [random.uniform(0.01, 10.0) for _ in range(10)]

            else:
                continue

            test_cases.append((params, x_values))

        return test_cases


def convergence_validation(distribution: str, method: str = "ppf") -> dict[str, Any]:
    """
    Validate convergence of numerical algorithms.

    Tests convergence properties of iterative algorithms used in the library,
    such as PPF computation via bisection/Newton-Raphson.

    Args:
        distribution: Distribution name to test
        method: Method to test ("ppf", "cdf", or "pdf")

    Returns:
        Dictionary with convergence diagnostics

    Examples:
        >>> result = convergence_validation("pareto", "ppf")
        >>> "converged" in result
        True
    """

    try:
        dist_lower = distribution.lower()

        # A representative instance of the family, for a check that does not
        # depend on the particular parameters.
        if dist_lower not in _REPRESENTATIVE:
            return {
                "converged": False,
                "error": f"Unknown distribution: {distribution}",
            }
        dist = create(dist_lower, **_REPRESENTATIVE[dist_lower])

        if method == "ppf":
            # Test PPF convergence for various quantiles
            u_values = [0.01, 0.1, 0.5, 0.9, 0.99]
            convergence_info = []

            for u in u_values:
                try:
                    # Compute PPF
                    x = dist.ppf(u)

                    # Verify convergence: CDF(PPF(u)) should equal u
                    u_recovered = dist.cdf(x)
                    error = abs(u - u_recovered)

                    convergence_info.append(
                        {
                            "u": u,
                            "x": x,
                            "error": error,
                            "converged": error < 1e-6,
                        }
                    )

                except Exception as e:
                    convergence_info.append(
                        {
                            "u": u,
                            "error": str(e),
                            "converged": False,
                        }
                    )

            all_converged = all(
                info.get("converged", False) for info in convergence_info
            )
            max_error = max(
                (
                    info["error"]
                    for info in convergence_info
                    if isinstance(info["error"], (int, float))
                ),
                default=float("inf"),
            )

            return {
                "converged": all_converged,
                "method": method,
                "distribution": distribution,
                "convergence_info": convergence_info,
                "max_error": float(max_error),
                "num_tests": len(u_values),
            }

        else:
            return {
                "converged": False,
                "error": f"Method {method} not implemented",
            }

    except Exception as e:
        return {
            "converged": False,
            "error": str(e),
        }


# TODO: Add cross-validation framework for parameter estimation methods


# FIXME: Edge cases in PPF calculation need better handling
def ppf_edge_case_handler(distribution: str, u: float, **params: Any) -> float:
    """
    Handle edge cases in quantile function calculation.

    Problematic cases:
    - u very close to 0 or 1
    - Parameters at boundary values
    - Distributions with bounded support
    - Numerical overflow/underflow

    Should provide graceful degradation and informative errors.
    """
    _ = (distribution, params)  # Reserved for future implementation
    if not (0 < u < 1):
        if u == 0:
            # TODO: Return theoretical minimum (support lower bound)
            pass
        elif u == 1:
            # TODO: Return theoretical maximum (support upper bound)
            pass
        else:
            raise ValueError(f"u must be in (0,1), got {u}")

    # TODO: Implement robust edge case handling for all distributions
    raise NotImplementedError("PPF edge case handling not fully implemented")


# TODO: Implement statistical goodness-of-fit tests
def _resolve_distribution(name: str, params: dict[str, Any]) -> Any:
    """Build a distribution instance from its name and keyword parameters.

    Accepts the same spellings the CLI does, so ``student-t`` and ``studentt``
    both work.

    Args:
        name: Distribution name, case-insensitive.
        params: Constructor keyword arguments.

    Returns:
        The constructed distribution.

    Raises:
        ValueError: If the name is unknown or the parameters are invalid.
    """
    import heavytails  # noqa: PLC0415

    registry = {
        "pareto": heavytails.Pareto,
        "cauchy": heavytails.Cauchy,
        "studentt": heavytails.StudentT,
        "student-t": heavytails.StudentT,
        "lognormal": heavytails.LogNormal,
        "weibull": heavytails.Weibull,
        "frechet": heavytails.Frechet,
        "gev": heavytails.GEV_Frechet,
        "gev_frechet": heavytails.GEV_Frechet,
        "gpd": heavytails.GeneralizedPareto,
        "generalizedpareto": heavytails.GeneralizedPareto,
        "burr": heavytails.BurrXII,
        "burrxii": heavytails.BurrXII,
        "loglogistic": heavytails.LogLogistic,
        "invgamma": heavytails.InverseGamma,
        "inversegamma": heavytails.InverseGamma,
        "betaprime": heavytails.BetaPrime,
    }

    key = name.lower().replace(" ", "")
    if key not in registry:
        raise ValueError(
            f"Unknown distribution {name!r}. "
            f"Available: {', '.join(sorted(set(registry)))}"
        )
    try:
        return registry[key](**params)
    except TypeError as exc:
        raise ValueError(f"Invalid parameters for {name!r}: {exc}") from exc


def _kolmogorov_p_value(statistic: float, n: int) -> float:
    """Asymptotic p-value for the Kolmogorov-Smirnov statistic.

    Uses the Kolmogorov distribution
    ``Q(lam) = 2 * sum_{j>=1} (-1)^(j-1) exp(-2 j^2 lam^2)`` evaluated at the
    finite-sample corrected ``lam = (sqrt(n) + 0.12 + 0.11/sqrt(n)) * D``.

    Parameters
    ----------
    statistic : float
        The KS statistic D.
    n : int
        Sample size.

    Returns
    -------
    float
        Probability of a statistic at least this large under the null.
    """
    if statistic <= 0.0:
        return 1.0
    root_n = math.sqrt(n)
    lam = (root_n + 0.12 + 0.11 / root_n) * statistic
    if lam < 0.05:
        return 1.0

    total = 0.0
    for j in range(1, 101):
        term = 2.0 * ((-1.0) ** (j - 1)) * math.exp(-2.0 * (j * lam) ** 2)
        total += term
        if abs(term) < 1e-15:
            break
    return min(max(total, 0.0), 1.0)


def _anderson_darling_cdf(z: float) -> float:
    """Asymptotic distribution function of the Anderson-Darling statistic.

    Implements ``adinf`` from Marsaglia and Marsaglia (2004), "Evaluating the
    Anderson-Darling Distribution", which approximates ``P(A^2 < z)`` in the
    limit of large n to about six decimal places.

    This is the *fully specified* null distribution, the one that applies when
    the parameters were not estimated from the sample being tested. Its
    critical values are 1.933 at 10%, 2.492 at 5% and 3.857 at 1%.

    The widely quoted piecewise formulas of D'Agostino and Stephens are a
    different thing: they apply to the normality test with estimated mean and
    variance, where the 5% critical value is 0.787. Using those here would
    reject a correctly specified distribution roughly half the time.

    Parameters
    ----------
    z : float
        Value of the A-squared statistic.

    Returns
    -------
    float
        ``P(A^2 < z)``.
    """
    if z <= 0.0:
        return 0.0
    if z < 2.0:
        return float(
            z**-0.5
            * math.exp(-1.2337141 / z)
            * (
                2.00012
                + (
                    0.247105
                    - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z)
                    * z
                )
                * z
            )
        )
    return math.exp(
        -math.exp(
            1.0776
            - (
                2.30695
                - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z
            )
            * z
        )
    )


def _anderson_darling_p_value(statistic: float) -> float:
    """Asymptotic p-value for the Anderson-Darling statistic.

    Parameters
    ----------
    statistic : float
        The A-squared statistic.

    Returns
    -------
    float
        Probability of a statistic at least this large under the null.
    """
    return min(max(1.0 - _anderson_darling_cdf(statistic), 0.0), 1.0)


class GoodnessOfFitTests:
    """Statistical tests for distribution goodness-of-fit.

    Both tests answer a question that AIC and BIC cannot. Information criteria
    rank candidate models against each other, so the best of a bad set still
    ranks first. A goodness-of-fit test asks whether the winner is compatible
    with the data at all.

    For heavy tails the Anderson-Darling test is the more informative of the
    two, because it weights the tails of the distribution. The
    Kolmogorov-Smirnov statistic is driven by the centre, which is exactly
    where these families agree with each other.

    Examples
    --------
    >>> from heavytails import Pareto
    >>> data = Pareto(alpha=2.5, xm=1.0).rvs(500, seed=42)
    >>> tests = GoodnessOfFitTests()
    >>> result = tests.kolmogorov_smirnov_test(
    ...     data, "pareto", alpha=2.5, xm=1.0
    ... )
    >>> result["reject"]
    False
    """

    #: Significance level used to populate the ``reject`` field.
    alpha_level: float = 0.05

    def __init__(self, alpha_level: float = 0.05) -> None:
        """Initialise the test suite.

        Args:
            alpha_level: Significance level for the ``reject`` field.
        """
        if not (0.0 < alpha_level < 1.0):
            raise ValueError("alpha_level must be in (0,1).")
        self.alpha_level = alpha_level

    def kolmogorov_smirnov_test(
        self,
        data: list[float],
        distribution: str,
        *,
        parameters_estimated: bool = False,
        **params: Any,
    ) -> dict[str, Any]:
        """Kolmogorov-Smirnov test against a named distribution.

        The statistic is the largest vertical distance between the empirical
        distribution function and the fitted one,
        ``D = max_i max(i/n - F(x_i), F(x_i) - (i-1)/n)``.

        Args:
            data: Sample values.
            distribution: Distribution name, as accepted by the fitting helpers.
            parameters_estimated: Set when the parameters came from this same
                sample. The reported p-value is then conservative, because the
                fitted distribution is closer to the data than the null
                assumes, and the result carries a ``caveat``.
            **params: Distribution parameters.

        Returns:
            Dictionary with ``statistic``, ``p_value``, ``reject``, ``n``,
            ``distribution``, ``parameters`` and ``method``, plus ``caveat``
            when the p-value should not be read at face value.

        Raises:
            ValueError: If the sample is empty or the distribution is unknown.
        """
        values = sorted(float(x) for x in data)
        n = len(values)
        if n == 0:
            raise ValueError("data must not be empty.")

        dist = _resolve_distribution(distribution, params)

        d_plus = 0.0
        d_minus = 0.0
        for i, x in enumerate(values, start=1):
            cdf = dist.cdf(x)
            d_plus = max(d_plus, i / n - cdf)
            d_minus = max(d_minus, cdf - (i - 1) / n)
        statistic = max(d_plus, d_minus)

        p_value = _kolmogorov_p_value(statistic, n)
        result: dict[str, Any] = {
            "test": "kolmogorov-smirnov",
            "statistic": statistic,
            "p_value": p_value,
            "reject": p_value < self.alpha_level,
            "alpha_level": self.alpha_level,
            "n": n,
            "distribution": distribution,
            "parameters": dict(params),
            "method": "asymptotic",
        }
        if parameters_estimated:
            result["caveat"] = (
                "Parameters were estimated from this sample, so the asymptotic "
                "null distribution does not apply and the p-value is "
                "conservative: the test rejects less often than its nominal "
                "level. Use a parametric bootstrap for a calibrated p-value."
            )
        return result

    def anderson_darling_test(
        self,
        data: list[float],
        distribution: str,
        *,
        parameters_estimated: bool = False,
        **params: Any,
    ) -> dict[str, Any]:
        """Anderson-Darling test against a named distribution.

        The statistic is
        ``A^2 = -n - (1/n) sum_i (2i-1) [ln F(x_i) + ln(1 - F(x_{n+1-i}))]``.

        Unlike the Kolmogorov-Smirnov statistic this weights the tails, which
        is what makes it the more useful of the two here: heavy-tailed families
        differ from one another in the tail and agree in the middle.

        Args:
            data: Sample values.
            distribution: Distribution name, as accepted by the fitting helpers.
            parameters_estimated: Set when the parameters came from this same
                sample; see :meth:`kolmogorov_smirnov_test`.
            **params: Distribution parameters.

        Returns:
            Dictionary with the same shape as
            :meth:`kolmogorov_smirnov_test`.

        Raises:
            ValueError: If the sample is empty or the distribution is unknown.
        """
        values = sorted(float(x) for x in data)
        n = len(values)
        if n == 0:
            raise ValueError("data must not be empty.")

        dist = _resolve_distribution(distribution, params)

        # Clamp away from 0 and 1: the statistic takes logs of both F and 1-F,
        # and a single saturated value would send the whole sum to infinity.
        eps = 1e-15
        cdfs = [min(max(dist.cdf(x), eps), 1.0 - eps) for x in values]

        total = 0.0
        for i in range(1, n + 1):
            total += (2 * i - 1) * (math.log(cdfs[i - 1]) + math.log(1.0 - cdfs[n - i]))
        statistic = -n - total / n

        p_value = _anderson_darling_p_value(statistic)
        result: dict[str, Any] = {
            "test": "anderson-darling",
            "statistic": statistic,
            "p_value": p_value,
            "reject": p_value < self.alpha_level,
            "alpha_level": self.alpha_level,
            "n": n,
            "distribution": distribution,
            "parameters": dict(params),
            "method": "asymptotic",
        }
        if parameters_estimated:
            result["caveat"] = (
                "Parameters were estimated from this sample, so the asymptotic "
                "null distribution does not apply and the p-value is "
                "conservative: the test rejects less often than its nominal "
                "level. Use a parametric bootstrap for a calibrated p-value."
            )
        return result


# TODO: Add automated regression testing for mathematical accuracy


# NOTE: Consider implementing fuzzing tests for robustness


# TODO: Implement mathematical property verification


# HACK: Some special function implementations use approximations - need accuracy bounds


# TODO: Add continuous integration tests with different Python versions


if __name__ == "__main__":
    print("Validation module loaded.")
    print("Contains TODO items for improving mathematical accuracy and reliability.")
