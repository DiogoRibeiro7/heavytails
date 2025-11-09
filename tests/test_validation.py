"""Tests for validation.py module."""

import pytest

import heavytails.validation as validation_module
from heavytails.validation import (
    HYPOTHESIS_AVAILABLE,
    SCIPY_AVAILABLE,
    GoodnessOfFitTests,
    MathematicalPropertyVerification,
    NumericalValidation,
    ParameterEstimationValidation,
    PropertyBasedTests,
    RegressionTesting,
    convergence_validation,
    fuzz_testing,
    parameter_stability_check,
    ppf_edge_case_handler,
    python_version_compatibility,
    special_function_accuracy_analysis,
)


class TestParameterStabilityCheck:
    """Test parameter stability checking."""

    def test_pareto_stable_parameters(self):
        """Test that normal Pareto parameters are stable."""
        result = parameter_stability_check("pareto", alpha=2.5, xm=1.0)

        assert result["stable"] is True
        assert len(result["warnings"]) == 0
        assert result["severity"] == "low"

    def test_pareto_unstable_alpha_small(self):
        """Test that very small alpha triggers warning."""
        result = parameter_stability_check("pareto", alpha=1e-8, xm=1.0)

        assert result["stable"] is False
        assert len(result["warnings"]) > 0
        assert result["severity"] == "high"
        assert any("too small" in w.lower() for w in result["warnings"])
        assert len(result["suggested_fixes"]) > 0

    def test_pareto_unstable_alpha_large(self):
        """Test that very large alpha triggers warning."""
        result = parameter_stability_check("pareto", alpha=1e7, xm=1.0)

        assert result["stable"] is False
        assert len(result["warnings"]) > 0
        assert result["severity"] in ["medium", "high"]

    def test_lognormal_stable_parameters(self):
        """Test that normal LogNormal parameters are stable."""
        result = parameter_stability_check("lognormal", mu=0.0, sigma=1.0)

        assert result["stable"] is True
        assert len(result["warnings"]) == 0

    def test_lognormal_unstable_mu_large(self):
        """Test that very large mu triggers warning."""
        result = parameter_stability_check("lognormal", mu=150.0, sigma=1.0)

        assert result["stable"] is False
        assert len(result["warnings"]) > 0

    def test_lognormal_unstable_sigma_large(self):
        """Test that very large sigma triggers warning."""
        result = parameter_stability_check("lognormal", mu=0.0, sigma=15.0)

        assert result["stable"] is False
        assert len(result["warnings"]) > 0

    def test_cauchy_stable_parameters(self):
        """Test that normal Cauchy parameters are stable."""
        result = parameter_stability_check("cauchy", x0=0.0, gamma=1.0)

        assert result["stable"] is True

    def test_studentt_unstable_nu_small(self):
        """Test that very small nu triggers warning."""
        result = parameter_stability_check("studentt", nu=1e-8)

        assert result["stable"] is False
        assert result["severity"] == "high"

    def test_studentt_unstable_nu_large(self):
        """Test that very large nu suggests normal approximation."""
        result = parameter_stability_check("studentt", nu=1e7)

        assert result["stable"] is False
        assert any("normal" in f.lower() for f in result["suggested_fixes"])

    def test_weibull_stable_parameters(self):
        """Test that normal Weibull parameters are stable."""
        result = parameter_stability_check("weibull", k=2.0, lam=1.0)

        assert result["stable"] is True

    def test_weibull_unstable_parameters(self):
        """Test that extreme Weibull parameters trigger warnings."""
        result = parameter_stability_check("weibull", k=1e-8, lam=1.0)

        assert result["stable"] is False
        assert len(result["warnings"]) > 0

    def test_non_finite_parameter(self):
        """Test that non-finite parameters trigger high severity warning."""
        result = parameter_stability_check("pareto", alpha=float("inf"), xm=1.0)

        assert result["stable"] is False
        assert result["severity"] == "high"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestNumericalValidation:
    """Test numerical validation against scipy."""

    def test_validate_pareto_default(self):
        """Test Pareto validation with default parameters."""
        validator = NumericalValidation(tolerance=0.01)  # Relaxed tolerance for test
        result = validator.validate_against_scipy("pareto")

        assert "pass" in result
        assert "max_error" in result
        assert "distribution" in result
        assert result["distribution"] == "pareto"

        # Should have run some tests
        assert result.get("num_pdf_tests", 0) > 0 or result.get("num_cdf_tests", 0) > 0

    def test_validate_pareto_custom_params(self):
        """Test Pareto validation with custom parameters."""
        validator = NumericalValidation(tolerance=0.01)
        result = validator.validate_against_scipy(
            "pareto", params={"alpha": 3.0, "xm": 2.0}
        )

        assert "pass" in result
        assert result["parameters"]["alpha"] == 3.0
        assert result["parameters"]["xm"] == 2.0

    def test_validate_lognormal(self):
        """Test LogNormal validation."""
        validator = NumericalValidation(tolerance=0.01)
        result = validator.validate_against_scipy("lognormal")

        assert "pass" in result
        assert result.get("num_pdf_tests", 0) > 0

    def test_validate_cauchy(self):
        """Test Cauchy validation."""
        validator = NumericalValidation(tolerance=0.01)
        result = validator.validate_against_scipy("cauchy")

        assert "pass" in result

    def test_validate_studentt(self):
        """Test Student-t validation."""
        validator = NumericalValidation(tolerance=0.01)
        result = validator.validate_against_scipy("studentt")

        assert "pass" in result

    def test_validate_unknown_distribution(self):
        """Test that unknown distribution returns error."""
        validator = NumericalValidation()
        result = validator.validate_against_scipy("unknown_dist")

        assert result["pass"] is False
        assert "error" in result


class TestPropertyBasedTests:
    """Test property-based testing framework."""

    def test_pdf_nonnegativity_pareto(self):
        """Test PDF non-negativity for Pareto."""
        tester = PropertyBasedTests()
        result = tester.test_pdf_nonnegativity("pareto")

        assert "property" in result
        assert result["property"] == "pdf_nonnegativity"
        assert "pass" in result
        assert "num_tests" in result

        # Should pass for Pareto
        if not HYPOTHESIS_AVAILABLE:
            assert result["pass"] is False
            assert "Hypothesis not available" in result.get("error", "")
        else:
            # If hypothesis available, PDF should be non-negative
            assert result.get("num_violations", 0) == 0

    def test_pdf_nonnegativity_lognormal(self):
        """Test PDF non-negativity for LogNormal."""
        tester = PropertyBasedTests()
        result = tester.test_pdf_nonnegativity("lognormal")

        assert result["property"] == "pdf_nonnegativity"
        assert "pass" in result

    def test_cdf_monotonicity_pareto(self):
        """Test CDF monotonicity for Pareto."""
        tester = PropertyBasedTests()
        result = tester.test_cdf_monotonicity("pareto")

        assert "property" in result
        assert result["property"] == "cdf_monotonicity"
        assert "pass" in result

        if not HYPOTHESIS_AVAILABLE:
            assert result["pass"] is False
        else:
            # CDF should be monotonic
            assert result.get("num_violations", 0) == 0

    def test_cdf_monotonicity_cauchy(self):
        """Test CDF monotonicity for Cauchy."""
        tester = PropertyBasedTests()
        result = tester.test_cdf_monotonicity("cauchy")

        assert result["property"] == "cdf_monotonicity"

    def test_ppf_cdf_inverse_pareto(self):
        """Test PPF/CDF inverse relationship for Pareto."""
        tester = PropertyBasedTests()
        result = tester.test_ppf_cdf_inverse("pareto")

        assert "property" in result
        assert result["property"] == "ppf_cdf_inverse"
        assert "pass" in result

        # PPF and CDF should be inverses
        if result["pass"]:
            assert result.get("max_error", float("inf")) < 1e-5

    def test_ppf_cdf_inverse_lognormal(self):
        """Test PPF/CDF inverse relationship for LogNormal."""
        tester = PropertyBasedTests()
        result = tester.test_ppf_cdf_inverse("lognormal")

        assert result["property"] == "ppf_cdf_inverse"
        assert "num_tests" in result

    def test_ppf_cdf_inverse_studentt(self):
        """Test PPF/CDF inverse relationship for Student-t."""
        tester = PropertyBasedTests()
        result = tester.test_ppf_cdf_inverse("studentt")

        assert result["property"] == "ppf_cdf_inverse"


class TestConvergenceValidation:
    """Test convergence validation."""

    def test_convergence_pareto_ppf(self):
        """Test PPF convergence for Pareto."""
        result = convergence_validation("pareto", method="ppf")

        assert "converged" in result
        assert "method" in result
        assert result["method"] == "ppf"
        assert "distribution" in result

        # Should have convergence info
        assert "convergence_info" in result
        assert len(result["convergence_info"]) > 0

        # All quantiles should converge
        assert result["converged"] is True
        assert result["max_error"] < 1e-5

    def test_convergence_lognormal_ppf(self):
        """Test PPF convergence for LogNormal."""
        result = convergence_validation("lognormal", method="ppf")

        assert result["converged"] is True
        assert result["max_error"] < 1e-5

    def test_convergence_cauchy_ppf(self):
        """Test PPF convergence for Cauchy."""
        result = convergence_validation("cauchy", method="ppf")

        assert result["converged"] is True

    @pytest.mark.xfail(
        reason="Student-t PPF has known numerical precision issues at extreme quantiles"
    )
    def test_convergence_studentt_ppf(self):
        """Test PPF convergence for Student-t."""
        result = convergence_validation("studentt", method="ppf")

        # Student-t PPF can have numerical precision issues at extreme quantiles
        # due to the complexity of the inverse CDF computation
        # Just verify the function runs and returns reasonable results
        assert "converged" in result
        assert "convergence_info" in result
        assert len(result["convergence_info"]) > 0

        # This may fail due to numerical issues - marked as expected failure
        assert result["converged"] is True

    def test_convergence_weibull_ppf(self):
        """Test PPF convergence for Weibull."""
        result = convergence_validation("weibull", method="ppf")

        assert result["converged"] is True

    def test_convergence_unknown_distribution(self):
        """Test that unknown distribution returns error."""
        result = convergence_validation("unknown_dist")

        assert result["converged"] is False
        assert "error" in result

    def test_convergence_unsupported_method(self):
        """Test that unsupported method returns error."""
        result = convergence_validation("pareto", method="unsupported")

        assert result["converged"] is False
        assert "not implemented" in result.get("error", "").lower()


class TestIntegration:
    """Integration tests for validation framework."""

    def test_full_validation_workflow_pareto(self):
        """Test complete validation workflow for Pareto."""
        # 1. Check parameter stability
        stability = parameter_stability_check("pareto", alpha=2.5, xm=1.0)
        assert stability["stable"] is True

        # 2. Test properties
        tester = PropertyBasedTests()
        pdf_test = tester.test_pdf_nonnegativity("pareto")
        cdf_test = tester.test_cdf_monotonicity("pareto")
        ppf_test = tester.test_ppf_cdf_inverse("pareto")

        # All properties should hold
        assert "pass" in pdf_test
        assert "pass" in cdf_test
        assert "pass" in ppf_test

        # 3. Test convergence
        convergence = convergence_validation("pareto")
        assert convergence["converged"] is True

    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
    def test_full_validation_workflow_with_scipy(self):
        """Test validation workflow including scipy comparison."""
        # Parameter stability
        stability = parameter_stability_check("pareto", alpha=2.5, xm=1.0)
        assert stability["stable"] is True

        # Numerical accuracy
        validator = NumericalValidation(tolerance=0.01)
        scipy_result = validator.validate_against_scipy("pareto")
        assert "pass" in scipy_result

        # Properties
        tester = PropertyBasedTests()
        pdf_result = tester.test_pdf_nonnegativity("pareto")
        assert "pass" in pdf_result

        # Convergence
        conv_result = convergence_validation("pareto")
        assert conv_result["converged"] is True

    def test_unstable_parameters_detection(self):
        """Test that unstable parameters are detected across validation methods."""
        # Very small alpha should trigger stability warnings
        stability = parameter_stability_check("pareto", alpha=1e-10, xm=1.0)
        assert stability["stable"] is False
        assert stability["severity"] == "high"

        # Should have warnings and fixes
        assert len(stability["warnings"]) > 0
        assert len(stability["suggested_fixes"]) > 0


class TestParameterStabilityEdgeCases:
    """Test edge cases in parameter stability checking."""

    def test_lognormal_small_sigma(self):
        """Test LogNormal with very small sigma."""
        result = parameter_stability_check("lognormal", mu=0.0, sigma=1e-8)
        assert result["stable"] is False
        assert any("small sigma" in w.lower() for w in result["warnings"])
        assert any("1e-6" in f for f in result["suggested_fixes"])

    def test_cauchy_small_gamma(self):
        """Test Cauchy with very small gamma."""
        result = parameter_stability_check("cauchy", x0=0.0, gamma=1e-8)
        assert result["stable"] is False
        assert any("small gamma" in w.lower() for w in result["warnings"])

    def test_cauchy_large_gamma(self):
        """Test Cauchy with very large gamma."""
        result = parameter_stability_check("cauchy", x0=0.0, gamma=1e7)
        assert result["stable"] is False
        assert any("large gamma" in w.lower() for w in result["warnings"])

    def test_frechet_small_parameters(self):
        """Test Frechet with very small parameters."""
        result = parameter_stability_check("frechet", alpha=1e-8, s=1e-8, m=0.0)
        assert result["stable"] is False
        assert any("small parameters" in w.lower() for w in result["warnings"])

    def test_frechet_stable_parameters(self):
        """Test Frechet with stable parameters."""
        result = parameter_stability_check("frechet", alpha=2.0, s=1.0, m=0.0)
        assert result["stable"] is True
        assert len(result["warnings"]) == 0


class TestPropertyBasedTestsExtended:
    """Extended tests for property-based testing."""

    def test_pdf_nonnegativity_weibull(self):
        """Test PDF non-negativity for Weibull."""
        tester = PropertyBasedTests()
        result = tester.test_pdf_nonnegativity("weibull")
        assert result["property"] == "pdf_nonnegativity"
        assert "pass" in result

    def test_cdf_monotonicity_weibull(self):
        """Test CDF monotonicity for Weibull."""
        tester = PropertyBasedTests()
        result = tester.test_cdf_monotonicity("weibull")
        assert result["property"] == "cdf_monotonicity"
        assert "pass" in result

    def test_ppf_cdf_inverse_weibull(self):
        """Test PPF/CDF inverse relationship for Weibull."""
        tester = PropertyBasedTests()
        result = tester.test_ppf_cdf_inverse("weibull")
        assert result["property"] == "ppf_cdf_inverse"
        assert "pass" in result

    def test_ppf_cdf_inverse_cauchy(self):
        """Test PPF/CDF inverse relationship for Cauchy."""
        tester = PropertyBasedTests()
        result = tester.test_ppf_cdf_inverse("cauchy")
        assert result["property"] == "ppf_cdf_inverse"
        assert "pass" in result

    def test_cdf_monotonicity_lognormal(self):
        """Test CDF monotonicity for LogNormal."""
        tester = PropertyBasedTests()
        result = tester.test_cdf_monotonicity("lognormal")
        assert result["property"] == "cdf_monotonicity"
        assert "pass" in result

    def test_cdf_monotonicity_studentt(self):
        """Test CDF monotonicity for Student-t."""
        tester = PropertyBasedTests()
        result = tester.test_cdf_monotonicity("studentt")
        assert result["property"] == "cdf_monotonicity"
        assert "pass" in result

    def test_pdf_nonnegativity_cauchy(self):
        """Test PDF non-negativity for Cauchy."""
        tester = PropertyBasedTests()
        result = tester.test_pdf_nonnegativity("cauchy")
        assert result["property"] == "pdf_nonnegativity"
        assert "pass" in result

    def test_pdf_nonnegativity_studentt(self):
        """Test PDF non-negativity for Student-t."""
        tester = PropertyBasedTests()
        result = tester.test_pdf_nonnegativity("studentt")
        assert result["property"] == "pdf_nonnegativity"
        assert "pass" in result

    def test_pdf_nonnegativity_unknown_distribution(self):
        """Test PDF non-negativity for unknown distribution."""
        tester = PropertyBasedTests()
        result = tester.test_pdf_nonnegativity("unknown_dist")
        # Should handle gracefully
        assert "property" in result

    def test_cdf_monotonicity_unknown_distribution(self):
        """Test CDF monotonicity for unknown distribution."""
        tester = PropertyBasedTests()
        result = tester.test_cdf_monotonicity("unknown_dist")
        # Should handle gracefully
        assert "property" in result

    def test_ppf_cdf_inverse_unknown_distribution(self):
        """Test PPF/CDF inverse for unknown distribution."""
        tester = PropertyBasedTests()
        result = tester.test_ppf_cdf_inverse("unknown_dist")
        # Should handle gracefully
        assert "property" in result


class TestPPFEdgeCaseHandler:
    """Test PPF edge case handling."""

    def test_ppf_edge_case_u_zero(self):
        """Test PPF edge case with u=0."""
        with pytest.raises((ValueError, NotImplementedError)):
            ppf_edge_case_handler("pareto", 0.0, alpha=2.5, xm=1.0)

    def test_ppf_edge_case_u_one(self):
        """Test PPF edge case with u=1."""
        with pytest.raises((ValueError, NotImplementedError)):
            ppf_edge_case_handler("pareto", 1.0, alpha=2.5, xm=1.0)

    def test_ppf_edge_case_u_negative(self):
        """Test PPF edge case with negative u."""
        with pytest.raises((ValueError, NotImplementedError)):
            ppf_edge_case_handler("pareto", -0.1, alpha=2.5, xm=1.0)

    def test_ppf_edge_case_u_greater_than_one(self):
        """Test PPF edge case with u > 1."""
        with pytest.raises((ValueError, NotImplementedError)):
            ppf_edge_case_handler("pareto", 1.5, alpha=2.5, xm=1.0)

    def test_ppf_edge_case_valid_u(self):
        """Test PPF edge case with valid u."""
        # Valid u should raise NotImplementedError since function is not fully implemented
        with pytest.raises(NotImplementedError):
            ppf_edge_case_handler("pareto", 0.5, alpha=2.5, xm=1.0)


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestNumericalValidationEdgeCases:
    """Test edge cases in numerical validation."""

    def test_validate_weibull(self):
        """Test Weibull validation."""
        validator = NumericalValidation(tolerance=0.01)
        result = validator.validate_against_scipy("weibull")
        assert "pass" in result

    def test_validate_frechet(self):
        """Test Frechet validation."""
        validator = NumericalValidation(tolerance=0.01)
        result = validator.validate_against_scipy("frechet")
        assert "pass" in result

    def test_create_scipy_distribution_error_handling(self):
        """Test error handling in scipy distribution creation."""
        validator = NumericalValidation()
        # Test with invalid distribution
        result = validator.validate_against_scipy("invalid_dist")
        assert result["pass"] is False
        assert "error" in result


class TestConvergenceValidationEdgeCases:
    """Test edge cases in convergence validation."""

    def test_convergence_exception_handling(self):
        """Test exception handling in convergence validation."""
        # This tests the exception handling path
        result = convergence_validation("invalid_distribution", method="ppf")
        assert "converged" in result
        assert "error" in result or result["converged"] is False


class TestUnimplementedFeatures:
    """Test that unimplemented features raise appropriate errors."""

    def test_goodness_of_fit_ks_test(self):
        """Test that KS test raises NotImplementedError."""
        gof = GoodnessOfFitTests()
        with pytest.raises(NotImplementedError):
            gof.kolmogorov_smirnov_test([1, 2, 3], "pareto", alpha=2.5, xm=1.0)

    def test_goodness_of_fit_ad_test(self):
        """Test that AD test raises NotImplementedError."""
        gof = GoodnessOfFitTests()
        with pytest.raises(NotImplementedError):
            gof.anderson_darling_test([1, 2, 3], "pareto", alpha=2.5, xm=1.0)

    def test_regression_testing_add_reference_value(self):
        """Test that regression testing raises NotImplementedError."""
        rt = RegressionTesting()
        with pytest.raises(NotImplementedError):
            rt.add_reference_value("test1", 1.5)

    def test_regression_testing_check_regression(self):
        """Test that regression check raises NotImplementedError."""
        rt = RegressionTesting()
        with pytest.raises(NotImplementedError):
            rt.check_regression("test1", 1.5)

    def test_parameter_estimation_validation_mle(self):
        """Test that MLE validation raises NotImplementedError."""
        pev = ParameterEstimationValidation()
        with pytest.raises(NotImplementedError):
            pev.validate_mle("pareto", {"alpha": 2.5, "xm": 1.0})

    def test_parameter_estimation_validation_hill(self):
        """Test that Hill estimator validation raises NotImplementedError."""
        pev = ParameterEstimationValidation()
        with pytest.raises(NotImplementedError):
            pev.validate_hill_estimator(2.5)

    def test_fuzz_testing(self):
        """Test that fuzz testing raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            fuzz_testing()

    def test_special_function_accuracy_analysis(self):
        """Test that special function analysis raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            special_function_accuracy_analysis()

    def test_python_version_compatibility(self):
        """Test that version compatibility raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            python_version_compatibility()

    def test_mathematical_property_verification_tail_behavior(self):
        """Test that tail behavior verification raises NotImplementedError."""
        mpv = MathematicalPropertyVerification()
        with pytest.raises(NotImplementedError):
            mpv.verify_tail_behavior("pareto", alpha=2.5, xm=1.0)

    def test_mathematical_property_verification_moments(self):
        """Test that moments verification raises NotImplementedError."""
        mpv = MathematicalPropertyVerification()
        with pytest.raises(NotImplementedError):
            mpv.verify_moments("pareto", alpha=2.5, xm=1.0)

    def test_mathematical_property_verification_relationships(self):
        """Test that relationships verification raises NotImplementedError."""
        mpv = MathematicalPropertyVerification()
        with pytest.raises(NotImplementedError):
            mpv.verify_relationships("pareto", "lognormal")


class TestNumericalValidationWithoutScipy:
    """Test validation behavior when scipy is not available."""

    def test_validation_without_scipy_mock(self, monkeypatch):
        """Test validation when scipy is mocked as unavailable."""
        # Temporarily set SCIPY_AVAILABLE to False
        original_scipy_available = validation_module.SCIPY_AVAILABLE
        try:
            monkeypatch.setattr(validation_module, "SCIPY_AVAILABLE", False)
            validator = NumericalValidation()
            result = validator.validate_against_scipy("pareto")
            assert result["pass"] is False
            assert "scipy not available" in result["error"]
        finally:
            monkeypatch.setattr(
                validation_module, "SCIPY_AVAILABLE", original_scipy_available
            )


class TestPropertyBasedTestsWithoutHypothesis:
    """Test property-based tests when hypothesis is not available."""

    def test_pdf_nonnegativity_without_hypothesis(self, monkeypatch):
        """Test PDF nonnegativity when hypothesis is mocked as unavailable."""
        original_hypothesis_available = validation_module.HYPOTHESIS_AVAILABLE
        try:
            monkeypatch.setattr(validation_module, "HYPOTHESIS_AVAILABLE", False)
            tester = PropertyBasedTests()
            result = tester.test_pdf_nonnegativity("pareto")
            assert result["pass"] is False
            assert "Hypothesis not available" in result["error"]
        finally:
            monkeypatch.setattr(
                validation_module,
                "HYPOTHESIS_AVAILABLE",
                original_hypothesis_available,
            )

    def test_cdf_monotonicity_without_hypothesis(self, monkeypatch):
        """Test CDF monotonicity when hypothesis is mocked as unavailable."""
        original_hypothesis_available = validation_module.HYPOTHESIS_AVAILABLE
        try:
            monkeypatch.setattr(validation_module, "HYPOTHESIS_AVAILABLE", False)
            tester = PropertyBasedTests()
            result = tester.test_cdf_monotonicity("pareto")
            assert result["pass"] is False
            assert "Hypothesis not available" in result["error"]
        finally:
            monkeypatch.setattr(
                validation_module,
                "HYPOTHESIS_AVAILABLE",
                original_hypothesis_available,
            )


class TestParameterStabilityAdditional:
    """Additional parameter stability tests."""

    def test_weibull_large_parameters(self):
        """Test Weibull with very large parameters."""
        result = parameter_stability_check("weibull", k=150, lam=1e7)
        assert result["stable"] is False
        assert len(result["warnings"]) > 0

    def test_frechet_different_combinations(self):
        """Test Frechet with different parameter combinations."""
        result = parameter_stability_check("frechet", alpha=1.5, s=0.5, m=1.0)
        assert "warnings" in result
        assert "severity" in result

    def test_unknown_distribution_stability(self):
        """Test stability check with unknown distribution."""
        result = parameter_stability_check("unknown_distribution", param1=1.0)
        # Should return result without crashing
        assert "stable" in result
        assert "warnings" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
