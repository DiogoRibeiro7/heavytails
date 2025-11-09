"""
Financial Applications of Heavy-Tailed Distributions

This module demonstrates practical applications of heavytails distributions
in quantitative finance, risk management, and econometrics.

Examples include:
- Value at Risk (VaR) estimation
- Expected Shortfall (ES) calculation
- Portfolio risk assessment
- Option pricing with heavy tails
- Extreme loss modeling
"""

from __future__ import annotations

from datetime import datetime
import math
import random
import statistics
from typing import Any

try:
    from scipy import optimize  # type: ignore[import-untyped]
    from scipy.stats import norm, t  # type: ignore[import-untyped]

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    optimize = None
    norm = None
    t = None

from heavytails import (
    GeneralizedPareto,
    GEV_Frechet,
    Pareto,
    StudentT,
)
from heavytails.tail_index import hill_estimator, moment_estimator


class RiskMetrics:
    """Calculate financial risk metrics with heavy-tailed distributions."""

    def __init__(self, returns: list[float]):
        """
        Initialize with return data.

        Parameters
        ----------
        returns : List[float]
            Historical return data (e.g., daily log returns)
        """
        self.returns = sorted(returns, reverse=True)  # Descending order
        self.n = len(returns)
        self._gpd_diagnostics: dict[str, Any] = {}  # Store GPD fit diagnostics

    def empirical_var(self, alpha: float) -> float:
        """Calculate empirical Value at Risk."""
        index = int(alpha * self.n)
        if index >= self.n:
            return self.returns[-1]
        return -self.returns[-(index + 1)]  # Negative for loss

    def empirical_es(self, alpha: float) -> float:
        """Calculate empirical Expected Shortfall."""
        var = self.empirical_var(alpha)
        tail_losses = [r for r in self.returns if -r >= var]
        return -statistics.mean(tail_losses) if tail_losses else var

    def fit_gpd_to_excesses(
        self, threshold: float, method: str = "moments"
    ) -> GeneralizedPareto:
        """
        Enhanced GPD fitting with multiple methods and diagnostics.

        Parameters
        ----------
        threshold : float
            Threshold for peaks-over-threshold model
        method : str
            Estimation method: "moments" (default) or "mle"

        Returns
        -------
        GeneralizedPareto
            Fitted GPD distribution
        """
        # Convert returns to losses (negative returns)
        losses = [-r for r in self.returns]
        excesses = [loss - threshold for loss in losses if loss > threshold]

        if len(excesses) < 50:  # Increased minimum requirement
            raise ValueError(
                f"Only {len(excesses)} excesses above threshold. "
                "Need at least 50 for reliable GPD fitting."
            )

        # Calculate basic statistics
        mean_excess = statistics.mean(excesses)

        # Method of moments estimation (with bias correction)
        if method == "moments":
            M1 = mean_excess
            M2 = sum(x**2 for x in excesses) / len(excesses)

            # More stable estimation (Hosking & Wallis, 1987)
            if M2 > 2 * M1**2:
                xi = 0.5 * (1 - M1**2 / (M2 - M1**2))
                sigma = M1 * (1 - xi)
            else:
                # Very heavy tail case
                xi = 0.5 * ((M1**2) / (M2 - M1**2) - 1)
                sigma = M1 * (1 + xi)

        elif method == "mle":
            # Simplified MLE using scipy optimization if available
            if SCIPY_AVAILABLE and optimize is not None:

                def gpd_neg_loglik(params: tuple[float, float]) -> float:
                    xi_mle, sigma_mle = params
                    if sigma_mle <= 0:
                        return 1e10
                    n = len(excesses)
                    if abs(xi_mle) < 1e-6:
                        # Exponential case
                        return n * math.log(sigma_mle) + sum(excesses) / sigma_mle
                    else:
                        # General case
                        term = 1 + xi_mle * sum(
                            math.log(1 + xi_mle * x / sigma_mle) for x in excesses
                        )
                        return n * math.log(sigma_mle) + (1 + 1 / xi_mle) * term

                # Initial guess from method of moments
                M1 = mean_excess
                M2 = sum(x**2 for x in excesses) / len(excesses)
                xi0 = 0.5 * (1 - M1**2 / max(M2 - M1**2, M1**2 * 0.1))
                sigma0 = M1 * (1 - xi0)

                result = optimize.minimize(
                    gpd_neg_loglik,
                    [xi0, sigma0],
                    bounds=[(-0.5, 1.0), (1e-6, None)],
                    method="L-BFGS-B",
                )
                xi, sigma = result.x
            else:
                # Fallback to method of moments if scipy not available
                method = "moments (mle fallback)"
                M1 = mean_excess
                M2 = sum(x**2 for x in excesses) / len(excesses)
                if M2 > 2 * M1**2:
                    xi = 0.5 * (1 - M1**2 / (M2 - M1**2))
                    sigma = M1 * (1 - xi)
                else:
                    xi = 0.5 * ((M1**2) / (M2 - M1**2) - 1)
                    sigma = M1 * (1 + xi)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Parameter bounds checking and adjustment
        if xi < -0.5:
            xi = -0.49  # Ensure finite variance exists
        if xi > 1.0:
            xi = 0.99  # Practical upper bound
        if sigma <= 0:
            sigma = mean_excess * 0.5  # Emergency fallback

        gpd = GeneralizedPareto(xi=xi, sigma=max(sigma, 1e-6), mu=threshold)

        # Store diagnostics
        self._gpd_diagnostics = {
            "n_excesses": len(excesses),
            "threshold": threshold,
            "mean_excess": mean_excess,
            "xi_estimate": xi,
            "sigma_estimate": sigma,
            "method": method,
            "excess_rate": len(excesses) / len(losses),
        }

        return gpd

    def var_gpd(self, alpha: float, threshold: float) -> float:
        """
        Calculate VaR using GPD tail model.

        Uses Peaks-Over-Threshold (POT) methodology where:
        VaR_alpha = threshold + GPD.ppf(1 - alpha/prob_exceed)
        """
        n_excesses = sum(1 for r in self.returns if -r > threshold)
        prob_exceed = n_excesses / self.n

        if prob_exceed == 0 or alpha > prob_exceed:
            # Not enough data in tail, use empirical
            return self.empirical_var(alpha)

        gpd = self.fit_gpd_to_excesses(threshold)

        # Calculate conditional probability using POT formula
        conditional_prob = 1 - alpha / prob_exceed
        return gpd.ppf(conditional_prob)

    def es_gpd(self, alpha: float, threshold: float) -> float:
        """Calculate Expected Shortfall using GPD tail model."""
        var = self.var_gpd(alpha, threshold)
        gpd = self.fit_gpd_to_excesses(threshold)

        # ES formula for GPD
        if gpd.xi < 1:
            es = var + (gpd.sigma - gpd.xi * (var - gpd.mu)) / (1 - gpd.xi)
        else:
            # Infinite mean case - use empirical
            es = self.empirical_es(alpha)

        return es

    def var_es_gpd_with_confidence(
        self,
        alpha: float,
        threshold: float,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> dict[str, Any]:
        """
        VaR and ES with confidence intervals using GPD.

        Parameters
        ----------
        alpha : float
            Probability level (e.g., 0.01 for 99% VaR)
        threshold : float
            Threshold for GPD fitting
        confidence_level : float
            Confidence level for intervals (default 0.95)
        n_bootstrap : int
            Number of bootstrap samples (default 1000)

        Returns
        -------
        Dict[str, Any]
            Dictionary with VaR, ES, and their confidence intervals
        """
        gpd = self.fit_gpd_to_excesses(threshold)
        excess_rate = self._gpd_diagnostics["excess_rate"]

        # Point estimates
        if alpha <= excess_rate:
            # In the tail region - use GPD with POT formula
            conditional_prob = 1 - alpha / excess_rate
            var_estimate = gpd.ppf(conditional_prob)

            # Expected Shortfall calculation for GPD
            if gpd.xi < 1:
                es_estimate = var_estimate + (
                    gpd.sigma - gpd.xi * (var_estimate - gpd.mu)
                ) / (1 - gpd.xi)
            else:
                es_estimate = float("inf")  # Infinite mean case
        else:
            # Below threshold - use empirical
            var_estimate = self.empirical_var(alpha)
            es_estimate = self.empirical_es(alpha)

        # Bootstrap confidence intervals (if requested)
        var_ci: tuple[float | None, float | None]
        es_ci: tuple[float | None, float | None]
        if confidence_level > 0 and n_bootstrap > 0:
            var_ci, es_ci = self._bootstrap_var_es_ci(
                alpha, threshold, confidence_level, n_bootstrap
            )
        else:
            var_ci = (None, None)
            es_ci = (None, None)

        return {
            "VaR": var_estimate,
            "ES": es_estimate,
            "VaR_CI_lower": var_ci[0],
            "VaR_CI_upper": var_ci[1],
            "ES_CI_lower": es_ci[0],
            "ES_CI_upper": es_ci[1],
            "method": "GPD",
            "threshold": threshold,
            "gpd_params": {"xi": gpd.xi, "sigma": gpd.sigma},
        }

    def _bootstrap_var_es_ci(
        self, alpha: float, threshold: float, confidence_level: float, n_bootstrap: int
    ) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None]]:
        """
        Bootstrap confidence intervals for VaR and ES.

        Parameters
        ----------
        alpha : float
            Probability level
        threshold : float
            Threshold for GPD fitting
        confidence_level : float
            Confidence level for intervals
        n_bootstrap : int
            Number of bootstrap samples

        Returns
        -------
        Tuple[Tuple[float, float], Tuple[float, float]]
            VaR CI and ES CI as (lower, upper) tuples
        """
        var_estimates = []
        es_estimates = []

        for _ in range(n_bootstrap):
            # Resample returns with replacement
            bootstrap_returns = random.choices(self.returns, k=len(self.returns))

            # Create bootstrap risk metrics object
            bootstrap_rm = RiskMetrics(bootstrap_returns)

            try:
                # Fit GPD to bootstrap sample
                bootstrap_result = bootstrap_rm.var_es_gpd_with_confidence(
                    alpha, threshold, confidence_level=0, n_bootstrap=0
                )
                var_estimates.append(bootstrap_result["VaR"])
                if bootstrap_result["ES"] != float("inf"):
                    es_estimates.append(bootstrap_result["ES"])
            except Exception:
                continue  # Skip failed bootstrap samples

        if not var_estimates:
            return (None, None), (None, None)

        # Calculate confidence intervals
        alpha_ci = (1 - confidence_level) / 2
        var_estimates_sorted = sorted(var_estimates)
        var_lower = var_estimates_sorted[int(alpha_ci * len(var_estimates_sorted))]
        var_upper = var_estimates_sorted[
            int((1 - alpha_ci) * len(var_estimates_sorted))
        ]

        if es_estimates:
            es_estimates_sorted = sorted(es_estimates)
            es_lower = es_estimates_sorted[int(alpha_ci * len(es_estimates_sorted))]
            es_upper = es_estimates_sorted[
                int((1 - alpha_ci) * len(es_estimates_sorted))
            ]
        else:
            es_lower = es_upper = None

        return (var_lower, var_upper), (es_lower, es_upper)


class TailRiskAnalyzer:
    """Analyze tail risk characteristics of financial time series."""

    def __init__(self, data: list[float]):
        self.data = data
        self.losses = [-x for x in data]  # Convert to losses

    def estimate_tail_index(
        self, method: str = "hill", k: int | None = None
    ) -> dict[str, Any]:
        """Estimate tail index using various methods."""
        if k is None:
            k = min(len(self.data) // 10, 250)  # Adaptive k selection

        results: dict[str, Any] = {}

        if method in {"hill", "all"}:
            try:
                gamma_hill = hill_estimator(self.losses, k)
                results["hill_gamma"] = gamma_hill
                results["hill_alpha"] = 1.0 / gamma_hill
            except Exception as e:
                results["hill_error"] = str(e)

        if method in {"moment", "all"}:
            try:
                gamma_mom, alpha_mom = moment_estimator(self.losses, k)
                results["moment_gamma"] = gamma_mom
                results["moment_alpha"] = alpha_mom
            except Exception as e:
                results["moment_error"] = str(e)

        results["k"] = k
        results["n"] = len(self.data)

        return results

    def test_heavy_tail_hypothesis(self) -> dict[str, Any]:
        """Test various heavy-tail characteristics."""
        results: dict[str, Any] = {}

        # Tail index estimates
        tail_estimates = self.estimate_tail_index("all")
        results["tail_indices"] = tail_estimates

        # Kurtosis test
        kurt = self._calculate_kurtosis()
        results["kurtosis"] = kurt
        results["excess_kurtosis"] = kurt - 3
        results["heavy_tail_kurtosis"] = kurt > 6  # Rule of thumb

        # Hill plot stability (simplified)
        k_values = [50, 100, 150, 200, 250]
        hill_estimates: list[float | None] = []

        for k in k_values:
            if k < len(self.data):
                try:
                    gamma = hill_estimator(self.losses, k)
                    hill_estimates.append(gamma)
                except Exception:
                    hill_estimates.append(None)

        results["hill_stability"] = {
            "k_values": k_values,
            "estimates": hill_estimates,
            "stable": self._assess_stability(hill_estimates),
        }

        return results

    def _calculate_kurtosis(self) -> float:
        """Calculate sample kurtosis."""
        n = len(self.data)
        mean = statistics.mean(self.data)
        var = statistics.variance(self.data)

        if var == 0:
            return 0

        m4 = sum((x - mean) ** 4 for x in self.data) / n
        return m4 / (var**2)

    def _assess_stability(self, estimates: list[float | None]) -> bool:
        """Assess if Hill estimates are stable."""
        valid_estimates = [e for e in estimates if e is not None]
        if len(valid_estimates) < 3:
            return False

        cv = statistics.stdev(valid_estimates) / statistics.mean(valid_estimates)
        return cv < 0.3  # Coefficient of variation threshold


class ExtremeValueModeling:
    """Extreme value modeling for financial applications."""

    @staticmethod
    def block_maxima_analysis(
        returns: list[float], block_size: int = 252
    ) -> dict[str, Any]:
        """
        Analyze block maxima (e.g., annual maxima from daily data).

        Parameters
        ----------
        returns : List[float]
            Return time series
        block_size : int
            Size of each block (e.g., 252 for annual blocks of daily data)
        """
        losses = [-r for r in returns]  # Convert to losses

        # Extract block maxima
        maxima = []
        for i in range(0, len(losses), block_size):
            block = losses[i : i + block_size]
            if len(block) >= block_size // 2:  # At least half the block size
                maxima.append(max(block))

        if len(maxima) < 5:
            raise ValueError("Too few blocks for reliable analysis")

        # Fit GEV distribution using method of moments
        mean_max = statistics.mean(maxima)
        var_max = statistics.variance(maxima)
        std_max = math.sqrt(var_max)

        # Approximate GEV parameters (method of moments)
        # This is a simplified estimation - proper MLE would be better
        sigma_approx = std_max * math.sqrt(6) / math.pi
        mu_approx = mean_max - 0.5772 * sigma_approx  # Euler's constant
        xi_approx = 0.1  # Assume small positive shape parameter

        gev = GEV_Frechet(xi=xi_approx, mu=mu_approx, sigma=sigma_approx)

        return {
            "maxima": maxima,
            "n_blocks": len(maxima),
            "mean_maxima": mean_max,
            "std_maxima": std_max,
            "gev_params": {"xi": xi_approx, "mu": mu_approx, "sigma": sigma_approx},
            "gev_distribution": gev,
        }

    @staticmethod
    def return_level_calculation(
        gev_params: dict[str, float], return_period: int
    ) -> float:
        """
        Calculate return level for given return period.

        Parameters
        ----------
        gev_params : dict[str, float]
            GEV parameters
        return_period : int
            Return period in years
        """
        xi = float(gev_params["xi"])
        mu = float(gev_params["mu"])
        sigma = float(gev_params["sigma"])

        p = 1 - (1 / return_period)

        if abs(xi) < 1e-6:  # Gumbel case
            return float(mu - sigma * math.log(-math.log(p)))
        else:  # Fréchet/Weibull case
            return float(mu + (sigma / xi) * ((-math.log(p)) ** (-xi) - 1))


class PortfolioRiskAssessment:
    """Portfolio-level risk assessment with heavy-tailed models."""

    def __init__(self, portfolio_returns: list[float]):
        self.returns = portfolio_returns
        self.risk_metrics = RiskMetrics(portfolio_returns)

    def comprehensive_risk_report(
        self, confidence_levels: list[float] | None = None
    ) -> dict[str, Any]:
        """
        Complete implementation of comprehensive risk assessment.

        Parameters
        ----------
        confidence_levels : list[float] | None
            Confidence levels to analyze (default: [0.95, 0.99, 0.995, 0.999])

        Returns
        -------
        dict[str, Any]
            Comprehensive risk report with multiple methods and diagnostics
        """
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99, 0.995, 0.999]

        report: dict[str, Any] = {
            "metadata": {
                "portfolio_size": len(self.returns),
                "analysis_date": datetime.now().isoformat(),
                "time_period": "daily",
                "confidence_levels": confidence_levels,
            }
        }

        # Enhanced basic statistics with heavy-tail indicators
        basic_stats = self._calculate_enhanced_statistics()
        report["basic_statistics"] = basic_stats

        # Multi-method risk estimates for each confidence level
        risk_estimates: dict[str, Any] = {}

        for alpha in confidence_levels:
            level_results: dict[str, Any] = {
                "confidence_level": alpha,
                "tail_probability": 1 - alpha,
            }

            # Empirical estimates (always available)
            level_results["empirical"] = {
                "VaR": self.risk_metrics.empirical_var(1 - alpha),
                "ES": self.risk_metrics.empirical_es(1 - alpha),
            }

            # Parametric estimates with multiple distributions
            parametric_results: dict[str, Any] = {}

            # Normal assumption (for comparison)
            mean_ret = statistics.mean(self.returns)
            std_ret = statistics.stdev(self.returns)

            if SCIPY_AVAILABLE and norm is not None:
                normal_var = mean_ret + std_ret * norm.ppf(alpha)
                parametric_results["normal"] = {
                    "VaR": -normal_var,  # Convert to loss
                    "ES": -(
                        mean_ret + std_ret * norm.pdf(norm.ppf(alpha)) / (1 - alpha)
                    ),
                }
            else:
                # Fallback if scipy not available
                # Use approximate normal quantile
                z_alpha = self._approx_normal_quantile(alpha)
                normal_var = mean_ret + std_ret * z_alpha
                parametric_results["normal"] = {
                    "VaR": -normal_var,
                    "ES": -(
                        mean_ret
                        + std_ret
                        * math.exp(-(z_alpha**2) / 2)
                        / (math.sqrt(2 * math.pi) * (1 - alpha))
                    ),
                }

            # Student-t assumption
            try:
                nu_est = self._estimate_student_t_dof()
                if nu_est > 2 and SCIPY_AVAILABLE and t is not None:
                    scale = std_ret * math.sqrt((nu_est - 2) / nu_est)
                    t_var = mean_ret + scale * t.ppf(alpha, df=nu_est)
                    parametric_results["student_t"] = {
                        "VaR": -t_var,
                        "ES": self._student_t_es(alpha, nu_est, mean_ret, scale),
                        "dof": nu_est,
                    }
                elif nu_est > 2:
                    parametric_results["student_t"] = {
                        "error": "scipy not available for Student-t calculations"
                    }
            except Exception as e:
                parametric_results["student_t"] = {"error": str(e)}

            # GPD tail modeling (for extreme quantiles)
            if alpha >= 0.95:  # Only for tail estimates
                try:
                    threshold = self.risk_metrics.empirical_var(
                        0.1
                    )  # 90% empirical quantile
                    gpd_result = self.risk_metrics.var_es_gpd_with_confidence(
                        1 - alpha, threshold, confidence_level=0.95, n_bootstrap=500
                    )
                    parametric_results["gpd"] = gpd_result
                except Exception as e:
                    parametric_results["gpd"] = {"error": str(e)}

            level_results["parametric"] = parametric_results
            risk_estimates[f"alpha_{alpha}"] = level_results

        report["risk_estimates"] = risk_estimates

        # Tail analysis and heavy-tail diagnostics
        tail_analyzer = TailRiskAnalyzer(self.returns)
        report["tail_analysis"] = tail_analyzer.test_heavy_tail_hypothesis()

        # Model comparison and recommendations
        report["model_recommendations"] = self._generate_model_recommendations(report)

        return report

    def _calculate_skewness(self) -> float:
        """Calculate sample skewness."""
        n = len(self.returns)
        mean = statistics.mean(self.returns)
        std = statistics.stdev(self.returns)

        if std == 0:
            return 0

        m3 = sum((x - mean) ** 3 for x in self.returns) / n
        return m3 / (std**3)

    def _calculate_kurtosis(self) -> float:
        """Calculate sample kurtosis."""
        n = len(self.returns)
        mean = statistics.mean(self.returns)
        var = statistics.variance(self.returns)

        if var == 0:
            return 0

        m4 = sum((x - mean) ** 4 for x in self.returns) / n
        return m4 / (var**2)

    def _estimate_student_t_dof(self) -> float:
        """Rough estimation of Student-t degrees of freedom from kurtosis."""
        kurt = self._calculate_kurtosis()
        excess_kurt = kurt - 3

        if excess_kurt <= 0:
            return float("inf")  # Normal case

        # For Student-t: excess kurtosis = 6/(nu-4) for nu > 4
        if excess_kurt > 0:
            nu_est = 4 + 6 / excess_kurt
            return max(nu_est, 2.1)  # Ensure nu > 2

        return 5.0  # Default moderate value

    def _calculate_enhanced_statistics(self) -> dict[str, Any]:
        """Calculate enhanced statistics including heavy-tail indicators."""
        mean_ret = statistics.mean(self.returns)
        std_ret = statistics.stdev(self.returns)
        skew = self._calculate_skewness()
        kurt = self._calculate_kurtosis()

        return {
            "mean": mean_ret,
            "std": std_ret,
            "skewness": skew,
            "kurtosis": kurt,
            "excess_kurtosis": kurt - 3,
            "min": min(self.returns),
            "max": max(self.returns),
            "n_observations": len(self.returns),
            "sharpe_ratio": mean_ret / std_ret if std_ret > 0 else 0,
        }

    def _student_t_es(
        self, alpha: float, nu: float, mean: float, scale: float
    ) -> float:
        """
        Calculate Expected Shortfall for Student-t distribution.

        Parameters
        ----------
        alpha : float
            Confidence level
        nu : float
            Degrees of freedom
        mean : float
            Location parameter
        scale : float
            Scale parameter

        Returns
        -------
        float
            Expected Shortfall estimate
        """
        if SCIPY_AVAILABLE and t is not None:
            # ES formula for Student-t
            t_alpha = t.ppf(alpha, df=nu)
            pdf_value = t.pdf(t_alpha, df=nu)
            es_multiplier = pdf_value * (nu + t_alpha**2) / ((1 - alpha) * (nu - 1))
            return float(-(mean + scale * es_multiplier))
        else:
            # Fallback to empirical if scipy not available
            return self.risk_metrics.empirical_es(1 - alpha)

    def _generate_model_recommendations(self, report: dict[str, Any]) -> dict[str, str]:
        """
        Generate model recommendations based on analysis results.

        Parameters
        ----------
        report : Dict[str, Any]
            The comprehensive risk report

        Returns
        -------
        Dict[str, str]
            Recommendations for modeling approaches
        """
        recommendations = {}

        # Check kurtosis for heavy-tail indication
        kurtosis = report["basic_statistics"]["kurtosis"]
        if kurtosis > 6:
            recommendations["distribution"] = (
                "Heavy-tailed models recommended (Student-t, GPD)"
            )
        elif kurtosis > 4:
            recommendations["distribution"] = "Consider Student-t distribution"
        else:
            recommendations["distribution"] = "Normal distribution may be adequate"

        # Risk model recommendations
        tail_analysis = report.get("tail_analysis", {})
        if tail_analysis.get("heavy_tail_kurtosis", False):
            recommendations["risk_model"] = "Use GPD for tail risk, Student-t for VaR"
        else:
            recommendations["risk_model"] = "Standard parametric models sufficient"

        # Sample size recommendations
        n = len(self.returns)
        if n < 500:
            recommendations["sample_size"] = (
                "Sample too small for reliable tail analysis"
            )
        elif n < 2000:
            recommendations["sample_size"] = (
                "Adequate sample size, use bootstrap for confidence intervals"
            )
        else:
            recommendations["sample_size"] = (
                "Excellent sample size for robust estimation"
            )

        return recommendations

    def _approx_normal_quantile(self, alpha: float) -> float:
        """
        Approximate normal quantile using Beasley-Springer-Moro algorithm.

        Parameters
        ----------
        alpha : float
            Probability level

        Returns
        -------
        float
            Approximate standard normal quantile
        """
        # Simple approximation for common quantiles
        if alpha >= 0.5:
            # Use symmetry for upper tail
            return -self._approx_normal_quantile(1 - alpha)

        # Rational approximation (simplified)
        p = alpha
        if p < 0.5:
            # Lower tail
            t = math.sqrt(-2 * math.log(p))
            num = 2.30753 + t * 0.27061
            den = 1 + t * (0.99229 + t * 0.04481)
            return -(t - num / den)
        else:
            return 0.0


# Example usage functions
def demo_var_calculation() -> None:
    """Demonstrate VaR calculation with different methods."""
    # Simulate some heavy-tailed returns
    random.seed(42)

    # Generate Student-t returns
    student_t = StudentT(nu=4)
    returns = [x * 0.02 for x in student_t.rvs(1000, seed=42)]  # Scale to 2% volatility

    risk_metrics = RiskMetrics(returns)

    print("VaR Estimates (95% confidence):")
    print(f"Empirical VaR: {risk_metrics.empirical_var(0.05):.4f}")

    try:
        threshold = risk_metrics.empirical_var(0.1)  # 90% quantile
        gpd_var = risk_metrics.var_gpd(0.05, threshold)
        print(f"GPD VaR: {gpd_var:.4f}")
    except Exception as e:
        print(f"GPD VaR failed: {e}")


def demo_tail_analysis() -> None:
    """Demonstrate tail index estimation."""
    # Generate Pareto-distributed data
    pareto = Pareto(alpha=2.5, xm=1.0)
    data = pareto.rvs(2000, seed=42)

    analyzer = TailRiskAnalyzer(data)
    results = analyzer.test_heavy_tail_hypothesis()

    print("Tail Analysis Results:")
    print(f"Estimated tail indices: {results['tail_indices']}")
    print(f"Kurtosis: {results['kurtosis']:.2f}")
    print(f"Heavy tail (kurtosis): {results['heavy_tail_kurtosis']}")


def demo_portfolio_risk() -> None:
    """Demonstrate comprehensive portfolio risk assessment."""
    # Generate mixed distribution returns
    random.seed(42)

    # 70% normal returns, 30% heavy-tailed shocks
    normal_returns = [random.gauss(0, 0.01) for _ in range(700)]

    student_t = StudentT(nu=3)
    shock_returns = [x * 0.03 for x in student_t.rvs(300, seed=42)]

    portfolio_returns = normal_returns + shock_returns
    random.shuffle(portfolio_returns)

    assessor = PortfolioRiskAssessment(portfolio_returns)
    report = assessor.comprehensive_risk_report()

    print("Portfolio Risk Report:")
    print(
        f"Basic Stats: {report.get('basic_statistics', report.get('basic_stats', {}))}"
    )
    print("\nRisk Estimates:")
    for level, estimates in report["risk_estimates"].items():
        print(f"  {level}: {estimates}")


def demo_comprehensive_backtest() -> dict[str, Any]:
    """
    Complete backtesting framework for risk models.

    This function demonstrates a comprehensive VaR backtesting approach with:
    - Regime-switching return generation (normal + crisis periods)
    - Multiple risk models (empirical, normal, Student-t, GPD)
    - Statistical validation of model performance
    """
    print("VaR Model Backtesting Framework")
    print("=" * 60)

    # Generate realistic financial return data with regime switching
    random.seed(42)

    # Create regime-switching returns: 90% normal, 10% crisis
    normal_returns = [random.gauss(0.001, 0.02) for _ in range(1800)]  # Normal times

    # Crisis periods with Student-t(3) distribution
    student_t = StudentT(nu=3)
    crisis_returns = [x * 0.05 for x in student_t.rvs(200, seed=42)]  # Crisis times

    # Combine and shuffle
    all_returns = normal_returns + crisis_returns
    random.shuffle(all_returns)

    # Split into estimation and testing periods
    estimation_returns = all_returns[:1500]
    test_returns = all_returns[1500:]

    print(f"Estimation period: {len(estimation_returns)} observations")
    print(f"Testing period: {len(test_returns)} observations")
    print()

    # Estimate models on estimation period
    risk_assessor = PortfolioRiskAssessment(estimation_returns)

    # Confidence level for VaR
    confidence_level = 0.99
    alpha = 1 - confidence_level

    print(f"Testing 99% VaR models (expected violation rate: {alpha:.1%})")
    print()

    # Models to test
    models_config: dict[str, Any] = {
        "empirical": {
            "description": "Historical Simulation",
            "var_func": lambda: risk_assessor.risk_metrics.empirical_var(alpha),
        },
        "normal": {
            "description": "Normal Distribution",
            "var_func": lambda: calculate_normal_var(
                estimation_returns, confidence_level
            ),
        },
        "student_t": {
            "description": "Student-t Distribution",
            "var_func": lambda: calculate_student_t_var(
                estimation_returns, confidence_level
            ),
        },
        "gpd": {
            "description": "GPD Tail Model",
            "var_func": lambda: calculate_gpd_var(risk_assessor, alpha),
        },
    }

    backtest_results: dict[str, Any] = {}

    for model_name, config in models_config.items():
        try:
            # Get VaR forecast using the model
            var_forecast = config["var_func"]()

            # Count violations in test period
            violations = 0
            for actual_return in test_returns:
                # Check if actual return violates VaR (loss exceeds VaR)
                if actual_return < -var_forecast:
                    violations += 1

            # Calculate backtest statistics
            violation_rate = violations / len(test_returns)
            expected_violations = alpha * len(test_returns)

            # Kupiec's POF test (simplified)
            # Test passes if violation rate is within reasonable bounds
            tolerance = 2 * math.sqrt(alpha * (1 - alpha) / len(test_returns))
            test_passes = abs(violation_rate - alpha) < tolerance

            backtest_results[model_name] = {
                "description": config["description"],
                "var_forecast": var_forecast,
                "violations": violations,
                "violation_rate": violation_rate,
                "expected_violations": expected_violations,
                "expected_rate": alpha,
                "test_passes": test_passes,
                "tolerance": tolerance,
            }

        except Exception as e:
            backtest_results[model_name] = {
                "description": config["description"],
                "error": str(e),
            }

    # Display results
    print("VaR Model Backtesting Results:")
    print("-" * 60)
    for results in backtest_results.values():
        if "error" in results:
            print(f"{results['description']:25s}: ERROR - {results['error']}")
        else:
            status = "PASS" if results["test_passes"] else "FAIL"
            print(f"{results['description']:25s}:")
            print(f"  VaR Forecast: {results['var_forecast']:.4f}")
            print(
                f"  Violations: {results['violations']}/{len(test_returns)} "
                f"({results['violation_rate']:.2%})"
            )
            print(
                f"  Expected: {results['expected_violations']:.1f} ({results['expected_rate']:.2%})"
            )
            print(f"  Test Status: {status}")
            print()

    return backtest_results


def calculate_normal_var(returns: list[float], confidence_level: float) -> float:
    """Calculate VaR under normal distribution assumption (returns positive loss value)."""
    mean_ret = statistics.mean(returns)
    std_ret = statistics.stdev(returns)

    if SCIPY_AVAILABLE and norm is not None:
        # VaR is the negative of the quantile (convert to positive loss)
        return float(-(mean_ret + std_ret * norm.ppf(1 - confidence_level)))
    else:
        # Approximate for 99% and 95%
        z = -2.326 if confidence_level == 0.99 else -1.645
        return -(mean_ret + std_ret * z)


def calculate_student_t_var(returns: list[float], confidence_level: float) -> float:
    """Calculate VaR under Student-t distribution assumption (returns positive loss value)."""
    mean_ret = statistics.mean(returns)
    std_ret = statistics.stdev(returns)

    # Estimate degrees of freedom from kurtosis
    kurt = calculate_kurtosis(returns)
    excess_kurt = kurt - 3

    if excess_kurt > 0:
        nu_est = max(4 + 6 / excess_kurt, 2.1)
    else:
        return calculate_normal_var(returns, confidence_level)

    if SCIPY_AVAILABLE and t is not None:
        scale = std_ret * math.sqrt((nu_est - 2) / nu_est)
        # VaR is the negative of the quantile (convert to positive loss)
        return float(-(mean_ret + scale * t.ppf(1 - confidence_level, df=nu_est)))
    else:
        return calculate_normal_var(returns, confidence_level)


def calculate_gpd_var(risk_assessor: PortfolioRiskAssessment, alpha: float) -> float:
    """Calculate VaR using GPD tail model."""
    # Use a threshold that's well into the tail but has enough excesses
    # For 99% VaR (alpha=0.01), use 95% quantile (5% excesses)
    threshold_prob = max(0.05, alpha * 5)  # At least 5x alpha for stable GPD fit
    threshold = risk_assessor.risk_metrics.empirical_var(threshold_prob)
    return risk_assessor.risk_metrics.var_gpd(alpha, threshold)


def calculate_kurtosis(data: list[float]) -> float:
    """Calculate sample kurtosis."""
    n = len(data)
    mean_val = statistics.mean(data)
    var_val = statistics.variance(data)

    if var_val == 0:
        return 3.0

    m4 = sum((x - mean_val) ** 4 for x in data) / n
    return m4 / (var_val**2)


if __name__ == "__main__":
    print("=== Heavy-Tails Financial Applications Demo ===\n")

    print("1. VaR Calculation Demo:")
    demo_var_calculation()

    print("\n2. Tail Analysis Demo:")
    demo_tail_analysis()

    print("\n3. Portfolio Risk Demo:")
    demo_portfolio_risk()

    print("\n4. Comprehensive Backtesting Demo:")
    demo_comprehensive_backtest()
