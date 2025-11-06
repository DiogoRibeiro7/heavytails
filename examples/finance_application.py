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

import math
import statistics
from typing import Any

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

    def fit_gpd_to_excesses(self, threshold: float) -> GeneralizedPareto:
        """
        Fit GPD to excesses over threshold using method of moments.
        
        Parameters
        ----------
        threshold : float
            Threshold for peaks-over-threshold model
            
        Returns
        -------
        GeneralizedPareto
            Fitted GPD distribution
        """
        # Convert returns to losses (negative returns)
        losses = [-r for r in self.returns]
        excesses = [loss - threshold for loss in losses if loss > threshold]

        if len(excesses) < 10:
            raise ValueError("Too few excesses for reliable fitting")

        # Method of moments estimation
        mean_excess = statistics.mean(excesses)
        var_excess = statistics.variance(excesses)

        # Method of moments estimates for GPD
        if var_excess <= mean_excess ** 2:
            # Light tail case
            xi_hat = 0.5 * (1 - (mean_excess ** 2) / var_excess)
            sigma_hat = mean_excess * (1 - xi_hat)
        else:
            # Heavy tail case
            xi_hat = 0.5 * ((mean_excess ** 2) / var_excess - 1)
            sigma_hat = mean_excess * (1 + xi_hat)

        return GeneralizedPareto(xi=xi_hat, sigma=max(sigma_hat, 1e-6), mu=threshold)

    def var_gpd(self, alpha: float, threshold: float) -> float:
        """Calculate VaR using GPD tail model."""
        n_excesses = sum(1 for r in self.returns if -r > threshold)
        prob_exceed = n_excesses / self.n

        if prob_exceed == 0:
            return self.empirical_var(alpha)

        gpd = self.fit_gpd_to_excesses(threshold)

        if alpha <= prob_exceed:
            # In the tail region
            conditional_prob = alpha / prob_exceed
            return gpd.ppf(conditional_prob)
        else:
            # Below threshold, use empirical
            return self.empirical_var(alpha)

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


class TailRiskAnalyzer:
    """Analyze tail risk characteristics of financial time series."""

    def __init__(self, data: list[float]):
        self.data = data
        self.losses = [-x for x in data]  # Convert to losses

    def estimate_tail_index(self, method: str = "hill", k: int = None) -> dict[str, float]:
        """Estimate tail index using various methods."""
        if k is None:
            k = min(len(self.data) // 10, 250)  # Adaptive k selection

        results = {}

        if method == "hill" or method == "all":
            try:
                gamma_hill = hill_estimator(self.losses, k)
                results["hill_gamma"] = gamma_hill
                results["hill_alpha"] = 1.0 / gamma_hill
            except Exception as e:
                results["hill_error"] = str(e)

        if method == "moment" or method == "all":
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
        results = {}

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
        hill_estimates = []

        for k in k_values:
            if k < len(self.data):
                try:
                    gamma = hill_estimator(self.losses, k)
                    hill_estimates.append(gamma)
                except:
                    hill_estimates.append(None)

        results["hill_stability"] = {
            "k_values": k_values,
            "estimates": hill_estimates,
            "stable": self._assess_stability(hill_estimates)
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
        return m4 / (var ** 2)

    def _assess_stability(self, estimates: list[float]) -> bool:
        """Assess if Hill estimates are stable."""
        valid_estimates = [e for e in estimates if e is not None]
        if len(valid_estimates) < 3:
            return False

        cv = statistics.stdev(valid_estimates) / statistics.mean(valid_estimates)
        return cv < 0.3  # Coefficient of variation threshold


class ExtremeValueModeling:
    """Extreme value modeling for financial applications."""

    @staticmethod
    def block_maxima_analysis(returns: list[float], block_size: int = 252) -> dict[str, Any]:
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
            block = losses[i:i + block_size]
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
            "gev_distribution": gev
        }

    @staticmethod
    def return_level_calculation(gev_params: dict[str, float], return_period: int) -> float:
        """
        Calculate return level for given return period.
        
        Parameters
        ----------
        gev_params : Dict[str, float]
            GEV parameters
        return_period : int
            Return period in years
        """
        xi, mu, sigma = gev_params["xi"], gev_params["mu"], gev_params["sigma"]

        p = 1 - (1 / return_period)

        if abs(xi) < 1e-6:  # Gumbel case
            return mu - sigma * math.log(-math.log(p))
        else:  # Fréchet/Weibull case
            return mu + (sigma / xi) * ((-math.log(p)) ** (-xi) - 1)


class PortfolioRiskAssessment:
    """Portfolio-level risk assessment with heavy-tailed models."""

    def __init__(self, portfolio_returns: list[float]):
        self.returns = portfolio_returns
        self.risk_metrics = RiskMetrics(portfolio_returns)

    def comprehensive_risk_report(self, confidence_levels: list[float] = None) -> dict[str, Any]:
        """Generate comprehensive risk assessment report."""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99, 0.995]

        report = {}

        # Basic statistics
        report["basic_stats"] = {
            "mean": statistics.mean(self.returns),
            "std": statistics.stdev(self.returns),
            "skewness": self._calculate_skewness(),
            "kurtosis": self._calculate_kurtosis(),
            "min": min(self.returns),
            "max": max(self.returns)
        }

        # Risk metrics for each confidence level
        risk_estimates = {}

        for alpha in confidence_levels:
            level_results = {}

            # Empirical estimates
            level_results["empirical_var"] = self.risk_metrics.empirical_var(alpha)
            level_results["empirical_es"] = self.risk_metrics.empirical_es(alpha)

            # Parametric estimates (assume Student-t)
            try:
                # Estimate degrees of freedom for Student-t
                nu = self._estimate_student_t_dof()
                if nu > 2:  # Finite variance
                    student_t = StudentT(nu=nu)
                    # Scale to match empirical std
                    scale = statistics.stdev(self.returns) * math.sqrt((nu - 2) / nu)
                    var_t = -scale * student_t.ppf(alpha)
                    level_results["student_t_var"] = var_t

            except Exception as e:
                level_results["student_t_error"] = str(e)

            # GPD tail modeling
            try:
                threshold = self.risk_metrics.empirical_var(0.9)  # 90% quantile as threshold
                level_results["gpd_var"] = self.risk_metrics.var_gpd(alpha, threshold)
                level_results["gpd_es"] = self.risk_metrics.es_gpd(alpha, threshold)
            except Exception as e:
                level_results["gpd_error"] = str(e)

            risk_estimates[f"alpha_{alpha}"] = level_results

        report["risk_estimates"] = risk_estimates

        # Tail analysis
        tail_analyzer = TailRiskAnalyzer(self.returns)
        report["tail_analysis"] = tail_analyzer.test_heavy_tail_hypothesis()

        return report

    def _calculate_skewness(self) -> float:
        """Calculate sample skewness."""
        n = len(self.returns)
        mean = statistics.mean(self.returns)
        std = statistics.stdev(self.returns)

        if std == 0:
            return 0

        m3 = sum((x - mean) ** 3 for x in self.returns) / n
        return m3 / (std ** 3)

    def _calculate_kurtosis(self) -> float:
        """Calculate sample kurtosis."""
        n = len(self.returns)
        mean = statistics.mean(self.returns)
        var = statistics.variance(self.returns)

        if var == 0:
            return 0

        m4 = sum((x - mean) ** 4 for x in self.returns) / n
        return m4 / (var ** 2)

    def _estimate_student_t_dof(self) -> float:
        """Rough estimation of Student-t degrees of freedom from kurtosis."""
        kurt = self._calculate_kurtosis()
        excess_kurt = kurt - 3

        if excess_kurt <= 0:
            return float('inf')  # Normal case

        # For Student-t: excess kurtosis = 6/(ν-4) for ν > 4
        if excess_kurt > 0:
            nu_est = 4 + 6 / excess_kurt
            return max(nu_est, 2.1)  # Ensure ν > 2

        return 5.0  # Default moderate value


# Example usage functions
def demo_var_calculation():
    """Demonstrate VaR calculation with different methods."""
    # Simulate some heavy-tailed returns
    import random
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


def demo_tail_analysis():
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


def demo_portfolio_risk():
    """Demonstrate comprehensive portfolio risk assessment."""
    # Generate mixed distribution returns
    import random
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
    print(f"Basic Stats: {report['basic_stats']}")
    print("\nRisk Estimates:")
    for level, estimates in report['risk_estimates'].items():
        print(f"  {level}: {estimates}")


if __name__ == "__main__":
    print("=== Heavy-Tails Financial Applications Demo ===\n")

    print("1. VaR Calculation Demo:")
    demo_var_calculation()

    print("\n2. Tail Analysis Demo:")
    demo_tail_analysis()

    print("\n3. Portfolio Risk Demo:")
    demo_portfolio_risk()
