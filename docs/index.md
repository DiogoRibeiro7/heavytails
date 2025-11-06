# HeavyTails Documentation

Welcome to the **HeavyTails** documentation! HeavyTails is a pure-Python library for heavy-tailed probability distributions, built from first principles for research, education, and practical applications.

## 🎯 Quick Start

```python
from heavytails import Pareto, StudentT, LogNormal
from heavytails.tail_index import hill_estimator

# Create a Pareto distribution
dist = Pareto(alpha=2.0, xm=1.0)

# Generate samples
samples = dist.rvs(1000, seed=42)

# Estimate tail index
gamma_hat = hill_estimator(samples, k=100)
alpha_hat = 1.0 / gamma_hat
print(f"Estimated tail index: {alpha_hat:.2f}")
```

## 📚 What's Inside

### Core Features

- **12+ Heavy-Tailed Distributions**: Pareto, Student-t, Cauchy, LogNormal, Weibull, Fréchet, GEV, GPD, Burr XII, and more
- **Tail Index Estimation**: Hill, Pickands, and moment estimators
- **Pure Python**: No external dependencies, perfect for education and understanding
- **Command-Line Interface**: Full CLI for analysis and visualization
- **Financial Applications**: Risk management tools for VaR, ES, and tail risk

### Mathematical Excellence

- **Numerical Stability**: Carefully implemented special functions
- **Academic Rigor**: Proper mathematical foundations and references
- **Validation**: Cross-validated against R and SciPy implementations
- **Performance**: Optimized algorithms with benchmarking

## 🧮 Mathematical Background

Heavy-tailed distributions are characterized by:

$$P(X > x) \sim x^{-\alpha} L(x) \text{ as } x \to \infty$$

where $L(x)$ is a slowly varying function and $\alpha > 0$ is the tail index.

**Applications include:**

- **Finance**: Stock returns, portfolio risk, extreme losses
- **Insurance**: Catastrophic claims, reinsurance modeling
- **Network Analysis**: Internet traffic, social networks
- **Environmental Science**: Extreme weather, natural disasters

## 🚀 Installation

```bash
# From PyPI (recommended)
pip install heavytails

# Development installation
git clone https://github.com/diogoribeiro7/heavytails.git
cd heavytails
poetry install
```

## 🔍 Examples by Domain

### Finance & Risk Management

```python
from heavytails import GeneralizedPareto, BurrXII
from heavytails.finance_applications import RiskMetrics

# Value-at-Risk estimation with GPD
risk = RiskMetrics(portfolio_returns)
var_95 = risk.var_gpd(alpha=0.05, threshold=0.1)
```

### Extreme Value Analysis

```python
from heavytails import GEV_Frechet, Frechet
from heavytails.tail_index import pickands_estimator

# Block maxima analysis
gev = GEV_Frechet(xi=0.2, mu=10, sigma=2)
annual_maxima = gev.rvs(50, seed=42)
```

### Academic Research

```python
from heavytails import Cauchy, LogNormal
import matplotlib.pyplot as plt

# Compare tail behavior
cauchy = Cauchy(x0=0, gamma=1)
lognorm = LogNormal(mu=0, sigma=1)

x = np.logspace(0, 3, 1000)
plt.loglog(x, cauchy.sf(x), label='Cauchy')
plt.loglog(x, lognorm.sf(x), label='LogNormal')
plt.legend()
plt.title('Tail Comparison')
```

## 📖 Documentation Structure

### For Beginners

- **[Getting Started](getting-started/installation.md)**: Installation and first steps
- **[Basic Concepts](getting-started/concepts.md)**: Heavy-tail theory primer
- **[Quick Tutorial](getting-started/quickstart.md)**: 10-minute introduction

### For Practitioners

- **[User Guide](guide/distributions.md)**: Comprehensive usage guide
- **[Examples Gallery](examples/)**: Real-world applications
- **[CLI Reference](guide/cli.md)**: Command-line interface

### For Researchers

- **[Mathematical Background](theory/heavy-tails.md)**: Theoretical foundations
- **[API Reference](reference/)**: Complete function documentation
- **[Validation Studies](theory/validation.md)**: Numerical accuracy

### For Developers

- **[Contributing](development/contributing.md)**: How to contribute
- **[Architecture](development/architecture.md)**: Code organization
- **[Performance](development/benchmarks.md)**: Benchmarks and optimization

## 🎓 Academic Usage

### Citation

If you use HeavyTails in academic work, please cite:

```bibtex
@software{ribeiro2025heavytails,
  author = {Ribeiro, Diogo},
  title = {HeavyTails: A Pure-Python Library for Heavy-Tailed Probability Distributions},
  url = {https://github.com/diogoribeiro7/heavytails},
  version = {0.1.0},
  year = {2025}
}
```

### Research Applications

- **Extreme Value Theory**: Block maxima, peaks-over-threshold
- **Financial Econometrics**: Tail risk, copula modeling
- **Reliability Engineering**: Failure time analysis
- **Environmental Statistics**: Climate extremes

## 🤝 Community & Support

- **📧 Contact**: <dfr@esmad.ipp.pt>
- **💬 Discussions**: [GitHub Discussions](https://github.com/diogoribeiro7/heavytails/discussions)
- **🐛 Bug Reports**: [GitHub Issues](https://github.com/diogoribeiro7/heavytails/issues)
- **🎓 Academic Collaboration**: [ORCID](https://orcid.org/0009-0001-2022-7072)

## 🌟 Key Features

### ✅ **Pure Python**

No external dependencies - perfect for understanding algorithms and educational use.

### ✅ **Mathematically Rigorous**

Proper implementation of special functions, numerical stability, academic references.

### ✅ **Production Ready**

Comprehensive testing, CI/CD pipeline, performance benchmarks.

### ✅ **Educational Focus**

Clear documentation, mathematical background, step-by-step examples.

### ✅ **Research Grade**

Validation against reference implementations, proper citations, reproducible results.

--------------------------------------------------------------------------------

## Next Steps

1. **[Install HeavyTails](getting-started/installation.md)** and try the quick start example
2. **[Explore Examples](examples/)** relevant to your domain
3. **[Read the Theory](theory/heavy-tails.md)** for mathematical background
4. **[Join the Community](https://github.com/diogoribeiro7/heavytails/discussions)** and ask questions

**Happy analyzing! 📊📈**
