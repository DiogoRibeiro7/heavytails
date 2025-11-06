# Contributing to HeavyTails

Thank you for your interest in contributing to HeavyTails! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Release Process](#release-process)

## Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/). By participating, you agree to uphold this code. Please report unacceptable behavior to <dfr@esmad.ipp.pt>.

## Getting Started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- Git for version control

### Development Setup

1. **Fork and clone the repository:**

  ```bash
  git clone https://github.com/yourusername/heavytails.git
  cd heavytails
  ```

2. **Install development dependencies:**

  ```bash
  poetry install --with dev,docs,examples
  ```

3. **Set up pre-commit hooks:**

  ```bash
  poetry run pre-commit install
  ```

4. **Verify installation:**

  ```bash
  poetry run pytest tests/ --tb=short
  ```

### Project Structure

```
heavytails/
├── heavytails/           # Main package
│   ├── heavy_tails.py    # Core distributions
│   ├── extra_distributions.py  # Extended distributions
│   ├── discrete.py       # Discrete distributions
│   ├── tail_index.py     # Tail index estimators
│   ├── plotting.py       # Diagnostic plotting
│   └── cli.py           # Command-line interface
├── tests/               # Test suite
├── examples/            # Example notebooks and scripts
├── docs/               # Documentation source
├── scripts/            # Utility scripts
└── .github/            # CI/CD workflows
```

## Development Workflow

### Branch Strategy

- **main**: Stable release branch
- **develop**: Integration branch for new features
- **feature/**: Feature development branches
- **hotfix/**: Critical bug fix branches

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Recommended Development Process

1. **Start with tests**: Write tests for new functionality first (TDD)
2. **Implement feature**: Write the minimal code to pass tests
3. **Add documentation**: Update docstrings and user documentation
4. **Run quality checks**: Ensure linting, type checking, and tests pass
5. **Submit PR**: Create a pull request with clear description

## Code Style Guidelines

### Python Style

We follow PEP 8 with some modifications enforced by our tools:

- **Line length**: 88 characters (Black default)
- **Import sorting**: isort configuration
- **Type annotations**: Required for all public APIs
- **Docstrings**: NumPy style for all public functions/classes

### Code Quality Tools

```bash
# Linting and formatting
poetry run ruff check .          # Check code style
poetry run ruff format .         # Format code
poetry run mypy heavytails/      # Type checking

# Run all checks
poetry run pre-commit run --all-files
```

### Mathematical Code Guidelines

- **Numerical stability**: Consider edge cases and parameter extremes
- **Documentation**: Explain mathematical formulas and cite sources
- **Accuracy**: Include numerical tolerance considerations
- **Performance**: Profile critical mathematical operations

Example of well-documented mathematical function:

```python
def incomplete_beta_reg(a: float, b: float, x: float) -> float:
    """
    Regularized incomplete beta function I_x(a,b).

    Uses continued fraction expansion with Lentz's algorithm for numerical
    stability. Implements symmetry reduction for improved convergence.

    Parameters
    ----------
    a, b : float
        Shape parameters, must be > 0
    x : float
        Upper integration limit, must be in [0, 1]

    Returns
    -------
    float
        Value of I_x(a,b) with relative accuracy ~1e-12

    References
    ----------
    Abramowitz & Stegun, Section 26.5
    Press et al., Numerical Recipes, Chapter 6.4
    """
```

## Testing

### Test Categories

1. **Unit tests**: Test individual functions/classes
2. **Property-based tests**: Use Hypothesis for mathematical properties
3. **Integration tests**: Test component interactions
4. **Numerical tests**: Validate against known results
5. **Performance tests**: Benchmark critical operations

### Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=heavytails --cov-report=html

# Run specific test categories
poetry run pytest -m "not slow"           # Skip slow tests
poetry run pytest -m "property"           # Property-based tests only
poetry run pytest tests/test_pareto.py    # Specific test file
```

### Writing Tests

#### Property-Based Testing Example

```python
from hypothesis import given, strategies as st
from heavytails import Pareto

@given(
    alpha=st.floats(min_value=0.1, max_value=10),
    xm=st.floats(min_value=0.1, max_value=100),
    u=st.floats(min_value=1e-6, max_value=1-1e-6)
)
def test_pareto_ppf_cdf_inverse(alpha: float, xm: float, u: float):
    """Test that PPF and CDF are inverses."""
    dist = Pareto(alpha=alpha, xm=xm)
    x = dist.ppf(u)
    recovered_u = dist.cdf(x)
    assert abs(recovered_u - u) < 1e-10
```

#### Numerical Validation Example

```python
def test_pareto_against_analytical():
    """Test Pareto implementation against analytical results."""
    dist = Pareto(alpha=2.0, xm=1.0)

    # Analytical CDF: F(x) = 1 - (xm/x)^alpha
    x = 2.0
    expected_cdf = 1 - (1.0/2.0)**2.0  # = 0.75
    actual_cdf = dist.cdf(x)

    assert abs(actual_cdf - expected_cdf) < 1e-15
```

### Test Data and Fixtures

- Store reference data in `tests/data/`
- Use `pytest` fixtures for common test setup
- Include data sources and generation methods in documentation

## Documentation

### API Documentation

- **Docstrings**: NumPy style for all public APIs
- **Type hints**: Complete type annotations
- **Examples**: Include usage examples in docstrings
- **Mathematical notation**: Use LaTeX for formulas

### User Documentation

- **Tutorials**: Step-by-step guides for common tasks
- **Examples**: Real-world applications with data
- **Theory**: Mathematical background and references
- **API Reference**: Auto-generated from docstrings

### Building Documentation

```bash
# Install docs dependencies
poetry install --with docs

# Build documentation
poetry run mkdocs build

# Serve locally for development
poetry run mkdocs serve
```

## Submitting Changes

### Pull Request Process

1. **Ensure tests pass**: All tests must pass on your branch
2. **Update documentation**: Include relevant doc updates
3. **Add changelog entry**: Update CHANGELOG.md
4. **Descriptive PR**: Clear title and description
5. **Link issues**: Reference related issues with keywords

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Breaking change

## Testing
- [ ] All tests pass
- [ ] Added new tests for new functionality
- [ ] Validated against reference implementations

## Documentation
- [ ] Updated docstrings
- [ ] Updated user documentation
- [ ] Added examples if applicable

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] CHANGELOG.md updated
```

### Commit Message Guidelines

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

<body>

<footer>
```

Examples:

- `feat(pareto): add bounded Pareto distribution`
- `fix(cli): handle edge case in parameter validation`
- `docs(api): add mathematical background for GPD`
- `test(integration): add cross-validation tests`

## Mathematical Contributions

### Distribution Implementations

When adding new distributions:

1. **Mathematical validation**: Verify formulas against multiple sources
2. **Parameter constraints**: Implement thorough parameter validation
3. **Special cases**: Handle edge cases and limiting behaviors
4. **Numerical stability**: Test with extreme parameter values
5. **Performance**: Benchmark against existing implementations

### Required Methods

All distributions must implement:

```python
class NewDistribution(Samplable):
    def pdf(self, x: float) -> float: ...
    def cdf(self, x: float) -> float: ...
    def ppf(self, u: float) -> float: ...  # If analytically available
    def _rvs_one(self, rng: RNG) -> float: ...
```

### Mathematical References

Include references for:

- Original papers defining the distribution
- Numerical algorithms used
- Parameter estimation methods
- Known applications and properties

## Performance Considerations

### Profiling

```bash
# Profile performance
poetry run python -m cProfile -o profile.prof examples/benchmark.py
poetry run snakeviz profile.prof

# Line profiling
poetry run kernprof -l -v script.py
```

### Optimization Guidelines

- **Algorithmic efficiency**: Choose appropriate algorithms for the problem scale
- **Numerical stability**: Prefer numerically stable formulations
- **Memory efficiency**: Avoid unnecessary array allocations
- **Caching**: Cache expensive computations when appropriate

## Release Process

### Version Bumping

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Commit changes**: `git commit -m "chore: bump version to X.Y.Z"`
4. **Create tag**: `git tag -a vX.Y.Z -m "Release vX.Y.Z"`
5. **Push**: `git push origin main --tags`

### Automated Release

GitHub Actions automatically:

- Runs full test suite
- Builds distribution packages
- Publishes to PyPI (on tag push)
- Deploys documentation
- Creates GitHub release

## Getting Help

- **Issues**: Use GitHub issues for bugs and feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact maintainer at <dfr@esmad.ipp.pt>
- **Documentation**: Check the [online documentation](https://diogoribeiro7.github.io/heavytails)

## Recognition

Contributors will be recognized in:

- AUTHORS.md file
- Release notes
- Documentation credits
- Academic papers (for significant mathematical contributions)

Thank you for contributing to HeavyTails! 🎯
