---
- >-
  Check [existing issues](https://github.com/diogoribeiro7/heavytails/issues)
  first
- Provide minimal reproducible examples
- Include Python version and operating system
- Describe expected vs. actual behavior
- >-
  Open a [feature
  request](https://github.com/diogoribeiro7/heavytails/issues/new)
- Explain the use case and motivation
- Provide examples or references
- Discuss implementation approach
- Fix typos and clarify explanations
- Add examples and tutorials
- Improve docstrings and type hints
- Translate documentation
- Propose new heavy-tailed families
- Implement discrete distributions
- Add multivariate extensions
- Contribute copula models
- Increase test coverage
- Add edge case tests
- Cross-validate against SciPy/R
- Add performance benchmarks
---

# Contributing to HeavyTails

Thank you for your interest in contributing to HeavyTails! This guide will help you get started with contributions, whether you're fixing bugs, adding features, improving documentation, or proposing new distributions.

## Development Setup

### Prerequisites

- **Python 3.8+**
- **Poetry** (recommended) or pip
- **Git**

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork:

```bash
git clone https://github.com/YOUR_USERNAME/heavytails.git
cd heavytails
```

1. Add upstream remote:

```bash
git remote add upstream https://github.com/diogoribeiro7/heavytails.git
```

### Install Development Dependencies

=== "Poetry (Recommended)"

````
```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install project and dev dependencies
poetry install --with dev,docs,test

# Activate virtual environment
poetry shell
```
````

=== "pip"

````
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e ".[dev,docs,test]"
```
````

--------------------------------------------------------------------------------

## Development Workflow

### Create a Feature Branch

```bash
# Update your main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Make Changes

Follow the coding standards below when making changes.

### Run Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=heavytails --cov-report=html

# Run specific test file
poetry run pytest tests/test_pareto.py

# Run with verbose output
poetry run pytest -v
```

### Check Code Quality

```bash
# Format code with ruff
poetry run ruff format .

# Lint code
poetry run ruff check .

# Fix auto-fixable issues
poetry run ruff check --fix .

# Type checking (if using mypy)
poetry run mypy heavytails
```

### Build Documentation Locally

```bash
# Build docs
poetry run mkdocs build

# Serve docs locally
poetry run mkdocs serve
# Open http://127.0.0.1:8000 in your browser
```

### Commit Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "feat: add Burr Type XII distribution

- Implement PDF, CDF, PPF, and sampling
- Add comprehensive tests
- Update documentation
"
```

**Commit message format:**

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `style:` Formatting changes
- `chore:` Build/maintenance tasks

### Push and Create Pull Request

```bash
# Push to your fork
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub:

1. Go to your fork on GitHub
2. Click "New Pull Request"
3. Fill in the template
4. Wait for review

--------------------------------------------------------------------------------

## Coding Standards

### Code Style

- **PEP 8** compliance (enforced by ruff)
- **Type hints** for all public functions
- **Docstrings** in NumPy style
- **No external dependencies** in core library (except standard library)

### Example Function

```python
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

@dataclass(frozen=True)
class MyDistribution:
    """
    Brief one-line description.

    Detailed description of the distribution, including mathematical
    formulation and use cases.

    Parameters
    ----------
    alpha : float
        Shape parameter, must be > 0.
    beta : float
        Scale parameter, must be > 0.

    Examples
    --------
    >>> dist = MyDistribution(alpha=2.0, beta=1.0)
    >>> samples = dist.rvs(100, seed=42)
    >>> print(dist.mean())
    1.0

    Notes
    -----
    Mathematical details, references to papers, etc.

    References
    ----------
    .. [1] Author, A. (Year). "Title". Journal, vol(issue), pages.
    """

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.alpha <= 0:
            raise ValueError("alpha must be > 0")
        if self.beta <= 0:
            raise ValueError("beta must be > 0")

    def pdf(self, x: float) -> float:
        """
        Probability density function.

        Parameters
        ----------
        x : float
            Evaluation point.

        Returns
        -------
        float
            PDF value at x.
        """
        if x <= 0:
            return 0.0
        return (self.alpha / self.beta) * (x / self.beta) ** (self.alpha - 1)
```

### Testing Standards

Every new feature must include tests:

```python
import pytest
from heavytails import MyDistribution

def test_my_distribution_creation():
    """Test distribution initialization."""
    dist = MyDistribution(alpha=2.0, beta=1.0)
    assert dist.alpha == 2.0
    assert dist.beta == 1.0

def test_my_distribution_invalid_params():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        MyDistribution(alpha=-1.0, beta=1.0)

    with pytest.raises(ValueError):
        MyDistribution(alpha=2.0, beta=0.0)

def test_pdf_values():
    """Test PDF against known values."""
    dist = MyDistribution(alpha=2.0, beta=1.0)

    assert dist.pdf(0.5) == pytest.approx(1.0, rel=1e-6)
    assert dist.pdf(1.0) == pytest.approx(2.0, rel=1e-6)
    assert dist.pdf(-1.0) == 0.0

def test_sampling_reproducibility():
    """Test that sampling with seed is reproducible."""
    dist = MyDistribution(alpha=2.0, beta=1.0)

    samples1 = dist.rvs(100, seed=42)
    samples2 = dist.rvs(100, seed=42)

    assert samples1 == samples2
```

--------------------------------------------------------------------------------

## Adding a New Distribution

### Step-by-Step Guide

1. **Research the distribution**

  - Mathematical properties
  - Parameter constraints
  - Tail behavior
  - Applications

2. **Implement the class**

  Create in `heavytails/extra_distributions.py` or a new module:

  ```python
  @dataclass(frozen=True)
  class NewDistribution(Samplable):
      """Distribution description."""

      param1: float
      param2: float

      def __post_init__(self) -> None:
          # Validate parameters
          pass

      def pdf(self, x: float) -> float:
          """Probability density function."""
          pass

      def cdf(self, x: float) -> float:
          """Cumulative distribution function."""
          pass

      def sf(self, x: float) -> float:
          """Survival function."""
          return 1.0 - self.cdf(x)

      def ppf(self, u: float) -> float:
          """Percent point function (quantile)."""
          pass

      def _rvs_one(self, rng: RNG) -> float:
          """Generate one random variate."""
          return self.ppf(rng.uniform_0_1())

      def mean(self) -> float:
          """Expected value (if exists)."""
          pass

      def variance(self) -> float:
          """Variance (if exists)."""
          pass
  ```

3. **Write comprehensive tests**

  Create `tests/test_new_distribution.py`:

  ```python
  def test_pdf_integrates_to_one():
      """Test that PDF integrates to 1."""
      pass

  def test_cdf_bounds():
      """Test CDF is in [0, 1]."""
      pass

  def test_quantile_inversion():
      """Test PPF inverts CDF."""
      pass

  def test_moments():
      """Test analytical moments match numerical."""
      pass
  ```

4. **Add documentation**

  Update:

  - `docs/guide/distributions.md` - Add to reference table
  - `docs/api/extra.md` - API documentation
  - `README.md` - Add to list if major distribution

5. **Update exports**

  In `heavytails/__init__.py`:

  ```python
  from .extra_distributions import NewDistribution

  __all__ = [
      ...,
      "NewDistribution",
  ]
  ```

6. **Add example**

  Create example in `examples/new_distribution_demo.py`

--------------------------------------------------------------------------------

## Pull Request Guidelines

### PR Checklist

- [ ] Code follows style guidelines (ruff passes)
- [ ] All tests pass (`pytest`)
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Docstrings added/updated
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No breaking changes (or clearly documented)
- [ ] Commit messages are clear and descriptive

### PR Description Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Motivation
Why is this change needed?

## Testing
How was this tested?

## Related Issues
Closes #123, Relates to #456

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] Code follows style guide
- [ ] All tests pass
```

--------------------------------------------------------------------------------

## Code Review Process

1. **Maintainer review** - Core team reviews code
2. **CI checks** - Automated tests must pass
3. **Discussion** - Feedback and iteration
4. **Approval** - At least one maintainer approval
5. **Merge** - Squash and merge to main

--------------------------------------------------------------------------------

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):

- **Major (1.0.0):** Breaking changes
- **Minor (0.1.0):** New features, backward compatible
- **Patch (0.0.1):** Bug fixes

--------------------------------------------------------------------------------

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Provide constructive feedback
- Focus on technical merits
- Assume good intentions

### Getting Help

- **Questions:** Use [GitHub Discussions](https://github.com/diogoribeiro7/heavytails/discussions)
- **Bugs:** Open an [issue](https://github.com/diogoribeiro7/heavytails/issues)
- **Email:** [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)

--------------------------------------------------------------------------------

## Recognition

Contributors are recognized in:

- `AUTHORS.md` file
- Release notes
- Documentation acknowledgments

Thank you for contributing to HeavyTails! 🎉
