# Testing Guide

This guide covers testing practices for the HeavyTails library, including writing tests, running the test suite, and ensuring code quality.

---

## Testing Philosophy

HeavyTails follows these testing principles:

1. **Comprehensive Coverage** - All code paths tested
2. **Mathematical Validation** - Cross-check against known values
3. **Numerical Stability** - Test edge cases and extremes
4. **Reproducibility** - Deterministic tests with seeds
5. **Fast Execution** - Quick feedback for developers

---

## Running Tests

### Run All Tests

```bash
# Using Poetry
poetry run pytest

# Using pip
pytest
```

### Run with Coverage

```bash
# Generate coverage report
poetry run pytest --cov=heavytails --cov-report=html

# Open report
open htmlcov/index.html  # On macOS
xdg-open htmlcov/index.html  # On Linux
start htmlcov/index.html  # On Windows
```

### Run Specific Tests

```bash
# Single test file
pytest tests/test_pareto.py

# Specific test function
pytest tests/test_pareto.py::test_pareto_pdf

# Tests matching pattern
pytest -k "pareto"

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

---

## Test Organization

### Directory Structure

```
tests/
├── __init__.py
├── test_pareto.py                 # Core Pareto tests
├── test_student_t.py              # Student-t tests
├── test_generalized_pareto.py     # GPD tests
├── test_tail_index.py             # Estimator tests
├── test_comprehensive.py          # Integration tests
└── test_performance.py            # Performance benchmarks
```

### Test Categories

- **Unit Tests** - Individual methods and functions
- **Integration Tests** - Cross-module functionality
- **Property Tests** - Mathematical properties
- **Validation Tests** - Against reference implementations
- **Performance Tests** - Speed and efficiency

---

## Writing Tests

### Basic Test Structure

```python
import pytest
from heavytails import Pareto

def test_pareto_creation():
    """Test Pareto distribution initialization."""
    dist = Pareto(alpha=2.0, xm=1.0)
    assert dist.alpha == 2.0
    assert dist.xm == 1.0

def test_pareto_invalid_parameters():
    """Test parameter validation."""
    with pytest.raises(ValueError):
        Pareto(alpha=-1.0, xm=1.0)

    with pytest.raises(ValueError):
        Pareto(alpha=2.0, xm=0.0)
```

### Testing Numerical Values

Use `pytest.approx()` for floating-point comparisons:

```python
def test_pareto_pdf():
    """Test PDF against known values."""
    dist = Pareto(alpha=2.0, xm=1.0)

    # Test at specific points
    assert dist.pdf(1.0) == pytest.approx(2.0, rel=1e-10)
    assert dist.pdf(2.0) == pytest.approx(0.5, rel=1e-10)
    assert dist.pdf(0.5) == 0.0  # Outside support

def test_pareto_mean():
    """Test analytical mean."""
    dist = Pareto(alpha=2.5, xm=1.0)
    expected_mean = 2.5 / 1.5  # α*xm/(α-1)

    assert dist.mean() == pytest.approx(expected_mean, rel=1e-10)
```

### Testing Probability Properties

```python
def test_cdf_bounds():
    """Test that CDF is in [0, 1]."""
    dist = Pareto(alpha=2.0, xm=1.0)

    test_points = [0.5, 1.0, 5.0, 10.0, 100.0]
    for x in test_points:
        cdf = dist.cdf(x)
        assert 0.0 <= cdf <= 1.0

def test_cdf_sf_complement():
    """Test that CDF + SF = 1."""
    dist = Pareto(alpha=2.0, xm=1.0)

    for x in [1.0, 2.0, 5.0, 10.0]:
        assert dist.cdf(x) + dist.sf(x) == pytest.approx(1.0, abs=1e-10)

def test_ppf_inverts_cdf():
    """Test that PPF inverts CDF."""
    dist = Pareto(alpha=2.0, xm=1.0)

    for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
        x = dist.ppf(p)
        assert dist.cdf(x) == pytest.approx(p, rel=1e-8)
```

### Testing Sampling

```python
def test_sampling_reproducibility():
    """Test that sampling with seed is reproducible."""
    dist = Pareto(alpha=2.0, xm=1.0)

    samples1 = dist.rvs(100, seed=42)
    samples2 = dist.rvs(100, seed=42)

    assert samples1 == samples2

def test_sampling_properties():
    """Test that samples have correct properties."""
    dist = Pareto(alpha=3.0, xm=1.0)
    samples = dist.rvs(10000, seed=42)

    # All samples above minimum
    assert all(x >= dist.xm for x in samples)

    # Sample mean close to analytical mean
    import statistics
    sample_mean = statistics.mean(samples)
    analytical_mean = dist.mean()

    assert sample_mean == pytest.approx(analytical_mean, rel=0.1)
```

### Parametrized Tests

Test multiple cases efficiently:

```python
@pytest.mark.parametrize("alpha,xm,expected_mean", [
    (2.0, 1.0, 2.0),
    (3.0, 1.0, 1.5),
    (2.5, 2.0, 10.0/3.0),
])
def test_pareto_mean_parametrized(alpha, xm, expected_mean):
    """Test mean for various parameters."""
    dist = Pareto(alpha=alpha, xm=xm)
    assert dist.mean() == pytest.approx(expected_mean, rel=1e-10)
```

---

## Validation Against References

### Cross-Validation with SciPy

```python
import pytest

try:
    from scipy.stats import pareto as scipy_pareto
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

@pytest.mark.skipif(not HAS_SCIPY, reason="SciPy not available")
def test_pareto_vs_scipy():
    """Validate against SciPy implementation."""
    from heavytails import Pareto

    # Our implementation
    our_dist = Pareto(alpha=2.5, xm=1.0)

    # SciPy implementation (note: different parameterization!)
    scipy_dist = scipy_pareto(b=2.5, scale=1.0)

    # Compare PDF values
    test_points = [1.0, 2.0, 5.0, 10.0]
    for x in test_points:
        our_pdf = our_dist.pdf(x)
        scipy_pdf = scipy_dist.pdf(x)
        assert our_pdf == pytest.approx(scipy_pdf, rel=1e-10)
```

### Mathematical Identities

```python
def test_pareto_tail_formula():
    """Test P(X > x) = (xm/x)^α."""
    dist = Pareto(alpha=2.5, xm=1.0)

    for x in [2.0, 5.0, 10.0]:
        # Our implementation
        sf = dist.sf(x)

        # Analytical formula
        expected_sf = (dist.xm / x) ** dist.alpha

        assert sf == pytest.approx(expected_sf, rel=1e-10)
```

---

## Testing Tail Index Estimators

### Hill Estimator Tests

```python
from heavytails import Pareto
from heavytails.tail_index import hill_estimator

def test_hill_estimator_pareto():
    """Test Hill estimator on Pareto data."""
    true_alpha = 2.5
    pareto = Pareto(alpha=true_alpha, xm=1.0)

    # Generate large sample
    data = pareto.rvs(5000, seed=42)

    # Estimate
    k = 500
    gamma_hat = hill_estimator(data, k)
    alpha_hat = 1.0 / gamma_hat

    # Should be close to true value (within 10%)
    assert alpha_hat == pytest.approx(true_alpha, rel=0.1)

def test_hill_estimator_consistency():
    """Test that Hill estimator is consistent."""
    true_alpha = 3.0
    pareto = Pareto(alpha=true_alpha, xm=1.0)

    # Increasing sample sizes
    sample_sizes = [1000, 5000, 10000]
    estimates = []

    for n in sample_sizes:
        data = pareto.rvs(n, seed=42)
        gamma = hill_estimator(data, k=int(n/10))
        estimates.append(1.0 / gamma)

    # Estimates should get closer to true value
    errors = [abs(est - true_alpha) for est in estimates]

    # Later estimates should have smaller error (on average)
    assert errors[-1] < errors[0] or abs(errors[-1] - errors[0]) < 0.1
```

---

## Performance Tests

### Benchmark Template

```python
import time
import pytest

@pytest.mark.benchmark
def test_pareto_sampling_speed():
    """Benchmark Pareto sampling."""
    from heavytails import Pareto

    dist = Pareto(alpha=2.0, xm=1.0)

    # Warm-up
    _ = dist.rvs(100, seed=42)

    # Benchmark
    start = time.time()
    samples = dist.rvs(100000, seed=42)
    elapsed = time.time() - start

    # Should generate at least 10,000 samples per second
    samples_per_second = len(samples) / elapsed
    assert samples_per_second > 10000

    print(f"\nSampling speed: {samples_per_second:.0f} samples/second")
```

---

## Test Fixtures

Use fixtures for common setup:

```python
import pytest
from heavytails import Pareto

@pytest.fixture
def standard_pareto():
    """Standard Pareto(2, 1) distribution."""
    return Pareto(alpha=2.0, xm=1.0)

@pytest.fixture
def pareto_samples(standard_pareto):
    """Sample data from Pareto."""
    return standard_pareto.rvs(1000, seed=42)

def test_with_fixture(standard_pareto):
    """Test using fixture."""
    assert standard_pareto.mean() == pytest.approx(2.0)

def test_with_samples_fixture(pareto_samples):
    """Test using samples fixture."""
    assert len(pareto_samples) == 1000
    assert all(x >= 1.0 for x in pareto_samples)
```

---

## Continuous Integration

Tests run automatically on GitHub Actions for:

- **Multiple Python versions** (3.8, 3.9, 3.10, 3.11, 3.12)
- **Multiple operating systems** (Ubuntu, macOS, Windows)
- **Code coverage** tracking
- **Linting** with ruff

### GitHub Actions Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Install dependencies
      run: |
        pip install poetry
        poetry install

    - name: Run tests
      run: poetry run pytest --cov=heavytails

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Coverage Goals

- **Overall coverage:** > 90%
- **Core distributions:** > 95%
- **Tail estimators:** > 90%
- **Utilities:** > 85%

### Check Coverage

```bash
# Generate HTML report
poetry run pytest --cov=heavytails --cov-report=html

# View missing coverage
poetry run pytest --cov=heavytails --cov-report=term-missing
```

---

## Best Practices

1. **Test edge cases** - Zero, infinity, NaN, negative values
2. **Use seeds** - Ensure reproducibility
3. **Test exceptions** - Verify parameter validation
4. **Compare to references** - SciPy, R packages, papers
5. **Document tests** - Clear docstrings
6. **Fast tests** - Keep test suite under 30 seconds
7. **Deterministic** - No random failures

---

## Debugging Failed Tests

### Run with Verbose Output

```bash
pytest -vv tests/test_pareto.py::test_failing_test
```

### Use Debugger

```bash
pytest --pdb tests/test_pareto.py::test_failing_test
```

### Print Debugging

```python
def test_with_debug():
    dist = Pareto(alpha=2.0, xm=1.0)
    result = dist.pdf(5.0)

    print(f"\nPDF(5.0) = {result}")  # Will show with pytest -s
    assert result > 0
```

---

## Next Steps

- **[Contributing Guide](contributing.md)** - Full contribution workflow
- **[Benchmarks](benchmarks.md)** - Performance optimization
- **[Code Review](../development/code-review.md)** - Review standards

---

**Questions?** Open an issue or discussion on GitHub!
