import math
import random

import pytest

from heavytails.tail_index import hill_estimator, moment_estimator, pickands_estimator


def test_hill_pareto():
    data = [((1 - random.random()) ** (-1 / 1.5)) for _ in range(5000)]
    gamma = hill_estimator(data, k=100)
    assert 0.4 < gamma < 1.0


def test_hill_estimator_invalid_k():
    """Test Hill estimator with invalid k values."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # k must be > 1
    with pytest.raises(ValueError, match="k must be between 1 and n-1"):
        hill_estimator(data, k=1)

    # k must be < n
    with pytest.raises(ValueError, match="k must be between 1 and n-1"):
        hill_estimator(data, k=5)

    # k must be < n
    with pytest.raises(ValueError, match="k must be between 1 and n-1"):
        hill_estimator(data, k=10)


def test_moment_estimator_consistency():
    data = [((1 - random.random()) ** (-1 / 2.0)) for _ in range(3000)]
    gamma, alpha = moment_estimator(data, k=150)
    assert math.isclose(alpha, 1 / gamma, rel_tol=1e-8)


def test_moment_estimator_invalid_k():
    """Test moment estimator with invalid k values."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    # k must be > 1
    with pytest.raises(ValueError, match="k must be between 1 and n-1"):
        moment_estimator(data, k=1)

    # k must be < n
    with pytest.raises(ValueError, match="k must be between 1 and n-1"):
        moment_estimator(data, k=5)


def test_pickands_estimator_basic():
    """Test Pickands estimator with valid data."""
    # Generate Pareto data
    data = [((1 - random.random()) ** (-1 / 2.0)) for _ in range(1000)]
    gamma = pickands_estimator(data, k=20, m=2)

    # Should give a reasonable estimate
    assert 0.1 < gamma < 2.0


def test_pickands_estimator_invalid_sample_size():
    """Test Pickands estimator with sample too small."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    with pytest.raises(ValueError, match="Sample too small"):
        pickands_estimator(data, k=2, m=2)
