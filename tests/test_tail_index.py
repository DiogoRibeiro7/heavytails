import math
import random

import pytest

from heavytails.tail_index import hill_estimator, moment_estimator, pickands_estimator


def _pareto_sample(rng: random.Random, alpha: float, n: int) -> list[float]:
    """Draw n Pareto(alpha) variates by inversion, from a generator we own.

    A local generator rather than the module-level one: seeding
    ``random.seed`` globally would leak into whatever ran next, and using the
    module-level generator unseeded is what made these tests intermittent.
    """
    return [(1 - rng.random()) ** (-1 / alpha) for _ in range(n)]


def test_hill_pareto():
    data = _pareto_sample(random.Random(20240517), 1.5, 5000)
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
    data = _pareto_sample(random.Random(20240517), 2.0, 3000)
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
    """The Pickands estimator is centred on the true index.

    Averaged over replications, not read off one. At k=20 on 1000 points the
    estimator has a standard deviation of about 0.44 around a true 0.5, and its
    first percentile is below zero -- a single draw carries almost no
    information about whether the implementation is right.

    This test used to draw one unseeded sample and require it to land in
    (0.1, 2.0). That fails for 18% of seeds, so it was a nearly one-in-five
    chance of a red build on every run, and it failed on a macOS job for
    exactly that reason. Averaging 200 replications divides the spread by
    about 14, which is what makes the bound below both meaningful and quiet.
    """
    rng = random.Random(20240517)
    estimates = [
        pickands_estimator(_pareto_sample(rng, 2.0, 1000), k=20, m=2)
        for _ in range(200)
    ]
    mean = sum(estimates) / len(estimates)

    # Standard error of this mean is about 0.031, so 0.15 is roughly five of
    # them: wide enough never to fire on noise, tight enough that an estimator
    # centred on the wrong number could not pass.
    assert abs(mean - 0.5) < 0.15, f"Pickands is centred on {mean:.3f}, not 0.5"


def test_pickands_estimator_invalid_sample_size():
    """Test Pickands estimator with sample too small."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    with pytest.raises(ValueError, match="Sample too small"):
        pickands_estimator(data, k=2, m=2)
