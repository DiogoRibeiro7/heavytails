import math
import random

from heavytails.tail_index import hill_estimator, moment_estimator


def test_hill_pareto():
    data = [((1-random.random()) ** (-1/1.5)) for _ in range(5000)]
    gamma = hill_estimator(data, k=100)
    assert 0.4 < gamma < 1.0

def test_moment_estimator_consistency():
    data = [((1-random.random()) ** (-1/2.0)) for _ in range(3000)]
    gamma, alpha = moment_estimator(data, k=150)
    assert math.isclose(alpha, 1/gamma, rel_tol=1e-8)
