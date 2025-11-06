#!/usr/bin/env python3
"""
Generate sample datasets for HeavyTails library examples and testing.

This script creates various heavy-tailed datasets for demonstration,
validation, and testing purposes.
"""

import csv
import json
import math
from pathlib import Path
import random

from heavytails import (
    Frechet,
    GeneralizedPareto,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)
from heavytails.tail_index import hill_estimator


def generate_financial_returns(n: int = 2500, seed: int = 42) -> list[float]:
    """Generate simulated daily financial returns with heavy tails."""
    random.seed(seed)

    # Mix of normal and heavy-tailed periods
    returns = []

    # 80% Student-t(3) periods (financial crisis periods)
    student_t = StudentT(nu=3.0)
    heavy_returns = student_t.rvs(int(n * 0.8), seed=seed)
    returns.extend([r * 0.02 for r in heavy_returns])  # Scale to 2% volatility

    # 20% normal periods
    normal_returns = [random.gauss(0, 0.015) for _ in range(int(n * 0.2))]
    returns.extend(normal_returns)

    # Shuffle to mix periods
    random.shuffle(returns)
    return returns[:n]


def generate_insurance_claims(n: int = 1000, seed: int = 42) -> list[float]:
    """Generate insurance claim amounts with Pareto tail."""
    random.seed(seed)

    # Most claims are small (exponential)
    small_claims = [-math.log(random.random()) * 1000 for _ in range(int(n * 0.9))]

    # Large claims follow Pareto
    pareto = Pareto(alpha=1.8, xm=5000)
    large_claims = pareto.rvs(int(n * 0.1), seed=seed + 1)

    all_claims = small_claims + large_claims
    random.shuffle(all_claims)
    return all_claims[:n]


def generate_extreme_weather(n: int = 365, seed: int = 42) -> list[float]:
    """Generate extreme weather event magnitudes (wind speeds, rainfall, etc.)."""
    random.seed(seed)

    # Use Frechet distribution for extreme events
    frechet = Frechet(alpha=2.5, s=20.0, m=0.0)
    return frechet.rvs(n, seed=seed)


def generate_failure_times(n: int = 500, seed: int = 42) -> list[float]:
    """Generate component failure times with Weibull distribution."""
    random.seed(seed)

    # Weibull with k < 1 for heavy tail (early failures)
    weibull = Weibull(k=0.8, lam=1000.0)  # Mean time to failure ~1000 hours
    return weibull.rvs(n, seed=seed)


def generate_network_traffic(n: int = 1440, seed: int = 42) -> list[float]:
    """Generate network traffic bursts (packets/minute over a day)."""
    random.seed(seed)

    # Log-normal for typical traffic, Pareto for bursts
    lognormal = LogNormal(mu=6.0, sigma=1.0)  # Base traffic
    base_traffic = lognormal.rvs(int(n * 0.95), seed=seed)

    # Heavy bursts
    pareto = Pareto(alpha=1.5, xm=2000)
    bursts = pareto.rvs(int(n * 0.05), seed=seed + 1)

    all_traffic = list(base_traffic) + list(bursts)
    random.shuffle(all_traffic)
    return all_traffic[:n]


def generate_validation_data(seed: int = 42) -> dict:
    """Generate validation data with known parameters for testing."""
    random.seed(seed)

    validation_sets = {}

    # Pareto with known parameters
    pareto_true = Pareto(alpha=2.0, xm=1.0)
    pareto_data = pareto_true.rvs(5000, seed=seed)
    pareto_hill = hill_estimator(pareto_data, k=200)

    validation_sets['pareto_alpha_2'] = {
        'data': pareto_data[:100],  # Store subset for file size
        'true_parameters': {'alpha': 2.0, 'xm': 1.0},
        'estimated_gamma': pareto_hill,
        'estimated_alpha': 1.0 / pareto_hill,
        'sample_size': len(pareto_data),
        'distribution': 'Pareto'
    }

    # GPD with known parameters
    gpd_true = GeneralizedPareto(xi=0.3, sigma=1.0, mu=0.0)
    gpd_data = gpd_true.rvs(3000, seed=seed + 1)

    validation_sets['gpd_xi_03'] = {
        'data': gpd_data[:100],
        'true_parameters': {'xi': 0.3, 'sigma': 1.0, 'mu': 0.0},
        'sample_size': len(gpd_data),
        'distribution': 'GeneralizedPareto'
    }

    return validation_sets


def save_csv_data(data: list[float], filename: str, header: str = 'value'):
    """Save data to CSV file."""
    filepath = Path('data') / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([header, 'index'])
        for i, value in enumerate(data):
            writer.writerow([value, i])

    print(f"Generated {filename} with {len(data)} observations")


def save_json_data(data: dict, filename: str):
    """Save data to JSON file."""
    filepath = Path('data') / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open('w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {filename}")


def main():
    """Generate all sample datasets."""
    print("Generating sample datasets for HeavyTails library...")

    # Financial datasets
    financial_returns = generate_financial_returns(2500)
    save_csv_data(financial_returns, 'financial_returns.csv', 'log_return')

    insurance_claims = generate_insurance_claims(1000)
    save_csv_data(insurance_claims, 'insurance_claims.csv', 'claim_amount')

    # Environmental datasets
    extreme_weather = generate_extreme_weather(365)
    save_csv_data(extreme_weather, 'extreme_weather.csv', 'wind_speed_ms')

    # Engineering datasets
    failure_times = generate_failure_times(500)
    save_csv_data(failure_times, 'failure_times.csv', 'hours_to_failure')

    network_traffic = generate_network_traffic(1440)
    save_csv_data(network_traffic, 'network_traffic.csv', 'packets_per_minute')

    # Validation datasets
    validation_data = generate_validation_data()
    save_json_data(validation_data, 'validation/known_parameters.json')

    print(f"\n✅ Generated {6} datasets in the data/ directory")
    print("These datasets can be used for:")
    print("  - Testing and validation")
    print("  - Documentation examples")
    print("  - Performance benchmarking")
    print("  - Educational demonstrations")


if __name__ == "__main__":
    main()
