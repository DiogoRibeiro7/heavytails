# Sample Datasets for HeavyTails Library

This directory contains sample datasets for testing, validation, and examples.

## Dataset Categories

### 📈 **Financial Data**

- `financial_returns.csv` - Daily stock returns (simulated heavy-tailed data)
- `insurance_claims.csv` - Insurance claim amounts
- `portfolio_losses.csv` - Portfolio loss data for risk modeling

### 🌪️ **Environmental Data**

- `extreme_weather.csv` - Extreme weather events
- `wind_speeds.csv` - Wind speed measurements
- `flood_levels.csv` - Flood level data

### 🏭 **Engineering Data**

- `failure_times.csv` - Component failure times
- `network_traffic.csv` - Network traffic bursts
- `material_strengths.csv` - Material strength testing data

### 🧮 **Reference Data**

- `validation_data.json` - Reference values for testing against R/SciPy
- `known_distributions.json` - Datasets with known distribution parameters

## Data Generation

All datasets are either:

1. **Simulated** using known heavy-tailed distributions
2. **Anonymized** real data (when available)
3. **Public domain** datasets from academic sources

## Usage in Examples

```python
import pandas as pd
from pathlib import Path

# Load sample financial data
data_dir = Path("data")
returns = pd.read_csv(data_dir / "financial_returns.csv")

# Use with heavytails
from heavytails import Pareto
from heavytails.tail_index import hill_estimator

# Estimate tail index
tail_index = hill_estimator(returns['value'].tolist(), k=100)
```

## Data Sources & Citations

- **Financial data**: Simulated using Student-t and Pareto distributions
- **Environmental data**: Based on publicly available weather station data
- **Engineering data**: Simulated reliability and network performance data

## Validation Data

The `validation/` subdirectory contains:

- Reference calculations from R packages (`evd`, `fExtremes`)
- SciPy comparison data
- Known analytical results for parameter estimation

## Contributing Data

To contribute additional datasets:

1. Ensure data is anonymized/simulated
2. Include clear documentation of source and characteristics
3. Provide citation information if from published research
4. Include validation scripts if possible

--------------------------------------------------------------------------------

_All datasets are provided for educational and research purposes._
