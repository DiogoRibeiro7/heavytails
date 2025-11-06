# Installation Guide

This guide covers different methods for installing the HeavyTails library.

--------------------------------------------------------------------------------

## Requirements

HeavyTails has **zero external dependencies** and requires only:

- **Python 3.8 or higher**
- Python standard library (`math`, `random`)

!!! tip "Pure Python Advantage" Since HeavyTails uses only the Python standard library, there are no complex dependency chains to manage, making it ideal for:

```
- **Educational environments** - students can inspect all code
- **Restricted systems** - no external packages needed
- **Reproducibility** - minimal version conflicts
- **Understanding** - all algorithms are explicit
```

--------------------------------------------------------------------------------

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The simplest method is to install from the Python Package Index using pip:

```bash
pip install heavytails
```

To upgrade to the latest version:

```bash
pip install --upgrade heavytails
```

### Method 2: Install with Poetry

If you use Poetry for dependency management:

```bash
poetry add heavytails
```

### Method 3: Development Installation

For contributors or those who want to modify the source code:

```bash
# Clone the repository
git clone https://github.com/diogoribeiro7/heavytails.git
cd heavytails

# Install with Poetry (recommended for development)
poetry install

# Or install with pip in editable mode
pip install -e .
```

### Method 4: Direct Source Installation

Download and install directly from source:

```bash
# Download the latest release
wget https://github.com/diogoribeiro7/heavytails/archive/refs/heads/main.zip
unzip main.zip
cd heavytails-main

# Install
pip install .
```

--------------------------------------------------------------------------------

## Verify Installation

After installation, verify that HeavyTails is working correctly:

```python
# Test basic import
import heavytails

# Check version
print(heavytails.__version__)

# Test a simple distribution
from heavytails import Pareto

pareto = Pareto(alpha=2.0, xm=1.0)
samples = pareto.rvs(10, seed=42)
print(f"Sample values: {samples}")
```

Expected output:

```
0.1.0
Sample values: [1.234, 2.456, 1.789, ...]
```

--------------------------------------------------------------------------------

## Installation for Different Use Cases

### For Academic Research

```bash
# Standard installation
pip install heavytails

# Verify numerical accuracy
python -c "from heavytails import Pareto; p = Pareto(2.0, 1.0); print(p.mean())"
# Expected: 2.0
```

### For Financial Applications

```bash
# Install HeavyTails
pip install heavytails

# Optional: Install visualization tools (not required by HeavyTails)
pip install matplotlib pandas
```

### For Teaching

```bash
# Install in user space (no admin required)
pip install --user heavytails

# Or create a virtual environment for the course
python -m venv heavytails_course
source heavytails_course/bin/activate  # On Windows: heavytails_course\Scripts\activate
pip install heavytails
```

--------------------------------------------------------------------------------

## Virtual Environment Setup (Recommended)

Using a virtual environment keeps your Python installation clean:

=== "Linux/macOS"

````
```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate

# Install HeavyTails
pip install heavytails

# Deactivate when done
deactivate
```
````

=== "Windows"

````
```bash
# Create virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Install HeavyTails
pip install heavytails

# Deactivate when done
deactivate
```
````

=== "Poetry"

````
```bash
# Poetry automatically creates and manages virtual environments
poetry init
poetry add heavytails
poetry shell  # Activate the environment
```
````

--------------------------------------------------------------------------------

## Optional Dependencies

While HeavyTails itself has no dependencies, you may want to install these for enhanced functionality:

### For Visualization

```bash
pip install matplotlib
```

Enables plotting capabilities:

```python
from heavytails import Pareto
import matplotlib.pyplot as plt
import numpy as np

pareto = Pareto(alpha=2.0, xm=1.0)
x = np.logspace(0, 2, 100)
plt.loglog(x, [pareto.pdf(xi) for xi in x])
plt.xlabel('x')
plt.ylabel('PDF')
plt.title('Pareto PDF')
plt.show()
```

### For Data Analysis

```bash
pip install pandas numpy
```

Useful for working with real datasets:

```python
import pandas as pd
from heavytails import StudentT
from heavytails.tail_index import hill_estimator

# Load financial data
returns = pd.read_csv('stock_returns.csv')['return'].values

# Estimate tail index
gamma = hill_estimator(returns, k=100)
print(f"Tail index estimate: {1/gamma:.2f}")
```

### For Jupyter Notebooks

```bash
pip install jupyter notebook
```

Run HeavyTails in interactive notebooks:

```bash
jupyter notebook
```

--------------------------------------------------------------------------------

## Troubleshooting

### ImportError: No module named 'heavytails'

**Solution:** Ensure you're using the correct Python environment:

```bash
# Check which Python you're using
which python  # On Linux/macOS
where python  # On Windows

# Check if heavytails is installed
pip list | grep heavytails
```

### Permission Denied Error

**Solution:** Install in user space:

```bash
pip install --user heavytails
```

Or use a virtual environment (recommended).

### Python Version Error

**Solution:** HeavyTails requires Python 3.8+. Check your version:

```bash
python --version
```

If you have an older version, upgrade Python or use pyenv/conda to manage multiple versions.

--------------------------------------------------------------------------------

## Docker Installation

For containerized environments:

```docker
FROM python:3.11-slim

# Install HeavyTails
RUN pip install heavytails

# Copy your analysis scripts
COPY analysis.py /app/
WORKDIR /app

CMD ["python", "analysis.py"]
```

Build and run:

```bash
docker build -t heavytails-analysis .
docker run heavytails-analysis
```

--------------------------------------------------------------------------------

## Next Steps

Now that HeavyTails is installed:

1. **[Quick Start Tutorial](quickstart.md)** - Get started in 10 minutes
2. **[Basic Concepts](concepts.md)** - Understand heavy-tailed distributions
3. **[Examples](../examples/basic_usage.ipynb)** - See practical applications

--------------------------------------------------------------------------------

## Getting Help

If you encounter installation issues:

- Check the [GitHub Issues](https://github.com/diogoribeiro7/heavytails/issues) for similar problems
- Ask in [GitHub Discussions](https://github.com/diogoribeiro7/heavytails/discussions)
- Contact the maintainer at <dfr@esmad.ipp.pt>
