# heavytails

**A pure-Python library of heavy-tailed probability distributions**

`heavytails` implements a broad collection of heavy-tailed continuous distributions — Pareto, Cauchy, Student-t, Log-Normal, Weibull (k<1), Fréchet, GEV (ξ>0), and additional families such as Generalized Pareto, Burr XII, Log-Logistic, Inverse-Gamma, and Beta-Prime — **without using any third-party dependencies**.

The goal is a **transparent, educational, and mathematically rigorous** implementation suitable for research, teaching, or simulation studies in risk, finance, insurance, and extreme-value analysis.

---

## ✨ Features

* Continuous heavy-tailed families implemented from first principles
* Full PDF / CDF / survival / quantile / random-sampling interface
* No external dependencies (only `math` and `random`)
* Deterministic RNG wrapper for reproducibility
* Custom incomplete-gamma and incomplete-beta functions
* Optional safeguarded-Newton numeric PPF solver for closed-form-free distributions

---

## 📦 Installation

```bash
poetry add heavytails
```

*(or clone directly if you prefer local source use)*

```bash
git clone https://github.com/DiogoRibeiro7/heavytails.git
cd heavytails
poetry install
```

---

## 🧬 Example

```python
from heavytails import Pareto, Cauchy, LogNormal, BurrXII

pareto = Pareto(alpha=1.5, xm=1.0)
print("Pareto P(X>10) =", pareto.sf(10.0))
print("Random samples:", pareto.rvs(5, seed=42))

burr = BurrXII(c=1.2, k=2.5, s=3.0)
print("Burr 95% quantile =", burr.ppf(0.95))
```

---

## 📚 Available Distributions

| Module                   | Distribution        | Heavy-Tail Regime |
| ------------------------ | ------------------- | ----------------- |
| `heavy_tails.py`         | Pareto              | always            |
|                          | Cauchy              | always            |
|                          | Student-t           | ν small           |
|                          | LogNormal           | always            |
|                          | Weibull             | k < 1             |
|                          | Fréchet             | always            |
|                          | GEV (ξ>0)           | ξ>0               |
| `extra_distributions.py` | Generalized Pareto  | ξ>0               |
|                          | Burr XII            | always            |
|                          | Log-Logistic (Fisk) | always            |
|                          | Inverse-Gamma       | always            |
|                          | Beta-Prime          | always            |

---

## 🧠 Future Extensions

* Discrete heavy-tailed families (Zipf, Zeta, Yule–Simon)
* Tail-index estimation (Hill, Pickands, Moment)
* Empirical tail diagnostics and QQ-plots
* Monte-Carlo tail-risk estimators

---

## 🤪 Tests

```bash
poetry run pytest -v
```

---

## ⚖️ License

MIT License © 2025 Diogo Ribeiro

---

## 🧉 Citation

If you use this package in research or teaching, please cite:

> Ribeiro, D. (2025). *heavytails: Pure-Python heavy-tailed distribution library*.
> [https://github.com/DiogoRibeiro7/heavytails](https://github.com/DiogoRibeiro7/heavytails)
