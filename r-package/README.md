# heavytailsR

Native R implementations of heavy-tailed distributions and tail-risk methods.

This R package is developed alongside the [Python heavytails library](https://github.com/DiogoRibeiro7/heavytails). It is independent of Python, NumPy and reticulate. The R edition has its own API, tests and versioning.

**Naming:** The CRAN package `heavytails` belongs to another maintainer. Our provisional name is `heavytailsR`; verify it against current and archived CRAN packages and Bioconductor before submission.

## Installation from source

From the root of the repository:

```sh
R CMD INSTALL r-package
```

Or within R:

```r
remotes::install_github("DiogoRibeiro7/heavytails", subdir = "r-package")
```

## Quick start

```r
library(heavytailsR)

dpareto_ht(2, shape = 2)
ppareto_ht(1e10, shape = 2, lower.tail = FALSE)
qpareto_ht(0.99, shape = 2)

set.seed(2026)
x <- rpareto_ht(1000, shape = 2)

# Hill estimates gamma = 1/alpha, not the Pareto exponent alpha.
gamma <- hill_index(x, k = 80)
alpha_hat <- 1 / gamma

pareto_var(0.99, shape = 2) # 10
pareto_es(0.99, shape = 2)  # 20
```

The API follows R's d/p/q/r distribution conventions. All parameters are validated, and upper-tail probabilities are evaluated directly.

## Quality checks

```sh
R CMD build r-package
R CMD check --as-cran --no-manual heavytailsR_0.1.0.tar.gz
```

Our GitHub workflow checks the native R package on Linux, macOS and Windows. Check results must be reviewed before a CRAN submission. See [ROADMAP.md](ROADMAP.md).
