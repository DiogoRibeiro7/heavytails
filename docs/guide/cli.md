# Command-Line Interface

`heavytails` ships a command-line tool for sampling, tail estimation, fitting and
benchmarking, so common tasks need no Python file at all.

## Installation

The library itself has no runtime dependencies, but the CLI is built on
[Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/).
Install them with the `cli` extra:

```bash
pip install "heavytails[cli]"
```

Without the extra, the entry point exits with a message telling you how to
install it. The library API remains fully usable either way.

Verify the installation:

```bash
heavytails --version
```

The tool can also be invoked as a module, which is useful inside virtual
environments and CI:

```bash
python -m heavytails --help
```

## Distribution names and parameters

Every command that takes a distribution accepts one of the names printed by
`heavytails list-distributions`:

```bash
heavytails list-distributions
```

| Name          | Class                | Parameters             |
| ------------- | -------------------- | ---------------------- |
| `pareto`      | `Pareto`             | `alpha`, `xm`          |
| `cauchy`      | `Cauchy`             | `x0`, `gamma`          |
| `student-t`   | `StudentT`           | `nu`                   |
| `lognormal`   | `LogNormal`          | `mu`, `sigma`          |
| `weibull`     | `Weibull`            | `k`, `lam`             |
| `frechet`     | `Frechet`            | `alpha`, `s`, `m`      |
| `gev`         | `GEV_Frechet`        | `xi`, `mu`, `sigma`    |
| `gpd`         | `GeneralizedPareto`  | `xi`, `sigma`, `mu`    |
| `burr`        | `BurrXII`            | `c`, `k`, `s`          |
| `loglogistic` | `LogLogistic`        | `kappa`, `lam`         |
| `invgamma`    | `InverseGamma`       | `alpha`, `beta`        |
| `betaprime`   | `BetaPrime`          | `a`, `b`, `s`          |

Parameters are passed as a JSON object through `--params`, on the commands that
take one:

```bash
heavytails sample pareto --params '{"alpha": 2.0, "xm": 1.0}' -n 10
```

!!! note "Quoting on Windows"
    PowerShell does not strip single quotes the way POSIX shells do. Use double
    quotes and escape the inner ones:

    ```powershell
    heavytails sample pareto --params '{\"alpha\": 2.0, \"xm\": 1.0}' -n 10
    ```

## Commands

### `sample` — generate random variates

```bash
heavytails sample pareto --params '{"alpha": 2.0, "xm": 1.0}' -n 1000 --seed 42 -o samples.txt
```

| Option           | Default | Description                                      |
| ---------------- | ------- | ------------------------------------------------ |
| `--samples`/`-n` | `1000`  | Number of variates to draw                       |
| `--seed`/`-s`    | none    | Seed for reproducible output                     |
| `--output`/`-o`  | stdout  | Write one value per line to this file            |
| `--params`/`-p`  | `{}`    | Distribution parameters as JSON                  |

Passing `--seed` makes the run reproducible: the same seed and parameters always
produce the same sequence.

### `estimate-tail` — estimate the tail index

```bash
heavytails estimate-tail samples.txt --method hill
```

| Option          | Default   | Description                                     |
| --------------- | --------- | ----------------------------------------------- |
| `--method`/`-m` | `hill`    | `hill`, `pickands` or `moment`                  |
| `--k`           | see below | Number of top order statistics to use           |
| `--format`/`-f` | `table`   | `table` or `json`                               |

When `--k` is omitted it defaults to `min(n // 10, 200)`, a rule of thumb that
is a starting point rather than an answer.

The input file holds one numeric value per line. The report shows the
extreme-value index γ alongside the implied shape parameter α = 1/γ, and an
interpretation of which moments exist. See
[Tail Index Estimation](tail-estimation.md) for how to choose `k`.

Use `--format json` when the output feeds another program:

```bash
heavytails estimate-tail samples.txt --format json > tail.json
```

### `fit` — estimate parameters from data

```bash
heavytails fit samples.txt pareto --method mle
```

Fits by maximum likelihood or the method of moments and reports the estimated
parameters together with the achieved log-likelihood.

### `compare` — rank candidate distributions

```bash
heavytails compare samples.txt --dists pareto,lognormal,burr
```

Fits several families to the same data and ranks them by AIC and BIC, reporting
the best model and its parameters. This is the quickest way to ask "which
heavy-tailed family does this sample look like?".

`--dists` takes a comma-separated list and defaults to `pareto,lognormal`.

### `info` — describe a distribution

```bash
heavytails info pareto
```

Prints the parameters the family takes, its support, and the tail behaviour it
exhibits. It describes the family itself, so it takes no `--params`.

### `validate` — check an implementation

```bash
heavytails validate pareto --params '{"alpha": 2.0, "xm": 1.0}' --tests basic
```

`--tests` selects the suite to run and defaults to `basic`. The checks are those
described in
[Validation Studies](../theory/validation.md): that the CDF is monotone, that
`ppf` inverts `cdf`, that the survival function complements the CDF, and that the
density integrates to one.

### `benchmark` — measure performance

```bash
heavytails benchmark pareto --params '{"alpha": 2.0, "xm": 1.0}'
```

Times PDF, CDF, PPF evaluation and sampling, and reports throughput. `--samples`
sets the sampling workload, and `--memory` adds a `tracemalloc` peak-usage
measurement.

For tracked, comparable measurements over time, use the benchmark suite
described in [Benchmarking](../development/benchmarks.md) instead; this command
is for a quick local check.

## Exit codes

| Code | Meaning                                                        |
| ---- | -------------------------------------------------------------- |
| `0`  | Success                                                         |
| `1`  | Invalid arguments, unreadable input, or a failed computation     |

Errors are written to the console with the offending value, which makes the
commands safe to chain in a shell script with `set -e`.

## A worked pipeline

Draw a sample, recover its tail index, and check which family fits best:

```bash
heavytails sample pareto --params '{"alpha": 1.5, "xm": 1.0}' -n 10000 --seed 7 -o data.txt
heavytails estimate-tail data.txt --method hill --k 200
heavytails compare data.txt --dists pareto,lognormal
```

The estimated α should land near the 1.5 used to generate the data, and
`compare` should rank `pareto` first.

## See also

- [Quick Start](../getting-started/quickstart.md) for the equivalent Python API
- [Tail Index Estimation](tail-estimation.md) for the statistics behind
  `estimate-tail`
- [Parameter Fitting](fitting.md) for the estimators behind `fit` and `compare`
