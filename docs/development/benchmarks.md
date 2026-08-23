# Benchmarking

`heavytails` computes through NumPy, so the thing to watch is whether work is
being done in one call or one element at a time. The difference is not marginal:
inverting 50,000 uniforms in a single call takes about 0.8ms, and inverting them
one at a time takes about 0.30s. This page covers how performance is measured,
tracked, and defended against regressions.

Two places still run a Python loop, and both are deliberate. `LogNormal`,
`StudentT`, `InverseGamma` and `BetaPrime` need the error function or an
incomplete beta or gamma for their probabilities; NumPy has none of those, so
those methods go through `_array.elementwise`. Their densities are elementary
and are vectorised normally. And sampling draws its uniforms in a loop, because
the generator is Python's `random` and each draw depends on the last -- only the
transform afterwards is batched.

## Running the benchmark suite

The suite lives in `benchmarks/performance_tests.py` and writes machine-readable
JSON:

```bash
poetry install --with benchmarks
poetry run python benchmarks/performance_tests.py --output results.json
```

| Option         | Default    | Description                                   |
| -------------- | ---------- | --------------------------------------------- |
| `--output`     | *required* | JSON file to write results to                 |
| `--baseline`   | none       | Earlier results file to compare against        |
| `--iterations` | `50`       | Repetitions per benchmark                     |

Comparing against a stored baseline reports the change for every measurement:

```bash
poetry run python benchmarks/performance_tests.py \
  --output current.json \
  --baseline baseline.json \
  --iterations 50
```

## Micro-benchmarks in the test suite

Finer-grained benchmarks run through
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/) and are marked so
they stay out of ordinary test runs:

```bash
poetry run pytest -m benchmark
```

`pytest-benchmark` handles calibration, warm-up and outlier rejection, and stores
results under `.benchmarks/` for comparison across runs:

```bash
poetry run pytest -m benchmark --benchmark-autosave
poetry run pytest -m benchmark --benchmark-compare
```

## A quick single-distribution check

For an interactive look at one family, the CLI is faster than either suite:

```bash
heavytails benchmark pareto --params '{"alpha": 2.0, "xm": 1.0}' --memory
```

It reports throughput for PDF, CDF, PPF and sampling, and with `--memory` a
`tracemalloc` peak. Use it while iterating; use the suites when the number needs
to be comparable to yesterday's.

## Continuous monitoring

The `Performance Monitoring` workflow runs the suite on every push to `main` and
`develop`, on pull requests, and weekly on a schedule. It:

1. Downloads the stored baseline artifact for the interpreter version.
2. Runs the suite against the current commit.
3. Compares the two and fails the job on a significant regression.
4. Comments the comparison on the pull request.
5. Updates the baseline when the run is on `main`.

Because the baseline is per interpreter version, a slowdown introduced by a new
Python release is visible separately from one introduced by a code change.

## Writing a benchmark that means something

Timing pure-Python numerical code is easy to get wrong. The rules the existing
suite follows:

**Use `time.perf_counter()`, never `time.time()`.** `time.time()` has roughly
16 ms resolution on Windows, so anything faster than that measures as exactly
zero — and any throughput calculation then divides by zero. This has bitten the
project before.

```python
import time

start = time.perf_counter()
result = do_work()
elapsed = time.perf_counter() - start
```

**Guard the division anyway.** Even `perf_counter` can return a zero interval for
a trivially small workload.

```python
rate = count / elapsed if elapsed > 0 else float("inf")
```

**Size the workload so it runs for milliseconds, not microseconds.** Below that,
you measure loop overhead rather than the operation.

**Keep the measured region free of setup.** Build inputs, construct the
distribution, and warm any cache before starting the clock.

**Seed the RNG.** Sampling benchmarks must be reproducible, otherwise run-to-run
variation swamps the effect you are trying to measure:

```python
samples = dist.rvs(100_000, seed=42)
```

**Never assert on absolute timings in the test suite.** A shared CI runner is not
a controlled environment; a test asserting "under 100 ms" will fail on a noisy
neighbour and teach everyone to ignore it. Assert on correctness in tests, and
track timings in the benchmark suite where a regression check compares like with
like.

## Where the time goes

Some structural facts that shape optimisation work here:

- **Sampling** is usually dominated by the inverse-transform call, so a
  closed-form `ppf` is dramatically faster than the numeric solver. Families
  without a closed form pay for it on every draw.
- **The numeric PPF** is a safeguarded Newton iteration. Its cost is the iteration
  count, so a better initial bracket helps more than a faster inner loop.
- **Special functions** — incomplete gamma and incomplete beta — dominate the
  Student-t, Inverse-Gamma and Beta-Prime families. `heavytails.performance`
  caches where the arguments repeat.
- **Attribute lookup and function call overhead** are a real fraction of the cost
  in tight pure-Python loops. Hoisting a bound method out of a loop is a
  legitimate and measurable optimisation.

## Before optimising

Profile first; intuition about pure-Python hot spots is unreliable.

```bash
poetry run python -m cProfile -s cumtime benchmarks/performance_tests.py --output /dev/null
```

For line-level detail, `line-profiler` and `memory-profiler` are in the
`benchmarks` dependency group.

Then: measure, change one thing, measure again, and record the numbers in the
pull request. An optimisation without a before-and-after measurement is a guess,
and the review will ask for one — see [Code Review](code-review.md).

## See also

- [Testing](testing.md) — the test suite and its markers
- [Architecture](architecture.md) — where the performance-critical code lives
- [CLI Reference](../guide/cli.md) — the `benchmark` command
