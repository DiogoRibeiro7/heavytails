"""
Command-line interface for the heavytails package.

Provides utilities for:
- Distribution parameter estimation
- Tail index estimation
- Data generation and simulation
- Diagnostic plotting
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Annotated

from rich.console import Console
from rich.table import Table
import typer

from heavytails import (
    BetaPrime,
    BurrXII,
    Cauchy,
    Frechet,
    GeneralizedPareto,
    GEV_Frechet,
    InverseGamma,
    LogLogistic,
    LogNormal,
    Pareto,
    StudentT,
    Weibull,
)
from heavytails.tail_index import hill_estimator, moment_estimator, pickands_estimator

app = typer.Typer(
    name="heavytails",
    help="Heavy-tailed probability distributions toolkit",
    add_completion=False,
)
console = Console()

# Distribution mapping
DISTRIBUTIONS = {
    "pareto": Pareto,
    "cauchy": Cauchy,
    "student-t": StudentT,
    "lognormal": LogNormal,
    "weibull": Weibull,
    "frechet": Frechet,
    "gev": GEV_Frechet,
    "gpd": GeneralizedPareto,
    "burr": BurrXII,
    "loglogistic": LogLogistic,
    "invgamma": InverseGamma,
    "betaprime": BetaPrime,
}


@app.command()
def sample(
    distribution: Annotated[str, typer.Argument(help="Distribution name")],
    n: Annotated[int, typer.Option("--samples", "-n", help="Number of samples")] = 1000,
    output: Annotated[
        Path | None, typer.Option("--output", "-o", help="Output file")
    ] = None,
    seed: Annotated[
        int | None, typer.Option("--seed", "-s", help="Random seed")
    ] = None,
    params: Annotated[
        str, typer.Option("--params", "-p", help="Distribution parameters as JSON")
    ] = "{}",
) -> None:
    """Generate samples from a heavy-tailed distribution."""
    if distribution not in DISTRIBUTIONS:
        console.print(f"[red]Error:[/red] Unknown distribution '{distribution}'")
        console.print(f"Available: {', '.join(DISTRIBUTIONS.keys())}")
        raise typer.Exit(1) from None

    try:
        param_dict = json.loads(params)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error:[/red] Invalid JSON parameters: {e}")
        raise typer.Exit(1) from e

    try:
        dist_class = DISTRIBUTIONS[distribution]
        dist = dist_class(**param_dict)
        samples = dist.rvs(n, seed=seed)

        if output:
            with output.open("w") as f:
                for sample_val in samples:
                    f.write(f"{sample_val}\n")
            console.print(f"[green]Success:[/green] Wrote {n} samples to {output}")
        else:
            for sample_val in samples:
                console.print(f"{sample_val}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def estimate_tail(
    data_file: Annotated[
        Path, typer.Argument(help="File containing data (one value per line)")
    ],
    method: Annotated[
        str, typer.Option("--method", "-m", help="Estimation method")
    ] = "hill",
    k: Annotated[
        int | None, typer.Option("--k", help="Number of top order statistics")
    ] = None,
    output_format: Annotated[
        str, typer.Option("--format", "-f", help="Output format (table|json)")
    ] = "table",
) -> None:
    """Estimate tail index from data."""
    if not data_file.exists():
        console.print(f"[red]Error:[/red] File {data_file} not found")
        raise typer.Exit(1) from None

    # Read data
    try:
        with data_file.open() as f:
            data = [float(line.strip()) for line in f if line.strip()]
    except (OSError, ValueError) as e:
        console.print(f"[red]Error:[/red] Could not read data: {e}")
        raise typer.Exit(1) from e

    if len(data) < 10:
        console.print(
            f"[red]Error:[/red] Need at least 10 data points, got {len(data)}"
        )
        raise typer.Exit(1)

    # Auto-select k if not provided
    if k is None:
        k = min(len(data) // 10, 200)  # Rule of thumb: ~10% of data, max 200

    if k >= len(data):
        console.print(
            f"[red]Error:[/red] k ({k}) must be less than data size ({len(data)})"
        )
        raise typer.Exit(1)

    try:
        if method == "hill":
            result = hill_estimator(data, k)
            alpha_hat = 1.0 / result
            results = {
                "method": "Hill",
                "gamma": result,
                "alpha": alpha_hat,
                "k": k,
                "n": len(data),
            }

        elif method == "pickands":
            gamma_hat = pickands_estimator(data, k)
            alpha_hat = 1.0 / gamma_hat
            results = {
                "method": "Pickands",
                "gamma": gamma_hat,
                "alpha": alpha_hat,
                "k": k,
                "n": len(data),
            }

        elif method == "moment":
            gamma_hat, alpha_hat = moment_estimator(data, k)
            results = {
                "method": "Moment (DEH)",
                "gamma": gamma_hat,
                "alpha": alpha_hat,
                "k": k,
                "n": len(data),
            }

        else:
            console.print(f"[red]Error:[/red] Unknown method '{method}'")
            console.print("Available methods: hill, pickands, moment")
            raise typer.Exit(1) from None

        # Output results
        if output_format == "json":
            console.print(json.dumps(results, indent=2))
        else:
            table = Table(title="Tail Index Estimation Results")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")

            table.add_row("Method", results["method"])
            table.add_row("Sample size (n)", str(results["n"]))
            table.add_row("Order statistics (k)", str(results["k"]))
            table.add_row("Tail index (gamma)", f"{results['gamma']:.4f}")
            table.add_row("Shape parameter (alpha)", f"{results['alpha']:.4f}")

            console.print(table)

            # Interpretation
            gamma = results["gamma"]
            console.print("\n[bold]Interpretation:[/bold]")
            if gamma > 0.5:
                console.print("• [red]Very heavy tail[/red] - infinite variance")
            elif gamma > 0:
                console.print(
                    "• [yellow]Heavy tail[/yellow] - finite variance, possible infinite higher moments"
                )
            else:
                console.print("• [green]Light tail or estimation error[/green]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def fit(
    data_file: Annotated[Path, typer.Argument(help="File containing data")],
    distribution: Annotated[str, typer.Argument(help="Distribution to fit")],
    method: Annotated[
        str, typer.Option("--method", "-m", help="Fitting method")
    ] = "mle",
) -> None:
    """Fit a distribution to data (placeholder for future MLE implementation)."""
    _ = (
        data_file,
        distribution,
        method,
    )  # Placeholder - will be used in future implementation
    console.print("[yellow]Warning:[/yellow] Parameter fitting not yet implemented")
    console.print("Future versions will include MLE and method-of-moments fitting")


@app.command()
def compare(
    data_file: Annotated[Path, typer.Argument(help="File containing data")],
    distributions: Annotated[
        str, typer.Option("--dists", help="Comma-separated distribution list")
    ] = "pareto,lognormal",
) -> None:
    """Compare multiple distributions against data (placeholder)."""
    _ = (
        data_file,
        distributions,
    )  # Placeholder - will be used in future implementation
    console.print(
        "[yellow]Warning:[/yellow] Distribution comparison not yet implemented"
    )
    console.print("Future versions will include AIC/BIC model comparison")


@app.command()
def info(
    distribution: Annotated[str, typer.Argument(help="Distribution name")],
) -> None:
    """Show information about a distribution."""

    if distribution not in DISTRIBUTIONS:
        console.print(f"[red]Error:[/red] Unknown distribution '{distribution}'")
        console.print(f"Available: {', '.join(DISTRIBUTIONS.keys())}")
        raise typer.Exit(1)

    DISTRIBUTIONS[distribution]

    # Get distribution info
    info_map = {
        "pareto": {
            "name": "Pareto Type I",
            "parameters": ["alpha (shape)", "xm (scale/minimum)"],
            "support": "x >= xm",
            "heavy_tail": "Always (alpha > 0)",
            "moments": "E[X^k] finite for k < alpha",
            "applications": ["Income distribution", "City sizes", "Firm sizes"],
        },
        "cauchy": {
            "name": "Cauchy",
            "parameters": ["x0 (location)", "gamma (scale)"],
            "support": "All real numbers",
            "heavy_tail": "Always (no finite moments)",
            "moments": "No finite moments",
            "applications": ["Physics (resonance)", "Ratio of normal RVs"],
        },
        "lognormal": {
            "name": "Log-Normal",
            "parameters": [
                "mu (underlying normal mean)",
                "sigma (underlying normal std)",
            ],
            "support": "x > 0",
            "heavy_tail": "Always (subexponential)",
            "moments": "All moments finite",
            "applications": ["Asset prices", "File sizes", "Environmental data"],
        },
        "weibull": {
            "name": "Weibull",
            "parameters": ["k (shape)", "lam (scale)"],
            "support": "x >= 0",
            "heavy_tail": "When k < 1",
            "moments": "All moments finite",
            "applications": ["Reliability", "Wind speeds", "Failure times"],
        },
        "student-t": {
            "name": "Student's t",
            "parameters": ["nu (degrees of freedom)"],
            "support": "All real numbers",
            "heavy_tail": "When nu is small",
            "moments": "E[X^k] finite for k < nu",
            "applications": ["Finance (returns)", "Robust statistics"],
        },
        "frechet": {
            "name": "Fréchet",
            "parameters": ["alpha (shape)", "s (scale)", "m (location)"],
            "support": "x > m",
            "heavy_tail": "Always",
            "moments": "E[X^k] finite for k < alpha",
            "applications": ["Extreme values", "Maximum of samples"],
        },
        "gev": {
            "name": "Generalized Extreme Value (Fréchet type)",
            "parameters": ["xi (shape, >0)", "mu (location)", "sigma (scale)"],
            "support": "x > mu - sigma/xi",
            "heavy_tail": "When ξ > 0",
            "moments": "E[X^k] finite for k < 1/ξ",
            "applications": ["Extreme weather", "Financial risk", "Insurance"],
        },
        "gpd": {
            "name": "Generalized Pareto",
            "parameters": ["xi (shape)", "sigma (scale)", "mu (location)"],
            "support": "x >= mu (if xi >= 0)",
            "heavy_tail": "When ξ > 0",
            "moments": "E[X^k] finite for k < 1/ξ",
            "applications": ["Peaks over threshold", "Insurance", "Finance"],
        },
        "burr": {
            "name": "Burr Type XII",
            "parameters": ["c (shape 1)", "k (shape 2)", "s (scale)"],
            "support": "x > 0",
            "heavy_tail": "Always",
            "moments": "E[X^k] finite for k < c",
            "applications": ["Income modeling", "Reliability", "Hydrology"],
        },
        "loglogistic": {
            "name": "Log-Logistic (Fisk)",
            "parameters": ["kappa (shape)", "lam (scale)"],
            "support": "x > 0",
            "heavy_tail": "Always",
            "moments": "E[X^k] finite for k < κ",
            "applications": ["Survival analysis", "Economics", "Hydrology"],
        },
        "invgamma": {
            "name": "Inverse Gamma",
            "parameters": ["alpha (shape)", "beta (scale)"],
            "support": "x > 0",
            "heavy_tail": "Always",
            "moments": "E[X^k] finite for k < alpha",
            "applications": ["Bayesian priors", "Reliability", "Variance modeling"],
        },
        "betaprime": {
            "name": "Beta Prime",
            "parameters": ["a (shape 1)", "b (shape 2)", "s (scale)"],
            "support": "x > 0",
            "heavy_tail": "Always",
            "moments": "E[X^k] finite for k < a",
            "applications": ["Economics", "Reliability", "Income modeling"],
        },
    }

    info = info_map.get(distribution, {})

    table = Table(title=f"{info.get('name', distribution.title())} Distribution")
    table.add_column("Property", style="cyan")
    table.add_column("Description", style="white")

    for key, value in info.items():
        if key == "name":
            continue

        if key == "parameters":
            table.add_row("Parameters", ", ".join(value))
        elif key == "applications":
            table.add_row("Applications", ", ".join(value))
        elif key == "heavy_tail":
            color = "red" if "Always" in value else "yellow"
            table.add_row("Heavy tail", f"[{color}]{value}[/{color}]")
        else:
            table.add_row(key.replace("_", " ").title(), str(value))

    console.print(table)


@app.command()
def list_distributions() -> None:
    """List all available distributions."""

    table = Table(title="Available Heavy-Tailed Distributions")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Heavy Tail", style="white")

    dist_info = [
        ("pareto", "Power law", "Always"),
        ("cauchy", "Symmetric", "Always"),
        ("student-t", "Symmetric", "Small nu"),
        ("lognormal", "Positive", "Always"),
        ("weibull", "Positive", "k < 1"),
        ("frechet", "Extreme value", "Always"),
        ("gev", "Extreme value", "ξ > 0"),
        ("gpd", "Threshold excess", "ξ > 0"),
        ("burr", "Flexible", "Always"),
        ("loglogistic", "Positive", "Always"),
        ("invgamma", "Positive", "Always"),
        ("betaprime", "Positive", "Always"),
    ]

    for name, dist_type, heavy_tail in dist_info:
        color = "red" if heavy_tail == "Always" else "yellow"
        table.add_row(name, dist_type, f"[{color}]{heavy_tail}[/{color}]")

    console.print(table)


@app.command()
def validate(
    distribution: Annotated[str, typer.Argument(help="Distribution name")],
    params: Annotated[
        str, typer.Option("--params", "-p", help="Distribution parameters as JSON")
    ] = "{}",
    tests: Annotated[
        str, typer.Option("--tests", "-t", help="Test suite to run")
    ] = "basic",
) -> None:
    """Validate distribution implementation."""
    _ = tests  # Reserved for future test suite selection
    if distribution not in DISTRIBUTIONS:
        console.print(f"[red]Error:[/red] Unknown distribution '{distribution}'")
        raise typer.Exit(1)

    try:
        param_dict = json.loads(params)
        dist_class = DISTRIBUTIONS[distribution]
        dist = dist_class(**param_dict)

        console.print("[green]✓[/green] Distribution created successfully")

        # Basic validation tests
        test_points = [0.1, 0.5, 0.9, 0.99]

        console.print("\n[bold]PDF/CDF Validation:[/bold]")
        for x in [1.0, 2.0, 5.0]:
            try:
                pdf = dist.pdf(x)
                cdf = dist.cdf(x)
                console.print(f"  x={x}: PDF={pdf:.6f}, CDF={cdf:.6f}")

                if pdf < 0:
                    console.print(f"  [red]✗[/red] Negative PDF at x={x}")
                if not (0 <= cdf <= 1):
                    console.print(f"  [red]✗[/red] CDF out of [0,1] at x={x}")
            except Exception as e:
                console.print(f"  [red]✗[/red] Error at x={x}: {e}")

        console.print("\n[bold]PPF/CDF Inverse Test:[/bold]")
        for u in test_points:
            try:
                x = dist.ppf(u)
                recovered_u = dist.cdf(x)
                error = abs(recovered_u - u)

                if error < 1e-6:
                    console.print(
                        f"  [green]✓[/green] u={u}: x={x:.4f}, error={error:.2e}"
                    )
                else:
                    console.print(f"  [red]✗[/red] u={u}: x={x:.4f}, error={error:.2e}")

            except Exception as e:
                console.print(f"  [red]✗[/red] Error at u={u}: {e}")

        console.print("\n[bold]Sampling Test:[/bold]")
        try:
            samples = dist.rvs(100, seed=42)
            console.print(f"  [green]✓[/green] Generated {len(samples)} samples")
            console.print(f"  Sample mean: {sum(samples) / len(samples):.4f}")
            console.print(f"  Sample range: [{min(samples):.4f}, {max(samples):.4f}]")
        except Exception as e:
            console.print(f"  [red]✗[/red] Sampling error: {e}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def benchmark(
    distribution: Annotated[str, typer.Argument(help="Distribution name")],
    params: Annotated[
        str, typer.Option("--params", "-p", help="Distribution parameters as JSON")
    ] = "{}",
    n_samples: Annotated[
        int, typer.Option("--samples", "-n", help="Number of samples for benchmark")
    ] = 10000,
) -> None:
    """Benchmark distribution performance."""
    if distribution not in DISTRIBUTIONS:
        console.print(f"[red]Error:[/red] Unknown distribution '{distribution}'")
        raise typer.Exit(1) from None

    try:
        param_dict = json.loads(params)
        dist_class = DISTRIBUTIONS[distribution]
        dist = dist_class(**param_dict)

        console.print(f"[bold]Benchmarking {distribution} distribution[/bold]\n")

        # PDF evaluation benchmark
        x_values = [1.0 + i * 0.01 for i in range(1000)]
        start_time = time.time()
        [dist.pdf(x) for x in x_values]
        pdf_time = time.time() - start_time

        console.print(
            f"PDF evaluation (1000 points): {pdf_time:.4f}s ({1000 / pdf_time:.0f} evals/sec)"
        )

        # CDF evaluation benchmark
        start_time = time.time()
        [dist.cdf(x) for x in x_values]
        cdf_time = time.time() - start_time

        console.print(
            f"CDF evaluation (1000 points): {cdf_time:.4f}s ({1000 / cdf_time:.0f} evals/sec)"
        )

        # PPF evaluation benchmark
        u_values = [i / 1000 for i in range(1, 1000)]
        start_time = time.time()
        try:
            [dist.ppf(u) for u in u_values]
            ppf_time = time.time() - start_time
            console.print(
                f"PPF evaluation (999 points): {ppf_time:.4f}s ({999 / ppf_time:.0f} evals/sec)"
            )
        except Exception as e:
            console.print(f"PPF benchmark failed: {e}")

        # Sampling benchmark
        start_time = time.time()
        dist.rvs(n_samples, seed=42)
        sampling_time = time.time() - start_time

        console.print(
            f"Sampling ({n_samples} samples): {sampling_time:.4f}s ({n_samples / sampling_time:.0f} samples/sec)"
        )

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


def main() -> None:
    """Main entry point for the CLI."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
