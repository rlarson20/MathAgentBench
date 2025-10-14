"""Command-line interface."""

import click
# from pathlib import Path


@click.group()
def cli():
    """MathAgent Bench: Evaluate LLM agents on math problems."""
    pass


@cli.command()
@click.argument("benchmark", type=click.Path(exists=True))
@click.option(
    "--model", default="openai/gpt-4o-mini", help="Model to use via OpenRouter"
)
@click.option(
    "--agent",
    type=click.Choice(["react", "reflexion"]),
    default="react",
    help="Agent type",
)
@click.option(
    "--output", type=click.Path(), default="results/", help="Output directory"
)
@click.option("--db", is_flag=True, help="Also save to SQLite database")
def run(benchmark: str, model: str, agent: str, output: str, db: bool):
    """Run evaluation on a benchmark."""
    # TODO: Implement run command
    # - Load benchmark
    # - Initialize agent with model
    # - Run evaluator
    # - Save results (JSONL + optionally SQLite)
    click.echo(f"Running {agent} agent with {model} on {benchmark}")
    raise NotImplementedError


@cli.command()
@click.argument("results", nargs=2, type=click.Path(exists=True))
def compare(results: tuple[str, str]):
    """Compare two result files for drift detection."""
    # TODO: Implement comparison
    # - Load both result files
    # - Compute deltas
    # - Print formatted comparison table
    # - Flag regressions and warnings
    click.echo(f"Comparing {results[0]} vs {results[1]}")
    raise NotImplementedError


@cli.command()
@click.argument("results", type=click.Path(exists=True))
def report(results: str):
    """Print detailed report from results file."""
    # TODO: Implement report generation
    # - Load results
    # - Print metrics table
    # - Show per-problem breakdown
    # - Optional: failure analysis
    click.echo(f"Generating report for {results}")
    raise NotImplementedError


@cli.command()
@click.argument("results", nargs=-1, type=click.Path(exists=True))
@click.option("--db", required=True, type=click.Path(), help="SQLite database path")
def export(results: tuple[str, ...], db: str):
    """Export JSONL results to SQLite database."""
    # TODO: Implement export
    # - Load all JSONL files
    # - Insert into SQLite
    # - Handle duplicates
    click.echo(f"Exporting {len(results)} result files to {db}")
    raise NotImplementedError


if __name__ == "__main__":
    cli()
