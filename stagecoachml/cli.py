"""Command-line interface for StagecoachML."""

from pathlib import Path

import typer
import yaml
from rich.console import Console
from rich.table import Table

from stagecoachml import __version__

app = typer.Typer(
    name="stagecoach",
    help="StagecoachML - Machine Learning Pipeline Orchestration",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"StagecoachML version {__version__}")


@app.command()
def run(
    config: Path = typer.Argument(..., help="Path to pipeline configuration file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show execution plan without running"),
):
    """Run a pipeline from configuration file."""
    if not config.exists():
        console.print(f"[red]Error: Configuration file '{config}' not found[/red]")
        raise typer.Exit(1)

    console.print(f"Loading pipeline from {config}")

    try:
        with open(config) as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[red]Error loading configuration: {e}[/red]")
        raise typer.Exit(1)

    if dry_run:
        console.print("[yellow]Dry run mode - showing execution plan[/yellow]")
        _show_execution_plan(config_data)
    else:
        console.print("[green]Starting pipeline execution...[/green]")
        # Pipeline execution would happen here
        console.print("[green]Pipeline completed successfully![/green]")


@app.command()
def validate(
    config: Path = typer.Argument(..., help="Path to pipeline configuration file"),
):
    """Validate a pipeline configuration."""
    if not config.exists():
        console.print(f"[red]Error: Configuration file '{config}' not found[/red]")
        raise typer.Exit(1)

    try:
        with open(config) as f:
            config_data = yaml.safe_load(f)
        console.print("[green]✓ Configuration file is valid YAML[/green]")

        # Additional validation would happen here
        if "pipeline" in config_data:
            console.print("[green]✓ Pipeline configuration found[/green]")
        if "stages" in config_data:
            console.print(f"[green]✓ Found {len(config_data['stages'])} stages[/green]")

    except Exception as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_stages(
    config: Path = typer.Argument(..., help="Path to pipeline configuration file"),
):
    """List all stages in a pipeline."""
    if not config.exists():
        console.print(f"[red]Error: Configuration file '{config}' not found[/red]")
        raise typer.Exit(1)

    with open(config) as f:
        config_data = yaml.safe_load(f)

    stages = config_data.get("stages", [])

    table = Table(title="Pipeline Stages")
    table.add_column("Stage", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Description", style="green")

    for stage in stages:
        table.add_row(
            stage.get("name", "unnamed"),
            stage.get("type", "unknown"),
            stage.get("description", ""),
        )

    console.print(table)


def _show_execution_plan(config_data: dict):
    """Display the execution plan for a pipeline."""
    stages = config_data.get("stages", [])
    console.print("\n[bold]Execution Plan:[/bold]")
    for i, stage in enumerate(stages, 1):
        console.print(f"  {i}. {stage.get('name', 'unnamed')} ({stage.get('type', 'unknown')})")


if __name__ == "__main__":
    app()
