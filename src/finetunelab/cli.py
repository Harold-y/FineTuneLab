"""FineTuneLab command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from finetunelab.config import load_config
from finetunelab.errors import FineTuneLabError
from finetunelab.runtime import environment_report
from finetunelab.workflows import (
    evaluate,
    export_model,
    generate_feedback,
    inspect_model,
    train,
    validate_data_workflow,
)

app = typer.Typer(help="Educational fine-tuning for Qwen3.5 LLMs and VLMs.", no_args_is_help=True)
model_app = typer.Typer(help="Inspect registered model families.")
data_app = typer.Typer(help="Validate and inspect datasets.")
feedback_app = typer.Typer(help="Generate auditable AI preference data.")
app.add_typer(model_app, name="model")
app.add_typer(data_app, name="data")
app.add_typer(feedback_app, name="feedback")
console = Console()

ConfigPath = Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)]


def _emit(value: Any) -> None:
    console.print_json(json.dumps(value, default=str))


def _guard(function: Any, *args: Any, **kwargs: Any) -> None:
    try:
        _emit(function(*args, **kwargs))
    except FineTuneLabError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(code=2) from exc


@app.command()
def doctor() -> None:
    """Report runtime versions and accelerator availability."""

    report = environment_report()
    table = Table(title="FineTuneLab environment")
    table.add_column("Item")
    table.add_column("Value")
    for key in ("finetunelab", "python", "platform", "cuda_available", "cuda_version", "gpu_count"):
        table.add_row(key, str(report[key]))
    for package, version in report["packages"].items():
        table.add_row(package, str(version))
    for index, gpu in enumerate(report["gpus"]):
        table.add_row(f"gpu_{index}", gpu)
    console.print(table)


@model_app.command("inspect")
def model_inspect(config: ConfigPath) -> None:
    """Load a model, apply the tuning strategy, and report trainable parameters."""

    _guard(inspect_model, load_config(config))


@data_app.command("validate")
def data_validate(config: ConfigPath) -> None:
    """Validate dataset structure without loading model weights."""

    _guard(validate_data_workflow, load_config(config))


@feedback_app.command("generate")
def feedback_generate(config: ConfigPath) -> None:
    """Generate candidates and collect local or API-based AI preferences."""

    _guard(generate_feedback, load_config(config))


@app.command("train")
def train_command(
    config: ConfigPath,
    resume: Annotated[str | None, typer.Option(help="Checkpoint directory or 'latest'.")] = None,
) -> None:
    """Run a configured training recipe."""

    _guard(train, load_config(config), resume)


@app.command("evaluate")
def evaluate_command(
    config: ConfigPath,
    checkpoint: Annotated[
        str | None, typer.Option(help="Model or adapter; defaults to run/final.")
    ] = None,
) -> None:
    """Evaluate the configured checkpoint on data.eval_split."""

    _guard(evaluate, load_config(config), checkpoint)


@app.command("export")
def export_command(config: ConfigPath) -> None:
    """Merge a PEFT adapter and save a standalone Hugging Face checkpoint."""

    _guard(export_model, load_config(config))


if __name__ == "__main__":  # pragma: no cover
    app()
