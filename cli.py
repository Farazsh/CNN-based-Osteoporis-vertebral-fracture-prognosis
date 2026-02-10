"""Command line interface for osteo fracture project."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import click

from osteo_fracture.experiments.cross_validation import run_cross_validation
from osteo_fracture.experiments.hyperparameter_search import run_hyperparameter_search
from scripts.preprocess_dicom import main as preprocess_dicom
from train import run_training


@click.group()
def cli() -> None:
    """Top-level command group."""


@cli.command("train")
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
def train_cmd(config_path: str) -> None:
    """Train and test a model from a YAML config."""
    run_training(config_path)


@cli.command("evaluate")
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
def evaluate_cmd(config_path: str) -> None:
    """Evaluate by running trainer test with configured checkpoint strategy."""
    run_training(config_path)


@cli.command("predict")
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
def predict_cmd(config_path: str) -> None:
    """Alias for inference/evaluation pipeline."""
    run_training(config_path)


@cli.command("preprocess")
@click.option("--dicom-root", required=True)
@click.option("--output-root", required=True)
def preprocess_cmd(dicom_root: str, output_root: str) -> None:
    """Convert DICOM to NIfTI and generate vertebral patches."""
    preprocess_dicom(dicom_root, output_root)


@cli.command("crossval")
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
@click.option("--splits", multiple=True, default=["Split1", "Split2", "Split3", "Split4"])
def crossval_cmd(config_path: str, splits: tuple[str, ...]) -> None:
    """Run cross-validation over provided split columns."""
    run_cross_validation(config_path, list(splits))


@cli.command("hp-search")
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
@click.option("--lr", "learning_rates", multiple=True, type=float, default=[1e-3, 1e-4])
@click.option("--model", "model_names", multiple=True, default=["fnet", "resnet18"])
def hp_search_cmd(config_path: str, learning_rates: tuple[float, ...], model_names: tuple[str, ...]) -> None:
    """Run simple grid hyperparameter search."""
    run_hyperparameter_search(config_path, list(learning_rates), list(model_names))


if __name__ == "__main__":
    cli()
