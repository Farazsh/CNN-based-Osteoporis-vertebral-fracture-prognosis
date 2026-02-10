"""Typed YAML configuration loader for experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    img_dir: str = "MrOs_dataset/patches_arbitary_sized"
    labels_file: str = "MrOs_dataset/MrOs_Label_from_meta_v12_1_relabelled2_newsplit.csv"
    batch_size: int = 16
    num_workers: int = 4
    label: str = "IF10SQ1"
    censor_label: str = "Censored_label"
    id_label: str = "ID"
    input_size: int = 47
    input_dimension: int = 3
    split_name: str = "Split1"
    inference_only: bool = False
    image_augmentation_prob: float = 0.33
    augmentation_magnitude: int = 8
    num_sequential_transforms: int = 3


@dataclass
class ModelConfig:
    name: str = "fnet"
    learning_rate: float = 1e-4
    optimizer: str = "adamw"
    pos_weight: float | None = None
    threshold: float = 0.5


@dataclass
class TrainerConfig:
    max_epochs: int = 50
    devices: int = 1
    accelerator: str = "auto"
    precision: str = "32"
    monitor_metric: str = "validation_auroc"
    monitor_mode: str = "max"
    early_stopping_patience: int = 8
    use_swa: bool = False
    swa_lrs: float = 1e-3


@dataclass
class LoggingConfig:
    project: str = "osteoporotic-fracture-prognosis"
    run_name: str = "baseline"
    use_wandb: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)



def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    default = ExperimentConfig()
    merged = _deep_merge(
        {
            "seed": default.seed,
            "data": default.data.__dict__,
            "model": default.model.__dict__,
            "trainer": default.trainer.__dict__,
            "logging": default.logging.__dict__,
        },
        raw,
    )
    return ExperimentConfig(
        seed=merged["seed"],
        data=DataConfig(**merged["data"]),
        model=ModelConfig(**merged["model"]),
        trainer=TrainerConfig(**merged["trainer"]),
        logging=LoggingConfig(**merged["logging"]),
    )
