"""Logging setup utilities."""

from __future__ import annotations

import logging

from pytorch_lightning.loggers import CSVLogger, WandbLogger

from osteo_fracture.config import LoggingConfig


def build_logger(cfg: LoggingConfig):
    """Build WandB logger when enabled, fallback to CSV logger."""
    if cfg.use_wandb:
        return WandbLogger(project=cfg.project, name=cfg.run_name)
    return CSVLogger(save_dir="logs", name=cfg.run_name)


def configure_console_logging(level: int = logging.INFO) -> None:
    """Initialize basic console logging."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
