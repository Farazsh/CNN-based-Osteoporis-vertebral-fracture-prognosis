"""Trainer callback factory."""

from __future__ import annotations

from pathlib import Path

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, StochasticWeightAveraging

from osteo_fracture.config import TrainerConfig


def build_callbacks(cfg: TrainerConfig, checkpoint_dir: Path) -> list:
    """Build early stopping, checkpointing, and optional SWA callbacks."""
    callbacks: list = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch}-{validation_auroc:.4f}",
            monitor=cfg.monitor_metric,
            mode=cfg.monitor_mode,
            save_top_k=1,
        ),
        EarlyStopping(
            monitor=cfg.monitor_metric,
            mode=cfg.monitor_mode,
            patience=cfg.early_stopping_patience,
        ),
    ]
    if cfg.use_swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=cfg.swa_lrs))
    return callbacks
