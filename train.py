"""Training entrypoint that wires config, datamodule, model, and trainer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytorch_lightning as pl

from osteo_fracture.config import load_config
from osteo_fracture.data.datamodule import FractureDataModule
from osteo_fracture.training.callbacks import build_callbacks
from osteo_fracture.training.lightning_module import FractureLightningModule
from osteo_fracture.utils.logging import build_logger
from osteo_fracture.utils.reproducibility import seed_everything


def run_training(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.seed)

    project_root = Path(__file__).resolve().parent
    datamodule = FractureDataModule(cfg.data, project_root)
    class_ratio = datamodule.get_class_ratio() if cfg.model.pos_weight is None else cfg.model.pos_weight

    model = FractureLightningModule(
        model_name=cfg.model.name,
        spatial_dims=cfg.data.input_dimension,
        learning_rate=cfg.model.learning_rate,
        optimizer_name=cfg.model.optimizer,
        threshold=cfg.model.threshold,
        class_ratio=class_ratio,
    )

    logger = build_logger(cfg.logging)
    checkpoint_dir = project_root / "checkpoints" / cfg.logging.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=build_callbacks(cfg.trainer, checkpoint_dir),
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == "__main__":
    run_training("configs/default.yaml")
