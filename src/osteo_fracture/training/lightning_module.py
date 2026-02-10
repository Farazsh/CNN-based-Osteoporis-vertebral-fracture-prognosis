"""PyTorch Lightning module for fracture prediction."""

from __future__ import annotations

import pytorch_lightning as pl
import torch

from osteo_fracture.models.backbones import build_backbone
from osteo_fracture.training.losses import build_bce_loss
from osteo_fracture.training.metrics import compute_binary_metrics


class FractureLightningModule(pl.LightningModule):
    """Binary classifier LightningModule for CT patch prognosis."""

    def __init__(
        self,
        model_name: str,
        spatial_dims: int,
        learning_rate: float,
        optimizer_name: str,
        threshold: float,
        class_ratio: float | None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_backbone(model_name, spatial_dims)
        self.loss_fn = build_bce_loss(class_ratio)
        self.sigmoid = torch.nn.Sigmoid()
        self.threshold = threshold

        self._train_pred: list[torch.Tensor] = []
        self._train_target: list[torch.Tensor] = []
        self._val_pred: list[torch.Tensor] = []
        self._val_target: list[torch.Tensor] = []
        self._test_pred: list[torch.Tensor] = []
        self._test_target: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def _common_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x).flatten()
        loss = self.loss_fn(logits, y.float())
        probs = self.sigmoid(logits)
        return loss, probs, y

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, probs, y = self._common_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self._train_pred.append(probs.detach().cpu())
        self._train_target.append(y.detach().cpu())
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, probs, y = self._common_step(batch)
        self.log("validation_loss", loss, prog_bar=True)
        self._val_pred.append(probs.detach().cpu())
        self._val_target.append(y.detach().cpu())
        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, probs, y = self._common_step(batch)
        self.log("test_loss", loss)
        self._test_pred.append(probs.detach().cpu())
        self._test_target.append(y.detach().cpu())
        return loss

    def _log_epoch_metrics(self, split: str, preds: list[torch.Tensor], targets: list[torch.Tensor]) -> None:
        if not preds:
            return
        p = torch.cat(preds)
        t = torch.cat(targets)
        m = compute_binary_metrics(p, t, self.threshold)
        self.log(f"{split}_auroc", m.auroc, prog_bar=(split != "test"))
        self.log(f"{split}_auprc", m.auprc)
        self.log(f"{split}_accuracy", m.accuracy)
        self.log(f"{split}_precision", m.precision)
        self.log(f"{split}_recall", m.recall)
        self.log(f"{split}_f1", m.f1)
        self.log(f"{split}_specificity", m.specificity)
        self.log(f"{split}_c_index", m.c_index)
        self.log(f"{split}_shr", m.shr)

    def on_train_epoch_end(self) -> None:
        self._log_epoch_metrics("train", self._train_pred, self._train_target)
        self._train_pred.clear()
        self._train_target.clear()

    def on_validation_epoch_end(self) -> None:
        self._log_epoch_metrics("validation", self._val_pred, self._val_target)
        self._val_pred.clear()
        self._val_target.clear()

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics("test", self._test_pred, self._test_target)
        self._test_pred.clear()
        self._test_target.clear()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        opt_name = self.hparams.optimizer_name.lower()
        if opt_name == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        if opt_name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
