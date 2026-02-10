"""Loss utilities for binary fracture prediction."""

from __future__ import annotations

import torch
import torch.nn as nn


def compute_pos_weight(class_ratio: float | None) -> torch.Tensor | None:
    """Build a positive-class weight tensor for BCEWithLogitsLoss."""
    if class_ratio is None:
        return None
    return torch.tensor([class_ratio], dtype=torch.float32)


def build_bce_loss(class_ratio: float | None = None) -> nn.BCEWithLogitsLoss:
    """Create BCE-with-logits loss with optional class weighting."""
    pos_weight = compute_pos_weight(class_ratio)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
