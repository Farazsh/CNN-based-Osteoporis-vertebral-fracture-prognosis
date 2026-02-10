"""Metrics for binary classification and risk analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


@dataclass
class BinaryMetrics:
    """Container for computed metrics."""

    auroc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    c_index: float
    shr: float


def concordance_index(event_times: np.ndarray, scores: np.ndarray) -> float:
    """Simple Harrell's C-index implementation."""
    n_concordant = 0
    n_comparable = 0
    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            if event_times[i] == event_times[j]:
                continue
            n_comparable += 1
            if (event_times[i] < event_times[j] and scores[i] > scores[j]) or (
                event_times[j] < event_times[i] and scores[j] > scores[i]
            ):
                n_concordant += 1
    return float(n_concordant / n_comparable) if n_comparable else float("nan")


def subdistribution_hazard_ratio(target: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Approximate subdistribution hazard ratio via odds ratio from thresholded risk groups."""
    pred = (scores >= threshold).astype(int)
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    tn = np.sum((pred == 0) & (target == 0))
    return float(((tp + 1) * (tn + 1)) / ((fp + 1) * (fn + 1)))


def compute_binary_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    event_times: np.ndarray | None = None,
) -> BinaryMetrics:
    """Compute model quality metrics from probabilities and labels."""
    probs = prediction.flatten().detach().cpu()
    y = target.flatten().long().detach().cpu()

    auroc = float(BinaryAUROC()(probs, y))
    auprc = float(BinaryAveragePrecision()(probs, y))

    pred = (probs >= threshold).long()
    tp = int(torch.sum(y * pred).item())
    tn = int(torch.sum((1 - y) * (1 - pred)).item())
    fp = int(torch.sum((1 - y) * pred).item())
    fn = int(torch.sum(y * (1 - pred)).item())

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    probs_np = probs.numpy()
    y_np = y.numpy()
    c_index = concordance_index(event_times, probs_np) if event_times is not None else float("nan")
    shr = subdistribution_hazard_ratio(y_np, probs_np, threshold)

    return BinaryMetrics(
        auroc=auroc,
        auprc=auprc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        c_index=c_index,
        shr=shr,
    )
