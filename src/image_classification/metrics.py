"""Metrics with explicit class coverage and validation-only temperature fitting."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_logits(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> None:
    if logits.ndim != 2 or logits.shape != (labels.numel(), num_classes) or labels.ndim != 1:
        raise ValueError("Expected logits [N, C] and labels [N]")
    if labels.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("Metrics require nonempty, finite logits")
    if labels.dtype != torch.long or labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError("Labels must be int64 class indices in [0, C)")


def classification_metrics(
    logits: torch.Tensor, labels: torch.Tensor, classes: list[str], *, temperature: float = 1.0
) -> dict:
    logits, labels = logits.detach().float().cpu(), labels.detach().cpu()
    validate_logits(logits, labels, len(classes))
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    scaled = logits / temperature
    probabilities = scaled.softmax(dim=1)
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted == labels
    count = len(classes)
    confusion = torch.bincount(labels * count + predicted, minlength=count * count).reshape(
        count, count
    )
    true_positive = confusion.diag().float()
    support = confusion.sum(dim=1)
    precision = true_positive / confusion.sum(dim=0).clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    present = support > 0
    # Each confidence belongs to exactly one bin, including boundary values.
    bin_indices = (confidence * 15).ceil().long().sub(1).clamp(0, 14)
    ece = torch.tensor(0.0)
    for bin_index in range(15):
        mask = bin_indices == bin_index
        if mask.any():
            ece += (
                mask.float().mean() * (correct[mask].float().mean() - confidence[mask].mean()).abs()
            )
    one_hot = F.one_hot(labels, num_classes=count).float()
    return {
        "samples": labels.numel(),
        "accuracy": correct.float().mean().item(),
        "balanced_accuracy": recall[present].mean().item(),
        "macro_f1": f1.mean().item(),
        "negative_log_likelihood": F.cross_entropy(scaled, labels).item(),
        "brier_score": ((probabilities - one_hot) ** 2).sum(dim=1).mean().item(),
        "expected_calibration_error": ece.item(),
        "confusion_matrix": confusion.tolist(),
        "class_order": classes,
        "per_class": {
            name: {
                "precision": precision[i].item(),
                "recall": recall[i].item(),
                "f1": f1[i].item(),
                "support": support[i].item(),
            }
            for i, name in enumerate(classes)
        },
    }


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one scalar on validation logits; never accept a worse validation NLL."""
    logits, labels = logits.detach().float().cpu(), labels.detach().long().cpu()
    validate_logits(logits, labels, logits.shape[1])
    log_temperature = nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=50, line_search_fn="strong_wolfe"
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_temperature.clamp(math.log(0.05), math.log(20.0)).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = log_temperature.detach().clamp(math.log(0.05), math.log(20.0)).exp().item()
    calibrated = F.cross_entropy(logits / temperature, labels).item()
    original = F.cross_entropy(logits, labels).item()
    return temperature if math.isfinite(calibrated) and calibrated <= original else 1.0
