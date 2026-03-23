from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_logits_to_target(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

from __future__ import annotations

import torch


def calculate_iou(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def calculate_dice(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds_flat = preds.reshape(preds.size(0), -1)
    targets_flat = targets.reshape(targets.size(0), -1)
    
    intersection = (preds_flat * targets_flat).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (preds_flat.sum(dim=1) + targets_flat.sum(dim=1) + smooth)
    return dice.mean().item()
