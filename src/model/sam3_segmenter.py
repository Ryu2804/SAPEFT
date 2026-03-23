from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def resolve_backbone_embed_dim(sam_model, vision_backbone) -> int:
    candidates = [
        getattr(getattr(vision_backbone, "config", None), "hidden_size", None),
        getattr(getattr(sam_model, "config", None), "hidden_size", None),
        getattr(
            getattr(getattr(sam_model, "config", None), "vision_config", None),
            "hidden_size",
            None,
        ),
    ]
    for value in candidates:
        if isinstance(value, int) and value > 0:
            return value
    return 1024


class UnpromptedSAMSegmenter(nn.Module):
    def __init__(self, backbone, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone

        if not isinstance(embed_dim, int) or embed_dim <= 0:
            raise RuntimeError("embed_dim must be a positive integer for SAM3 backbone outputs.")

        self.mask_head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def _reshape_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features
        if features.ndim == 3:
            batch, tokens, channels = features.shape
            side = int(np.sqrt(tokens))
            if side * side != tokens:
                raise RuntimeError(f"Cannot reshape sequence length {tokens} into square feature map")
            return features.transpose(1, 2).contiguous().view(batch, channels, side, side)
        raise RuntimeError(f"Unexpected backbone feature shape: {tuple(features.shape)}")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if hasattr(self.backbone, "base_model"):
            backbone_outputs = self.backbone.base_model(pixel_values=pixel_values)
        else:
            backbone_outputs = self.backbone(pixel_values=pixel_values)

        features = backbone_outputs.last_hidden_state
        features = self._reshape_features(features)

        features = features.to(next(self.mask_head.parameters()).dtype)

        logits = self.mask_head(features)
        return logits
