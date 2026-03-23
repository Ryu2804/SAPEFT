from __future__ import annotations

import cv2
import numpy as np


def expand_mask_with_context(mask_np: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Expand each connected region to include surrounding context.

    scale=2.0 means bbox width/height become ~2x. Use 1.5-3.0 as practical range.
    """
    if mask_np.dtype != np.uint8:
        mask_np = mask_np.astype(np.uint8)
    binary = (mask_np > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    expanded = np.zeros_like(binary, dtype=np.uint8)
    height, width = binary.shape

    for label_idx in range(1, num_labels):
        x, y, bw, bh, area = stats[label_idx]
        if area <= 0:
            continue

        cx = x + bw / 2.0
        cy = y + bh / 2.0
        new_w = max(1, int(round(bw * scale)))
        new_h = max(1, int(round(bh * scale)))

        x1 = max(0, int(round(cx - new_w / 2.0)))
        y1 = max(0, int(round(cy - new_h / 2.0)))
        x2 = min(width, int(round(cx + new_w / 2.0)))
        y2 = min(height, int(round(cy + new_h / 2.0)))

        expanded[y1:y2, x1:x2] = 1

    return expanded
