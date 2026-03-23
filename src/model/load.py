from __future__ import annotations

from pathlib import Path
import torch


def load_lora_weights(
    model: torch.nn.Module,
    weights_path: str | Path,
    *,
    device: torch.device | str | None = None,
) -> bool:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        print(f"Weight path '{weights_path}' not found. Please ensure the file is present.")
        return False

    state = torch.load(weights_path, map_location=device or "cpu")
    best_val_iou = None

    if isinstance(state, dict) and "model_state_dict" in state:
        best_val_iou = state.get("best_val_iou")
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)

    if best_val_iou is not None:
        print(
            f"Loaded weights from {weights_path} successfully! Best Val IoU (from training): {best_val_iou}"
        )
    else:
        print(f"Loaded weights from {weights_path} successfully!")

    return True
