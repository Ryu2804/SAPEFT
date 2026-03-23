from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.amp import autocast
from tqdm.auto import tqdm

from metric_utils.eval_metric import resize_logits_to_target,calculate_dice, calculate_iou


def _amp_context(device: torch.device, use_amp: bool, amp_dtype: torch.dtype | None):
    if use_amp and device.type == "cuda":
        return autocast(device_type="cuda", dtype=amp_dtype or torch.float16, enabled=True)
    return nullcontext()


def run_inference_and_evaluate(
    model: torch.nn.Module,
    dataloader,
    *,
    device: torch.device,
    num_visualize: int = 4,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> Tuple[float, float]:
    import matplotlib.pyplot as plt

    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    num_batches = len(dataloader) if hasattr(dataloader, "__len__") else 0

    shown = 0
    if num_visualize > 0:
        plt.figure(figsize=(12, 4 * num_visualize))

    print("Starting Evaluation on Test Set...")
    with torch.inference_mode():
        test_pbar = tqdm(dataloader, desc="EVALUATING")
        for images, masks in test_pbar:
            images = images.to(device)
            masks = masks.to(device)

            with _amp_context(device, use_amp, amp_dtype):
                logits = model(pixel_values=images)
                logits_resized = resize_logits_to_target(logits, masks)

            iou = calculate_iou(logits_resized, masks, threshold)
            dice = calculate_dice(logits_resized, masks, threshold)

            total_iou += iou
            total_dice += dice

            test_pbar.set_postfix(iou=f"{iou:.4f}", dice=f"{dice:.4f}")

            if shown < num_visualize:
                pred_mask = (torch.sigmoid(logits_resized) > threshold).float()
                pred_np = pred_mask[0, 0].detach().cpu().numpy().astype(np.uint8)

                img_np = images[0].detach().cpu().permute(1, 2, 0).numpy()
                gt_np = masks[0, 0].detach().cpu().numpy()

                plt.subplot(num_visualize, 2, 2 * shown + 1)
                plt.imshow(img_np)
                plt.imshow(gt_np, alpha=0.45, cmap="Reds")
                plt.title(f"Ground Truth (Test Image {shown+1})")
                plt.axis("off")

                plt.subplot(num_visualize, 2, 2 * shown + 2)
                plt.imshow(img_np)
                plt.imshow(pred_np, alpha=0.45, cmap="Blues")
                plt.title(f"Predicted Mask (IoU: {iou:.3f}, Dice: {dice:.3f})")
                plt.axis("off")

                shown += 1

    avg_iou = total_iou / max(num_batches, 1)
    avg_dice = total_dice / max(num_batches, 1)

    print("-" * 50)
    print(f"EVALUATION RESULTS OVER {len(dataloader.dataset)} TEST IMAGES:")
    print(f"Average Test IoU  : {avg_iou:.4f}")
    print(f"Average Test Dice : {avg_dice:.4f}")
    print("-" * 50)

    if num_visualize > 0:
        plt.tight_layout()
        plt.show()

    return avg_iou, avg_dice


def generate_masked_images_for_depth(
    model: torch.nn.Module,
    dataloader,
    *,
    output_dir: str | Path,
    device: torch.device,
    threshold: float = 0.5,
    use_amp: bool = False,
    amp_dtype: torch.dtype | None = None,
) -> int:
    """Run inference and save masked images for depth estimation."""
    model.eval()
    output_dir = Path(output_dir)

    print(f"Saving masked images to: {output_dir}")
    saved_count = 0

    with torch.inference_mode():
        val_pbar = tqdm(dataloader, desc="GENERATING MASKED IMAGES")
        for i, (images, masks) in enumerate(val_pbar):
            images = images.to(device)
            masks = masks.to(device)

            img_np = (images[0].detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(
                np.uint8
            )
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            with _amp_context(device, use_amp, amp_dtype):
                logits = model(pixel_values=images)
                logits_resized = resize_logits_to_target(logits, masks)

            pred_mask = (torch.sigmoid(logits_resized) > threshold).float()
            mask_np = pred_mask[0, 0].detach().cpu().numpy().astype(np.uint8)

            masked_img = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_np)

            try:
                orig_filename = dataloader.dataset.image_filenames[i]
                save_path = output_dir / orig_filename
            except AttributeError:
                save_path = output_dir / f"masked_{i:04d}.png"

            cv2.imwrite(str(save_path), masked_img)
            saved_count += 1

    print(f"Mask generation completed. {saved_count} images saved to {output_dir}")
    return saved_count
