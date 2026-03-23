from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ImageSegmentationDataset(Dataset):
    def __init__(self, base_dir: str | Path, split: str, transform=None):
        super().__init__()
        self.split_dir = Path(base_dir) / split
        self.image_dir = self.split_dir / "images"
        self.mask_dir = self.split_dir / "masks"
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Missing mask directory: {self.mask_dir}")

        self.image_filenames = sorted(
            [p.name for p in self.image_dir.iterdir() if p.is_file()]
        )

        self.mask_paths = {}
        for mask_name in self.mask_dir.iterdir():
            if mask_name.is_file():
                self.mask_paths[mask_name.stem] = mask_name

        self.image_filenames = [
            name for name in self.image_filenames if Path(name).stem in self.mask_paths
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = self.image_dir / img_name
        mask_path = self.mask_paths[img_path.stem]

        image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            transformed = self.transform(image=image_rgb, mask=mask)
            image_rgb = transformed["image"]
            mask = transformed["mask"]

        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)

        return image_tensor, mask_tensor


def resolve_split_base_dir(data_dir: str | Path, split: str) -> Path:
    data_dir = Path(data_dir)
    seg_dir = data_dir / "seg"
    if (seg_dir / split).exists():
        return seg_dir
    return data_dir


def build_segmentation_dataloader(
    *,
    base_dir: str | Path,
    split: str,
    transform=None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> Tuple[ImageSegmentationDataset, DataLoader]:
    dataset = ImageSegmentationDataset(base_dir=base_dir, split=split, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return dataset, loader
