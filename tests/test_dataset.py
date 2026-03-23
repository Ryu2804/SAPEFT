import cv2
import numpy as np
import albumentations as A

from dataset.get_dataset import (
    ImageSegmentationDataset,
    build_segmentation_dataloader,
    resolve_split_base_dir,
)


def _write_pair(image_path, mask_path, image, mask):
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)


def test_image_segmentation_dataset_basic(tmp_path):
    split_dir = tmp_path / "testing"
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    img1 = np.zeros((6, 8, 3), dtype=np.uint8)
    img2 = np.full((6, 8, 3), 255, dtype=np.uint8)
    mask1 = np.zeros((6, 8), dtype=np.uint8)
    mask2 = np.ones((6, 8), dtype=np.uint8) * 255

    _write_pair(image_dir / "sample1.jpg", mask_dir / "sample1.png", img1, mask1)
    _write_pair(image_dir / "sample2.jpg", mask_dir / "sample2.png", img2, mask2)

    dataset = ImageSegmentationDataset(base_dir=tmp_path, split="testing")
    assert len(dataset) == 2

    image_tensor, mask_tensor = dataset[0]
    assert image_tensor.shape[0] == 3
    assert mask_tensor.shape[0] == 1


def test_build_segmentation_dataloader_with_transform(tmp_path):
    split_dir = tmp_path / "testing"
    image_dir = split_dir / "images"
    mask_dir = split_dir / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    img = np.zeros((6, 8, 3), dtype=np.uint8)
    mask = np.zeros((6, 8), dtype=np.uint8)
    _write_pair(image_dir / "sample.jpg", mask_dir / "sample.png", img, mask)

    transform = A.Compose([A.Resize(height=4, width=4)])
    dataset, loader = build_segmentation_dataloader(
        base_dir=tmp_path,
        split="testing",
        transform=transform,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    image_tensor, mask_tensor = next(iter(loader))
    assert image_tensor.shape[-2:] == (4, 4)
    assert mask_tensor.shape[-2:] == (4, 4)


def test_resolve_split_base_dir(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "seg" / "validation").mkdir(parents=True)
    resolved = resolve_split_base_dir(data_dir, "validation")
    assert resolved == data_dir / "seg"
