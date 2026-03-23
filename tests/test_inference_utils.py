import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from inference.runner import run_inference_and_evaluate


class DummySegDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image = torch.zeros(3, 8, 8)
        mask = torch.zeros(1, 8, 8)
        return image, mask


class DummyModel(torch.nn.Module):
    def forward(self, pixel_values):
        return pixel_values.mean(dim=1, keepdim=True)


def test_run_inference_and_evaluate_cpu():
    dataset = DummySegDataset()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = DummyModel()

    avg_iou, avg_dice = run_inference_and_evaluate(
        model,
        loader,
        device=torch.device("cpu"),
        num_visualize=0,
        threshold=0.5,
        use_amp=False,
        amp_dtype=None,
    )

    assert avg_iou == pytest.approx(1.0, rel=1e-6)
    assert avg_dice == pytest.approx(1.0, rel=1e-6)
