import pytest
import torch

from metric_utils.eval_metric import resize_logits_to_target
from metric_utils.severity_metric import calculate_dice, calculate_iou


def test_iou_dice_perfect_match():
    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    assert calculate_iou(logits, targets) == pytest.approx(1.0, rel=1e-6)
    assert calculate_dice(logits, targets) == pytest.approx(1.0, rel=1e-6)


def test_resize_logits_to_target():
    logits = torch.randn(1, 1, 4, 4)
    target = torch.randn(1, 1, 2, 2)
    resized = resize_logits_to_target(logits, target)
    assert resized.shape[-2:] == target.shape[-2:]
