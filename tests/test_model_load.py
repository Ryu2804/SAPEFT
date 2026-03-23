import torch

from model.load import load_lora_weights


def test_load_lora_weights_from_state_dict(tmp_path):
    model = torch.nn.Linear(4, 2)
    weights_path = tmp_path / "weights.pth"
    torch.save(model.state_dict(), weights_path)

    loaded = load_lora_weights(model, weights_path, device="cpu")
    assert loaded is True


def test_load_lora_weights_missing_path(tmp_path):
    model = torch.nn.Linear(4, 2)
    missing_path = tmp_path / "missing.pth"
    loaded = load_lora_weights(model, missing_path, device="cpu")
    assert loaded is False
