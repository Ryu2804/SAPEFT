import torch

from model.sam3_segmenter import UnpromptedSAMSegmenter, resolve_backbone_embed_dim


class Dummy:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class DummyOutput:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class DummyBackbone(torch.nn.Module):
    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        features = torch.zeros(batch, 16 * 16, 8)
        return DummyOutput(features)


def test_resolve_backbone_embed_dim():
    sam_model = Dummy(config=Dummy(hidden_size=512, vision_config=Dummy(hidden_size=256)))
    vision_backbone = Dummy(config=Dummy(hidden_size=768))

    assert resolve_backbone_embed_dim(sam_model, vision_backbone) == 768

    sam_model = Dummy(config=Dummy(hidden_size=640, vision_config=Dummy(hidden_size=256)))
    vision_backbone = Dummy(config=Dummy(hidden_size=None))
    assert resolve_backbone_embed_dim(sam_model, vision_backbone) == 640


def test_unprompted_segmenter_forward():
    backbone = DummyBackbone()
    model = UnpromptedSAMSegmenter(backbone=backbone, embed_dim=8, num_classes=1)

    pixels = torch.zeros(2, 3, 16, 16)
    logits = model(pixel_values=pixels)
    assert logits.shape == (2, 1, 16, 16)


def test_unprompted_segmenter_reshape_features():
    backbone = DummyBackbone()
    model = UnpromptedSAMSegmenter(backbone=backbone, embed_dim=8, num_classes=1)

    features = torch.randn(2, 16 * 16, 8)
    reshaped = model._reshape_features(features)
    assert reshaped.shape == (2, 8, 16, 16)
