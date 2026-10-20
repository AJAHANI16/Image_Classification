import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from image_classification.models import (
    Classifier,
    build_model,
    make_transform,
    optimizer_groups,
    preprocessing_for,
)


@pytest.mark.parametrize("architecture", ["efficientnet_v2_s", "convnext_tiny", "tiny_cnn"])
def test_all_architectures_forward_and_backward_without_downloads(architecture):
    model = build_model(architecture, 3, pretrained=False)
    output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 3)
    assert torch.isfinite(output).all()
    nn.functional.cross_entropy(output, torch.tensor([0, 2])).backward()
    assert model.classifier[-1].weight.grad.abs().sum() > 0
    assert next(model.features.parameters()).grad.abs().sum() > 0


def test_freezing_protects_batchnorm_buffers_and_backbone_gradients():
    features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.Dropout(0.5))
    model = Classifier(
        features, nn.AdaptiveAvgPool2d(1), nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    )
    before = features[1].running_mean.clone()
    model.set_backbone_trainable(False)
    model.train()
    model(torch.ones(2, 3, 16, 16)).sum().backward()
    assert not features.training
    assert torch.equal(before, features[1].running_mean)
    assert all(parameter.grad is None for parameter in features.parameters())
    assert model.classifier[-1].weight.grad is not None
    model.zero_grad(set_to_none=True)
    model.set_backbone_trainable(True)
    model.train()
    model(torch.ones(2, 3, 16, 16)).sum().backward()
    assert features.training
    assert features[0].weight.grad is not None
    assert not torch.equal(before, features[1].running_mean)


@pytest.mark.parametrize("architecture", ["efficientnet_v2_s", "convnext_tiny"])
def test_eval_preprocessing_matches_official_weights(architecture):
    from image_classification.models import WEIGHTS

    pixels = np.random.default_rng(7).integers(0, 256, size=(421, 503, 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    actual = make_transform(preprocessing_for(architecture))(image)
    expected = WEIGHTS[architecture].transforms()(image)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert torch.equal(actual, make_transform(preprocessing_for(architecture))(image))


def test_optimizer_groups_have_no_missing_or_repeated_parameters():
    model = build_model("tiny_cnn", 2, pretrained=False)
    model.set_backbone_trainable(False)
    groups = optimizer_groups(model, learning_rate=0.01, backbone_lr_ratio=0.1, weight_decay=0.1)
    parameters = [parameter for group in groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {id(p) for p in model.parameters()}
    assert {group["lr"] for group in groups} == {0.001, 0.01}
    for group in groups:
        if any(parameter.ndim <= 1 for parameter in group["params"]):
            assert group["weight_decay"] == 0.0
