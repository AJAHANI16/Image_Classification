"""Pretrained backbones and checkpoint-owned preprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models
from torchvision.transforms import InterpolationMode, v2

ARCHITECTURES = ("efficientnet_v2_s", "convnext_tiny", "tiny_cnn")
WEIGHTS = {
    "efficientnet_v2_s": models.EfficientNet_V2_S_Weights.DEFAULT,
    "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
}


@dataclass(frozen=True)
class Preprocessing:
    image_size: int
    resize_size: int
    mean: tuple[float, ...] = (0.485, 0.456, 0.406)
    std: tuple[float, ...] = (0.229, 0.224, 0.225)
    interpolation: str = "bilinear"


def preprocessing_for(architecture: str, image_size: int | None = None) -> Preprocessing:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    if architecture == "tiny_cnn":
        size = image_size or 32
        spec = Preprocessing(size, round(size * 1.125))
    else:
        preset = WEIGHTS[architecture].transforms()
        size = image_size or preset.crop_size[0]
        spec = Preprocessing(
            size,
            round(size * preset.resize_size[0] / preset.crop_size[0]),
            tuple(preset.mean),
            tuple(preset.std),
            preset.interpolation.value,
        )
    minimum = 16 if architecture == "tiny_cnn" else 64
    if spec.image_size < minimum:
        raise ValueError(f"image_size must be at least {minimum} for {architecture}")
    return spec


def make_transform(
    spec: Preprocessing, *, training: bool = False, augment: bool = True, hflip: bool = True
) -> v2.Compose:
    interpolation = InterpolationMode(spec.interpolation)
    ops: list = []
    if training and augment:
        ops.append(v2.ToImage())
        ops.append(
            v2.RandomResizedCrop(
                spec.image_size, scale=(0.7, 1.0), interpolation=interpolation, antialias=True
            )
        )
        if hflip:
            ops.append(v2.RandomHorizontalFlip())
        ops.append(v2.RandAugment(num_ops=2, magnitude=9, interpolation=interpolation))
    else:
        ops.extend(
            [
                v2.Resize(spec.resize_size, interpolation=interpolation, antialias=True),
                v2.CenterCrop(spec.image_size),
                # Match the official weights' PIL resize/crop path before tensor conversion.
                v2.ToImage(),
            ]
        )
    ops.extend([v2.ToDtype(torch.float32, scale=True), v2.Normalize(spec.mean, spec.std)])
    return v2.Compose(ops)


class Classifier(nn.Module):
    def __init__(self, features: nn.Module, pool: nn.Module, classifier: nn.Module):
        super().__init__()
        self.features = features
        self.pool = pool
        self.classifier = classifier
        self.backbone_frozen = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))

    def set_backbone_trainable(self, trainable: bool) -> None:
        self.backbone_frozen = not trainable
        self.features.requires_grad_(trainable)
        self.features.train(self.training and trainable)

    def train(self, mode: bool = True) -> Classifier:
        super().train(mode)
        # Frozen BatchNorm buffers and stochastic layers must stay frozen as well.
        if self.backbone_frozen:
            self.features.eval()
        return self


def build_model(architecture: str, num_classes: int, *, pretrained: bool = True) -> Classifier:
    if num_classes < 2:
        raise ValueError("At least two classes are required")
    if architecture == "tiny_cnn":
        features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        return Classifier(
            features,
            nn.AdaptiveAvgPool2d(1),
            nn.Sequential(nn.Flatten(), nn.Linear(32, num_classes)),
        )
    if architecture not in WEIGHTS:
        raise ValueError(f"Unknown architecture: {architecture}")
    weights = WEIGHTS[architecture] if pretrained else None
    backbone = getattr(models, architecture)(weights=weights)
    old_head = backbone.classifier[-1]
    backbone.classifier[-1] = nn.Linear(old_head.in_features, num_classes)
    classifier = backbone.classifier
    if architecture == "efficientnet_v2_s":
        # EfficientNet flattens in its original forward(), outside its classifier module.
        classifier = nn.Sequential(nn.Flatten(1), *classifier.children())
    return Classifier(backbone.features, backbone.avgpool, classifier)


def optimizer_groups(
    model: Classifier, *, learning_rate: float, backbone_lr_ratio: float, weight_decay: float
) -> list[dict]:
    groups: dict[tuple[bool, bool], list[nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        # Biases and normalization scales do not receive weight decay.
        key = (name.startswith("features."), parameter.ndim <= 1)
        groups.setdefault(key, []).append(parameter)
    return [
        {
            "params": parameters,
            "lr": learning_rate * (backbone_lr_ratio if is_backbone else 1.0),
            "weight_decay": 0.0 if no_decay else weight_decay,
        }
        for (is_backbone, no_decay), parameters in groups.items()
    ]
