from __future__ import annotations
import warnings
import torch
import torch.nn as nn


class NormalizedModel(nn.Module):
    """Takes images in [0,1]; normalisation is inside, so every budget is in raw pixel space."""
    def __init__(self, backbone, mean, std):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([mean] * 3).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([std] * 3).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


class Smoothed(nn.Module):
    """3x3 average pooling pre-filter. Differentiable, so attacks go THROUGH it (adaptive)."""
    def __init__(self, model):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 1, 1)
        self.model = model

    def forward(self, x):
        return self.model(self.pool(x))


class FeatureSqueeze(nn.Module):
    """Bit-depth reduction with a straight-through gradient (BPDA), so the attack is adaptive."""
    def __init__(self, model, bits=5):
        super().__init__()
        self.model, self.levels = model, 2 ** bits - 1

    def forward(self, x):
        q = torch.round(x * self.levels) / self.levels
        return self.model(x + (q - x).detach())


def build_model(cfg: dict, num_classes: int = 3) -> nn.Module:
    from torchvision import models
    arch = cfg["model"].get("arch", "resnet18")
    m = None
    if cfg["model"]["pretrained"]:
        try:
            m = models.get_model(arch, weights="DEFAULT")
        except Exception as e:
            warnings.warn(f"!!! Could not load ImageNet weights ({e}). Training from scratch - REPORT THIS !!!")
    if m is None:
        m = models.get_model(arch, weights=None)
    if hasattr(m, "fc"):
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return NormalizedModel(m, cfg["model"]["mean"], cfg["model"]["std"])