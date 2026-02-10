"""Model backbones used for fracture prognosis."""

from __future__ import annotations

from abc import ABC, abstractmethod

import monai
import torch
import torch.nn as nn
import torchvision


class Backbone(ABC, nn.Module):
    """Common interface for all backbones."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return logits."""


class FNet2D(Backbone):
    """Compact 2D convolutional network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FNet3D(Backbone):
    """Compact 3D convolutional network."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3 * 3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18_2D(Backbone):
    """Torchvision ResNet-18 adapted for single-logit binary prediction."""

    def __init__(self) -> None:
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet18_3D(Backbone):
    """MONAI ResNet18 3D adapted for binary output."""

    def __init__(self) -> None:
        super().__init__()
        self.net = monai.networks.nets.resnet18(
            spatial_dims=3,
            n_input_channels=1,
            num_classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNet50_2D(Backbone):
    """Torchvision ResNet-50 adapted for single-channel input."""

    def __init__(self) -> None:
        super().__init__()
        net = torchvision.models.resnet50(weights=None)
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        net.fc = nn.Linear(net.fc.in_features, 1)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_backbone(name: str, spatial_dims: int) -> Backbone:
    """Factory to build model backbones by name."""
    key = name.lower()
    if key == "fnet" and spatial_dims == 2:
        return FNet2D()
    if key == "fnet" and spatial_dims == 3:
        return FNet3D()
    if key == "resnet18" and spatial_dims == 2:
        return ResNet18_2D()
    if key == "resnet18" and spatial_dims == 3:
        return ResNet18_3D()
    if key == "resnet50" and spatial_dims == 2:
        return ResNet50_2D()
    if key == "seresnext50":
        return monai.networks.nets.SEResNext50(spatial_dims=spatial_dims, in_channels=1, num_classes=1)  # type: ignore[return-value]
    raise ValueError(f"Unsupported model '{name}' for spatial_dims={spatial_dims}")
