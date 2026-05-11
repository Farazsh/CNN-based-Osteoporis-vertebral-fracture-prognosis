import torch

from monai.networks.blocks import SEResNetBottleneck
from torch import nn
from monai.networks.nets import SEResNet50


class CustomSEResNet50(SEResNet50):
    r"""**A customized version of SEResNet50.**

    We only need to get:
        the number of channels_in (can be read via self.backbone.inplanes)
        the number of channels_out [number_of_classes] (can be read via self.backbone.channels_out)
    """
    def __init__(self) -> None:

        self.channels_in = 512 * SEResNetBottleneck.expansion  # TODO: Why's that?

        super().__init__(
            input_3x3=False,  # If True, the first layer is 3x3 conv
            in_channels=1,  # 1=grayscale, 3=rgb, default = 1
            spatial_dims=3,  # 2=2D imgage data , 3=3D volumetric data, default = 3
            progress=False,  # Show progressbar when downloading pretrained encoder
            num_classes=0,  # src.architecture.utilities.get_channels_out(config=self.config),  # num_classes used as **kwargs
            dropout_prob=None  # Dropout probability, default is already None
        )
        self.classifier = nn.LazyLinear(1)
        del self.last_linear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x