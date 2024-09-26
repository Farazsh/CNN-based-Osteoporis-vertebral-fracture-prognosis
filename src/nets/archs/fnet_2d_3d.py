import torch
import torch.nn as nn
from auxillary_functions import SELayer
from torchvision.models.feature_extraction import create_feature_extractor
from auxillary_functions import count_and_print_num_of_parameters


class Fnet2D(nn.Module):
    def __init__(self):
        super(Fnet2D, self).__init__()
        conv_output_size = 3 * 3
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * conv_output_size, 1),
        )

    def forward(self, x):
        return self.backbone(x)


class Fnet3D(nn.Module):
    def __init__(self):
        super(Fnet3D, self).__init__()
        conv_output_size = 3 * 3 * 3
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=0, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=0, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * conv_output_size, 1),
        )

    def forward(self, x):
        return self.backbone(x)


class SEnet3D(nn.Module):
    def __init__(self):
        super(SEnet3D, self).__init__()
        conv_output_size = 5 * 3 * 2
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=3, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm3d(3),
            torch.nn.Conv3d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm3d(64),
            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm3d(128),
            SELayer(128),
            torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm3d(128),
            torch.nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.InstanceNorm3d(256),
            SELayer(256),
            torch.nn.Flatten(),
            torch.nn.Linear(256 * conv_output_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.backbone(x)


if __name__ == "__main__":
    model = ResnetMedicalPT()
    count_and_print_num_of_parameters(model)
