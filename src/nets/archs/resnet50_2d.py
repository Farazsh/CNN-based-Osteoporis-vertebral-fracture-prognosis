import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicBlock2D(nn.Module):
    """
    A basic building block for the 2D ResNet, implementing two 2D convolutional layers with BatchNorm and ReLU activations.
    """

    # expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.shortcut = nn.Sequential()
        # if stride != 1 or in_channels != self.expansion*out_channels:
        #     self.shortcut = nn.Sequential(
        #         nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(self.expansion*out_channels)
        #     )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2D(nn.Module):
    """
    Implements a 2D ResNet architecture from scratch with a modular design.
    This model adapts the ResNet architecture for 2D inputs.
    """

    def __init__(self, block, num_blocks, num_classes=1):
        super(ResNet2D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """
        Creates a layer by stacking `num_blocks` of the specified block type, adjusting the stride for downsampling.
        """
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # self.in_channels = out_channels * block.expansion
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the model.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet50_2d():
    """
    Constructs a 2D ResNet-50 model.
    """
    return ResNet2D(BasicBlock2D, [3, 4, 6, 3])


if __name__ == "__main__":
    model = resnet50_2d()
    image_batch = torch.tensor(np.random.rand(16, 1, 47, 47), dtype=torch.float32)
    print(image_batch.shape)
    output = model(image_batch)
    print(output)
