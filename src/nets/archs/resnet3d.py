import torch
import torch.nn as nn
import numpy as np


class BasicBlock(nn.Module):
    """
    BasicBlock used in ResNet-18.

    Each block has two 3D convolutional layers with the same number of output channels.
    Each convolution is followed by batch normalization and ReLU activation.
    A shortcut connection is added to the output of the second convolution.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = nn.ReLU()(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.ReLU()(out)
        return out


class ResNet3D(nn.Module):
    """
    ResNet3D for 3D image inputs.

    The network starts with a single 3D convolutional layer, followed by a max pooling.
    This is followed by 4 blocks of residual layers (BasicBlock), with 2, 2, 2, and 2 layers respectively.
    Finally, a global average pooling is applied followed by a fully connected layer.
    """

    def __init__(self, block, layers, num_classes=1):
        super(ResNet3D, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        """
        Create a layer consisting of multiple residual blocks.

        Args:
        - block: The class representing a single block (BasicBlock).
        - out_channels: The number of output channels in this layer.
        - blocks: The number of blocks to be stacked in this layer.
        - stride: The stride to be used in the convolutional layer of the blocks.
        """
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.fc(x)
        return x


def Resnet18_3d():
    """
    Constructs a ResNet-18 model for 3D inputs.
    """
    model = ResNet3D(BasicBlock, [2, 2, 2, 2])
    return model


if __name__ == "__main__":
    model = Resnet18_3d()
    image_batch = torch.tensor(np.random.rand(16, 1, 47, 47, 47), dtype=torch.float32)
    print(image_batch.shape)
    output = model(image_batch)
    print(output)