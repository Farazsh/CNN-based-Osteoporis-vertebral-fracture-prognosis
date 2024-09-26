import torch
import torch.nn as nn


class Resnet50_2d(nn.Module):

    def __init__(self, dropout1=0.5, dropout2=0.5, num_classes=1):
        super(Resnet50_2d, self).__init__()
        self.dropout1 = dropout1
        self.dropout2 = dropout2

        # Load the ResNet50 backbone
        self.backbone = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)

        # Define the classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout1),
            nn.ReLU(),
            nn.Linear(1000, 256),  # Assuming the output features of ResNet50 is 1000
            nn.Dropout(self.dropout2),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Define the forward pass
        features = self.backbone(x)
        output = self.classifier(features)
        return output
