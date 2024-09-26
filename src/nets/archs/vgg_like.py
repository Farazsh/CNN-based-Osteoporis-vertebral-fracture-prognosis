from typing import Final

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO

from torch.nn import functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy, HingeLoss

from src.experiment_config.experiment_config import ExperimentConfig

data_flag = 'pathmnist'
# data_flag = 'breastmnist'
download = True

info = INFO[data_flag]
task = info['task']

DataClass = getattr(medmnist, info['python_class'])


class MedNet(LightningModule):
    def __init__(self, config: ExperimentConfig):
        super().__init__()

        # Set experiment config and required hyperparameters
        self.config: Final[ExperimentConfig] = config
        self.learning_rate = self.config.learning_rate
        self.momentum = self.config.momentum

        # Define Data Transformations
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5]),
        ])

        # Define Metrics
        self.accuracy = Accuracy()

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.config.num_input_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.config.num_classes))

        self.save_hyperparameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_nb):
        inputs, targets = batch
        predictions = self(inputs).type(torch.FloatTensor)  # needs to be casted to a float tensor for cce
        targets = targets.squeeze().type(torch.LongTensor)  # dont really understand why but gives error otherwise
        train_loss = F.cross_entropy(predictions, targets)
        train_acc = self.accuracy(predictions, targets)

        self.log('Train Loss', train_loss, on_epoch=False, on_step=True)
        self.log('Train Acc', train_acc, on_epoch=False, on_step=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs).type(torch.FloatTensor)  # needs to be casted to a float tensor for cce
        targets = targets.squeeze().type(torch.LongTensor)
        validation_loss = F.cross_entropy(predictions, targets)
        validation_acc = self.accuracy(predictions, targets)

        self.log('Val Loss', validation_loss, on_epoch=True, on_step=False)
        self.log('Val Acc', validation_acc, on_epoch=True, on_step=False)
        return predictions

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        predictions = self(inputs).type(torch.FloatTensor)
        targets = targets.squeeze().type(torch.LongTensor)
        test_loss = F.cross_entropy(predictions, targets)
        test_acc = self.accuracy(predictions, targets)

        self.log('Test Loss', test_loss, on_epoch=True, on_step=False)
        self.log('Test Acc', test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        # return optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        return optim.Adam(self.parameters(), lr=self.config.learning_rate)

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_set = DataClass(split='train', transform=self.transform, download=True)
            self.val_set = DataClass(split='val', transform=self.transform, download=True)

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test_set = DataClass(split='test', transform=self.transform, download=True)

    def train_dataloader(self):
        """Persistent Workers in combination with pin memory speed up training by a factor of 3. Always use."""
        return data.DataLoader(dataset=self.train_set,
                               batch_size=self.config.batch_size,
                               shuffle=True,
                               num_workers=12,
                               pin_memory=True,
                               persistent_workers=True)

    def val_dataloader(self):
        return data.DataLoader(dataset=self.val_set,
                               batch_size=self.config.batch_size * 2,
                               num_workers=2,
                               pin_memory=True,
                               persistent_workers=True)

    def test_dataloader(self):
        return data.DataLoader(dataset=self.test_set,
                               batch_size=self.config.batch_size * 2,
                               num_workers=2,
                               pin_memory=True,
                               persistent_workers=True)