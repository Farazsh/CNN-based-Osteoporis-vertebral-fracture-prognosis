from pathlib import Path
from typing import Dict, Final, Type, List

import torch
import pickle

from monai.networks.blocks import SEResNetBottleneck
from torch import nn
from torch.nn import Dropout
from monai.networks.nets import SENet, SEResNeXt50, SEResNext101, SEResNet50
from monai.networks.nets.senet import SEResNeXtBottleneck, SEResNet101, SEResNet152
from src.nets.util import PTH_DIR
from src.io_tools import CHECKPOINTS_DIR
import src.evaluate_checkpoint

import configs.config as c


class SerResNext50(SEResNeXt50):
    def __init__(self, config):
        super(SerResNext50, self).__init__(in_channels=1,
                         spatial_dims=3,
                         progress=False,
                         num_classes=config.net.num_classes,
                         dropout_prob=None)
        self._general_config: c.Config = config
        self._config: c.NetConfig = config.net
        self._num_classes = self._config.num_classes
        self._use_dropout = self._config.use_dropout
        self._experiment_name = self._general_config.model.experiment_name

        self.in_planes = 512 * SEResNeXtBottleneck.expansion
        del self.last_linear
        if self._general_config.simclr_training:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.Dropout(self._config.dropout),
                nn.ReLU(),
                # nn.Linear(self._config['num_hidden_units'], self._config['num_hidden_units']),
                # nn.Dropout(self._config['dropout']),
                # nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().features(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # when using binary classification the bcewithlogits loss should be used
        # this increases numercial stability
        # therefore the raw logits should be the outputs
        # this is also true for the cce loss

        return logits

    def freeze_all_conv_layers(self):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = False


class SerResNext101(SEResNext101):
    def __init__(self, config):
        super().__init__(in_channels=1,
                         spatial_dims=3,
                         progress=False,
                         num_classes=config.net.num_classes,
                         dropout_prob=None)
        self._general_config: c.Config = config
        self._config: c.NetConfig = config.net
        self._num_classes = self._config.num_classes
        self._layer_to_freeze = self._config.freeze_layers
        self._use_dropout = self._config.use_dropout
        self._experiment_name = self._general_config.model.experiment_name

        self.in_planes = 512 * SEResNetBottleneck.expansion
        del self.last_linear
        if self._general_config.simclr_training:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.Dropout(self._config.dropout),
                nn.ReLU(),
                # nn.Linear(self._config['num_hidden_units'], self._config['num_hidden_units']),
                # nn.Dropout(self._config['dropout']),
                # nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().features(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # when using binary classification the bcewithlogits loss should be used
        # this increases numercial stability
        # therefore the raw logits should be the outputs
        # this is also true for the cce loss

        return logits

    def freeze_all_conv_layers(self):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = False


class SeResNet50(SEResNet50):
    def __init__(self, config):
        super().__init__(in_channels=1,
                         spatial_dims=3,
                         progress=False,
                         num_classes=config.net.num_classes,
                         dropout_prob=None)

        self._general_config: c.Config = config
        self._config: c.NetConfig = config.net
        self._num_classes = self._config.num_classes
        self._use_dropout = self._config.use_dropout
        self._experiment_name = self._general_config.model.experiment_name

        self.in_planes = 512 * SEResNetBottleneck.expansion
        del self.last_linear

        if self._config.finetune:
            self._load_pretrained_weights()

        elif self._general_config.simclr_training:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )
        elif self._use_dropout:
            assert self._num_classes < 100, 'Do you really have that many classes or are you trying to use SimCLR? ' \
                                            'Should not use Dropout with SimCLR'
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.Dropout(self._config.dropout),
                nn.ReLU(),
                #nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
                #nn.Dropout(self._config.dropout),
                #nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().features(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # when using binary classification the bcewithlogits loss should be used
        # this increases numercial stability
        # therefore the raw logits should be the outputs
        # this is also true for the cce loss

        return logits

    def freeze_all_conv_layers(self):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def _load_pretrained_weights(self):
        ckpt_path, config_simclr = src.evaluate_checkpoint.get_ckpt_path_and_config('SeRes50_fullaugment_0.8_data')

        pretrain = torch.load(str(ckpt_path))
        state_dict = {k[1:].replace('net.', ''): v for k, v in pretrain['state_dict'].items()}
        num_hiden_units_simclr = config_simclr.net.num_hidden_units
        num_classes_simclr = config_simclr.net.num_classes
        # this is what the typical simmclr head looked like

        self.classifier = nn.Sequential(
            nn.Linear(self.in_planes, num_hiden_units_simclr),
            nn.ReLU(),
            nn.Linear(num_hiden_units_simclr, num_hiden_units_simclr),
            nn.ReLU(),
            nn.Linear(num_hiden_units_simclr, num_classes_simclr),
        )
        # load all weights into empty simclr net
        self.load_state_dict(state_dict, strict=True)

        del self.classifier

        self.classifier = nn.Sequential(
            nn.Linear(self.in_planes, self._config.num_hidden_units),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            # nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
            # nn.Dropout(self._config.dropout),
            # nn.ReLU(),
            nn.Linear(self._config.num_hidden_units, self._num_classes))

    def __str__(self):
        return 'SeResNet50'


class SeResNet101(SEResNet101):
    def __init__(self, config):
        super().__init__(in_channels=1,
                         spatial_dims=3,
                         progress=False,
                         num_classes=config.net.num_classes,
                         dropout_prob=None)

        self._general_config: c.Config = config
        self._config: c.NetConfig = config.net
        self._num_classes = self._config.num_classes
        self._layer_to_freeze = self._config.freeze_layers
        self._use_dropout = self._config.use_dropout
        self._experiment_name = self._general_config.model.experiment_name

        self.in_planes = 512 * SEResNetBottleneck.expansion
        del self.last_linear
        if self._general_config.simclr_training:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.Dropout(self._config.dropout),
                nn.ReLU(),
                # nn.Linear(self._config['num_hidden_units'], self._config['num_hidden_units']),
                # nn.Dropout(self._config['dropout']),
                # nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().features(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # when using binary classification the bcewithlogits loss should be used
        # this increases numercial stability
        # therefore the raw logits should be the outputs
        # this is also true for the cce loss

        return logits

    def freeze_all_conv_layers(self):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def __str__(self):
        return 'SeResNet101'

class SeResNet152(SEResNet152):
    def __init__(self, config):
        super().__init__(in_channels=1,
                         spatial_dims=3,
                         progress=False,
                         num_classes=config.net.num_classes,
                         dropout_prob=None)

        self._general_config: c.Config = config
        self._config: c.NetConfig = config.net
        self._num_classes = self._config.num_classes
        self._layer_to_freeze = self._config.freeze_layers
        self._use_dropout = self._config.use_dropout
        self._experiment_name = self._general_config.model.experiment_name

        self.in_planes = 512 * SEResNetBottleneck.expansion
        del self.last_linear
        if self._general_config.simclr_training:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
                nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(self.in_planes, self._config.num_hidden_units),
                nn.Dropout(self._config.dropout),
                nn.ReLU(),
                # nn.Linear(self._config['num_hidden_units'], self._config['num_hidden_units']),
                # nn.Dropout(self._config['dropout']),
                # nn.ReLU(),
                nn.Linear(self._config.num_hidden_units, self._num_classes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().features(x)
        x = self.adaptive_avg_pool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)

        # when using binary classification the bcewithlogits loss should be used
        # this increases numercial stability
        # therefore the raw logits should be the outputs
        # this is also true for the cce loss

        return logits

    def freeze_all_conv_layers(self):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def __str__(self):
        return 'SeResNet152'