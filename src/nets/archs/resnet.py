from pathlib import Path
from typing import Dict, Final, Type, List

import torch
import pickle
from torch import nn
from torch.nn import Dropout
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock, get_inplanes, ResNetBottleneck
from src.nets.util import PTH_DIR
from src.io_tools import CHECKPOINTS_DIR
import src.evaluate_checkpoint

import configs.config as c


class BaseResNet(ResNet):
    def __init__(self, config, depth: int):
        super().__init__(block=self._choose_res_block_from_depth(depth),
                         layers=self._choose_layers_on_depth(depth),
                         block_inplanes=get_inplanes(),
                         n_input_channels=1,
                         spatial_dims=3,
                         num_classes=config.net.num_classes,
                         feed_forward=False)  # num_classes => num output neurons?

        self._general_config: c.Config = config
        self._config: c.NetConfig = config.net
        self._depth = depth
        self._num_classes = self._config.num_classes
        self._use_dropout = self._config.use_dropout
        self._experiment_name = self._general_config.model.experiment_name

        if self._config.pretrained_med_net:
            self._pretrained_filename = f"resnet_{self._depth}_23dataset.pth"
            self._load_weights_med_net(self._pretrained_filename)
            #self.freeze_all_conv_layers()
            self._replace_fc_layer()
        elif self._config.finetune:
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

    def _choose_res_block_from_depth(self, depth):
        """
        Depending on the resnet type a different resnet base block is chosen.
        """
        if depth >= 50:
            block = ResNetBottleneck
        else:
            block = ResNetBlock
        return block

    def _choose_layers_on_depth(self, depth: int) -> List[int]:
        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        elif depth == 101:
            layers = [3, 4, 23, 3]
        elif depth == 152:
            layers = [3, 8, 36, 3]
        else:
            raise NotImplementedError(f"ResNet of depth '{depth} not implemented!")
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        x = self.classifier(x)

        # when using binary classification the bcewithlogits loss should be used
        # this increases numercial stability
        # therefore the raw logits should be the outputs
        # this is also true for the cce loss
        logits = x
        return logits

    def _load_weights(self):
        """Loads the weights form the ckpt file inside the current net"""
        # Check if the pretrained net and net in config are the same
        config_used = self._load_pretrain_model_config()
        assert config_used.net.net_class == self._config.net_class, \
            'Current net and net used for pre training are not the same'

        pretrain = torch.load(self._get_ckpt_path())  # load weights from checkpoint file
        # Change names of layers as they differ
        state_dict = {k[1:].replace('net.', ''): v for k, v in pretrain['state_dict'].items()}
        # add new simclr head
        self.classifier = nn.Sequential(
            nn.Linear(self.in_planes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 512)
        )
        # load all weights into empty simclr net
        self.load_state_dict(state_dict, strict=True)
        # delete every layer except the first
        del self.classifier
        # add new relu and classification head

    def _get_ckpt_path(self):
        """Iterates over all files inside the directory for the current experiment
        and returns the filepath to the top...ckpt file"""
        experiment_dir = Path.joinpath(CHECKPOINTS_DIR, self._experiment_name)
        for file_path in experiment_dir.iterdir():
            if file_path.is_file() and file_path.name.startswith('top') and file_path.name.endswith('.ckpt') and not 'fine_tune' in file_path.name:
                return file_path
        raise FileNotFoundError(f'No top ckpt file could be found in {experiment_dir}')

    def _load_pretrain_model_config(self):
        """Iterates over all files inside the directory for the current experiment
        and returns the config used as a dictionary"""
        experiment_dir = Path.joinpath(CHECKPOINTS_DIR, self._experiment_name)
        for file_path in experiment_dir.iterdir():
            if file_path.is_file() and file_path.name == 'config_fine_tune.pkl':
                with open(file_path, "rb") as f:
                    config_dict = pickle.load(f)
                return config_dict
        raise FileNotFoundError(f'No config file could be found in {experiment_dir}')

    def _load_weights_med_net(self, pth_filename: str):
        """
        From https://github.com/Borda/kaggle_vol-3D-classify/blob/037c1c3f3a2d601ed272cf7c2199a5d9fed0eb04/kaggle_brain3d/models.py#L16
        Args:
            pth_filename: The path to the pretrained weights file

        Returns:

        """
        full_path: Path = PTH_DIR.joinpath(pth_filename)
        if not full_path.is_file():
            raise FileNotFoundError(f"PTH-file does not exist in pretrain folder: '{full_path}'")
        #self.load_state_dict(torch.load(full_path, map_location='cuda:0'), strict=False)

        net_dict = self.state_dict()
        pretrain = torch.load(full_path)
        pretrain['state_dict'] = {k.replace('module.', ''): v for k, v in pretrain['state_dict'].items()}
        missing = tuple({k for k in net_dict.keys() if k not in pretrain['state_dict']})
        inside = tuple({k for k in pretrain['state_dict'] if k in net_dict.keys()})
        unused = tuple({k for k in pretrain['state_dict'] if k not in net_dict.keys()})
        assert len(inside) > len(missing)
        assert len(inside) > len(unused)

        pretrain['state_dict'] = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        # this throws an error when strict is set to false, as the downsample.0.bias for each of the for layer does not exist.
        # Otherwise, all weights are properly loaded
        self.load_state_dict(pretrain['state_dict'], strict=False)
        print("loaded weights")

    def _load_pretrained_weights(self):
        ckpt_path, config_simclr = src.evaluate_checkpoint.get_ckpt_path_and_config('SimCLR_blur_30_25_20_shift_04_noise003')

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

    def freeze_conv_layers(self):
        """
        Looks into all parameters of the network and finds the layers that match names inside the config.net.freeze_layers.
        When found, turns of gradient calculation, which essentially "freezes" that layer.
        """
        for name, child in self.named_children():
            if name in self._layer_to_freeze:
                print(name + ' is frozen')
                for param in child.parameters():
                    param.requires_grad = False
            else:
                print(name + ' is not frozen')
                for param in child.parameters():
                    param.requires_grad = True

    def freeze_all_conv_layers(self):
        for name, child in self.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def _replace_fc_layer(self):
        self.classifier = nn.Sequential(
            nn.Linear(self.in_planes, self._config.num_hidden_units),
            nn.Dropout(self._config.dropout),
            nn.ReLU(),
            # nn.Linear(self._config.num_hidden_units, self._config.num_hidden_units),
            # nn.Dropout(self._config.dropout),
            # nn.ReLU(),
            nn.Linear(self._config.num_hidden_units, self._num_classes)
        )
        for param in self.classifier.parameters():
            param.requires_grad = True


class ResNet18(BaseResNet):
    def __init__(self, config):
        super(ResNet18, self).__init__(config, 18)


class ResNet50(BaseResNet):
    def __init__(self, config):
        super(ResNet50, self).__init__(config, 50)

    def __str__(self):
        return 'ResNet50'


class ResNet101(BaseResNet):
    def __init__(self, config):
        super(ResNet101, self).__init__(config, 101)

    def __str__(self):
        return 'ResNet101'


class ResNet152(BaseResNet):
    def __init__(self, config):
        super(ResNet152, self).__init__(config, 152)

    def __str__(self):
        return 'ResNet152'
