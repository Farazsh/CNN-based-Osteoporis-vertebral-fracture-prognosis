import numpy as np
import torch
import torchvision
from torch._refs import nn

from merlin_old.models.i3res import I3ResNet
from copy import deepcopy
# from lightning_fabric import Fabric
# from pytorch_lightning.strategies import FSDPStrategy
# import pytorch_lightning
# strategy = FSDPStrategy()
# fabric = Fabric(accelerator="cuda", devices=1, precision='bf16-mixed')
# fabric.launch()

class I3ResnetCustom(torch.nn.Module):

    def __init__(self, conv=True, resnet_pt=True):
        super().__init__()
        resnet = torchvision.models.resnet152(weights=None)
        if resnet_pt:
            resnet.load_state_dict(torch.load(r"merlin_old/resnet152_weights.pth"))
        if conv:
            self.i3_resnet = I3ResNet(deepcopy(resnet), class_nb=1, conv_class=True)
            self.i3_resnet.contrastive_head = torch.nn.Conv3d(2048, 1, kernel_size=(1, 1, 1), bias=True)
        else:
            print("Not Implemented")

    def forward(self, image):
        image_out, ehr_out = self.i3_resnet(image)
        return image_out


def test_network():
    image_batch = torch.tensor(np.random.rand(16, 1, 47, 47, 47), dtype=torch.float32).to('cuda')
    label = torch.randint(0, 1, (16,))

    # with fabric.init_module():
    # resnet = torchvision.models.resnet152(weights=None)
    #     model = torchvision.models.resnet152(weights=None)
    model = I3ResnetCustom().to('cuda')
    # model.load_state_dict(torch.load(r"merlin/resnet152_weights.pth"))
    image_out = model(image_batch)
    print(model)
    print("Output Shape: ", image_out.shape)
    print("Labels Shape: ", label.shape)

# test_network()
#
# print(pytorch_lightning.__version__)