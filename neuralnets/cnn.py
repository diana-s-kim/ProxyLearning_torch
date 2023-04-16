import torch
from torch import nn
from torchvision import models
class NerualNet:
    def __init__(self,name='vgg16',drop=1):
        backbones={'vgg16_bn':models.vgg16_bn,
                   'vgg16':models.vgg16}
        self.name=name
        self.basenet=nn.Sequential(*list(backbones[self.name]().children())[:-drop])


        
        


