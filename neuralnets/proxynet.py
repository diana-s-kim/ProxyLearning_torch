import torch
from torch import nn
from . import neuralnet

class ProxyNet(nn.Module):
    def __init__(self,name=None,drop=None,mlp=None,dropout=None,activations=None):
        super().__init__()
        self.net=neuralnet.BaseNet(name=name,drop=drop)
        self.fc=neuralnet.MLP(mlp,dropout,activations)
        self.visual_element=nn.Sequential(self.net, self.fc)#neuralnet.BaseNet(name=name,drop=1),neuralnet.MLP(mlp,dropout,activations))
        self.G=neuralnet.SVD_G()

    def forward(self,x):
        x=self.visual_element(x)
        logits=self.G(x)
        return logits
        
