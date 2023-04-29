"""
Proxy Learning pyTorch version--Proxy Learning of Visual Concepts of Fine Art Paintings from Styles through Language Models, Vol. 36 No. 4: AAAI-22
The MIT License (MIT)                                                                                                    
Originally created in 2023, for Python 3.x                                                                                    
Copyright (c) 2023 Diana S. Kim (diana.se.kim@gmail.com)
"""

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
        
