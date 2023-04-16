import torch
import numpy as np
from torch import nn

class mlp(nn.Module):
    def __init__(self,layers,dropout,activations)
        fc=[]
        for layer,drop,relu zip(layers,dropout,activations):
            fc.append(nn.Linear(layer[0],layer[1])
            if drop>0:
                fc.append(torch.nn.Dropout(p=0.5, inplace=False))
            if relu is not None:
                fc.append(nn.ReLU(inplace=True))
        self.visual_element=nn.Sequential(*fc)
    def forward(self,x):
        out=self.visual_elements(self.x)
        return out

        
        
