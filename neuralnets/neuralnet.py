import torch
from torch import nn
from torchvision import models
import sys
sys.path.append("/ibex/scratch/kimds/Research/P1/ProxyLearning/ProxyLearning")
import style_attribute
import pandas as pd
import numpy as np
from numpy.linalg import inv
#back bone net
class BaseNet(nn.Module):
    def __init__(self,name=None,drop=None):
        super().__init__()
        backbones={'vgg16_bn':models.vgg16_bn,
                   'vgg16':models.vgg16,
                   'resnet50':models.resnet50,
                   'resnet101':models.resnet101}
        self.name=name
        self.backbone=backbones[self.name]()
        self.backbone.load_state_dict(torch.load("./tensorflow_torch/tensorflow_"+name+".pt"),strict=True)#tensorflow model migrating
        self.basenet=nn.Sequential(*list(self.backbone.children())[:-drop])
        print(list(self.basenet.children()))
    def forward(self,x):
        x=self.basenet(x)
        return x #keep batch dim


#fc layer
class MLP(nn.Module):
    def __init__(self,layers=None,dropout=None,activations=None):
        super().__init__()
        fc=[nn.Flatten()]
        for layer,drop,relu in zip(layers,dropout,activations):
            linear_layer=nn.Linear(layer[0],layer[1])
            fc.append(linear_layer)
            if drop is not None:
                dropout_layer=nn.Dropout(p=drop)
                fc.append(dropout_layer)
            if relu is not None:
                relu_layer=nn.ReLU(inplace=True)
                fc.append(relu_layer)
        self.elements_layer=nn.Sequential(*fc)
    def forward(self,x):
        return self.elements_layer(x)
                    

    
class SVD_G(nn.Module):
    def __init__(self,g_path="./GT_matrix/G.csv",scale_factor=1.0):
        super().__init__()
        self.test=nn.Linear(58,20)
        self.pre_computedG=pd.read_csv(g_path)[style_attribute.ATTRIBUTES].to_numpy().transpose().astype(float)/scale_factor
        self.u,self.s,_= np.linalg.svd(self.pre_computedG)
        self.svd=np.matmul(inv(np.diag(self.s)),self.u[:,:self.s.shape[0]].transpose())
        self.transformedG=np.matmul(self.svd,self.pre_computedG).transpose()
        self.svd_visual_elements=nn.Parameter(torch.from_numpy(self.svd).to("cuda:0"),requires_grad=False).type(torch.float)
        self.svd_G=nn.Parameter(torch.from_numpy(self.transformedG).to("cuda:0"),requires_grad=False).type(torch.float)
    def forward(self,x):
        x=nn.functional.linear(x,self.svd_visual_elements)
        x=nn.functional.linear(x,self.svd_G)
        return x

