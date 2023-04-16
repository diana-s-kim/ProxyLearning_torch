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
    def __init__(self,name='vgg16',drop=1):
        super().__init__()
        backbones={'vgg16_bn':models.vgg16_bn,
                   'vgg16':models.vgg16}
        self.name=name
#        self.backbone=backbones[self.name](weights="IMAGENET1K_V1")
        self.backbone=backbones[self.name]()
        self.backbone.load_state_dict(torch.load("./tensorflow_torch/tensorflow_vgg16.pt"),strict=True)#tensorflow model migrating
        self.basenet=nn.Sequential(*list(self.backbone.children())[:-drop])
        print(list(self.basenet.children()))
    def forward(self,x):
        x=self.basenet(x)
        return x #keep batch dim


#fc layer
class MLP(nn.Module):
    def __init__(self,layers,dropout,activations):
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
    def __init__(self,g_path="/ibex/scratch/kimds/Research/P1/ProxyLearning/ProxyLearning/GT_matrix/G.csv",scale_factor=1.0):
        super().__init__()
        self.test=nn.Linear(58,20)
        self.pre_computedG=pd.read_csv(g_path)[style_attribute.ATTRIBUTES].to_numpy().transpose().astype(float)/scale_factor
#        print(self.pre_computedG)
        self.u,self.s,_= np.linalg.svd(self.pre_computedG)
#        print(self.u,self.s)
        self.svd=np.matmul(inv(np.diag(self.s)),self.u[:,:self.s.shape[0]].transpose())
#        self.svd_1=np.matmul(self.u[:,:self.s.shape[0]],inv(np.diag(self.s)))
#        self.svd_2=np.matmul(inv(np.diag(self.s)),self.u[:,:self.s.shape[0]]).transpose())
#        self.svd_adjustment=np.matmul(self.u[:,:self.s.shape[0]],np.matmul(inv(np.matmul(np.diag(self.s),np.diag(self.s))),self.u[:,:self.s.shape[0]].transpose()))
#        print(self.s.shape[0])
        self.transformedG=np.matmul(self.svd,self.pre_computedG).transpose()
#        print(np.matmul(self.transformedG.transpose(),self.transformedG))
        self.svd_visual_elements=nn.Parameter(torch.from_numpy(self.svd).to("cuda:0"),requires_grad=False).type(torch.float)
        self.svd_G=nn.Parameter(torch.from_numpy(self.transformedG).to("cuda:0"),requires_grad=False).type(torch.float)
        #self.Gmatrix=nn.Parameter(torch.from_numpy(self.pre_computedG).to("cuda:0"),requires_grad=False).type(torch.float)

    def forward(self,x):
        #x=torch.mm(x,self.svd_visual_elements)
        #x=torch.mm(x,self.Gmatrix)
        x=nn.functional.linear(x,self.svd_visual_elements)
        x=nn.functional.linear(x,self.svd_G)
        return x

