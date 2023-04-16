import torch
from torch import nn
from . import neuralnet

class ProxyNet(nn.Module):
    def __init__(self,name='vgg16',drop=1,mlp=[[25088,2048],[2048,2048],[2048,1024],[1024,58]],dropout=[0.5,None,None,None],activations=['relu','relu','relu','relu',None]):
        super().__init__()
        self.net=neuralnet.BaseNet(name=name,drop=1)
        self.fc=neuralnet.MLP(mlp,dropout,activations)
        self.visual_element=nn.Sequential(self.net, self.fc)#neuralnet.BaseNet(name=name,drop=1),neuralnet.MLP(mlp,dropout,activations))
        self.G=neuralnet.SVD_G(g_path="/ibex/scratch/kimds/Research/P1/ProxyLearning/ProxyLearning/GT_matrix/G.csv")
    def forward(self,x):
        x=self.visual_element(x)
        logits=self.G(x)
#        print(self.net(x).detach().numpy().shape)
        return logits
        
