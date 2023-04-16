from data.wiki import WikiArt
from neuralnets.proxynet import ProxyNet 
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Lambda
from torch import optim
import presets

##parameters (no use)
num_style=20
crop_size=224
learning_rate=0.0001
num_epochs=20
num_batch=32
wiki_csv="./data/wiki.csv"
img_dir="/ibex/scratch/kimds/Research/P2/data/wikiart/"


netname='vgg16'
drop=1
mlp=[[25088,2048],[2048,2048],[2048,1024],[1024,58]]
dropout=[0.5,None,None,None]
activations=['relu','relu','relu',None]
model_ckpt="./model/proxy_0.pt"

##device 
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
model=ProxyNet(name=netname,drop=drop,mlp=mlp,dropout=dropout,activations=activations).to(device)
model.load_state_dict(torch.load(model_ckpt),strict=True)
model=model.visual_element


def collect_embedding(dataloader):
    for batch, (X,_,fileinfo) in enumerate(dataloader):
        X = X.to(device)
        visual_elements = model(X)
        print(X)
        print(fileinfo)
        print(visual_elements)
	
##data
transform_cllct=presets.ClassificationPresetCllct(crop_size=crop_size)
wikiart_cllct=WikiArt(annotations_file=wiki_csv,img_dir="/ibex/scratch/kimds/Research/P2/data/wikiart/",transform=transform_cllct,target_transform=None,split='test')
cllct_dataloader = DataLoader(wikiart_cllct, batch_size=num_batch, shuffle=False)


##train

for t in range(num_epochs):
    print(f"Epoch {t+1}\n------------------------------- \n")
    collect_embedding(cllct_dataloader)
