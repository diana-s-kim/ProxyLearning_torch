"""
Proxy Learning pyTorch version--Proxy Learning of Visual Concepts of Fine Art Paintings from Styles through Language Models, Vol. 36 No. 4: AAAI-22
The MIT License (MIT)                                                                                                    
Originally created in 2023, for Python 3.x                                                                                    
Copyright (c) 2023 Diana S. Kim (diana.se.kim@gmail.com)
"""

from data.wiki import WikiArt
from neuralnets.proxynet import ProxyNet 
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Lambda
from torch import optim
import presets
import numpy as np

model_config={"vgg16":
             {"name":"vgg16","drop":1, "mlp":[[25088,2048],[2048,2048],[2048,1024],[1024,58]],"dropout":[0.5,None,None,None],"activation":['relu','relu','relu',None]}}
#add resnet later

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y,_) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def evaluate(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y,_) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            current = (batch + 1) * len(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            print(f"{current:>5d}/{size:>5d}")
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    

def collect(dataloader,args, model_visual_elements, model_styles, device):
    try:
        os.system("mkdir "+args.do_cllct[2]) #folder to save embedding"
    except:
        print("...folder exists already")
    visual_elements=np.empty((0,58))
    styles=np.empty((0,20))
    for batch, (X,_,_) in enumerate(dataloader):
        X = X.to(device)
        temp_visual_elements = model_visual_elements(X).detach().cpu().numpy()
        visual_elements=np.append(visual_elements,temp_visual_elements,axis=0)
        temp_styles=nn.functional.softmax(model_styles(X),dim=1).detach().cpu().numpy()
        styles=np.append(styles,temp_styles,axis=0)
        print("...collect", batch)        
    path=args.do_cllct[2]+"embedding_"+args.do_cllct[1]+".npz"
    np.savez(path,visual_elements=visual_elements,styles=styles)
    

def main(args):
    device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
    
    #data and pre-processing transform
    transform_train=presets.ClassificationPresetTrain(crop_size=args.crop_size)
    transform_val=presets.ClassificationPresetEval(crop_size=args.crop_size)

    wikiart_train=WikiArt(data_csv=args.csv_dir+"train.csv",img_dir=args.img_dir,transform=transform_train,target_transform=None)
    wikiart_val=WikiArt(data_csv=args.csv_dir+"val.csv",img_dir=args.img_dir,transform=transform_val,target_transform=None)
    train_dataloader = DataLoader(wikiart_train, batch_size=args.num_batches, shuffle=True)
    val_dataloader = DataLoader(wikiart_val, batch_size=args.num_batches, shuffle=False)
    
    #model
    model=ProxyNet(name=model_config[args.backbone_net]["name"],drop=model_config[args.backbone_net]["drop"],mlp=model_config[args.backbone_net]["mlp"],dropout=model_config[args.backbone_net]["dropout"],activations=model_config[args.backbone_net]["activation"]).to(device)
    for param in model.net.parameters():
        param.requires_grad = False
    

    #optimization
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)
    
    if args.do_eval:
        checkpoint=torch.load(args.do_eval,map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        evaluate(val_dataloader, model, criterion, device)
        return 

    if args.do_cllct: #cllct last hidden embedding and style representation
        checkpoint=torch.load(args.do_cllct[0],map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"],strict=True)
        model_visual_elements=model.visual_element
        model_styles=model
        if args.do_cllct[1]=="train":
           collect(train_dataloader, args, model_visual_elements, model_styles, device)
        else:#val
           collect(val_dataloader, args, model_visual_elements, model_styles, device)
        return

    #resume-part#
    if args.resume:
        checkpoint=torch.load(args.resume,map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
    
    print("start training....")
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(train_dataloader, model, criterion, optimizer, device)
        scheduler.step()
        if epoch%20==0:
#            torch.save(model.state_dict(),args.save_model_dir+"proxy_"+str(epoch)+".pt")
             torch.save({'epoch':epoch,'model_state_dict':model.state_dict(),'optimizer':optimizer.state_dict(),'lr_scheduler':scheduler.state_dict()},args.save_model_dir+"proxy_"+str(epoch)+".pt")
        evaluate(val_dataloader,model, criterion, device=device)
    return
        

def get_args_parser():
    import argparse
    parser = argparse.ArgumentParser(description="train,eval,cllct-embedding proxylearning")

    #data#
    parser.add_argument("--img_dir",default="/ibex/scratch/kimds/Research/P2/data/wikiart_resize/",type=str)
    parser.add_argument("--csv_dir",default="./data/",type=str) 
    
    #model#
    parser.add_argument("--backbone_net",default="vgg16", type=str)
    parser.add_argument("--crop_size",default=224,type=int)
    
    #train#
    parser.add_argument("--learning_rate",default=1.e-4,type=float)
    parser.add_argument("--epochs",default=100,type=int)
    parser.add_argument("--num_batches",default=32,type=int)
    parser.add_argument("--start_epoch",default=0,type=int)

    #options
    parser.add_argument("--save_model_dir",default="./model/",type=str)
    parser.add_argument("--do_cllct",default=None,nargs="*")#collect-embedding ["model.pt","val","./cllct_embedding/"]
    parser.add_argument("--resume",default=None,type=str,help="model dir to resume training")#resume-training
    parser.add_argument("--do_eval",default=None,type=str,help="model dir to eval")#just-evaluation
    return parser

if __name__== "__main__":
    args = get_args_parser().parse_args()
    main(args)
