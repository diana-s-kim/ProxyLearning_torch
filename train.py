from data.wiki import WikiArt
from neuralnets.proxynet import ProxyNet 
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Lambda
from torch import optim
import presets

##parameters
num_style=20
crop_size=224
learning_rate=0.0001
num_epochs=20
num_batch=32#64#128#32
wiki_csv="./data/wiki.csv"
img_dir="/ibex/scratch/kimds/Research/P2/data/wikiart_kaust/" #***#


##model
netname='vgg16'
drop=1
mlp=[[25088,2048],[2048,2048],[2048,1024],[1024,58]]
dropout=[0.5,None,None,None]
activations=['relu','relu','relu',None]


##device 
device = torch.device("cuda:" + str(0) if torch.cuda.is_available() else "cpu")
model=ProxyNet(name=netname,drop=drop,mlp=mlp,dropout=dropout,activations=activations).to(device)

for param in model.net.parameters():
    param.requires_grad = False
print("model complete")
##loss softmax
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.96)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y,_) in enumerate(dataloader):
#    for batch in range(2):
#        X,y,_=next(iter(train_dataloader))
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #if batch % 100 == 0:
        loss, current = loss.item(), (batch + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
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

##data
transform_train=presets.ClassificationPresetTrain(crop_size=crop_size)
transform_test=presets.ClassificationPresetEval(crop_size=crop_size)
target_transform = Lambda(lambda y: torch.zeros(num_style, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
wikiart_train=WikiArt(annotations_file=wiki_csv,img_dir=img_dir,transform=transform_train,target_transform=None,split='train')
wikiart_test=WikiArt(annotations_file=wiki_csv,img_dir=img_dir,transform=transform_test,target_transform=None,split='test')
train_dataloader = DataLoader(wikiart_train, batch_size=num_batch, shuffle=True)
test_dataloader = DataLoader(wikiart_test, batch_size=num_batch, shuffle=False)


##train

for t in range(num_epochs):
    print(f"Epoch {t+1}\n------------------------------- \n")
    train(train_dataloader, model,criterion,optimizer)
    torch.save(model.state_dict(),"./model/proxy_"+str(t)+".pt")
test(test_dataloader, model,criterion)




