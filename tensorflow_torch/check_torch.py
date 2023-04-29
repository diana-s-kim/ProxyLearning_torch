
from torchvision.models import vgg16
import torch

#load_state_dict(wieghts.get_state_dict)
test_dict={"a":[1,2,3],"b":[2,3]}
for item in test_dict:
    print(item)

model=vgg16("IMAGENET1K_V1")
param=model.state_dict()
#for k,v in param.items():
#    print(k,v.shape)

## transplant tensorflow model ##

import numpy as np
tensorflow_wieghts=np.load("tensorflow_weights.npz",allow_pickle=True)["weights"]

for k,w in zip(param.keys(),tensorflow_wieghts):
    print(k)
    print(param[k].shape,w.shape)
    param[k]=torch.from_numpy(w)

print(param)    
torch.save(param,"tensorflow_vgg16.pt")
torch.load("tensorflow_vgg16.pt")
