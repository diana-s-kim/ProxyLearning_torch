import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image #compare
from PIL import Image

STYLES=['early-renaissance','high-renaissance','mannerism-late-renaissance','northern-renaissance','baroque','rococo','romanticism','impressionism','post-impressionism','realism','art-nouveau-modern','cubism','expressionism','fauvism','abstract-expressionism','color-field-painting','minimalism','na-ve-art-primitivism','ukiyo-e','pop-art']

class WikiArt(Dataset):
    def __init__(self,annotations_file,img_dir,transform=None, target_transform=None,split='train'):
        print("here")
        self.img_labels = pd.read_csv(annotations_file,header=None)
        self.img_labels = self.img_labels[self.img_labels.iloc[:,-1]==split]#train or test
        self.img_labels.iloc[:,1]=self.img_labels.iloc[:,1].apply(lambda x: STYLES.index(x))
#        print(pd.unique(self.img_labels.iloc[:,1]))
        self.img_dir = img_dir
#        print(self.img_dir)
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx,0])
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            try:
                image = self.transform(image)
            except:
                print(img_path)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path
