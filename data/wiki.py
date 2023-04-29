import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image #compare
from PIL import Image
import style_attribute

class WikiArt(Dataset):
    def __init__(self,data_csv,img_dir,transform=None, target_transform=None):
        self.img_labels = pd.read_csv(data_csv,header=0)
        self.img_labels.iloc[:,1]=self.img_labels.iloc[:,1].apply(lambda x: style_attribute.STYLE_MERGE[x])
        self.img_labels.iloc[:,1]=self.img_labels.iloc[:,1].apply(lambda x: style_attribute.STYLES.index(x))
        self.img_dir = img_dir
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
