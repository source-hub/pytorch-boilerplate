import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,datasets

class ImageDataset(Dataset):
        def __init__(self,root_path,transforms=None):
            self.root=root_path
            if transforms:
                self.transforms=transforms
            self.data=os.listdir(root_path)
        def __len__(self):
            return len(self.data)
        def __getitem__(self,idx):
            image_path=os.path.join(self.root,self.data[idx])
            print(image_path)
            img=io.imread(image_path)
            sample={"image":img,"path":image_path}
            return sample

class NewDataset(datasets.ImageFolder):
        def __init__(self,root=None,transform=None,**kwargs):
            if transform==None:
                data_transforms=transforms.Compose([
		        transforms.RandomResizedCrop(200),
        	    transforms.RandomHorizontalFlip(),
        	    transforms.ToTensor(),
        	    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    		    ])
                transform=data_transforms
            super(NewDataset,self).__init__(root,transform,**kwargs)