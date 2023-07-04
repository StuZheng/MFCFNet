import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

import pandas as pd
import numpy as np
from PIL import Image 



data_transform = {

    "train": transforms.Compose( [lambda x: Image.open(x).convert('RGB'),
                                                 
                                                 transforms.Resize((128, 128)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.044171255, 0.044171255, 0.044171255), (0.048645314, 0.048645314, 0.048645314)) # OpenSAR


                                                 ]),

    "test": transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                               
                                                 transforms.Resize((128, 128)),
                                                 transforms.ToTensor(),
                                                  transforms.Normalize((0.044171255, 0.044171255, 0.044171255), (0.048645314, 0.048645314, 0.048645314)) # OpenSAR

                                                 ])

}




class MyDataSet(Dataset):
    def __init__(self,mode,data,labels):
        self.data = data
        self.labels = labels

        self.transforms = data_transform[mode] 


    def __getitem__(self, index):
        

        data = self.transforms(self.data[index])
        label = torch.tensor(int(self.labels[index]))
        return data,label

    def __len__(self):
         return len(self.labels)

