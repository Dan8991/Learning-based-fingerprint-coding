from torch.utils.data import Dataset
import os
import PIL
from glob import glob
from itertools import product
import torch

class Fingerprint(Dataset):
    def __init__(self, root, transform=None, mode = "test", extension="bmp"):
        self.root = root
        self.transform = transform

        if mode == "train":
            self.images = glob(os.path.join(root,"[0-2][0-9][0-9]"))
        elif mode == "val":
            self.images = glob(os.path.join(root,"3[0-9][0-9]"))
        elif mode == "test":
            self.images = glob(os.path.join(root,"4[0-9][0-9]"))
        self.extension = extension
        self.mode = mode
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if self.mode == "test":
            idx += 400
        elif self.mode == "val":
            idx += 300
        imgs = []
        paths = []
        for i, hand in enumerate(["L", "R"]):
            imgs.append([])
            for finger in range(4):
                imgs[i].append([])
                for impression in range(5):
                    path = os.path.join(self.root, f"{idx:03}", hand, f"{idx:03}_{hand}{finger}_{impression}.{self.extension}")
                    paths.append(path)
                    img = PIL.Image.open(path)
                    if self.transform:
                        img = self.transform(img)
                    imgs[i][finger].append(img)
        imgs = torch.stack([torch.stack([torch.stack(finger) for finger in hand]) for hand in imgs]).float() 
        return imgs, paths

class FingerprintTraining(Dataset):
    def __init__(self, root, transform=None, split="train"):
        self.root = root
        self.transform = transform
        self.images = glob(os.path.join(root,"[0-9][0-9][0-9]"))
        if split == "train":
            idxs = range(0, 300)
        elif split == "val":
            idxs = range(300, 320)
        elif split == "test":
            idxs = range(400, 500)
        combs = product(idxs, ["L", "R"], range(4), range(5))
        self.images = [os.path.join(self.root, f"{idx:03}", hand, f"{idx:03}_{hand}{finger}_{impression}.bmp") for idx, hand, finger, impression in combs]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):

        path = self.images[idx]
        img = PIL.Image.open(path)
        if self.transform:
            img = self.transform(img)
        return img

class FinetuningDataset(Dataset):
    def __init__(self, root, transform=None):
        
        self.root = root
        self.transform = transform
        self.images = []
        
        for dataset in ["FVC2000", "FVC2002"]:
            for db in ["DB1", "DB2", "DB3", "DB4"]:
                self.images += glob(os.path.join(root, dataset, db, "*.tif"))
        
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = PIL.Image.open(self.images[idx])
        if self.transform:
            img = self.transform(img)
        if img.shape[0] == 1:
            img = torch.stack([img.squeeze(0)]*3)
        return img 