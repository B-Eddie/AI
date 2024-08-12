import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from PIL import Image

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.labels.iloc[idx, 0])
        img = Image.open(img_name).convert('RGB')

        if self.transform:
            img = self.transform(img)

        age = torch.tensor(self.labels.iloc[idx, 1], dtype=torch.float32)
        sex = torch.tensor(self.labels.iloc[idx, 2], dtype=torch.float32)
        emotion = torch.tensor(self.labels.iloc[idx, 3], dtype=torch.long)

        return img, age, sex, emotion
