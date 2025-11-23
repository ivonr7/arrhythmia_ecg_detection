import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pytorch_datasets.annot_dataset import Annot_Dataset
import numpy as np
import os
class ArrCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(12, 32, 5, padding=2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(64 * 586, num_classes)
    def forward(self,x):
        x= x.permute(0,2,1)
        x=self.relu1(self.conv1(x))
        x=self.relu2(self.conv2(x))
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x
def datasetloader():
    ds = Annot_Dataset(
        extracted_folder="data/combined",
        dataset_folder="data",
        window_duration=0.3,
        classification_task="binary",
    )

    print(f"Loaded dataset with {len(ds)} samples and {ds.nclasses} classes.")
    x0, y0 = ds[0]
    print(f"Window shape:{x0.shape}")
    print(f"label vector shape: {y0.shape}")
    return ds

if __name__ == "__main__":
    datasetfull = datasetloader()
    num_classes= datasetfull.nclasses
    model = ArrCNN(num_classes=num_classes)
    x0, y0 = datasetfull[0]
    x0 = x0.unsqueeze(0)
    x0 = x0.float()
    out=model(x0)
    print("Model output shape:", out.shape)