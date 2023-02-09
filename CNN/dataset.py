import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MusicTaggingDataset(Dataset):
    def __init__(self, label_list, feats_list):
        self.label_list = label_list
        self.feats_list = feats_list
        
    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        # Load the npy file at the given index
        labels = np.load(self.label_list[index])
        features = np.load(self.feats_list[index])
        return torch.tensor(features), torch.tensor(labels)
