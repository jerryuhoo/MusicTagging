import torch
from torch.utils.data import Dataset
import numpy as np
import os

class MusicTaggingDataset(Dataset):
    def __init__(self, folder_path, feats_type):
        self.folder_path = folder_path
        self.feats_type = feats_type
        self.label_list = []
        self.feats_list = []
        for filename in os.listdir(os.path.join(self.folder_path, "label")):
            if filename.endswith(".npy"):
                 self.label_list.append(os.path.join(self.folder_path, "label", filename))
        for filename in os.listdir(os.path.join(self.folder_path, self.feats_type)):
            if filename.endswith(".npy"):
                 self.feats_list.append(os.path.join(self.folder_path, self.feats_type, filename))

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        # Load the npy file at the given index
        labels = np.load(self.label_list[index])
        features = np.load(self.feats_list[index])
        return torch.tensor(features), torch.tensor(labels)
