import torch
from torch.utils.data import Dataset

class MusicTaggingDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        features, labels = self.data_list[index]
        return torch.tensor(features), torch.tensor(labels)
