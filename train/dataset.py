import torch

import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class HDF5Dataset(Dataset):
    def __init__(self, feature_h5_path, label_h5_path, feature_type):
        self.feature_type = feature_type
        self.feature_h5 = h5py.File(feature_h5_path, "r")
        self.label_h5 = h5py.File(label_h5_path, "r")
        self.total_samples = self.feature_h5[feature_type].shape[0]
    
    def __getitem__(self, idx):
        features = np.array(self.feature_h5[self.feature_type][idx])
        label = np.array(self.label_h5["label"][idx])
        return features, label
    
    def __len__(self):
        return self.total_samples

class HDF5DataLoader(DataLoader):
    def __init__(self, train_dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = train_dataset
        super().__init__(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)