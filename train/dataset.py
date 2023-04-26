import torch

import h5py
from torch.utils.data import Dataset, DataLoader


class HDF5Dataset(Dataset):
    def __init__(self, feature_h5_path, label_h5_path, feature_type):
        self.feature_type = feature_type
        self.feature_h5 = h5py.File(feature_h5_path, "r")
        self.label_h5 = h5py.File(label_h5_path, "r")
        self.total_samples = self.feature_h5[feature_type].shape[0]

    def __getitem__(self, idx):
        features = torch.from_numpy(self.feature_h5[self.feature_type][idx])
        label = torch.from_numpy(self.label_h5["label"][idx])
        return features, label

    def __len__(self):
        return self.total_samples

    def __del__(self):
        if self.feature_h5 is not None:
            self.feature_h5.close()
        if self.label_h5 is not None:
            self.label_h5.close()


class HDF5DataLoader(DataLoader):
    def __init__(self, train_dataset, batch_size, shuffle=True, num_workers=0):
        self.dataset = train_dataset
        super().__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
