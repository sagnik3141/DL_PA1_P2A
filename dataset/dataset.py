import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """
    This defines the dataset class.

    data_path: Path to text file of input data.
    labels_path: Path to text file of labels.
    """
    def __init__(self, args):

        self.data_arr = np.loadtxt(args.data_path)
        self.labels_arr = np.loadtxt(args.labels_path)

        self.indices = self.labels_arr<=4 # Classes Assigned to our Group
        self.data_arr = self.data_arr[self.indices]
        self.labels_arr = self.labels_arr[self.indices]

        assert self.data_arr.shape[0] == self.labels_arr.shape[0], "Data and Labels mismatch."

    def __len__(self):

        return self.data_arr.shape[0]

    def __getitem__(self, index):

        data = self.data_arr[index]
        label = self.labels_arr[index]

        data = torch.from_numpy(data)
        label = torch.from_numpy(np.asarray(label))

        return (data, label)