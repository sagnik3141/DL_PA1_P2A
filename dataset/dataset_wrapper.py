import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from dataset.dataset import ImageDataset

def create_dataloaders(args):
    """
    Returns train, val and test dataloaders for given splits.
    """
    dataset = ImageDataset(args)
    dataset_len = len(dataset)
    indices = list(range(dataset_len))

    val_split = int(np.floor(args.val_split * dataset_len))
    test_split = int(np.floor(args.test_split * dataset_len))

    if args.shuffle_data:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)

    val_indices, test_indices, train_indices = indices[:val_split], \
        indices[val_split:test_split], \
            indices[test_split:]

    # Create Samplers
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # Create Dataloaders
    train_loader = DataLoader(dataset, batch_size = args.batch_size, sampler = train_sampler)
    val_loader = DataLoader(dataset, batch_size = args.batch_size, sampler = val_sampler)
    test_loader = DataLoader(dataset, batch_size = args.batch_size, sampler = test_sampler)

    return train_loader, val_loader, test_loader