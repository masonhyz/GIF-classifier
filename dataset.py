
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Subset
import os
from utils_new import load_gif_compressed
import json
import h5py
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class GIFDataset(Dataset):
    """GIF dataset"""

    def __init__(
        self,
        data_file=os.path.join(CURR_DIR, "preprocessing/processed_data.hdf5"),
        transform=None,
    ):
        # with h5py.File(data_file, "r") as h5f:
        #     gif_group = h5f["gif_data"]
        #     target_group = h5f["targets"]

        #     d = defaultdict(list)

        #     # Iterate over all datasets in the 'gif_data' group
        #     for dataset_name in gif_group:
        #         # Access the dataset
        #         data = np.array(gif_group[dataset_name]).tobytes()
        #         target = target_group[dataset_name][()].decode("utf-8")

        #         # Store data in dictionary using dataset name as key
        #         d[dataset_name].insert(0, data)
        #         d[dataset_name].append(target)
        # self.data = list(d.values())

        self.data_file = data_file
        self.actions = ["dancing", "playing", "walking", "looking", "talking", "singing", "doing", "kissing", "holding", "running"]
        self.transform = transform

    def __len__(self):
        with h5py.File(self.data_file, "r") as h5f:
            return len(h5f["gif_data"])

    def __getitem__(self, idx):
        if isinstance(idx, list):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            samples = [self.get_single_item(i) for i in idx]
            return samples
        else:
            return self.get_single_item(idx)

    def get_single_item(self, idx):
        with h5py.File(self.data_file, "r") as h5f:
            if str(idx) not in h5f["gif_data"]:
                idx = 0
            gif_data = np.array(h5f["gif_data"][str(idx)]).tobytes()
            target = h5f["targets"][str(idx)][()].decode("utf-8")

        # (num_frames, 3, h, w) and (num_frames, h, w)
        gif_tensor, attention_mask = load_gif_compressed(gif_data)

        # Tuple of three sets of strings
        action_in_target = [action for action in self.actions if action in target]

        # 1 hot encoding, shape [10]
    
        target_vector = [1 if action in target.split() else 0 for action in self.actions]
        found = False  
        target_vector = []
        for action in self.actions:
            if action in target.split() and not found:
                target_vector.append(1)
                found = True 
            else:
                target_vector.append(0)
        assert len(target_vector) == 10 and sum(target_vector) == 1

        sample = {
            "gif": gif_tensor,
            "attention_mask": attention_mask,
            "target": target_vector,
        }
        return sample


def train_val_sklearn_split(dataset: GIFDataset, test_size=0.2):
    indices = list(range(len(dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=test_size, random_state=42)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    return train_set, val_set


if __name__ == "__main__":
    dataset = GIFDataset()
    train_split, val_split = train_val_sklearn_split(dataset, test_size=0.2)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == 3:
            break