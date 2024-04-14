import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,Subset
import os
from utils import load_gif, target_to_subjects_and_objects, get_gif_len
from collections import defaultdict
import json
import h5py
from preprocessing.preprocess import ACTIONS, SUBJECTS
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
        with h5py.File(data_file, "r") as h5f:
            gif_group = h5f["gif_data"]
            target_group = h5f["targets"]

            d = defaultdict(list)

            # Iterate over all datasets in the 'gif_data' group
            for dataset_name in gif_group:
                # Access the dataset
                data = np.array(gif_group[dataset_name]).tobytes()
                target = target_group[dataset_name][()].decode("utf-8")

                # Store data in dictionary using dataset name as key
                d[dataset_name].insert(0, data)
                d[dataset_name].append(target)

        self.data = list(d.values())
        self.all_subjects = SUBJECTS
        self.all_actions = ACTIONS
        self.transform = transform

    def create_binary_vector(self, subjects, actions):
        subject_vector = [1 if subject in subjects else 0 for subject in self.all_subjects]
        action_vector = [1 if action in actions else 0 for action in self.all_actions]

        # Combine the vectors to create a single target vector
        return subject_vector + action_vector

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gif_data, target = self.data[idx]
        gif_tensor, attention_mask = load_gif(gif_data)

        # Tuple of three sets of strings
        (subjects, actions) = target_to_subjects_and_objects(target)

        # n hot encoding, shape (num_subjects + num_actions = [16]
        target_vector = self.create_binary_vector(subjects, actions)
        assert len(target_vector) == 17

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

if __name__ == '__main__':
    dataset = GIFDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == 3:
            break
