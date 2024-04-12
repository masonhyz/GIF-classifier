import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from utils import load_gif, target_to_subjects_and_objects, get_gif_len
from collections import defaultdict
import json

import warnings
warnings.filterwarnings("ignore")

CURR_DIR = os.path.dirname(os.path.realpath(__file__))


class GIFDataset(Dataset):
    """GIF dataset"""

    def __init__(
        self,
        data_file=os.path.join(CURR_DIR, "data/tgif-v1.0.tsv"),
        subj_obj_dir=os.path.join(CURR_DIR, "subj_obj_data/"),
        transform=None,
    ):
        self.gif_df = pd.read_csv(data_file, sep="\t")
        self.all_subjects = json.load(open(subj_obj_dir + "subject_frequency.json")).keys()
        self.all_actions = json.load(open(subj_obj_dir + "action_frequency.json")).keys()
        self.all_objects = json.load(open(subj_obj_dir + "object_frequency.json")).keys()
        self.transform = transform

    def create_binary_vector(self, subjects, actions, objects):
        subject_vector = [1 if subject in subjects else 0 for subject in self.all_subjects]
        action_vector = [1 if action in actions else 0 for action in self.all_actions]
        object_vector = [1 if object_ in objects else 0 for object_ in self.all_objects]

        # Combine the vectors to create a single target vector
        return subject_vector + action_vector + object_vector

    def __len__(self):
        return len(self.gif_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gif_url = self.gif_df.iloc[idx, 0]
        target = self.gif_df.iloc[idx, 1]

        # Shape (40, 3, 500, 500) and (40, 500, 500)
        gif_tensor, attention_mask, is_resized = load_gif(gif_url)
        
        # Tuple of three sets of strings
        (subjects, actions, objects) = target_to_subjects_and_objects(target)

        # n hot encoding, shape (num_subjects + num_actions + num_objects) ~ (12,000)
        target_vector = self.create_binary_vector(subjects, actions, objects)

        sample = {
            "gif": gif_tensor,
            "attention_mask": attention_mask,
            "target": target_vector,
        }
        return sample


batch_size = 5
gif_dataset = GIFDataset()
gif_loader = DataLoader(gif_dataset, batch_size=1, shuffle=False)

for i, sample in enumerate(gif_loader):
    print(sample['gif'].shape)
    break
