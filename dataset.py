import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from utils import load_gif, target_to_subjects_and_objects
from collections import defaultdict

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class GIFDataset(Dataset):
    """GIF dataset"""

    def __init__(
        self,
        data_file=os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "data/tgif-v1.0.tsv"
        ),
        transform=None,
    ):
        self.gif_df = pd.read_csv(data_file, sep="\t")
        self.transform = transform

    def __len__(self):
        return len(self.gif_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        gif_url = self.gif_df.iloc[idx, 0]
        target = self.gif_df.iloc[idx, 1]

        gif_tensor, attention_mask = load_gif(
            gif_url
        )  # Shape (40, 3, H, W) and (40, H, W)
        (subjects, objects) = target_to_subjects_and_objects(
            target
        )  # Tuple of two sets of strings

        sample = {
            "gif": gif_tensor,
            "attention_mask": attention_mask,
            "target": (subjects, objects),
        }
        return sample


gif_dataset = GIFDataset()
