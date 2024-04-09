import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
from utils import load_gif
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
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.gif_df = pd.read_csv(data_file, sep="\t")
        self.transform = transform

    def __len__(self):
        return len(self.gif_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        gif_url = self.gif_df.iloc[idx, 0]
        target = self.gif_df.iloc[idx, 1]
        
        gif_tensor, attention_mask = load_gif(gif_url) # Shape (num_frames, 3, H, W)

        sample = {"gif": gif_tensor, "attention_mask": attention_mask, "target": target}
        return sample

gif_dataset = GIFDataset()

plot = plt.figure()
d = defaultdict(int)

for i in range(len(gif_dataset)):
    print(i)
    sample = gif_dataset[i]
    shape = sample["gif"].shape
    print(shape)
