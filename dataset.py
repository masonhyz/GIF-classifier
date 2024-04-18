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
from PIL import Image, ImageSequence
from io import BytesIO
import requests
from PIL import Image, ImageTk, ImageSequence
import tkinter as tk

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
        self.data_file = data_file
        self.h5f = h5py.File(self.data_file, "r")
        self.actions = ["dancing", "playing", "walking", "looking", "talking", "singing", "doing", "kissing", "holding", "running"]
        self.transform = transform

    def __len__(self):
        with h5py.File(self.data_file, "r") as file:
            return len(file["gif_data"])

    def __getitem__(self, idx):
        if isinstance(idx, list):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            samples = [self.get_single_item(i) for i in idx]
            return samples
        else:
            return self.get_single_item(idx)

    def __del__(self):
        self.h5f.close()

    def show_gif(self, gif_bytes):
        root = tk.Tk()
        gif_stream = BytesIO(gif_bytes)
        gif = Image.open(gif_stream)
        frames = [ImageTk.PhotoImage(image=frame.copy()) for frame in ImageSequence.Iterator(gif)]
        frame_label = tk.Label(root)
        frame_label.pack()

        def update_frame(num=0):
            frame = frames[num]
            frame_label.config(image=frame)
            num = (num + 1) % len(frames)  
            root.after(250, update_frame, num)  

        update_frame() 
        root.mainloop()
        
    def create_target_vector(self, actions, target):
        target_components = set(target.split())
        
        target_vector = [0] * len(actions)
        
        for i, action in enumerate(actions):
            if action in target_components:
                target_vector[i] = 1
                break  
        
        return torch.tensor(target_vector)

    def get_single_item(self, idx):
        gif_data = np.array(self.h5f["gif_data"][str(idx)]).tobytes()
        # self.show_gif(gif_data)
        
        target = self.h5f["targets"][str(idx)][()].decode("utf-8")

        # (num_frames, 3, h, w) and (num_frames, h, w)
        gif_tensor, attention_mask = load_gif_compressed(gif_data)

        # 1 hot encoding, shape [10]
        target_vector = self.create_target_vector(self.actions, target)
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
    # train_split, val_split = train_val_sklearn_split(dataset, test_size=0.2)
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    gif = dataset[1]["gif"]
