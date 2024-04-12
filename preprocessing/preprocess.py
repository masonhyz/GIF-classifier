import pandas as pd
import numpy as np
import h5py
import os
import requests
from tqdm import tqdm

SUBJECTS = ["man", "woman", "people", "cat", "dog", "car", "boy", "girl", "men"]
ACTIONS = ["looking", "dancing", "walking", "smiling", "playing", "standing", "wearing", "holding"]
HDF5_FILE = "preprocessing/processed_data.hdf5"

data_file = "data/tgif-v1.0.tsv"
data = pd.read_csv(data_file, sep="\t")


def contains_subject_and_action(text: str):
    words = text.lower().split()
    has_subject = any(word in SUBJECTS for word in words)
    has_action = any(word in ACTIONS for word in words)
    return has_subject and has_action


subset_data = data[data.iloc[:, 1].apply(contains_subject_and_action)]
print(subset_data.shape)


def load_progress(hdf5_file):
    if not os.path.exists(hdf5_file) or os.path.getsize(hdf5_file) == 0:
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)
        return 0
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            next_index = len(h5f["gif_data"])
            return next_index
    except (OSError, KeyError) as e:
        print(f"Error reading HDF5 file: {e}")
        os.remove(hdf5_file)  # Remove corrupted file
        return 0  # Start from scratch


def contains_subject_and_action(text: str):
    words = text.lower().split()
    has_subject = any(word in SUBJECTS for word in words)
    has_action = any(word in ACTIONS for word in words)
    return has_subject and has_action


def preprocess(start_index, data: pd.DataFrame, hdf5_file):
    with h5py.File(hdf5_file, "a") as h5f:
        gif_group = h5f.require_group("gif_data")
        target_group = h5f.require_group("targets")

        for i in tqdm(range(start_index, len(data))):
            gif_url, target = data.iloc[i]
            gif_data = requests.get(gif_url).content
            dataset_name = f"{start_index + i}"
            gif_group.create_dataset(dataset_name, data=np.frombuffer(gif_data, dtype=np.uint8))
            target_group.create_dataset(dataset_name, data=target.encode("utf-8"))


def main(data=subset_data, hdf5_file=HDF5_FILE):
    start_index = load_progress(hdf5_file)
    if start_index == 0:
        print("Starting from scratch")
    else:
        print(f"Resuming preprocessing from index {start_index}")
    preprocess(start_index, data, hdf5_file)


if __name__ == "__main__":
    main()
