import pandas as pd
import numpy as np
import h5py
import os
import requests
from tqdm import tqdm

ACTIONS = {'dancing', 'playing', 'walking', 'looking', 'talking', 'singing', 'doing', 'kissing', 'holding', 'running'}
HDF5_FILE = "preprocessing/processed_data.hdf5"

subset_file = "data/subset_data.csv"
subset_data = pd.read_csv(subset_file)

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


def preprocess(start_index, data: pd.DataFrame, hdf5_file):
    with h5py.File(hdf5_file, "w") as h5f:
        gif_group = h5f.require_group("gif_data")
        target_group = h5f.require_group("targets")

        for i in tqdm(range(start_index, len(data))):
            gif_url, target = data.iloc[i]
            gif_data = requests.get(gif_url).content
            dataset_name = f"{start_index + i}"
            gif_group.create_dataset(dataset_name, data=np.frombuffer(gif_data, dtype=np.uint8))
            target_group.create_dataset(dataset_name, data=target.encode("utf-8"))
            break


def main(data=subset_data, hdf5_file=HDF5_FILE):
    start_index = load_progress(hdf5_file)
    if start_index == 0:
        print("Starting from scratch")
    else:
        print(f"Resuming preprocessing from index {start_index}")
    preprocess(start_index, data, hdf5_file)


if __name__ == "__main__":
    main()
