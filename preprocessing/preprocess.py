import pandas as pd
import numpy as np
import h5py
import os
import requests
import time
from tqdm import tqdm

ACTIONS = {'dancing', 'playing', 'walking', 'looking', 'talking', 'singing', 'doing', 'kissing', 'holding', 'running'}
HDF5_FILE = "preprocessing/processed_data.hdf5"

data_file = "data/tgif-v1.0.tsv"
data = pd.read_csv(data_file, sep="\t")


def contains_single_action_in_list(text: str):
    words = text.lower().split()
    return sum(word in ACTIONS for word in words) == 1


subset_data = data[data.iloc[:, 1].apply(contains_single_action_in_list)]
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

def preprocess(start_index, data: pd.DataFrame, hdf5_file):
    with h5py.File(hdf5_file, "a") as h5f:
        gif_group = h5f.require_group("gif_data")
        target_group = h5f.require_group("targets")

        for i in tqdm(range(start_index, len(data))):
            success = False
            attempt_count = 0
            max_attempts = 20 
            gif_url, target = data.iloc[i]

            while not success and attempt_count < max_attempts:
                try:
                    gif_data = requests.get(gif_url, timeout=10).content  
                    success = True
                except requests.RequestException as e:
                    attempt_count += 1
                    time.sleep(2 ** attempt_count)  
                    print(f"Failed to fetch {gif_url}, attempt {attempt_count}. Error: {e}")

            if not success:
                print(f"Failed to fetch data after {max_attempts} attempts for URL: {gif_url}")
                continue  
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