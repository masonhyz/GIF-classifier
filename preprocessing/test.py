import pandas as pd
import requests
from io import BytesIO
import torch
import numpy as np
import time
import io
from io import BytesIO
import unittest
import os
import h5py
import tracemalloc
from tqdm import tqdm


class TestPreprocess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_curr_preprocessed(self):
        if not os.path.exists("preprocessing/processed_data.hdf5"):
            print("No preprocessed data found, running main()")
            return

        subset_data = pd.read_csv("data/subset_data.csv")

        with h5py.File("preprocessing/processed_data.hdf5", "r") as h5f:

            gif_group = h5f["gif_data"]
            target_group = h5f["targets"]

            for dataset_name in gif_group.keys():
                if dataset_name in target_group:
                    gif_data = np.array(h5f["gif_data"][dataset_name]).tobytes()
                    original_data = requests.get(subset_data.iloc[int(dataset_name)]["gif_url"]).content
                    
                    h5f_target = target_group[dataset_name][()].decode("utf-8")
                    original_target = subset_data.iloc[int(dataset_name)]["target"][:-1]
                    
                    self.assertEqual(gif_data, original_data)
                    self.assertEqual(h5f_target, original_target)
                    
                else:
                    print(f"Target not found for dataset {dataset_name}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test.hdf5"):
            print("Cleaning up, removing test.hdf5")
            os.remove("test.hdf5")


if __name__ == "__main__":
    unittest.main()
