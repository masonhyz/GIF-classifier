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
from preprocess import main, contains_subject_and_action
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

        data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t")

        with h5py.File("preprocessing/processed_data.hdf5", "r") as h5f:

            gif_group = h5f["gif_data"]
            target_group = h5f["targets"]

            for dataset_name in gif_group.keys():
                if dataset_name in target_group:
                    gif_data = gif_group[dataset_name][()]
                    target_text = target_group[dataset_name][()].decode("utf-8")
                    self.assertTrue(contains_subject_and_action(target_text))
                else:
                    print(f"Target not found for dataset {dataset_name}")

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test.hdf5"):
            print("Cleaning up, removing test.hdf5")
            os.remove("test.hdf5")


if __name__ == "__main__":
    unittest.main()
