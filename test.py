from utils import load_gif
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
from preprocess import main
import tracemalloc
from tqdm import tqdm


class TestPreprocess(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def test_stop_start(self):
        if os.path.exists("test.hdf5"):
            os.remove("test.hdf5")

        data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t", nrows=50)
        data_split_1 = data.iloc[:23]

        main(data=data_split_1, batch_size=5, hdf5_file="test.hdf5")
        main(data=data, batch_size=5, hdf5_file="test.hdf5")

        with h5py.File("test.hdf5", "r") as h5f:
            self.assertEqual(len(list(h5f.keys())), 50)
            all_keys = sorted([int(x) for x in h5f.keys()])
            assert all_keys == list(range(max(all_keys) + 1))
        print("test_stop_start passed")

    def test_curr_preprocessed(self):
        if not os.path.exists("processed_data.hdf5"):
            return

        data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t")

        with h5py.File("processed_data.hdf5", "r") as h5f:

            all_keys = sorted([int(x) for x in h5f.keys()])
            assert all_keys == list(range(max(all_keys) + 1))
            print("Keys line up for processed_data.hdf5")

            for ds_name in tqdm(h5f.keys()):
                read_data = h5f[ds_name][:]
                read_gif = BytesIO(read_data)

                data_gif = data.iloc[int(ds_name), 0]
                response = requests.get(data_gif).content
                gif = BytesIO(response)

                assert read_gif.getvalue() == gif.getvalue()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test.hdf5"):
            print("Cleaning up, removing test.hdf5")
            os.remove("test.hdf5")


if __name__ == "__main__":
    unittest.main()
