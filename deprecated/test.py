import unittest
from preprocessing.preprocess import main, process_gif
import h5py
import pandas as pd
import os
import requests


class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_small_dataset_to_complete(self):
        print("Running test_small_dataset_to_complete")
        if os.path.exists("test.hdf5"):
            os.remove("test.hdf5")

        data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t", nrows=23)
        main(data=data, batch_size=5, hdf5_file="test.hdf5")
        with h5py.File("test.hdf5", "r") as h5f:
            self.assertEqual(len(h5f["gif_tensors"]), 23)
            self.assertEqual(len(h5f["attn_masks"]), 23)
            self.assertEqual(len(h5f["tgt_embeddings"]), 23)

    def test_stop_and_resume(self):
        print("Running test_stop_and_resume")
        if os.path.exists("test.hdf5"):
            os.remove("test.hdf5")

        data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t", nrows=50)
        data_split_1 = data.iloc[:23]

        main(
            data=data_split_1, batch_size=5, hdf5_file="test.hdf5"
        )  # Simulating stopping early
        main(data=data, batch_size=5, hdf5_file="test.hdf5")

        with h5py.File("test.hdf5", "r") as h5f:
            self.assertEqual(len(h5f["gif_tensors"]), 50)
            self.assertEqual(len(h5f["attn_masks"]), 50)
            self.assertEqual(len(h5f["tgt_embeddings"]), 50)

            manual_tensors = []
            manual_masks = []

            for index, row in data.iterrows():
                gif_url = row[0]
                text = row[1]
                response = requests.get(gif_url)
                gif_data = response.content

                gif_tensor, attn_mask, _ = process_gif(gif_data, text)
                manual_tensors.append(gif_tensor)
                manual_masks.append(attn_mask)

                self.assertTrue(
                    (gif_tensor.numpy() == h5f["gif_tensors"][str(index)]).all()
                )
                self.assertTrue(
                    (attn_mask.numpy() == h5f["attn_masks"][str(index)]).all()
                )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists("test.hdf5"):
            print("Cleaning up, removing test.hdf5")
            os.remove("test.hdf5")


if __name__ == "__main__":
    unittest.main()
