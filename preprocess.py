import pandas as pd
import torchtext
import torch
from torchvision import transforms
from PIL import Image, ImageSequence
import requests
from io import BytesIO
import torch
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np
import aiohttp
import asyncio
import h5py
import os

data_file = "data/tgif-v1.0.tsv"
hdf5_file = "processed_data.hdf5"

data = pd.read_csv(data_file, sep="\t")

urls = data.iloc[:, 0].tolist()


def load_progress(hdf5_file):
    if not os.path.exists(hdf5_file) or os.path.getsize(hdf5_file) == 0:
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)
        return 0
    try:
        with h5py.File(hdf5_file, "r") as h5f:
            next_index = len(h5f.keys())
            return next_index
    except OSError as e:
        print(f"Error reading HDF5 file: {e}")
        os.remove(hdf5_file)
        next_index = 0
    return next_index


async def download_gif(session, url):
    async with session.get(url) as response:
        return await response.read()


async def download_gifs_in_batch(urls):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*(download_gif(session, url) for url in urls))


async def preprocess(start_index, data, batch_size, hdf5_file):
    with h5py.File(hdf5_file, "a") as h5f:
        for i in range(start_index, len(data), batch_size):
            batch_urls = data.iloc[i : i + batch_size, 0].tolist()
            gifs = await download_gifs_in_batch(batch_urls)

            print(f"Processed up to index {i}")

            for j, gif in enumerate(gifs):
                if gif:
                    dataset_name = f"{i + j}"
                    gif_array = np.frombuffer(gif, dtype=np.uint8)
                    h5f.create_dataset(dataset_name, data=gif_array, dtype=np.uint8)


def main(data, batch_size, hdf5_file):
    print("Starting preprocessing")
    if data is None:
        print("Loading data from file...")
        data = pd.read_csv(data_file, sep="\t")

    next_index = load_progress(hdf5_file)
    print(f"Starting from index {next_index}")
    asyncio.run(preprocess(next_index, data, batch_size, hdf5_file))


if __name__ == "__main__":
    batch_size = 5
    main(data, batch_size, hdf5_file)
