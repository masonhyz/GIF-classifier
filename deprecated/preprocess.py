import pandas as pd
import os
import torchtext
from utils import __crop_gif_frame
import torch
import aiohttp
import asyncio
from torchvision import transforms
from PIL import Image, ImageSequence
import torch
import io
import h5py


EMBEDDING_MODEL = torchtext.vocab.GloVe(name="6B", dim=100)
MAX_WIDTH = 555
MAX_HEIGHT = 810


def load_progress(hdf5_file):
    if not os.path.exists(hdf5_file) or os.path.getsize(hdf5_file) == 0:
        if os.path.exists(hdf5_file):
            os.remove(hdf5_file)
        return 0

    try:
        with h5py.File(hdf5_file, "r") as h5f:
            if "gif_tensors" in h5f:
                next_index = len(h5f["gif_tensors"])
            else:
                next_index = 0
    except OSError as e:
        print(f"Error reading HDF5 file: {e}")
        os.remove(hdf5_file)
        next_index = 0

    return next_index


def get_embedding(text: str, embedding: torchtext.vocab.Vectors):
    return embedding.get_vecs_by_tokens(text.lower().split())


async def download_gif(session, url):
    async with session.get(url) as response:
        return await response.read()


async def download_gifs_in_batch(urls):
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*(download_gif(session, url) for url in urls))


def process_gif(gif_data, text, embedding_model=EMBEDDING_MODEL):
    gif = Image.open(io.BytesIO(gif_data))
    frames_as_tensors = []
    attention_masks = []

    for frame in ImageSequence.Iterator(gif):
        rgb_frame = frame.convert("RGB")
        original_width, original_height = rgb_frame.size

        if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
            target_width = min(MAX_WIDTH, original_width)
            target_height = min(MAX_HEIGHT, original_height)
            resized_frame = __crop_gif_frame(rgb_frame, target_width, target_height)
        else:
            resized_frame = rgb_frame

        mask = torch.ones((resized_frame.size[1], resized_frame.size[0]))
        padded_mask = transforms.functional.pad(
            mask,
            (
                0,
                0,
                MAX_WIDTH - resized_frame.size[0],
                MAX_HEIGHT - resized_frame.size[1],
            ),
        )
        padded_frame = transforms.functional.pad(
            resized_frame,
            (
                0,
                0,
                MAX_WIDTH - resized_frame.size[0],
                MAX_HEIGHT - resized_frame.size[1],
            ),
        )
        frame_tensor = transforms.ToTensor()(padded_frame)

        frames_as_tensors.append(frame_tensor)
        attention_masks.append(padded_mask)

    gif_tensor = torch.stack(frames_as_tensors, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)
    target_embedding = get_embedding(text, embedding_model)

    return gif_tensor, attention_mask, target_embedding


async def process_gifs_and_targets(
    batch: pd.DataFrame,
    embedding_model: torchtext.vocab.Vectors,
    h5f: h5py.File,
    start_idx: int,
):
    urls, texts = batch.iloc[:, 0].tolist(), batch.iloc[:, 1].tolist()
    batch_data = await download_gifs_in_batch(urls)

    for idx, (gif_data, text) in enumerate(zip(batch_data, texts)):
        hdf5_idx = start_idx + idx
        gif_tensor, attn_mask, tgt_embedding = process_gif(
            gif_data, text, embedding_model
        )

        h5f["gif_tensors"].create_dataset(str(hdf5_idx), data=gif_tensor.numpy())
        h5f["attn_masks"].create_dataset(str(hdf5_idx), data=attn_mask.numpy())
        h5f["tgt_embeddings"].create_dataset(str(hdf5_idx), data=tgt_embedding.numpy())

    print(f"Processed {len(batch)} items")


async def preprocess(
    next_index,
    data,
    embedding_model,
    hdf5_file,
    batch_size,
):
    with h5py.File(hdf5_file, "a") as h5f:
        if "gif_tensors" not in h5f:
            h5f.create_group("gif_tensors")
            h5f.create_group("attn_masks")
            h5f.create_group("tgt_embeddings")

        offset_idx = next_index
        data_batch = data.iloc[offset_idx : offset_idx + batch_size]

        while data_batch.shape[0] > 0:
            await process_gifs_and_targets(data_batch, embedding_model, h5f, offset_idx)
            offset_idx += len(data_batch)
            data_batch = data.iloc[offset_idx : offset_idx + batch_size]


def main(
    data=None,
    embedding_model=EMBEDDING_MODEL,
    hdf5_file="preprocessing_progress.hdf5",
    batch_size=5,
):
    print("Starting preprocessing")
    if data is None:
        print("Loading data from file...")
        data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t")

    print(f"Data shape: {data.shape}")
    next_index = load_progress(hdf5_file)
    print(f"Starting from index {next_index}")
    asyncio.run(preprocess(next_index, data, embedding_model, hdf5_file, batch_size))


if __name__ == "__main__":
    main()
