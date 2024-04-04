import pandas as pd
import torchtext
import torch
from torchvision import transforms
from PIL import Image, ImageSequence
import requests
from io import BytesIO
import torch

DATA_PATH = "data/tgif-v1.0.tsv"
data = pd.read_csv(DATA_PATH, sep="\t")
embedding = torchtext.vocab.GloVe(name="6B", dim=100)
PROGRESS_FILE = "preprocessing_progress.pt"
MAX_WIDTH = 555
MAX_HEIGHT = 810


def crop(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    cropped_image = image.crop((left, top, right, bottom))
    cropped_rgb_image = Image.new("RGB", (target_width, target_height))
    cropped_rgb_image.paste(cropped_image)
    return cropped_rgb_image


def load_gif(path):
    response = requests.get(path)
    gif_data = BytesIO(response.content)
    gif = Image.open(gif_data)
    frames_as_tensors = []
    attention_masks = []

    for frame in ImageSequence.Iterator(gif):
        # frame.show()
        rgb_frame = frame.convert("RGB")
        original_width, original_height = rgb_frame.size

        if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
            # Resize the frame to fit within the constraints
            target_width = min(MAX_WIDTH, original_width)
            target_height = min(MAX_HEIGHT, original_height)
            print(
                f"Resizing frame from {original_width}x{original_height} to {target_width}x{target_height}"
            )
            resized_frame = crop(frame, target_width, target_height)
            assert resized_frame.size == (target_width, target_height)
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
        assert frame_tensor.shape[0] == 3

        frames_as_tensors.append(frame_tensor)
        attention_masks.append(padded_mask)
        assert frame_tensor.shape[1:] == (MAX_HEIGHT, MAX_WIDTH)
        assert padded_mask.shape == (MAX_HEIGHT, MAX_WIDTH)

    gif_tensor = torch.stack(frames_as_tensors, dim=0)  # (num_frames, 3, 810, 540)
    attention_mask = torch.stack(attention_masks, dim=0)  # (num_frames, 810, 540)
    return gif_tensor, attention_mask
