import pandas as pd
import torchtext
import torch
from torchvision import transforms
from PIL import Image, ImageSequence
import requests
from io import BytesIO
import torch

MAX_WIDTH = 555
MAX_HEIGHT = 810


def __crop_gif_frame(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    cropped_image = image.crop((left, top, right, bottom))
    cropped_rgb_image = Image.new("RGB", (target_width, target_height))
    cropped_rgb_image.paste(cropped_image)
    return cropped_rgb_image


def __resize_gif_frame(rgb_frame, frame):
    original_width, original_height = rgb_frame.size

    if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
        target_width = min(MAX_WIDTH, original_width)
        target_height = min(MAX_HEIGHT, original_height)
        print(
            f"Resizing frame from {original_width}x{original_height} to {target_width}x{target_height}"
        )
        resized_frame = __crop_gif_frame(frame, target_width, target_height)
        assert resized_frame.size == (target_width, target_height)
    else:
        resized_frame = rgb_frame
    return resized_frame


def load_gif(path, target_num_frames=40):
    response = requests.get(path)
    gif_data = BytesIO(response.content)
    gif = Image.open(gif_data)
    frames_as_tensors = []
    attention_masks = []
    
    # Only take the first target_num_frames frames
    frames = list(ImageSequence.Iterator(gif))[:target_num_frames]

    for frame in frames:
        # frame.show()
        rgb_frame = frame.convert("RGB")

        resized_frame = __resize_gif_frame(rgb_frame, frame)

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
        assert (
            frame_tensor.shape[0] == 3
            and frame_tensor.shape[1:] == (MAX_HEIGHT, MAX_WIDTH)
            and padded_mask.shape == (MAX_HEIGHT, MAX_WIDTH)
        )

        frames_as_tensors.append(frame_tensor)
        attention_masks.append(padded_mask)
        
    # Pad frames and masks if necessary
    while len(frames_as_tensors) < target_num_frames:
        frames_as_tensors.append(frames_as_tensors[-1])
        attention_masks.append(torch.zeros((MAX_HEIGHT, MAX_WIDTH)))

    gif_tensor = torch.stack(frames_as_tensors, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)
    assert gif_tensor.shape == (target_num_frames, 3, MAX_HEIGHT, MAX_WIDTH)
    assert attention_mask.shape == (target_num_frames, MAX_HEIGHT, MAX_WIDTH)
    return gif_tensor, attention_mask
