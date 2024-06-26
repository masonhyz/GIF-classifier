import pandas as pd
import torchtext
import torch
from torchvision import transforms
from PIL import Image, ImageSequence
import requests
from io import BytesIO
import torch
import spacy
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np

nlp = spacy.load("en_core_web_sm")

MAX_WIDTH = 500
MAX_HEIGHT = 500
TARGET_NUM_FRAMES = 50


def __crop_gif_frame(image, target_width, target_height):
    width, height = image.size
    left = (width - target_width) // 2
    top = (height - target_height) // 2
    right = left + target_width
    bottom = top + target_height
    cropped_image = image.crop((left, top, right, bottom))
    # cropped_image.show()
    cropped_rgb_image = Image.new("RGB", (target_width, target_height))
    cropped_rgb_image.paste(cropped_image)
    return cropped_rgb_image


def __resize_gif_frame(rgb_frame, frame):
    original_width, original_height = rgb_frame.size

    if original_width > MAX_WIDTH or original_height > MAX_HEIGHT:
        target_width = min(MAX_WIDTH, original_width)
        target_height = min(MAX_HEIGHT, original_height)
        # print(f"Resizing frame from {original_width}x{original_height} to {target_width}x{target_height}")
        resized_frame = __crop_gif_frame(frame, target_width, target_height)
        assert resized_frame.size == (target_width, target_height)
    else:
        resized_frame = rgb_frame
    return resized_frame


def load_gif(gif_data, target_num_frames=TARGET_NUM_FRAMES):
    gif_data = BytesIO(gif_data)
    gif = Image.open(gif_data)

    frames_as_tensors = []
    attention_masks = []
    count = 0

    for frame in ImageSequence.Iterator(gif):
        # frame.show()
        rgb_frame = frame.convert("RGB")

        resized_frame = __resize_gif_frame(rgb_frame, frame)

        mask = torch.ones((resized_frame.size[1], resized_frame.size[0]))

        padded_mask = transforms.functional.pad(mask, (0, 0, MAX_WIDTH - resized_frame.size[0], MAX_HEIGHT - resized_frame.size[1]))
        padded_frame = transforms.functional.pad(resized_frame, (0, 0, MAX_WIDTH - resized_frame.size[0], MAX_HEIGHT - resized_frame.size[1]))

        frame_tensor = transforms.ToTensor()(padded_frame)
        frames_as_tensors.append(frame_tensor)
        attention_masks.append(padded_mask)

        count += 1
        if count == target_num_frames:
            break

    while count < target_num_frames:
        frames_as_tensors.append(frames_as_tensors[-1])
        attention_masks.append(torch.zeros((MAX_HEIGHT, MAX_WIDTH)))
        count += 1

    gif_tensor = torch.stack(frames_as_tensors, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)
    assert gif_tensor.shape == (target_num_frames, 3, MAX_HEIGHT, MAX_WIDTH)
    assert attention_mask.shape == (target_num_frames, MAX_HEIGHT, MAX_WIDTH)
    return gif_tensor, attention_mask
    

def target_to_subjects_and_objects(target_str: str):
    """
    subjects: the person or objects performing the action
    objects: the person or objects that the action is being performed on
    """
    doc = nlp(target_str)

    actions = []

    for token in doc:
        if token.pos_ == "VERB":
            actions.append(token.text)

    if len(actions) != 1:
        return None
    return actions


def get_subj_obj_frequency():
    data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t")
    acts = defaultdict(int)
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        target = row[1]
        actions = target_to_subjects_and_objects(target)
        if actions is not None:
            acts[actions[0]] += 1
            
    with open("subj_obj_data/action_frequency.json", "w") as f:
        json.dump(acts, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # get_subj_obj_frequency()
    # load_subject_object_frequency()
    get_subj_obj_frequency()