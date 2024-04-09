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

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

MAX_WIDTH = 555
MAX_HEIGHT = 810
TARGET_NUM_FRAMES = 40


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


def load_gif(path, target_num_frames=TARGET_NUM_FRAMES):
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


def target_to_subjects_and_objects(target_str: str) -> tuple[set[str], set[str]]:
    """
    subjects: the person or objects performing the action
    objects: the person or objects that the action is being performed on
    """
    doc = nlp(target_str)

    subjects = set()
    objects = set()

    for token in doc:
        if "subj" in token.dep_:
            subjects.add(token.text)
        elif "obj" in token.dep_:
            objects.add(token.text)

    return subjects, objects


def get_subj_obj_frequency():
    data = pd.read_csv("data/tgif-v1.0.tsv", sep="\t")
    subjs = defaultdict(int)
    objs = defaultdict(int)
    for _, row in tqdm(data.iterrows(), total=data.shape[0]):
        target = row[1]
        subjects, objects = target_to_subjects_and_objects(target)
        for subj in subjects:
            subjs[subj] += 1
        for obj in objects:
            objs[obj] += 1
            
    with open('subj_obj_data/subject_frequency.json', 'w') as f:
        json.dump(subjs, f, ensure_ascii=False, indent=4)
        
    with open('subj_obj_data/object_frequency.json', 'w') as f:
        json.dump(objs, f, ensure_ascii=False, indent=4)
        
def load_subject_object_frequency():
    with open('subj_obj_data/subject_frequency.json') as f:
        subj_freq = json.load(f)

    with open('subj_obj_data/object_frequency.json') as f:
        obj_freq = json.load(f)
        
    return subj_freq, obj_freq
        
if __name__ == "__main__":
    # get_subj_obj_frequency()
    load_subject_object_frequency()