from torchvision import transforms
from PIL import Image, ImageSequence
import torch
from io import BytesIO

MAX_WIDTH = 128
MAX_HEIGHT = 128
TARGET_NUM_FRAMES = 40

def resize_and_pad_frame(frame: Image.Image):
    # Resize the frame while maintaining aspect ratio
    frame.thumbnail((MAX_WIDTH, MAX_HEIGHT))
    padding = (0, 0, MAX_WIDTH - frame.width, MAX_HEIGHT - frame.height)
    return transforms.functional.pad(frame, padding)

def load_gif_compressed(gif_data, target_num_frames=TARGET_NUM_FRAMES):
    gif_data = BytesIO(gif_data)
    gif = Image.open(gif_data)

    frames_as_tensors = []
    attention_masks = []

    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        if i >= target_num_frames:
            break
        rgb_frame = frame.convert("RGB")
        resized_and_padded_frame = resize_and_pad_frame(rgb_frame)
        
        frame_tensor = transforms.ToTensor()(resized_and_padded_frame)
        frames_as_tensors.append(frame_tensor)

        mask = torch.ones_like(frame_tensor[:1])  
        attention_masks.append(mask)

    while len(frames_as_tensors) < target_num_frames:
        frames_as_tensors.append(torch.zeros_like(frames_as_tensors[0]))
        attention_masks.append(torch.zeros_like(attention_masks[0]))

    gif_tensor = torch.stack(frames_as_tensors, dim=0)
    attention_mask = torch.stack(attention_masks, dim=0)
    assert gif_tensor.shape == (target_num_frames, 3, MAX_HEIGHT, MAX_WIDTH), f"Expected shape {(target_num_frames, 3, MAX_HEIGHT, MAX_WIDTH)}, got {gif_tensor.shape}"
    assert attention_mask.shape == (target_num_frames, 1, MAX_HEIGHT, MAX_WIDTH), f"Expected shape {(target_num_frames, 1, MAX_HEIGHT, MAX_WIDTH)}, got {attention_mask.shape}"
    return gif_tensor, attention_mask