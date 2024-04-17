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
    padded_frame = transforms.functional.pad(frame, padding, fill=0)  

    return padded_frame, frame.width, frame.height

def load_gif_compressed(gif_data, target_num_frames=TARGET_NUM_FRAMES):
    gif_data = BytesIO(gif_data)
    gif = Image.open(gif_data)

    # Pre allocate memory
    frames_as_tensors = torch.zeros(target_num_frames, 3, MAX_HEIGHT, MAX_WIDTH)
    attention_masks = torch.zeros(target_num_frames, 1, MAX_HEIGHT, MAX_WIDTH)

    for i, frame in enumerate(ImageSequence.Iterator(gif)):
        if i >= target_num_frames:
            break
        rgb_frame = frame.convert("RGB")
        resized_and_padded_frame, frame_width, frame_height = resize_and_pad_frame(rgb_frame)

        frame_tensor = transforms.ToTensor()(resized_and_padded_frame)
        frames_as_tensors[i] = frame_tensor

        mask_tensor = torch.zeros(1, MAX_HEIGHT, MAX_WIDTH)
        mask_tensor[:, :frame_height, :frame_width] = 1
        attention_masks[i] = mask_tensor

    return frames_as_tensors, attention_masks
