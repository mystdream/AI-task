from PIL import Image
import torch
from torchvision.transforms import functional as F

def load_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image_tensor