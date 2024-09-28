import numpy as np
from PIL import Image

def extract_object_image(original_image, mask):
    # Extract the object from the original image using the mask
    mask = mask.squeeze().numpy()
    masked_image = original_image * mask[:, :, np.newaxis]
    object_image = Image.fromarray((masked_image * 255).astype(np.uint8))
    return object_image