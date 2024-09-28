import cv2
import numpy as np

def load_image(image_path):
    """Load an image from the given path."""
    return cv2.imread(image_path)

def resize_image(image, max_size=1024):
    """Resize the image while maintaining aspect ratio."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        if h > w:
            new_h, new_w = max_size, int(max_size * w / h)
        else:
            new_h, new_w = int(max_size * h / w), max_size
        image = cv2.resize(image, (new_w, new_h))
    return image
