import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

def extract_objects(image, segmentation_output):
    """Extract individual objects from the segmented image."""
    masks = segmentation_output["instances"].pred_masks.cpu().numpy()
    objects = []
    for i, mask in enumerate(masks):
        obj_image = image.copy()
        obj_image[mask == 0] = 0
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        cropped_obj = obj_image[y:y+h, x:x+w]
        objects.append(cropped_obj)
    return objects

def save_object(object_image, object_id, master_id, output_dir):
    """Save an extracted object as an image file."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{master_id}_{object_id}.png"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, object_image)
    return filepath

def store_metadata(conn, object_id, master_id, filepath):
    """Store object metadata in the SQLite database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO objects (object_id, master_id, filepath)
        VALUES (?, ?, ?)
    ''', (object_id, master_id, filepath))
    conn.commit()
