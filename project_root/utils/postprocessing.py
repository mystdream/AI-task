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

def process_object(object_image, object_id, master_id, output_dir, conn, 
                   id_model, char_model, text_model, sum_model):
    # Identify object
    object_class = id_model.identify_object(object_image)

    # Detect text regions
    text_regions = char_model.detect_text_regions(object_image)

    # Extract text from regions
    extracted_text = ""
    for (startX, startY, endX, endY) in text_regions:
        roi = object_image[startY:endY, startX:endX]
        text = text_model.extract_text(roi)
        extracted_text += text + " "

    # Summarize attributes
    summary = sum_model.summarize_text(f"{object_class}. {extracted_text}")

    # Save object and store metadata
    filepath = save_object(object_image, object_id, master_id, output_dir)
    store_metadata(conn, object_id, master_id, filepath, object_class, extracted_text, summary)

def store_metadata(conn, object_id, master_id, filepath, object_class, extracted_text, summary):
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO objects 
        (object_id, master_id, filepath, object_class, extracted_text, summary)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (object_id, master_id, filepath, object_class, extracted_text, summary))
    conn.commit()
