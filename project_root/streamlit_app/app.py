import streamlit as st
import cv2
import numpy as np
import sqlite3
import os
from models.segmentation_model import SegmentationModel
from models.identification_model import ObjectIdentificationModel
from models.character_detection_model import CharacterDetectionModel
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel
from utils.preprocessing import resize_image
from utils.postprocessing import extract_objects, process_object
from utils.data_mapping import map_data_to_objects, create_dataframe
from utils.visualization import draw_bounding_boxes, create_summary_table

# Initialize models
@st.cache(allow_output_mutation=True)
def load_models():
    return {
        'segmentation': SegmentationModel(),
        'identification': ObjectIdentificationModel(),
        'character_detection': CharacterDetectionModel(),
        'text_extraction': TextExtractionModel(),
        'summarization': SummarizationModel()
    }

models = load_models()

# Streamlit app
st.title('AI Image Analysis Pipeline')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = resize_image(image)

    # Perform segmentation
    segmentation_output = models['segmentation'].segment_image(image)

    # Extract objects
    objects = extract_objects(image, segmentation_output)

    # Draw bounding boxes
    image_with_boxes = draw_bounding_boxes(image, objects)

    # Display the original image with bounding boxes
    st.image(image_with_boxes, caption='Segmented Image', use_column_width=True)

    # Process objects
    conn = sqlite3.connect(':memory:')  # Use in-memory database for demo
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS objects
        (object_id TEXT PRIMARY KEY, master_id TEXT, filepath TEXT,
         object_class TEXT, extracted_text TEXT, summary TEXT)
    ''')
    conn.commit()

    for i, obj in enumerate(objects):
        object_id = f"obj_{i}"
        master_id = "uploaded_image"
        process_object(obj, object_id, master_id, 'temp', conn,
                       models['identification'], models['character_detection'],
                       models['text_extraction'], models['summarization'])

    # Map data
    mapped_data = map_data_to_objects(conn)
    df = create_dataframe(mapped_data)

    # Display summary table
    st.subheader('Object Summary')
    st.dataframe(df)

    # Create and display summary table as an image
    fig = create_summary_table(df)
    st.pyplot(fig)

    conn.close()
