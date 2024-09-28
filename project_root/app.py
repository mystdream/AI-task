import streamlit as st
from PIL import Image
from models.segmentation_model import load_segmentation_model, segment_image
from models.identification_model import load_identification_model, identify_object
from models.text_extraction_model import extract_text
from models.summarization_model import load_summarization_model, summarize_text
from utils.preprocessing import load_image
from utils.postprocessing import extract_object_image
from utils.data_mapping import map_data
from utils.visualization import visualize_segmentation
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    st.title("AI Pipeline for Image Segmentation and Object Analysis")

    # Load models
    segmentation_model = load_segmentation_model()
    identification_model = load_identification_model()
    summarization_model = load_summarization_model()

    # File upload
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Process image
        image_tensor = load_image(uploaded_file)
        masks = segment_image(segmentation_model, image_tensor)

        # Visualize segmentation
        st.subheader("Segmentation Results")
        visualize_segmentation(image_tensor, masks)

        # Process each segmented object
        objects_data = []
        for i, mask in enumerate(masks):
            object_image = extract_object_image(image, mask)

            # Object identification
            object_class = identify_object(identification_model, object_image)

            # Text extraction
            extracted_text = extract_text(object_image)

            # Summarization
            summary = summarize_text(summarization_model, extracted_text)

            # Store object data
            object_data = {
                "id": f"obj_{i}",
                "class": object_class,
                "extracted_text": extracted_text,
                "summary": summary
            }
            objects_data.append(object_data)

            # Display object details
            st.subheader(f"Object {i + 1}")
            st.image(object_image, caption=f'Segmented Object {i + 1}', use_column_width=True)
            st.write(f"Class: {object_class}")
            st.write(f"Extracted Text: {extracted_text}")
            st.write(f"Summary: {summary}")

        # Data mapping
        mapped_data = map_data("master_001", objects_data)

        # Display final output
        st.subheader("Final Output")
        st.json(mapped_data)


if __name__ == "__main__":
    main()