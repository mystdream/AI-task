import pytest
import cv2
import numpy as np
from models.text_extraction_model import TextExtractionModel

@pytest.fixture
def sample_image_with_text():
    image = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Hello World", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image

def test_text_extraction(sample_image_with_text):
    model = TextExtractionModel()
    extracted_text = model.extract_text(sample_image_with_text)
    assert "Hello World" in extracted_text
