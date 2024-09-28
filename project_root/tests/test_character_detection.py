import pytest
import cv2
import numpy as np
from models.character_detection_model import CharacterDetectionModel

@pytest.fixture
def sample_image_with_text():
    image = np.ones((300, 300, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Hello", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image

def test_character_detection(sample_image_with_text):
    model = CharacterDetectionModel()
    regions = model.detect_text_regions(sample_image_with_text)
    assert len(regions) > 0
    assert all(isinstance(region, tuple) and len(region) == 4 for region in regions)
