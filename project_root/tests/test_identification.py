import pytest
from models.identification_model import ObjectIdentificationModel
from PIL import Image
import numpy as np

@pytest.fixture
def sample_image():
    return Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

def test_object_identification(sample_image):
    model = ObjectIdentificationModel()
    result = model.identify_object(sample_image)
    assert isinstance(result, str)
    assert len(result) > 0
