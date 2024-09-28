import pytest
import cv2
import numpy as np
from models.segmentation_model import SegmentationModel
from utils.preprocessing import load_image, resize_image
from utils.postprocessing import extract_objects, save_object, store_metadata

@pytest.fixture
def sample_image():
    # Create a sample image for testing
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 10), (90, 90), (255, 255, 255), -1)
    return image

def test_load_image(tmp_path):
    # Test image loading
    image_path = tmp_path / "test_image.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 100, 3), dtype=np.uint8))
    loaded_image = load_image(str(image_path))
    assert loaded_image is not None
    assert loaded_image.shape == (100, 100, 3)

def test_resize_image(sample_image):
    # Test image resizing
    resized_image = resize_image(sample_image, max_size=50)
    assert max(resized_image.shape[:2]) == 50

def test_segmentation_model():
    # Test segmentation model initialization
    model = SegmentationModel()
    assert model.predictor is not None

def test_extract_objects(sample_image):
    # Mock segmentation output
    mock_output = {
        "instances": type('', (), {
            "pred_masks": type('', (), {"cpu": lambda: None})()
        })()
    }
    mock_output["instances"].pred_masks.cpu = lambda: np.array([np.ones((100, 100), dtype=bool)])
    
    objects = extract_objects(sample_image, mock_output)
    assert len(objects) == 1
    assert objects[0].shape[:2] == (100, 100)

def test_save_object(tmp_path):
    # Test object saving
    object_image = np.zeros((50, 50, 3), dtype=np.uint8)
    filepath = save_object(object_image, "obj1", "master1", str(tmp_path))
    assert os.path.exists(filepath)

def test_store_metadata(tmp_path):
    # Test metadata storage
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE objects
        (object_id TEXT, master_id TEXT, filepath TEXT)
    ''')
    
    store_metadata(conn, "obj1", "master1", "/path/to/image.png")
    
    cursor.execute("SELECT * FROM objects")
    result = cursor.fetchone()
    assert result == ("obj1", "master1", "/path/to/image.png")
    
    conn.close()
