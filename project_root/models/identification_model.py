from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
from PIL import Image

class ObjectIdentificationModel:
    def __init__(self):
        self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

    def identify_object(self, image):
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        return self.model.config.id2label[predicted_class_idx]
