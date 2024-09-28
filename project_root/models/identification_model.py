import torch
from torchvision.models import resnet50
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize


def load_identification_model():
    # Load a pre-trained ResNet model
    model = resnet50(pretrained=True)
    model.eval()
    return model


def identify_object(model, image):
    # Preprocess the image
    preprocess = Compose([
        Resize(256),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    _, predicted_idx = torch.max(output, 1)

    # You would need a class mapping to convert idx to label
    # For simplicity, we'll return the index here
    return predicted_idx.item()