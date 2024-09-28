import torch
import torchvision


def load_segmentation_model():
    # Load a pre-trained Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


def segment_image(model, image):
    # Perform segmentation on the input image
    with torch.no_grad():
        prediction = model([image])

    masks = prediction[0]['masks']
    scores = prediction[0]['scores']

    # Filter out low-confidence predictions
    mask_threshold = 0.5
    high_confidence_masks = masks[scores > mask_threshold]

    return high_confidence_masks
