import matplotlib.pyplot as plt
import numpy as np

def visualize_segmentation(image, masks):
    # Visualize the segmented objects on the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image.permute(1, 2, 0))
    for mask in masks:
        plt.imshow(mask.squeeze().numpy(), alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.show()