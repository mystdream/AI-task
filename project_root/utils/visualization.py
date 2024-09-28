import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_bounding_boxes(image, objects):
    image_copy = image.copy()
    for obj in objects:
        x, y, w, h = cv2.boundingRect(obj)
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image_copy

def create_summary_table(df):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    plt.tight_layout()
    return fig
