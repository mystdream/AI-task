import pytesseract
from PIL import Image

def extract_text(image):
    # Use Tesseract OCR to extract text from the image
    text = pytesseract.image_to_string(image)
    return text