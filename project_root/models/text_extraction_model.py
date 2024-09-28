import pytesseract
import easyocr
from pyzbar.pyzbar import decode
import cv2

class TextExtractionModel:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract_text(self, image):
        # OCR using EasyOCR
        results = self.reader.readtext(image)
        extracted_text = ' '.join([result[1] for result in results])

        # Barcode/QR code detection
        barcodes = decode(image)
        for barcode in barcodes:
            extracted_text += f" Barcode: {barcode.data.decode('utf-8')}"

        return extracted_text
