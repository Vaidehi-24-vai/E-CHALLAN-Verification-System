# text_recognition.py

import easyocr

def perform_ocr(image_path, lang="en"):
    # Initialize EasyOCR reader
    reader = easyocr.Reader([lang])

    # Perform OCR on the image
    result = reader.readtext(image_path)

    # Extract text from the result
    text = [entry[1] for entry in result]

    return text
