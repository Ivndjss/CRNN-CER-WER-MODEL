import pytesseract
from PIL import Image

# Set the path to the Tesseract-OCR executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Example: Convert an image to text
img = Image.open('AA-001_1.jpg')
text = pytesseract.image_to_string(img)
print(text)
