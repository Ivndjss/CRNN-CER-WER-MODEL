import torch
import torchvision
from PIL import Image
import numpy as np
import pytesseract

print("Torch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)
print("Pillow version:", Image.__version__)
print("NumPy version:", np.__version__)
print("Pytesseract version:", pytesseract.get_tesseract_version())
