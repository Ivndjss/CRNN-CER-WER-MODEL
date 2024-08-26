import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pytesseract
import numpy as np

# Custom Dataset
class HandwritingDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('L')
        if self.transform:
            image = self.transform(image)
        label = self.get_label(image)  # Get label using OCR
        return image, label

    def get_label(self, image):
        # Use Tesseract to extract text from image
        text = pytesseract.image_to_string(image)
        return text.strip()

# Model Definition
class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.rnn = nn.Sequential(
            nn.LSTM(128, nh, bidirectional=True),
            nn.LSTM(nh * 2, nh, bidirectional=True)
        )
        self.fc = nn.Linear(nh * 2, nclass)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output, _ = self.rnn(conv)
        output = self.fc(output)
        return output

# Training Loop
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            # Assuming labels are already in a suitable format for CTCLoss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}/{num_epochs}, Loss: {loss.item()}')

# Dataset and DataLoader
image_dir = r'C:\Users\Arcly-za B Aguinaldo\Desktop\Datasets\Datasets (first batch)'
transform = transforms.Compose([transforms.Resize((32, 128)), transforms.ToTensor()])
dataset = HandwritingDataset(image_dir, transform)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Model, Loss, Optimizer
model = CRNN(imgH=32, nc=1, nclass=37, nh=256)  # Adjust 'nclass' based on your character set
criterion = nn.CTCLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# Function to Convert Dataset to IAM Format
def convert_to_iam_format(image_dir, model, transform):
    iam_data_dir = "IAM_dataset"
    os.makedirs(iam_data_dir, exist_ok=True)
    
    for idx, img_file in enumerate(os.listdir(image_dir)):
        if img_file.endswith('.jpg'):
            img_path = os.path.join(image_dir, img_file)
            image = Image.open(img_path).convert('L')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            output = model(image)
            pred_text = decode_output(output)  # Decode the model's output into text
            
            # Create IAM format file
            iam_file_path = os.path.join(iam_data_dir, f'{img_file}.txt')
            with open(iam_file_path, 'w') as f:
                f.write(pred_text)

def decode_output(output):
    # This function will decode the CRNN model output into readable text
    # Implement decoding logic here (e.g., Greedy/Beam Search, etc.)
    decoded_text = "decoded text from model output"
    return decoded_text

# Convert to IAM format after training
convert_to_iam_format(image_dir, model, transform)
