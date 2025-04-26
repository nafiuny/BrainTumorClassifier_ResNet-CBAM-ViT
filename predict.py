import argparse
import torch
import os
import numpy as np
import cv2
from model import create_model
from PIL import Image
import pywt


class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Parameters
parser = argparse.ArgumentParser(description="Predict tumor class from an input image.")
parser.add_argument('--image_path', type=str, required=True, help="Path to input image")
parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_70.pth', help="Path to checkpoint file")
parser.add_argument('--num_classes', type=int, default=4, help="Number of classes")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Model
model = create_model(num_classes=args.num_classes).to(device)
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()


img_size = (224, 224)

# Preprocess
def preprocess_image(image_path):
    # Load image
    img = Image.open(image_path).convert('RGB')
    img = np.array(img)
    img = cv2.resize(img, img_size)

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply wavelet transform
    coeffs2 = pywt.dwt2(img_gray, 'haar')
    LL, (LH, HL, HH) = coeffs2
    LL_resized = cv2.resize(LL, img_size)

    # تبدیل به 3 کانال
    img_wavelet_rgb = np.repeat(LL_resized[..., np.newaxis], 3, axis=-1)

    # Normalize
    img_wavelet_rgb = img_wavelet_rgb / 255.0

    # Change (H,W,C) -> (C,H,W)
    img_wavelet_rgb = np.transpose(img_wavelet_rgb, (2, 0, 1))

    # Convert to tensor
    img_tensor = torch.tensor(img_wavelet_rgb, dtype=torch.float32)

    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

# Load Image
input_image = preprocess_image(args.image_path).to(device)

# predict
with torch.no_grad():
    outputs = model(input_image)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

predicted_class_idx = predicted.item()
predicted_class_name = class_names[predicted_class_idx]
confidence_percent = confidence.item() * 100

# results
# print(f"\n Predicted Class Index: {predicted_class_idx}")
print(f"\n Predicted Tumor Type: {predicted_class_name} ({confidence_percent:.2f}% confidence)\n")
