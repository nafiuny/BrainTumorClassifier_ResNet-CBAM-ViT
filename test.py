import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import argparse
import time
from preprocess_dataset import BrainTumorDataset
from model import create_model

# -------- آرگومان‌ها -------- #
parser = argparse.ArgumentParser(description="Test a trained model on test dataset")
parser.add_argument('--x_test', type=str, default='data_preprocessed/X_test_final.npy',
                    help='Path to X_test.npy file')
parser.add_argument('--y_test', type=str, default='data_preprocessed/y_test_final.npy',
                    help='Path to y_test.npy file')
parser.add_argument('--checkpoint', type=str, default='checkpoints/model_epoch_70.pth',
                    help='Path to model checkpoint (.pth)')

args = parser.parse_args()

# -------- دستگاه پردازش -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# -------- بارگذاری داده‌ها -------- #
test_dataset = BrainTumorDataset(args.x_test, args.y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------- انتخاب مدل -------- #
model = create_model(num_classes=4).to(device)

# -------- لود کردن وزن مدل -------- #
checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# -------- شروع تست -------- #
start_time = time.time()
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# -------- پایان زمان‌گیری -------- #
end_time = time.time()
inference_time = end_time - start_time
avg_time_per_image = inference_time / len(test_loader.dataset)

# -------- ارزیابی -------- #
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

accuracy = accuracy_score(all_labels, all_preds)
print(f"\n🕒 Total Inference Time: {inference_time:.2f} sec")
print(f"⏱ Average Time per Image: {avg_time_per_image:.4f} sec")
print(f"✅ Accuracy: {accuracy * 100:.2f}%\n")
print("📋 Classification Report:")
print(classification_report(all_labels, all_preds))
print("📊 Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
