import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from model import create_model  # یا هر مدلی که می‌خواهی استفاده کنی
from preprocess_dataset import get_dataloaders

# -------- دریافت ورودی از خط فرمان -------- #
parser = argparse.ArgumentParser(description="Train a brain tumor classifier model.")
parser.add_argument('--x_train', type=str, required=True, help="Path to X_train.npy")
parser.add_argument('--y_train', type=str, required=True, help="Path to y_train.npy")
parser.add_argument('--x_val', type=str, required=True, help="Path to X_val.npy")
parser.add_argument('--y_val', type=str, required=True, help="Path to y_val.npy")
parser.add_argument('--epochs', type=int, default=80, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help="Directory to save model checkpoints")

args = parser.parse_args()

# -------- تنظیمات پایه -------- #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.checkpoint_dir, exist_ok=True)

# -------- بارگذاری داده‌ها -------- #
train_loader, val_loader = get_dataloaders(
    args.x_train, args.y_train, args.x_val, args.y_val, batch_size=args.batch_size
)

# -------- تعریف مدل -------- #
model = create_model(num_classes=4).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# -------- بررسی checkpoint قبلی -------- #
start_epoch = 0
checkpoint_files = [f for f in os.listdir(args.checkpoint_dir) if f.endswith('.pth')]
if checkpoint_files:
    latest_ckpt = sorted(checkpoint_files)[-1]
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, latest_ckpt), map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"🔄 Checkpoint loaded: {latest_ckpt}")

# -------- آموزش -------- #
for epoch in range(start_epoch, args.epochs):
    model.train()
    total, correct, running_loss = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = correct / total
    train_loss = running_loss / len(train_loader)

    # -------- ارزیابی -------- #
    model.eval()
    val_correct, val_total, val_loss_total = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss_total += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_acc = val_correct / val_total
    val_loss = val_loss_total / len(val_loader)

    print(f"Epoch [{epoch+1}/{args.epochs}] | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    # -------- ذخیره مدل هر 10 epoch -------- #
    if (epoch + 1) % 10 == 0:
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }, ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

# -------- ارزیابی نهایی -------- #
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("\n✅ Train Compeleted.")

