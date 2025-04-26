import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split

# پارس کردن آرگومان‌ها
parser = argparse.ArgumentParser(description="Split training data into train and validation sets.")
parser.add_argument('--x_path', type=str, required=True, help="Path to X_train_final.npy")
parser.add_argument('--y_path', type=str, required=True, help="Path to y_train_final.npy")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save split data")
parser.add_argument('--val_ratio', type=float, default=0.2, help="Validation set ratio (default=0.2)")

args = parser.parse_args()

# بارگذاری داده‌ها
X = np.load(args.x_path)
y = np.load(args.y_path)

# تقسیم داده‌ها
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=args.val_ratio, random_state=42)

# ذخیره داده‌های تقسیم‌شده
os.makedirs(args.output_dir, exist_ok=True)
np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
np.save(os.path.join(args.output_dir, "X_val.npy"), X_val)
np.save(os.path.join(args.output_dir, "y_val.npy"), y_val)

# چاپ اطلاعات
print("✅ Data has been split into train and validation sets.")
print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
