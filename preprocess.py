import os
import argparse
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pywt

# Parameters
parser = argparse.ArgumentParser(description="Preprocess brain tumor MRI images.")
parser.add_argument('--train_dir', type=str, help="Path to training data")
parser.add_argument('--test_dir', type=str, help="Path to testing data")
parser.add_argument('--output_dir', type=str, required=True, help="Directory to save processed files")
parser.add_argument('--batch_size', type=int, default=200, help="Batch size for saving intermediate files")
args = parser.parse_args()

# data path
data_dir = args.train_dir if args.train_dir else args.test_dir
mode = "train" if args.train_dir else "test"

img_size = (224, 224)
batch_size = args.batch_size
save_dir = args.output_dir
os.makedirs(save_dir, exist_ok=True)


# Wavelet transform
def wavelet_transform(image):
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    LL_resized = cv2.resize(LL, img_size)
    return LL_resized

# Apply wavelet to all images
def process_with_wavelet(images):
    processed_images = []
    for img in images:
        img_wavelet = wavelet_transform(img)
        img_wavelet_rgb = np.repeat(img_wavelet[..., np.newaxis], 3, axis=-1)
        processed_images.append(img_wavelet_rgb)
    return np.array(processed_images)

# Start preprocessing
X_data, y_data = [], []
batch_count = 0
classes = os.listdir(data_dir)

for label, category in enumerate(classes):
    img_dir = os.path.join(data_dir, category)
    img_files = os.listdir(img_dir)

    for img_file in img_files:
        img_path = os.path.join(img_dir, img_file)
        img = load_img(img_path, target_size=img_size)
        img = img_to_array(img) / 255.0
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        img_wavelet = process_with_wavelet([img_gray])[0]
        img_wavelet = np.transpose(img_wavelet, (2, 0, 1))

        X_data.append(img_wavelet)
        y_data.append(label)

        if len(X_data) >= batch_size:
            np.save(f"{save_dir}/X_{mode}_batch_{batch_count}.npy", np.array(X_data))
            np.save(f"{save_dir}/y_{mode}_batch_{batch_count}.npy", np.array(y_data))
            print(f"Batch {batch_count} saved.")
            X_data, y_data = [], []
            batch_count += 1

if len(X_data) > 0:
    np.save(f"{save_dir}/X_{mode}_batch_{batch_count}.npy", np.array(X_data))
    np.save(f"{save_dir}/y_{mode}_batch_{batch_count}.npy", np.array(y_data))
    print(f"Final batch {batch_count} saved.")

# Merge batches
X_batches, y_batches = [], []
for i in range(batch_count + 1):
    X = np.load(f"{save_dir}/X_{mode}_batch_{i}.npy")
    y = np.load(f"{save_dir}/y_{mode}_batch_{i}.npy")
    X_batches.append(X)
    y_batches.append(y)
    os.remove(f"{save_dir}/X_{mode}_batch_{i}.npy")
    os.remove(f"{save_dir}/y_{mode}_batch_{i}.npy")
    print(f"Batch {i} deleted.")

X_final = np.concatenate(X_batches, axis=0)
y_final = np.concatenate(y_batches, axis=0)

np.save(f"{save_dir}/X_{mode}_final.npy", X_final)
np.save(f"{save_dir}/y_{mode}_final.npy", y_final)

print("✅ All batches merged and saved.")
print(f"Final {mode.capitalize()} Samples: {X_final.shape[0]}")
