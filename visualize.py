import os
import argparse
import torch 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from model import create_model
import pandas as pd

# Parameters
parser = argparse.ArgumentParser(description="Visualize evaluation results and metrics.")
parser.add_argument('--checkpoint_path', type=str, default='checkpoints/model_epoch_70.pth',
                    help="Path to the model checkpoint file")
parser.add_argument('--data_dir', type=str, default='data_preprocessed',
                    help="Directory containing X_test_final.npy and y_test_final.npy")
parser.add_argument('--output_dir', type=str, default='plots',
                    help="Directory to save plots")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32

# Load test data
x_test = np.load(os.path.join(args.data_dir, 'X_test_final.npy'))
y_test = np.load(os.path.join(args.data_dir, 'y_test_final.npy'))

x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load model
model = create_model(num_classes=4).to(device)
checkpoint = torch.load(args.checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Model prediction
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.to(device))
        _, preds = torch.max(outputs, 1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Evaluation
accuracy = accuracy_score(all_labels, all_preds)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

conf_matrix = confusion_matrix(all_labels, all_preds)
print("\n Confusion Matrix:")
print(conf_matrix)


os.makedirs(args.output_dir, exist_ok=True)

class_names = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# Confusion matrix plot
conf_matrix_path = os.path.join(args.output_dir, 'confusion_matrix_full.png')
plt.figure(figsize=(12, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 18})
plt.title('Confusion Matrix', fontsize=18)
plt.xlabel('Predicted Labels', fontsize=16)
plt.ylabel('True Labels', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.savefig(conf_matrix_path)
plt.show()
print(f"Confusion matrix saved to: {conf_matrix_path}")


if 'train_accuracies' in checkpoint:
    train_acc_path = os.path.join(args.output_dir, 'train_val_acc.png')
    train_loss_path = os.path.join(args.output_dir, 'train_val_loss.png')

    plt.figure(figsize=(12, 7))
    plt.plot(checkpoint['train_accuracies'], label='Train Accuracy')
    plt.plot(checkpoint['val_accuracies'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(train_acc_path)
    plt.show()
    print(f"📈 Accuracy plot saved to: {train_acc_path}")

    plt.figure(figsize=(12, 7))
    plt.plot(checkpoint['train_losses'], label='Train Loss')
    plt.plot(checkpoint['val_losses'], label='Validation Loss')
    plt.title('Loss over Epochs', fontsize=18)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.savefig(train_loss_path)
    plt.show()
    print(f"Loss plot saved to: {train_loss_path}")
else:
    print(" No training/validation metrics found in checkpoint.")

#Table of results for each class
class_report = classification_report(all_labels, all_preds, output_dict=True)
df = pd.DataFrame(class_report).transpose()
print("\nClass-wise Metrics Table:")
print(df.round(2))

# Metrics chart
bar_plot_path = os.path.join(args.output_dir, 'final_metrics_barplot.png')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
values = [
    accuracy * 100,
    df.loc['weighted avg']['precision'] * 100,
    df.loc['weighted avg']['recall'] * 100,
    df.loc['weighted avg']['f1-score'] * 100
]
colors = ['#4C72B0', '#5A82C1', '#6B93CF', '#7BA3DE']

plt.figure(figsize=(10, 7))
bars = plt.bar(metrics, values, color=colors, edgecolor='black', width=0.5)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f'{height:.2f}%', ha='center', fontsize=16)

plt.title('Final Evaluation Metrics of Model', fontsize=18)
plt.ylabel('Percentage (%)', fontsize=16)
plt.ylim(90, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.tight_layout()
plt.savefig(bar_plot_path)
plt.show()
print(f"Final metrics barplot saved to: {bar_plot_path}")
