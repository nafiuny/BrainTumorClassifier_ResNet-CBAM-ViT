import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class BrainTumorDataset(Dataset):
    def __init__(self, x_path, y_path):
        self.images = torch.tensor(np.load(x_path), dtype=torch.float32)
        self.labels = torch.tensor(np.load(y_path), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

def get_dataloaders(x_train_path, y_train_path, x_val_path, y_val_path, batch_size=32, num_workers=2):
    train_dataset = BrainTumorDataset(x_train_path, y_train_path)
    val_dataset = BrainTumorDataset(x_val_path, y_val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

