# src/dataset.py (FIXED)
# PyTorch dataset wrapper for RetinaMNIST (medmnist).
# Robust: handles when medmnist returns PIL.Image or numpy array.

from medmnist import RetinaMNIST
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import numpy as np

class RetinaRegressionDataset(Dataset):
    def __init__(self, split='train', img_size=64):
        """
        split: 'train', 'val' or 'test' (medmnist usually has 'train','val','test' splits)
        img_size: resize images to square size
        """
        assert split in ('train', 'val', 'test'), "split must be 'train','val' or 'test'"
        self.ds = RetinaMNIST(split=split, download=True)
        # transforms: assume input is PIL.Image or numpy array
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # output in [0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]  # img may be PIL.Image or numpy array
        # normalize input type: if numpy array, convert to PIL first
        if isinstance(img, np.ndarray):
            # medmnist sometimes returns HxWxC numpy array (uint8)
            img = Image.fromarray(img)
        elif not isinstance(img, Image.Image):
            # fallback: try to convert whatever to PIL
            img = Image.fromarray(np.array(img))

        img = self.transform(img)  # now safe
        target = float(label[0])   # label shape (1,) -> scalar
        return img, torch.tensor([target], dtype=torch.float32)

def get_loaders(batch_size=64, img_size=64, num_workers=2):
    train_ds = RetinaRegressionDataset(split='train', img_size=img_size)
    val_ds = RetinaRegressionDataset(split='val', img_size=img_size)
    test_ds = RetinaRegressionDataset(split='test', img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
