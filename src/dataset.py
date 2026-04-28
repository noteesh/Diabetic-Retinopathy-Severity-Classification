"""
APTOS 2019 Dataset loader for Diabetic Retinopathy Severity Classification.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


class APTOSDataset(Dataset):
    """APTOS 2019 Blindness Detection dataset."""

    def __init__(self, df, img_dir, transform=None):
        """
        Args:
            df: DataFrame with 'id_code' and 'diagnosis' columns
            img_dir: Path to directory containing images
            transform: Optional transform to apply to images
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.loc[idx, 'id_code'] + '.png')
        # Try .jpeg if .png not found
        if not os.path.exists(img_name):
            img_name = os.path.join(self.img_dir, self.df.loc[idx, 'id_code'] + '.jpeg')
        if not os.path.exists(img_name):
            img_name = os.path.join(self.img_dir, self.df.loc[idx, 'id_code'] + '.jpg')

        image = Image.open(img_name).convert('RGB')
        label = int(self.df.loc[idx, 'diagnosis'])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(img_size=224):
    """Return basic train and val transforms — intentionally minimal for baseline."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


def get_improved_transforms(img_size=224):
    """
    Richer augmentation pipeline suited for retinal fundus images.

    Additions over baseline:
    - Vertical flip (valid: fundus images have no canonical orientation)
    - Rotation up to 15 deg (cameras/patients vary)
    - ColorJitter: accounts for inter-camera brightness/contrast differences
    - RandomResizedCrop: mild scale and crop variation
    """
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),  # slight oversize before crop
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return train_transform, val_transform


def compute_class_weights(train_df, num_classes=5):
    """
    Inverse-frequency class weights for CrossEntropyLoss.
    Returns a float32 tensor of shape (num_classes,).
    """
    counts = (
        train_df['diagnosis']
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .sort_index()
    )
    total = len(train_df)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts.values > 0
    weights[nonzero] = total / (num_classes * counts.values[nonzero].astype(float))
    return torch.tensor(weights, dtype=torch.float32)


def create_weighted_sampler(train_df, num_classes=5):
    """
    Build a WeightedRandomSampler so minority classes are seen more often.
    """
    class_weights = compute_class_weights(train_df, num_classes=num_classes)
    sample_weights = class_weights[train_df['diagnosis'].to_numpy()].double()
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def load_data(csv_path, img_dir, img_size=224, batch_size=32, val_size=0.15, test_size=0.15, seed=42):
    """
    Load APTOS dataset and return DataLoaders for train/val/test splits.

    Uses stratified splitting to preserve class distribution.
    No oversampling — intentionally left out for baseline to highlight class imbalance effect.
    """
    df = pd.read_csv(csv_path)

    # Stratified train/val/test split
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df['diagnosis'], random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test, stratify=temp_df['diagnosis'], random_state=seed
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train class distribution:\n{train_df['diagnosis'].value_counts().sort_index()}\n")

    train_transform, val_transform = get_transforms(img_size)

    train_dataset = APTOSDataset(train_df, img_dir, transform=train_transform)
    val_dataset = APTOSDataset(val_df, img_dir, transform=val_transform)
    test_dataset = APTOSDataset(test_df, img_dir, transform=val_transform)

    import platform
    # num_workers > 0 causes slowdowns on macOS with MPS; pin_memory not supported on MPS
    num_workers = 0 if platform.system() == 'Darwin' else 2
    pin_memory = False if platform.system() == 'Darwin' else True

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader, test_loader, train_df, val_df, test_df
