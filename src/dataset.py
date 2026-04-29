import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

def _crop_fundus(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_np
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img_np[y:y + h, x:x + w]

def preprocess_fundus(img_np, sigmaX=10):
    img_np = _crop_fundus(img_np)
    blurred = cv2.GaussianBlur(img_np, (0, 0), sigmaX)
    img_np = cv2.addWeighted(img_np, 4, blurred, -4, 128)
    return np.clip(img_np, 0, 255).astype(np.uint8)

class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, preprocess=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.preprocess = preprocess

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, 'id_code']
        img_name = None
        # Support multiple common extensions
        for ext in ['.png', '.jpeg', '.jpg']:
            temp_path = os.path.join(self.img_dir, id_code + ext)
            if os.path.exists(temp_path):
                img_name = temp_path
                break
        
        if img_name is None:
            raise FileNotFoundError(f"Image not found: {id_code}")

        image = Image.open(img_name).convert('RGB')
        label = int(self.df.loc[idx, 'diagnosis'])
        
        if self.preprocess:
            image = Image.fromarray(preprocess_fundus(np.array(image)))
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_transforms(img_size=224):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform

def get_improved_transforms(img_size=224):
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        transforms.Resize((img_size + 16, img_size + 16)),
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])
    return train_transform, val_transform

def compute_class_weights(train_df, num_classes=5):
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
    class_weights = compute_class_weights(train_df, num_classes=num_classes)
    sample_weights = class_weights[train_df['diagnosis'].to_numpy()].double()
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def load_data(csv_path, img_dir, img_size=224, batch_size=32, val_size=0.15, test_size=0.15, seed=42):
    df = pd.read_csv(csv_path)
    
    # Split into train and "temp" (val + test)
    train_df, temp_df = train_test_split(
        df, test_size=(val_size + test_size), stratify=df['diagnosis'], random_state=seed
    )
    
    # Split "temp" into validation and test
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, stratify=temp_df['diagnosis'], random_state=seed
    )
    
    train_transform, val_transform = get_transforms(img_size)
    
    train_dataset = APTOSDataset(train_df, img_dir, transform=train_transform)
    val_dataset = APTOSDataset(val_df, img_dir, transform=val_transform)
    test_dataset = APTOSDataset(test_df, img_dir, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    return train_loader, val_loader, test_loader, train_df, val_df, test_df
