"""
Train and evaluate the improved EfficientNet-based DR classifier from the terminal.
Format: py -3 src/run_improved.py --csv-path "C:\path\to\train.csv" --img-dir "C:\path\to\train_images"

"""

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import (
    APTOSDataset,
    compute_class_weights,
    create_weighted_sampler,
    get_improved_transforms,
)
from improved_model import ImprovedDRClassifier
from train import evaluate, print_metrics, train_improved


CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv-path', default='data/train.csv')
    parser.add_argument('--img-dir', default='data/train_images')
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--val-size', type=float, default=0.15)
    parser.add_argument('--test-size', type=float, default=0.15)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--freeze-backbone', action='store_true')
    parser.add_argument('--no-sampler', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--preprocess', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def make_loaders(args):
    df = pd.read_csv(args.csv_path)

    train_df, temp_df = train_test_split(
        df,
        test_size=(args.val_size + args.test_size),
        stratify=df['diagnosis'],
        random_state=args.seed,
    )
    relative_test = args.test_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        stratify=temp_df['diagnosis'],
        random_state=args.seed,
    )

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print(f"Train class distribution:\n{train_df['diagnosis'].value_counts().sort_index()}\n")

    train_transform, val_transform = get_improved_transforms(args.img_size)
    train_dataset = APTOSDataset(train_df, args.img_dir, transform=train_transform, preprocess=args.preprocess)
    val_dataset = APTOSDataset(val_df, args.img_dir, transform=val_transform, preprocess=args.preprocess)
    test_dataset = APTOSDataset(test_df, args.img_dir, transform=val_transform, preprocess=args.preprocess)

    sampler = None if args.no_sampler else create_weighted_sampler(train_df, num_classes=5)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, train_df


def plot_training_curves(history, output_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].plot(epochs, history['train_loss'], label='Train', marker='o', ms=3)
    axes[0].plot(epochs, history['val_loss'], label='Val', marker='s', ms=3)
    axes[0].set_title('Loss - Improved Model')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], label='Train', marker='o', ms=3)
    axes[1].plot(epochs, history['val_acc'], label='Val', marker='s', ms=3)
    axes[1].set_title('Accuracy - Improved Model')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, history['val_f1'], label='Val Macro F1', marker='^', ms=3, color='darkorange')
    axes[2].set_title('Macro F1 - Improved Model')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix(cm, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    short_names = ['No DR\n(0)', 'Mild\n(1)', 'Moderate\n(2)', 'Severe\n(3)', 'Proliferative\n(4)']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_title('Confusion Matrix - Improved Model (Test Set)', fontsize=13)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    set_seed(args.seed)
    device = get_device()
    print(f'Using device: {device}')

    train_loader, val_loader, test_loader, train_df = make_loaders(args)
    class_weights = compute_class_weights(train_df, num_classes=5).to(device)
    print('Class weights:')
    for name, weight in zip(CLASS_NAMES, class_weights.cpu().tolist()):
        print(f'  {name}: {weight:.3f}')

    model = ImprovedDRClassifier(
        num_classes=5,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable parameters: {trainable_params:,}')

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = train_improved(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        use_amp=not args.no_amp,
    )

    model_path = os.path.join(args.results_dir, 'improved_efficientnet.pth')
    torch.save(model.state_dict(), model_path)
    print(f'Saved model to {model_path}')

    plot_training_curves(history, os.path.join(args.results_dir, 'improved_training_curves.png'))

    test_loss, test_acc, test_labels, test_preds = evaluate(
        model,
        test_loader,
        criterion,
        device,
        use_amp=not args.no_amp,
    )
    print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}')
    cm = print_metrics(test_labels, test_preds)
    plot_confusion_matrix(cm, os.path.join(args.results_dir, 'improved_confusion_matrix.png'))


if __name__ == '__main__':
    main()
