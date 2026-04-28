"""
Training and evaluation utilities for the baseline DR classification model.
"""

import time
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)


CLASS_NAMES = ['No DR (0)', 'Mild (1)', 'Moderate (2)', 'Severe (3)', 'Proliferative (4)']


def _maybe_autocast(device, use_amp):
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, use_amp=False, max_grad_norm=None):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with _maybe_autocast(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with _maybe_autocast(device, use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, device, num_epochs=20):
    """Full training loop. Returns history dict."""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

    print(f"\nBest Val Accuracy: {best_val_acc:.4f}")
    model.load_state_dict(best_state)
    return history


def train_improved(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler,
    device,
    num_epochs=30,
    use_amp=True,
    max_grad_norm=1.0,
):
    """
    Training loop for the improved model.

    Key differences from train():
    - Best checkpoint is saved by macro F1, not accuracy (right metric for imbalanced classes)
    - scheduler.step() called with no argument (compatible with CosineAnnealingLR)
    - val_f1 is logged each epoch
    """
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [],   'val_acc': [],  'val_f1': []
    }
    best_val_f1 = 0.0
    best_state = None
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            use_amp=use_amp,
            max_grad_norm=max_grad_norm,
        )
        val_loss, val_acc, val_labels, val_preds = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
        )

        val_f1 = f1_score(val_labels, val_preds, average='macro')

        if scheduler is not None:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
              f"Time: {elapsed:.1f}s")

    print(f"\nBest Val Macro F1: {best_val_f1:.4f}")
    model.load_state_dict(best_state)
    return history


def print_metrics(labels, preds):
    """Print full classification report and confusion matrix."""
    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(labels, preds)
    # Print with row/col labels
    header = "       " + "  ".join(f"{i:>6}" for i in range(5))
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>6}" for v in row)
        print(f"  [{i}]  {row_str}")

    macro_f1 = f1_score(labels, preds, average='macro')
    weighted_f1 = f1_score(labels, preds, average='weighted')
    print(f"\nMacro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    return cm
