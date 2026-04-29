import time
import numpy as np
import torch
from contextlib import nullcontext
from sklearn.metrics import f1_score, accuracy_score

def _maybe_autocast(device, use_amp):
    if use_amp and device.type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, use_amp):
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
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(loader.dataset), accuracy_score(all_labels, all_preds)

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp):
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
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return running_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), np.array(all_labels), np.array(all_preds)

def train_two_stage(model, train_loader, val_loader, optimizer_fn, criterion, device, use_amp=True):
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device.type == 'cuda')
    best_f1 = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}

    # Stage 1: Head Only
    print("=== Stage 1: training head only (10 epochs) ===")
    model.freeze_features(True)
    optimizer = optimizer_fn(model, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    for epoch in range(1, 11):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        vl, va, vlab, vpred = evaluate(model, val_loader, criterion, device, use_amp)
        vf1 = f1_score(vlab, vpred, average='macro')
        
        scheduler.step()
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va); history['val_f1'].append(vf1)
        
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, 'results/best_model.pth')
            
        print(f"S1 Epoch [{epoch:02d}/10] Train Acc: {ta:.4f} | Val Acc: {va:.4f} F1: {vf1:.4f} | {time.time()-t0:.1f}s")

    # Stage 2: Full Model
    print("\n=== Stage 2: full fine-tune (20 epochs, lower LR) ===")
    model.freeze_features(False)
    # Re-initialize optimizer for all parameters
    optimizer = optimizer_fn(model, lr=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    for epoch in range(1, 21):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, use_amp)
        vl, va, vlab, vpred = evaluate(model, val_loader, criterion, device, use_amp)
        vf1 = f1_score(vlab, vpred, average='macro')
        
        scheduler.step()
        history['train_loss'].append(tl); history['train_acc'].append(ta)
        history['val_loss'].append(vl); history['val_acc'].append(va); history['val_f1'].append(vf1)
        
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, 'results/best_model.pth')
            
        print(f"S2 Epoch [{epoch:02d}/20] Train Acc: {ta:.4f} | Val Acc: {va:.4f} F1: {vf1:.4f} | {time.time()-t0:.1f}s")

    print(f"\nBest Val Macro F1: {best_f1:.4f}")
    model.load_state_dict(best_state)
    return history
