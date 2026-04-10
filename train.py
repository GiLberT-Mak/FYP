import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
import os
import glob
import argparse
import csv
import time
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit
import re

from config import (device, DATA_FOLDER, MODEL_DIR, TRAIN_RECORD_DIR,
                    BATCH_SIZE, NUM_OUTPUTS, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)
from model import TunedSNN
from dataset import LoadDataset

sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def natural_sort_key(s):
    """
    Sort strings containing numbers in numerical order (1, 2, 10 instead of 1, 10, 2).
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]


def stratified_split(dataset, val_ratio=0.2, seed=42):
    """
    Stratified train/validation split that preserves class distribution.

    Uses sklearn StratifiedShuffleSplit so every gesture class is
    proportionally represented in both subsets.

    Returns:
        train_subset, val_subset : torch.utils.data.Subset objects
    """
    labels   = dataset.get_labels()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idxs, val_idxs = next(splitter.split(np.zeros(len(labels)), labels))
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)


def compute_class_weights(dataset, num_classes, device):
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Classes with fewer samples receive a higher weight, countering the
    natural imbalance between gesture classes and the rest class.

    Returns:
        weights : float32 tensor of shape [num_classes] on `device`
    """
    labels  = dataset.get_labels()
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls in range(num_classes):
        if cls in counts:
            weights[cls] = total / (num_classes * counts[cls])
        # Classes absent from training data get weight 0
    return weights.to(device)


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train_single_model(target_filename):
    print(f"Running on: {device}")

    os.makedirs(MODEL_DIR,        exist_ok=True)
    os.makedirs(TRAIN_RECORD_DIR, exist_ok=True)

    base_name        = os.path.splitext(target_filename)[0]
    model_save_path  = os.path.join(MODEL_DIR,       f"snn_nina_trained_{base_name}.pth")
    csv_path         = os.path.join(TRAIN_RECORD_DIR, f"training_{base_name}.csv")

    # ── Dataset ───────────────────────────────────────────────
    full_dataset = LoadDataset(DATA_FOLDER, is_training=True,
                               target_filename=target_filename, augment=True)
    if len(full_dataset) == 0:
        return

    # Stratified 80/20 split
    train_dataset, val_dataset = stratified_split(full_dataset, val_ratio=0.2)
    print(f"   Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────
    net = TunedSNN().to(device)

    # Class-weighted loss to handle gesture imbalance
    class_weights = compute_class_weights(full_dataset, NUM_OUTPUTS, device)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    # CosineAnnealingLR smoothly anneals the LR over the full training budget
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    use_amp = (device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Training loop ─────────────────────────────────────────
    print(f"\n Starting Training  (patience={EARLY_STOPPING_PATIENCE} epochs)…")
    best_val_acc    = 0.0
    patience_counter = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'val_lat_ms'])

        for epoch in range(NUM_EPOCHS):

            # ── Train ──────────────────────────────────────────
            net.train()
            batch_loss, batch_acc, total_batches = 0.0, 0.0, 0

            for i, (data, targets) in enumerate(train_loader):
                data, targets = data.to(device), targets.to(device)
                data          = data.permute(1, 0, 2)   # [Time, Batch, Ch]

                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast('cuda'):
                        mem_rec = net(data)
                        loss    = loss_fn(mem_rec.mean(dim=0), targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    mem_rec = net(data)
                    loss    = loss_fn(mem_rec.mean(dim=0), targets)
                    loss.backward()
                    optimizer.step()

                _, pred   = torch.max(mem_rec.mean(dim=0), 1)
                acc        = (pred == targets).float().mean() * 100
                batch_loss += loss.item()
                batch_acc  += acc.item()
                total_batches += 1

                if i % 10 == 0:
                    print(f"   [Ep {epoch}] Batch {i} | Loss: {loss.item():.4f}", end='\r')

            scheduler.step()

            # ── Validate ───────────────────────────────────────
            net.eval()
            val_acc, val_loss, val_batches = 0.0, 0.0, 0
            val_total_time   = 0.0   # total inference wall-time (seconds)
            val_total_samples = 0

            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    data          = data.permute(1, 0, 2)
                    batch_sz      = targets.size(0)

                    # ── Timed forward pass ─────────────────────
                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()

                    mem_rec = net(data)

                    if device.type == 'cuda':
                        torch.cuda.synchronize()
                    val_total_time += time.perf_counter() - t0
                    val_total_samples += batch_sz
                    # ───────────────────────────────────────────

                    loss    = loss_fn(mem_rec.mean(dim=0), targets)

                    _, pred = torch.max(mem_rec.mean(dim=0), 1)
                    acc     = (pred == targets).float().mean() * 100

                    val_loss += loss.item()
                    val_acc  += acc.item()
                    val_batches += 1

            if total_batches > 0 and val_batches > 0:
                avg_tr_loss  = batch_loss  / total_batches
                avg_tr_acc   = batch_acc   / total_batches
                avg_val_loss = val_loss    / val_batches
                avg_val_acc  = val_acc     / val_batches
                current_lr   = optimizer.param_groups[0]['lr']
                avg_lat_ms   = (val_total_time / val_total_samples * 1000) if val_total_samples > 0 else 0.0

                lat_status = '✓' if avg_lat_ms < 0.05 else '✗'
                print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {avg_tr_loss:.4f} | Train Acc: {avg_tr_acc:6.2f}% | "
                    f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:6.2f}% | "
                    f"Latency: {avg_lat_ms:.4f} ms/sample [{lat_status}] | "
                    f"LR: {current_lr:.6f}"
                )

                # Log to CSV
                writer.writerow([epoch,
                                  f"{avg_tr_loss:.4f}", f"{avg_tr_acc:.2f}",
                                  f"{avg_val_loss:.4f}", f"{avg_val_acc:.2f}",
                                  f"{current_lr:.6f}", f"{avg_lat_ms:.4f}"])
                csv_file.flush()

                # Best model checkpoint + early stopping
                if avg_val_acc > best_val_acc:
                    best_val_acc      = avg_val_acc
                    patience_counter  = 0
                    torch.save(net.state_dict(), model_save_path)
                    print(f"   New best saved → {os.path.basename(model_save_path)} "
                          f"(Val Acc: {best_val_acc:.2f}%)")
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"\n    Early stopping at epoch {epoch} "
                              f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs).")
                        break

    print(f"\n Training complete. Best Val Acc: {best_val_acc:.2f}%")
    print(f"    Training log saved → {csv_path}")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SNN models on NinaProDB patient data")
    parser.add_argument('--file', type=str,
                        help='Target .mat file in Data/ to train on a specific patient.')
    parser.add_argument('--all', action='store_true',
                        help='Train sequentially on all .mat files in Data/.')
    args = parser.parse_args()

    if args.all:
        files = glob.glob(os.path.join(DATA_FOLDER, "*.mat"))
        if not files:
            print(" No .mat files found in Data/ directory.")
        for f_path in sorted(files, key=natural_sort_key):
            filename = os.path.basename(f_path)
            print(f"\n\n{'='*60}")
            print(f" Starting training for patient: {filename}")
            print(f"{'='*60}")
            train_single_model(filename)

    elif args.file:
        train_single_model(args.file)

    else:
        print(" Please provide exactly one of `--file <filename>` or `--all`")