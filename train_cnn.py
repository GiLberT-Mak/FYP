import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import sys
import os
import glob
import argparse
import csv
import time
import re
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedShuffleSplit

from config import (device, DATA_FOLDER, TRAIN_RECORD_DIR,
                    BATCH_SIZE, NUM_OUTPUTS, NUM_EPOCHS, EARLY_STOPPING_PATIENCE)
from model_cnn import MirrorCNN
from dataset import LoadDataset

sys.stdout.reconfigure(encoding='utf-8')

# Separate directory for CNN models
MODEL_DIR_CNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trained_CNN')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def stratified_split(dataset, val_ratio=0.2, seed=42):
    labels   = dataset.get_labels()
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idxs, val_idxs = next(splitter.split(np.zeros(len(labels)), labels))
    return Subset(dataset, train_idxs), Subset(dataset, val_idxs)

def compute_class_weights(dataset, num_classes, device):
    labels  = dataset.get_labels()
    counts  = Counter(labels)
    total   = sum(counts.values())
    weights = torch.zeros(num_classes)
    for cls in range(num_classes):
        if cls in counts:
            weights[cls] = total / (num_classes * counts[cls])
    return weights.to(device)

def train_single_model(target_filename):
    print(f"Running CNN training on: {device}")

    os.makedirs(MODEL_DIR_CNN,    exist_ok=True)
    os.makedirs(TRAIN_RECORD_DIR, exist_ok=True)

    base_name        = os.path.splitext(target_filename)[0]
    model_save_path  = os.path.join(MODEL_DIR_CNN,   f"cnn_nina_trained_{base_name}.pth")
    csv_path         = os.path.join(TRAIN_RECORD_DIR, f"training_cnn_{base_name}.csv")

    full_dataset = LoadDataset(DATA_FOLDER, is_training=True,
                               target_filename=target_filename, augment=True)
    if len(full_dataset) == 0:
        return

    train_dataset, val_dataset = stratified_split(full_dataset, val_ratio=0.2)
    print(f"   Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True, drop_last=True, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, drop_last=False, num_workers=0)

    net = MirrorCNN().to(device)

    class_weights = compute_class_weights(full_dataset, NUM_OUTPUTS, device)
    loss_fn       = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    print(f"\n Starting CNN Training (patience={EARLY_STOPPING_PATIENCE})…")
    best_val_acc = 0.0
    patience_counter = 0

    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'])

        for epoch in range(NUM_EPOCHS):
            net.train()
            batch_loss, batch_acc, total_batches = 0.0, 0.0, 0

            for i, (data, targets) in enumerate(train_loader):
                # data is [Batch, Time, Channels]
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = net(data)
                loss    = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

                _, pred = torch.max(outputs, 1)
                acc      = (pred == targets).float().mean() * 100
                batch_loss += loss.item()
                batch_acc  += acc.item()
                total_batches += 1

                if i % 10 == 0:
                    print(f"   [Ep {epoch}] Batch {i} | Loss: {loss.item():.4f}", end='\r')

            scheduler.step()

            net.eval()
            val_acc, val_loss, val_batches = 0.0, 0.0, 0
            with torch.no_grad():
                for data, targets in val_loader:
                    data, targets = data.to(device), targets.to(device)
                    outputs = net(data)
                    loss    = loss_fn(outputs, targets)
                    _, pred = torch.max(outputs, 1)
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

                print(f"Epoch {epoch:3d} | Train Acc: {avg_tr_acc:6.2f}% | Val Acc: {avg_val_acc:6.2f}% | LR: {current_lr:.6f}")
                writer.writerow([epoch, f"{avg_tr_loss:.4f}", f"{avg_tr_acc:.2f}", f"{avg_val_loss:.4f}", f"{avg_val_acc:.2f}", f"{current_lr:.6f}"])
                csv_file.flush()

                if avg_val_acc > best_val_acc:
                    best_val_acc = avg_val_acc
                    patience_counter = 0
                    torch.save(net.state_dict(), model_save_path)
                    print(f"   New best CNN saved → {os.path.basename(model_save_path)} ({best_val_acc:.2f}%)")
                else:
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print(f"\n Early stopping at epoch {epoch}.")
                        break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MirrorCNN on NinaProDB patient data")
    parser.add_argument('--file', type=str, help='Target .mat file.')
    parser.add_argument('--all', action='store_true', help='Train on all files.')
    args = parser.parse_args()

    if args.all:
        files = glob.glob(os.path.join(DATA_FOLDER, "*.mat"))
        for f_path in sorted(files, key=natural_sort_key):
            train_single_model(os.path.basename(f_path))
    elif args.file:
        train_single_model(args.file)
    else:
        print("Provide --file or --all")
