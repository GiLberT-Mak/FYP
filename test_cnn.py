import torch
import os
import argparse
import sys
import csv
import time
import glob
import re
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from config import device, DATA_FOLDER, NUM_OUTPUTS, BATCH_SIZE, SUMMARY_DIR
from model_cnn import MirrorCNN
from dataset import SingleFileLoader

MODEL_DIR_CNN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Trained_CNN')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def run_test(target_file):
    print(f"Testing CNN model on device: {device}")
    
    base_name = os.path.splitext(target_file)[0]
    model_path = os.path.join(MODEL_DIR_CNN, f"cnn_nina_trained_{base_name}.pth")
    
    if not os.path.exists(model_path):
        print(f" Error: Model not found at {model_path}")
        return

    net = MirrorCNN().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()
    print("CNN Model loaded.")

    test_dataset = SingleFileLoader(DATA_FOLDER, target_file)
    if len(test_dataset) == 0:
        return
    
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f" Loading Test File: {target_file} | Samples: {len(test_dataset)}")

    all_preds   = []
    all_targets = []
    total_infer_time = 0.0
    total_infer_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            # data is [Batch, Time, Channels]
            
            t0 = time.perf_counter()
            outputs = net(data)
            total_infer_time += time.perf_counter() - t0
            total_infer_samples += targets.size(0)
            
            _, pred = torch.max(outputs, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Calculate metrics
    active_mask = (all_targets != 0)
    active_acc = np.mean(all_preds[active_mask] == all_targets[active_mask]) * 100 if np.any(active_mask) else 0.0
    overall_acc = np.mean(all_preds == all_targets) * 100
    avg_lat_ms = (total_infer_time / total_infer_samples * 1000) if total_infer_samples > 0 else 0.0

    print(f" Active Acc: {active_acc:.2f}% | Overall Acc: {overall_acc:.2f}%")
    print(f" Avg Latency: {avg_lat_ms:.4f} ms/sample")

    # Append to a dedicated CNN summary
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_DIR, "cnn_results_summary.csv")
    file_exists = os.path.isfile(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['patient_file', 'overall_acc_%', 'active_acc_%', 'latency_ms_per_sample'])
        writer.writerow([target_file, f"{overall_acc:.2f}", f"{active_acc:.2f}", f"{avg_lat_ms:.4f}"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test MirrorCNN models")
    parser.add_argument('--file', type=str, help='Target .mat file.')
    parser.add_argument('--all', action='store_true', help='Test all trained models.')
    args = parser.parse_args()

    if args.all:
        model_files = glob.glob(os.path.join(MODEL_DIR_CNN, "cnn_nina_trained_*.pth"))
        mat_names = [os.path.basename(mf).replace("cnn_nina_trained_", "").replace(".pth", ".mat") for mf in model_files]
        for m in sorted(mat_names, key=natural_sort_key):
            run_test(m)
    elif args.file:
        run_test(args.file)
    else:
        print("Provide --file or --all")
