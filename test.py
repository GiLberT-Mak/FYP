import torch
import os
import argparse
import sys
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')          # Save figures without a display
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from torch.utils.data import DataLoader

from config import device, MODEL_DIR, DATA_FOLDER, NUM_OUTPUTS, BATCH_SIZE, CM_DIR, RASTER_DIR, SUMMARY_DIR
from model import TunedSNN
from dataset import SingleFileLoader

sys.stdout.reconfigure(encoding='utf-8')


# ─────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────

def save_confusion_matrix(all_targets, all_preds, target_file):
    """Save a high-resolution confusion matrix PNG to Result/."""
    cm   = confusion_matrix(all_targets, all_preds, labels=range(NUM_OUTPUTS))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_OUTPUTS))
    fig, ax = plt.subplots(figsize=(12, 12))
    disp.plot(cmap='Blues', ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — {target_file}", fontsize=14, pad=14)
    plt.tight_layout()

    os.makedirs(CM_DIR, exist_ok=True)
    save_path = os.path.join(CM_DIR, f"cm_{os.path.splitext(target_file)[0]}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved → {save_path}")
    return save_path


def save_spike_raster(spk_tensor, true_label, pred_label, target_file):
    """
    Plot and save a spike raster of the output layer for one sample.

    Args:
        spk_tensor  : [T, NUM_OUTPUTS] spike record on CPU
        true_label  : ground-truth class index (int)
        pred_label  : predicted class index (int)
        target_file : patient filename (used for titling / saving)
    """
    spk_np = spk_tensor.cpu().numpy()   # [T, C]
    T, C   = spk_np.shape

    fig, ax = plt.subplots(figsize=(14, 5))

    for neuron_idx in range(C):
        spike_times = np.where(spk_np[:, neuron_idx] > 0.5)[0]
        if len(spike_times) > 0:
            ax.scatter(spike_times,
                       np.ones_like(spike_times) * neuron_idx,
                       s=50, c='steelblue', marker='|', linewidths=2)

    ax.axhline(y=true_label, color='green',  lw=1.5, ls='--', label=f'True class : {true_label}')
    ax.axhline(y=pred_label, color='crimson', lw=1.5, ls=':',  label=f'Pred class : {pred_label}')
    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Output Neuron Index", fontsize=12)
    ax.set_title(f"Hidden Layer 3 Spike Raster — {target_file}  "
                 f"[True: {true_label} | Pred: {pred_label}]", fontsize=13)
    ax.legend(loc='upper right')
    ax.set_xlim(0, T)
    ax.set_ylim(-1, C)
    plt.tight_layout()

    os.makedirs(RASTER_DIR, exist_ok=True)
    save_path = os.path.join(RASTER_DIR,
                             f"spike_raster_{os.path.splitext(target_file)[0]}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Spike raster saved → {save_path}")


def append_summary_csv(target_file, overall_acc, active_acc):
    """Append this patient's summary to Result/Summary/results_summary.csv."""
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    summary_path = os.path.join(SUMMARY_DIR, "results_summary.csv")
    file_exists  = os.path.exists(summary_path)

    with open(summary_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['patient_file', 'overall_acc_%', 'active_acc_%'])
        writer.writerow([target_file, f"{overall_acc:.2f}", f"{active_acc:.2f}"])

    print(f"Summary appended → {summary_path}")


# ─────────────────────────────────────────────────────────────
# Main inference function
# ─────────────────────────────────────────────────────────────

def run_test(target_file):
    print(f"Running on: {device}")

    base_name       = os.path.splitext(target_file)[0]
    model_save_path = os.path.join(MODEL_DIR, f"snn_nina_trained_{base_name}.pth")

    if not os.path.exists(model_save_path):
        print(f"Trained model not found at {model_save_path}. Run train.py first.")
        return

    # ── Load model ────────────────────────────────────────────
    net = TunedSNN().to(device)
    net.load_state_dict(torch.load(model_save_path, map_location=device, weights_only=True))
    net.eval()
    print("Model loaded.")

    # ── Load test data ────────────────────────────────────────
    try:
        dataset = SingleFileLoader(DATA_FOLDER, target_file)
    except FileNotFoundError as e:
        print(e)
        return

    if len(dataset) == 0:
        print("No valid samples found in the test file.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Inference ─────────────────────────────────────────────
    print("Running inference…")
    all_preds   = []
    all_targets = []
    spike_sample = None    # (spk_tensor [T,C], true_label, pred_label) for raster

    with torch.no_grad():
        for i, (data, targets) in enumerate(loader):
            data, targets = data.to(device), targets.to(device)
            data          = data.permute(1, 0, 2)           # [Time, Batch, Ch]

            mem_out, hidden_spk = net(data, return_spikes=True)
            _, pred = torch.max(mem_out.mean(dim=0), 1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            # Capture a sample for spike raster (prefer an active gesture if possible)
            # We plot the first 50 hidden neurons since the output layer no longer spikes
            if spike_sample is None or spike_sample[1] == 0:
                for b in range(targets.size(0)):
                    lbl = int(targets[b].item())
                    if lbl > 0: # Found a gesture!
                        spike_sample = (hidden_spk[:, b, :50], lbl, int(pred[b].item()))
                        break
                # Fallback to the first sample if no active gestures in this batch yet
                if spike_sample is None:
                    spike_sample = (hidden_spk[:, 0, :50], int(targets[0].item()), int(pred[0].item()))

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ── Accuracy metrics ──────────────────────────────────────
    active_mask = all_targets > 0
    active_acc  = 0.0

    if np.sum(active_mask) > 0:
        active_acc = np.mean(all_preds[active_mask] == all_targets[active_mask]) * 100
        print(f"\n Active Accuracy  (excl. Rest) on '{target_file}': {active_acc:.2f}%")
    else:
        print(f" No active samples found in test split for '{target_file}'.")

    overall_acc = np.mean(all_preds == all_targets) * 100
    print(f"Overall Accuracy (incl. Rest) on '{target_file}': {overall_acc:.2f}%")

    # ── Per-class classification report ───────────────────────
    print(f"\n Classification Report — {target_file}")
    print(classification_report(
        all_targets, all_preds,
        labels=list(range(NUM_OUTPUTS)),
        zero_division=0
    ))

    # ── Save results ──────────────────────────────────────────
    save_confusion_matrix(all_targets, all_preds, target_file)
    append_summary_csv(target_file, overall_acc, active_acc)

    if spike_sample is not None:
        spk_t, true_lbl, pred_lbl = spike_sample
        save_spike_raster(spk_t, true_lbl, pred_lbl, target_file)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a trained SNN model on NinaProDB data")
    parser.add_argument('--file', type=str, default='S21_A1_E2.mat',
                        help='Target .mat file in Test_Data/ (default: S21_A1_E2.mat)')
    args = parser.parse_args()

    run_test(args.file)