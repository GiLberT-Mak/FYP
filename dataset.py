import os
import glob
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from config import NUM_STEPS, NUM_OUTPUTS, CACHE_DIR


# ─────────────────────────────────────────────────────────────
# Shared preprocessing helper
# ─────────────────────────────────────────────────────────────

def process_mat_file(f_path, cache_dir=None):
    """
    Load and preprocess a NinaProDB .mat file.

    Preprocessing steps:
      1. Forward-fill repetition labels so rest periods belong to the
         nearest active repetition.
      2. Per-channel 99th-percentile normalisation (each of the 10 EMG
         channels is normalised independently to [-5, 5]).

    Results are cached to disk as .npz files so repeated runs skip the
    expensive scipy.io.loadmat call.

    Args:
        f_path    : Absolute path to the .mat file.
        cache_dir : Directory to store/load .npz cache files (or None to disable).

    Returns:
        raw_emg     : float32 array [T, C]  — normalised EMG
        raw_labels  : int array   [T]       — gesture labels (restimulus)
        filled_reps : int array   [T]       — forward-filled repetition indices
    """
    # ── Cache hit ────────────────────────────────────────────
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(
            cache_dir,
            os.path.splitext(os.path.basename(f_path))[0] + ".npz"
        )
        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            return cached["emg"], cached["labels"], cached["reps"]

    # ── Load raw data ─────────────────────────────────────────
    mat        = scipy.io.loadmat(f_path)
    raw_emg    = mat["emg"].astype(np.float32)
    raw_labels = mat["restimulus"].flatten()
    raw_reps   = mat["repetition"].flatten()

    # ── Forward-fill repetitions ──────────────────────────────
    current_rep  = 1
    filled_reps  = np.zeros_like(raw_reps)
    for i in range(len(raw_reps)):
        if raw_reps[i] > 0:
            current_rep = raw_reps[i]
        filled_reps[i] = current_rep

    # ── Per-channel normalisation ─────────────────────────────
    #   Each of the 10 EMG channels is normalised independently
    #   using the channel's 99th percentile of absolute amplitude.
    for ch in range(raw_emg.shape[1]):
        p99 = np.percentile(np.abs(raw_emg[:, ch]), 99) + 1e-6
        raw_emg[:, ch] = raw_emg[:, ch] / p99 * 5.0

    # ── Cache write ───────────────────────────────────────────
    if cache_dir:
        np.savez(cache_file, emg=raw_emg, labels=raw_labels, reps=filled_reps)

    return raw_emg, raw_labels, filled_reps


# ─────────────────────────────────────────────────────────────
# Training / Validation Dataset
# ─────────────────────────────────────────────────────────────

class LoadDataset(Dataset):
    """
    Sliding-window EMG dataset used during training and validation.

    Each sample is a window of shape [NUM_STEPS, NUM_INPUTS] labelled
    by the majority gesture class within that window.

    Args:
        folder_path     : Folder containing .mat files.
        is_training     : If True, uses a stride of 50; else stride = NUM_STEPS.
        target_filename : If given, load only this single file.
        augment         : If True, apply on-the-fly data augmentation in __getitem__.
    """

    def __init__(self, folder_path, is_training=True, target_filename=None, augment=False):
        if target_filename:
            self.files = [os.path.join(folder_path, target_filename)]
        else:
            self.files = glob.glob(os.path.join(folder_path, "*.mat"))

        self.raw_data_list   = []
        self.raw_labels_list = []
        self.indices         = []          # (file_id, start_t, label)
        self.augment         = augment

        print(f" Analyzing Dataset in: {folder_path}")

        if not self.files:
            print(" No .mat files found! Check your folder.")

        for file_id, f_path in enumerate(self.files):
            try:
                raw_emg, raw_labels, filled_reps = process_mat_file(f_path, cache_dir=CACHE_DIR)
                self.raw_data_list.append(raw_emg)
                self.raw_labels_list.append(raw_labels)

                total_len = raw_emg.shape[0]
                stride    = 50 if is_training else NUM_STEPS

                for i in range(0, total_len - NUM_STEPS, stride):
                    rep_window = filled_reps[i : i + NUM_STEPS]
                    # Train on repetitions 1–7 (NinaProDB convention)
                    if rep_window[-1] > 7:
                        continue

                    lbl_win       = raw_labels[i : i + NUM_STEPS]
                    vals, counts  = np.unique(lbl_win, return_counts=True)
                    label         = int(vals[np.argmax(counts)])

                    if label >= NUM_OUTPUTS:
                        continue
                    # Down-sample the rest class (label 0) to ~15 %
                    if is_training and label == 0 and np.random.rand() > 0.15:
                        continue

                    self.indices.append((file_id, i, label))

            except Exception as e:
                print(f"   !  Error reading {f_path}: {e}")

        print(f" Dataset Ready. Total Windows: {len(self.indices)}")

    # ── Helper for stratified splitting ──────────────────────

    def get_labels(self):
        """Return a flat list of labels (one per sample) for stratified splitting."""
        return [item[2] for item in self.indices]

    # ── PyTorch Dataset interface ─────────────────────────────

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_id, start_t, label = self.indices[idx]
        data = self.raw_data_list[file_id][start_t : start_t + NUM_STEPS].copy()

        if self.augment:
            # Additive Gaussian noise (σ = 0.05)
            data += np.random.normal(0, 0.05, data.shape).astype(np.float32)
            # Random amplitude scaling in [0.8, 1.2]
            data *= float(np.random.uniform(0.8, 1.2))

        data_tensor  = torch.tensor(data, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor


# ─────────────────────────────────────────────────────────────
# Test / Inference Dataset
# ─────────────────────────────────────────────────────────────

class SingleFileLoader(Dataset):
    """
    Non-overlapping sliding-window dataset for inference.

    Uses repetitions 8+ (held-out from training) as the test split,
    consistent with NinaProDB cross-repetition evaluation protocol.
    No augmentation is applied.
    """

    def __init__(self, folder_path, target_filename):
        self.samples  = []
        full_path     = os.path.join(folder_path, target_filename)

        if not os.path.exists(full_path):
            raise FileNotFoundError(f" ERROR: File not found at {full_path}")

        print(f" Loading Test File: {target_filename}")

        raw_emg, raw_labels, filled_reps = process_mat_file(full_path, cache_dir=CACHE_DIR)

        total_len = raw_labels.shape[0]
        stride    = NUM_STEPS

        for i in range(0, total_len - NUM_STEPS, stride):
            rep_window = filled_reps[i : i + NUM_STEPS]
            # Test on repetitions 8+ (cross-repetition holdout)
            if rep_window[-1] <= 7:
                continue

            label_window = raw_labels[i : i + NUM_STEPS]
            vals, counts = np.unique(label_window, return_counts=True)
            label        = int(vals[np.argmax(counts)])

            if label < NUM_OUTPUTS:
                snippet = raw_emg[i : i + NUM_STEPS, :]
                self.samples.append((snippet, label))

        print(f" Samples Loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label  = self.samples[idx]
        data_tensor  = torch.tensor(data, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor
