# EMG Hand Gesture Classification with Spiking Neural Networks

A per-patient **Spiking Neural Network (SNN)** for classifying hand gestures from surface EMG signals, built with [snntorch](https://snntorch.readthedocs.io/) and PyTorch. Uses the [NinaProDB](http://ninapro.hevs.ch/) dataset.

---

## Project Structure

```
FYP/
├── Data/                   # All .mat files — train (reps 1–7) & test (reps 8+) are split internally
├── Trained_SNN/            # Saved per-patient model weights (.pth)
├── Result/                 # Categorized outputs:
│   ├── Training-Records/   # Per-epoch training logs (.csv)
│   ├── Confusion-Matrices/ # Test performance plots (.png)
│   ├── Spike-Rasters/      # Model firing patterns (.png)
│   ├── Efficiency-Metrics/ # Power/Real-time analysis plots & CSV
│   └── Summary/            # Overall accuracy summary table
├── .cache/                 # Preprocessed .npz cache (auto-generated, not committed)
│
├── config.py               # All hyperparameters and paths
├── model.py                # 3-layer TunedSNN architecture
├── dataset.py              # Data loading, preprocessing, augmentation
├── train.py                # Training pipeline
├── test.py                 # Inference pipeline
├── inspect_mat.py          # Utility to inspect raw .mat file contents
├── requirements.txt        # Python dependencies
└── REPORT.md               # Full technical report
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

### Train a single patient model
```bash
python train.py --file S1_A1_E1.mat
```

### Train all patients sequentially
```bash
python train.py --all
```

### Test a trained model
```bash
python test.py --file S21_A1_E2.mat
```

Outputs saved to `Result/`:
- `Confusion-Matrices/cm_<patient>.png` — Confusion matrix
- `Spike-Rasters/spike_raster_<patient>.png` — Output-layer spike raster
- `Training-Records/training_<patient>.csv` — Training logs
- `Summary/results_summary.csv` — Aggregated accuracy table
- `Efficiency-Metrics/` — Power and real-time analysis reports (from `analyze.py`)

---

## Key Configuration (`config.py`)

| Parameter | Value | Description |
|---|---|---|
| `NUM_INPUTS` | 10 | EMG channels |
| `NUM_OUTPUTS` | 18 | Gesture classes (incl. rest) |
| `NUM_STEPS` | 100 | Timesteps per window |
| `HIDDEN_SIZE` | 512 | Neurons per hidden layer |
| `BETA` | 0.90 | Initial membrane decay (learnable) |
| `THRESHOLD` | 0.7 | Initial firing threshold (learnable) |
| `EARLY_STOPPING_PATIENCE` | 15 | Epochs without improvement before stopping |

---

## Architecture

The `TunedSNN` model consists of three hidden Leaky-Integrate-and-Fire (LIF) layers followed by an output LIF layer:

```
Input [T, B, 10]
   → FC(10 → 512) → BN → LIF₁ (learn β, θ) → Dropout(0.25)
   → FC(512 → 512) → BN → LIF₂ (learn β, θ) → Dropout(0.25)
   → FC(512 → 256) → BN → LIF₃ (learn β, θ) → Dropout(0.25)
   → FC(256 → 18)  → LIF_out (learn β, θ)
Output spike counts [T, B, 18] → sum over T → argmax → class
```

---

## Cross-Repetition Evaluation Protocol

Following the NinaProDB standard:
- **Train**: Repetitions 1–7
- **Test**: Repetitions 8+ (held-out)

This ensures the model generalises to unseen repetitions of each gesture.
