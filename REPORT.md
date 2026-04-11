# Final Year Project Report

## EMG-Based Hand Gesture Classification Using Spiking Neural Networks

**Author:** FattMak  
**Model:** Per-patient Spiking Neural Network (SNN)  
**Dataset:** NinaProDB (`.mat` format)  
**Framework:** PyTorch + snntorch  

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Dataset & Preprocessing](#3-dataset--preprocessing)
4. [Model Architecture](#4-model-architecture)
5. [Training Strategy](#5-training-strategy)
6. [Evaluation Protocol](#6-evaluation-protocol)
7. [Results & Analysis](#7-results--analysis)
8. [Improvements Implemented](#8-improvements-implemented)
9. [Future Work](#9-future-work)
10. [References](#10-references)

---

## 1. Abstract

This project develops a **Spiking Neural Network (SNN)** system for classifying hand gestures from surface electromyography (sEMG) signals. Unlike conventional artificial neural networks (ANNs), SNNs communicate through discrete spike events that mimic biological neurons, offering potential energy efficiency advantages on neuromorphic hardware.

Per-patient models are trained on the NinaProDB dataset, where each model learns the specific EMG patterns of a single subject, avoiding cross-subject interference. The system classifies 17 distinct hand gestures plus a rest state (18 classes total) using 10 sEMG channels across 100 timesteps per inference window.

Key improvements over a baseline SNN include: learnable membrane dynamics, a three-layer architecture, per-channel normalisation, data augmentation, stratified splitting, class-weighted loss, cosine learning-rate annealing, and early stopping.

---

## 2. Introduction

### 2.1 Motivation

Hand gesture recognition from EMG signals has important applications in:
- **Prosthetic limb control** — enabling amputees to intuitively control prostheses
- **Human-computer interaction** — gesture-based interfaces without wearable cameras
- **Rehabilitation** — monitoring motor recovery progress

### 2.2 Why Spiking Neural Networks?

Traditional deep learning models process data as continuous-valued activations at every layer. SNNs instead communicate using binary **spike events** (0 or 1), which means:

| Property | ANN | SNN |
|---|---|---|
| Activation | Continuous float | Binary spike |
| Computation | Every timestep | Only on spike |
| Temporal encoding | Implicit | Explicit (spike timing) |
| Neuromorphic HW | Incompatible | Native |

For EMG, which is an inherently time-varying signal, the temporal dynamics of LIF neurons naturally align with the signal structure.

### 2.3 Per-Patient vs. Universal Models

Cross-subject EMG classification is notoriously difficult due to:
- Electrode placement variability
- Skin impedance differences
- Muscular anatomy differences

Training one model per patient sidesteps these issues at the cost of requiring calibration data per user — an acceptable trade-off for prosthetic applications.

---

## 3. Dataset & Preprocessing

### 3.1 NinaProDB

The NinaProDB dataset provides sEMG recordings from multiple subjects performing standardised hand and finger movements. Each `.mat` file contains:

| Field | Description |
|---|---|
| `emg` | `[T, 10]` Raw EMG signals from 10 Delsys Trigno electrodes |
| `restimulus` | `[T]` Gesture label at each timestep (0 = rest) |
| `repetition` | `[T]` Repetition index (1–10 per gesture) |

### 3.2 Repetition-Based Split

Following standard NinaProDB evaluation protocol:

```
Repetitions 1–7  →  Training + Validation
Repetitions 8+   →  Test (held-out)
```

This ensures the model generalises to unseen repetitions rather than memorising specific trials.

### 3.3 Repetition Forward-Fill

Rest periods in the raw data have `repetition = 0`. To correctly assign rest windows to a repetition bucket, forward-filling is applied: each rest sample inherits the most recent active repetition index. This prevents rest windows from leaking into the wrong split.

### 3.4 Sliding Window Segmentation

| Parameter | Value |
|---|---|
| Window length (`NUM_STEPS`) | 50 samples |
| Training stride | 10 samples (80% overlap) |
| Test stride | 50 samples (non-overlapping) |

Each window is labelled by the **majority class** within that window.

### 3.5 Normalisation

**Per-channel 99th-percentile normalisation** is applied independently to each of the 10 EMG channels:

$$x_{ch} \leftarrow \frac{x_{ch}}{p_{99}(|x_{ch}|) + \epsilon} \times 5.0$$

This is superior to global normalisation because individual electrode contacts may have very different amplitudes due to placement and skin contact quality.

### 3.6 Class Balancing

The rest class (label 0) dominates the dataset because subjects rest between gestures. To prevent the model from predicting rest by default, rest windows are randomly **down-sampled to 15%** during training dataset construction.

### 3.7 Disk Caching

Preprocessing (`.mat` loading, forward-fill, normalisation) is cached to `.npz` files in `.cache/`. Repeated training runs after the first are significantly faster because the expensive scipy I/O and NumPy operations are skipped.

---

## 4. Model Architecture

### 4.1 TunedSNN Architecture Evolution

Over the course of the project, a hyperparameter grid search was conducted to drastically reduce latency and computational cost while preserving high accuracy. The model transitioned from a heavy baseline to a high-speed optimized configuration utilising a non-spiking voltage readout.

#### Linear SNN
```text
Input signal: [T=100, B, C=10]

Layer 1:  Linear(10 → 512)  → BatchNorm1d(512) → LIF₁(β₁, θ₁) → Dropout(0.25)
Layer 2:  Linear(512 → 512) → BatchNorm1d(512) → LIF₂(β₂, θ₂) → Dropout(0.25)
Layer 3:  Linear(512 → 256) → BatchNorm1d(256) → LIF₃(β₃, θ₃) → Dropout(0.25)
Output:   Linear(256 → 18)  → LIF_out(β₄, θ₄) (Spiking)

Classification: argmax( Σ_t spk_out[t] )
```

#### Linear SNN with non-spiking output
```text
Input signal: [T=50, B, C=10]

Layer 1:  Linear(10 → 256)  → BatchNorm1d(256) → LIF₁(β₁, θ₁) → Dropout(0.25)
Layer 2:  Linear(256 → 256) → BatchNorm1d(256) → LIF₂(β₂, θ₂) → Dropout(0.25)
Layer 3:  Linear(256 → 128) → BatchNorm1d(128) → LIF₃(β₃, θ₃) → Dropout(0.25)
Output:   Linear(128 → 18)  → LIF_out(β₄, None) (Non-Spiking Membrane Integrator)

Classification: argmax( Mean_t mem_out[t] )
```

| Metric | Linear SNN | Linear SNN with non-spiking output | Improvement |
| :--- | :--- | :--- | :--- |
| **Temporal Window** | 100 timesteps | 50 timesteps | **50% faster signal sampling** |
| **Network Density** | 512 → 512 → 256 | 256 → 256 → 128 | **Massive parameter reduction** |
| **Output Type** | Discrete Spikes | Continuous Voltage | **Higher mathematical precision** |
| **Inference Latency** | ~68.8 ms | ~16.5 ms | **~76% absolute speedup** |

### 4.2 Leaky Integrate-and-Fire Neuron

Each LIF neuron follows the discrete-time membrane equation:

$$U[t] = \beta \cdot U[t-1] + I[t]$$
$$S[t] = \Theta(U[t] - \theta)$$
$$U[t] \leftarrow U[t] \cdot (1 - S[t]) \quad \text{(reset-by-subtraction)}$$

Where:
- $\beta$ = membrane decay constant (learnable, initialised at 0.90)
- $\theta$ = firing threshold (learnable, initialised at 0.70)
- $S[t]$ = spike output (0 or 1)

### 4.3 Surrogate Gradient

Because the Heaviside function $\Theta$ has zero gradient almost everywhere, direct backpropagation through spikes is impossible. The **fast-sigmoid surrogate**:

$$\tilde{\Theta}'(x) = \frac{1}{(1 + |\text{slope} \cdot x|)^2}$$

is substituted during the backward pass (`slope=25`). This provides a smooth, bounded gradient estimate that enables gradient-based optimisation of the SNN.

### 4.4 Learnable β and θ

Unlike fixed-parameter SNNs, each LIF layer here has its own **learnable** $\beta$ and $\theta$ scalar parameters (one per layer, not per neuron). This allows the network to adapt its:
- **Temporal integration window** (via $\beta$) — how much past activity influences current spiking
- **Excitability** (via $\theta$) — how "sensitive" the layer is to its input

### 4.5 Regularisation

| Technique | Where Applied | Purpose |
|---|---|---|
| BatchNorm1d | After each FC layer | Stabilises activations, reduces covariate shift |
| Dropout(0.25) | After each LIF layer | Prevents co-adaptation of neurons |
| Weight decay (1e-4) | Adam optimiser | L2 regularisation |

---

## 5. Training Strategy

### 5.1 Data Augmentation

On-the-fly augmentation is applied at `__getitem__` time during training only:

| Transform | Description |
|---|---|
| **Gaussian Noise** | Add $\mathcal{N}(0, 0.05)$ noise to all channels/timesteps |
| **Amplitude Scaling** | Multiply by random $U(0.8, 1.2)$ scalar |

These simulate real-world variability in electrode contact and muscular effort, improving generalisation without expanding disk storage.

### 5.2 Stratified Train/Validation Split

The 80/20 split uses `StratifiedShuffleSplit` from scikit-learn. Unlike `random_split`, this guarantees that every gesture class appears in both the training and validation subsets with proportional representation. This is critical because some gesture classes are rare.

### 5.3 Class-Weighted Cross-Entropy Loss

Each class $c$ receives a weight:

$$w_c = \frac{N}{K \cdot n_c}$$

Where $N$ = total training samples, $K$ = number of classes, $n_c$ = samples in class $c$.

Rare gestures receive proportionally higher loss weight, preventing the model from ignoring them.

### 5.4 Optimiser

**Adam** with:
- Learning rate: `0.001`
- Weight decay: `1e-4` (L2)
- Betas: default `(0.9, 0.999)`

### 5.5 Learning Rate Scheduler

**CosineAnnealingLR** smoothly decays the learning rate from `lr_max` to 0 following a cosine curve over `T_max = NUM_EPOCHS`. Compared to StepLR (which steps abruptly), cosine annealing provides a warm-cool decay that tends to find flatter minima.

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 + \cos\frac{t\pi}{T_{max}}\right)$$

### 5.6 Early Stopping

Training monitors validation accuracy. If there is no improvement for `EARLY_STOPPING_PATIENCE = 15` consecutive epochs, training terminates. The best model (highest val acc) is always checkpointed.

### 5.7 Mixed-Precision Training (CUDA)

On CUDA-capable GPUs, `torch.amp.autocast` + `GradScaler` are used for FP16 mixed-precision training, roughly halving memory usage and accelerating compute by ~1.5–2×.

---

## 6. Evaluation Protocol

### 6.1 Metrics

| Metric | Description |
|---|---|
| **Overall Accuracy** | `correct / total` including rest class |
| **Active Accuracy** | `correct / total` excluding rest class (label 0) |
| **Per-class F1 / Precision / Recall** | `sklearn.metrics.classification_report` |

Active accuracy is typically the more meaningful metric for gesture recognition because the rest class is easy to classify and inflates overall accuracy.

### 6.2 Outputs Saved to `Result/`

| File | Description |
|---|---|
| `Confusion-Matrices/cm_<patient>.png` | Confusion matrix (18×18, Blues colormap) |
| `Spike-Rasters/spike_raster_<patient>.png` | Output-layer spike raster (1 sample) |
| `Training-Records/training_<patient>.csv` | Per-epoch: train/val loss, accuracy, LR |
| `Summary/results_summary.csv` | Overall & active accuracy for every tested patient |

### 6.3 Spike Raster Interpretation

The spike raster shows which output neurons fired at which timesteps for a single test sample. In a well-trained SNN:
- The neuron corresponding to the **true class** should show the densest spiking activity
- If `True ≠ Pred`, the wrong neuron accumulated more spikes over 100 timesteps

---

## 7. Results & Analysis

> Results are populated automatically in `Result/Summary/results_summary.csv` after running `test.py`.

### 7.1 Linear SNN Results (Without Non-Spiking Output)

The following table summarises the historical baseline performance across all 25 subjects in the NinaProDB dataset prior to architecture reduction.

| Subject ID | Overall Acc (%) | Active Acc (%) |
|:---|:---|:---|
| **S1** | 84.42% | 72.61% |
| **S2** | 84.81% | 78.89% |
| **S3** | 83.68% | 64.07% |
| **S4** | 69.68% | 62.22% |
| **S5** | 71.33% | 57.50% |
| **S6** | 58.94% | 73.80% |
| **S7** | 70.19% | 61.14% |
| **S8** | 52.93% | 59.28% |
| **S9** | 75.81% | 59.14% |
| **S10** | 76.98% | 64.61% |
| **S11** | 81.15% | 71.63% |
| **S12** | 83.41% | 68.93% |
| **S13** | 56.05% | 66.00% |
| **S14** | 83.02% | 72.92% |
| **S15** | 67.44% | 58.42% |
| **S16** | 78.45% | 60.69% |
| **S17** | 75.58% | 67.22% |
| **S18** | 65.89% | 54.84% |
| **S19** | 77.05% | 59.28% |
| **S20** | 64.12% | 68.72% |
| **S22** | 72.52% | 59.00% |
| **S23** | 79.08% | 63.02% |
| **S24** | 72.92% | 59.67% |
| **S26** | 77.88% | 75.43% |
| **S27** | 64.57% | 64.86% |
| **Mean** | **73.12%** | **65.45%** |

### 7.2 Linear SNN Results (With Non-Spiking Output)

The following table will summarise the performance of the high-speed optimized configuration across all 25 subjects in the NinaProDB dataset.

| Subject ID | Overall Acc (%) | Active Acc (%) |
|:---|:---|:---|
| **S1** | 82.13% | 74.76% |
| **S2** | 78.62% | 74.73% |
| **S3** | 79.67% | 73.31% |
| **S4** | 74.39% | 64.54% |
| **S5** | 58.74% | 67.08% |
| **S6** | 65.28% | 75.93% |
| **S7** | 74.39% | 67.97% |
| **S8** | 79.74% | 72.73% |
| **S9** | 73.37% | 70.13% |
| **S10** | 76.69% | 71.27% |
| **S11** | 75.41% | 70.47% |
| **S12** | 77.71% | 72.52% |
| **S13** | 78.90% | 66.78% |
| **S14** | 79.65% | 71.76% |
| **S15** | 76.01% | 60.32% |
| **S16** | 79.21% | 58.45% |
| **S17** | 78.25% | 68.66% |
| **S18** | 58.35% | 57.58% |
| **S19** | 70.79% | 63.66% |
| **S20** | 80.85% | 65.12% |
| **S22** | 77.37% | 75.13% |
| **S23** | 71.02% | 73.83% |
| **S24** | 81.71% | 72.33% |
| **S26** | 82.28% | 82.75% |
| **S27** | 62.15% | 63.61% |
| **Mean** | **74.11%** | **69.81%** |

### 7.3 Performance Analysis

- **Baseline Comparison**: The mean **Active Accuracy of 69.81%** (excluding the rest class) demonstrates a significant improvement over the historical baseline (65.45%), despite a smaller network and shorter window.
- **Top Performer**: Subject **S26** achieved the highest active accuracy of **82.75%**, followed by S6 at 75.93%.
- **Robustness**: The model maintains an average overall accuracy of **74.11%**, showing that the non-spiking output integration provides more stable classification gradients.

---

## 8. Efficiency & Real-Time Analysis

Using the `analyze.py` tool, the SNN was compared against a theoretically equivalent dense ANN (ReLU-based) of the same architecture.

### 8.1 Synaptic Operations (SynOps) Reduction

Unlike ANNs which perform Multiply-Accumulate (MAC) operations for every connection every timestep, the SNN only performs spike-triggered additions (ADDs).

| Metric | Dense ANN | This SNN (S1) | Reduction |
|:---|:---|:---|:---|
| **Operations per window** | 40,294,400 MACs | 9,271,200 SynOps | **4.3×** |

### 8.2 Energy Consumption (Theoretical)

Based on a **45nm CMOS process** model (MAC = 4.6 pJ, ADD = 0.9 pJ):

- **Equivalent ANN Energy**: 185,354 nJ per inference
- **Tuned SNN Energy**: 10,699 nJ per inference
- **Performance Gain**: **~17.3× more energy efficient** than a standard ANN.

### 8.3 Real-Time Capability

- **Inference Latency**: ~2.0 ms per 50ms window.
- **Headroom**: **24.1× faster than real-time**.
- **Conclusion**: The model is highly suitable for deployment on low-power embedded processors for prosthetic control, as it uses only a small fraction of the available temporal budget.

### 8.4 State-of-the-Art Baseline: 1D-CNN Comparison

To rigorously evaluate the Spiking Neural Network, its performance was benchmarked against a **Mirror 1D-CNN** trained on the same data pipeline across all 25 subjects in the NinaProDB dataset.

| Metric (Mean, 25 Subjects) | Mirror 1D-CNN (Baseline) | Linear SNN (Our Model) | Comparison |
|:---|:---|:---|:---|
| **Active Accuracy** | **73.82%** | 69.81% | The SNN achieves competitive accuracy within **~4%** of the CNN counterpart. |
| **Overall Accuracy** | **79.04%** | 74.11% | Consistent performance across both architectures. |
| **Inference Latency** | **0.017 ms/sample** | 0.33 ms/sample | The CNN is ~20× faster on GPU (MPS) hardware due to temporal parallelism. |
| **Power Consumption** | Floating-point MACs | **Sparse Integer ADDs** | The SNN offers a theoretical **~17.3× energy reduction** by replacing MACs with spike-driven additions. |

**Conclusion:** The Mirror CNN provides a high-performance upper bound for accuracy and throughput on modern GPUs. However, the Linear SNN's ability to maintain high utility (nearly 70% active accuracy) while utilizing sparse, integer-only operations makes it the superior choice for **embedded neuromorphic hardware** where battery life and spatial constraints are paramount.

### 7.4 Confusion Matrix

Saved per patient to `Result/Confusion-Matrices/cm_<patient>.png`. Common patterns to look for:
- **Diagonal dominance** → good overall classification
- **Off-diagonal clusters** → gesture pairs that confuse the model (often biomechanically similar gestures)
- **Row of errors toward class 0** → model defaults to rest when uncertain

### 7.5 Expected Behaviour of Learnable β and θ

After training, the learned membrane decay $\beta$ and threshold $\theta$ per layer can be inspected:

```python
net = TunedSNN()
net.load_state_dict(torch.load("Trained_SNN/snn_nina_trained_S21_A1_E2.pth"))
print("β₁:", net.lif1.beta.item())
print("θ₁:", net.lif1.threshold.item())
```

Typically shallower layers converge to larger $\beta$ (longer memory), while deeper layers may develop lower thresholds (more sensitive to sparse upstream spikes).

---

## 9. Improvements Implemented

| # | Improvement | File | Impact |
|---|---|---|---|
| 1 | **3rd hidden layer** (512 → 256) | `model.py` | Deeper feature extraction |
| 2 | **Learnable β and θ** per LIF layer | `model.py` | Adaptive temporal dynamics |
| 3 | **Per-channel normalisation** | `dataset.py` | Corrects inter-electrode amplitude bias |
| 4 | **On-the-fly data augmentation** (noise + scaling) | `dataset.py` | Improved generalisation |
| 5 | **Disk caching** of preprocessed `.npz` | `dataset.py` | Faster repeated training runs |
| 6 | **Stratified train/val split** (sklearn) | `train.py` | Class-balanced validation |
| 7 | **Inverse-frequency class weights** | `train.py` | Handles gesture class imbalance |
| 8 | **CosineAnnealingLR** (replaces StepLR) | `train.py` | Smoother LR decay |
| 9 | **Early stopping** (patience=15) | `train.py` | Prevents overfitting |
| 10 | **Per-epoch CSV logging** | `train.py` | Training curve analysis |
| 11 | **Per-class classification report** | `test.py` | Detailed precision/recall/F1 |
| 12 | **Confusion matrix saved to PNG** | `test.py` | Persistent visual result |
| 13 | **Spike raster visualisation** | `test.py` | SNN interpretability |
| 14 | **Aggregated results CSV** | `test.py` | Cross-patient comparison |
| 15 | **requirements.txt** | root | Reproducible environment |
| 16 | **README.md** | root | Project documentation |

---

## 10. Future Work

### 9.1 Architecture
- **Recurrent LIF (RLIF)**: Replace `snn.Leaky` with `snn.RLeaky` to add lateral recurrent connections. This can capture longer temporal dependencies than feedforward LIF.
- **Convolutional front-end**: Add 1-D temporal convolutions before the LIF layers to automatically extract frequency-domain features from raw EMG.
- **Synaptic conductance model**: Replace simplified LIF with alpha/exponential synaptic models for richer pre-synaptic integration.

### 9.2 Training
- **Optuna hyperparameter search**: Automate search over `HIDDEN_SIZE`, `BETA`, `THRESHOLD`, `SLOPE`, dropout rate, and learning rate.
- **Parallel patient training**: Use `multiprocessing.Pool` or `concurrent.futures` to train all patient models simultaneously.
- **Cross-validation**: Replace the single 80/20 split with k-fold stratified cross-validation for more reliable accuracy estimates.

### 9.3 Results & Analysis
- **Cross-patient accuracy bar chart**: Visual comparison of all patients in a single figure using matplotlib.
- **Training curve plots**: Plot the per-epoch CSVs in `Result/` as loss/accuracy curves.
- **Energy efficiency analysis**: Count total spikes per inference, compute approximate multiply-accumulate (MAC) operations, and compare against an equivalent ANN.
- **SNN vs. ANN comparison**: Train the same 3-layer architecture as a standard ReLU network and compare accuracy and spiking efficiency.

### 9.4 Deployment
- **Real-time streaming inference**: Implement a sliding-window pipeline that reads from a live EMG device and classifies gestures with sub-200 ms latency.
- **Neuromorphic hardware export**: Use Intel Lava or BrainScale toolchains to compile the trained SNN weights for deployment on Intel Loihi or SpiNNaker.

---

## 11. References

1. Atzori, M. et al. (2014). *Electromyography data for non-invasive naturally-controlled robotic hand prostheses*. **Scientific Data**, 1, 140053. https://doi.org/10.1038/sdata.2014.53

2. Eshraghian, J. K. et al. (2023). *Training Spiking Neural Networks Using Lessons From Deep Learning*. **Proceedings of the IEEE**, 111(9), 1016–1054. https://doi.org/10.1109/JPROC.2023.3308088

3. Neftci, E. O., Mostafa, H., & Zenke, F. (2019). *Surrogate Gradient Learning in Spiking Neural Networks: Bringing the Power of Gradient-Based Optimization to Spiking Neural Networks*. **IEEE Signal Processing Magazine**, 36(6), 51–63.

4. snntorch Documentation. https://snntorch.readthedocs.io/

5. PyTorch Documentation. https://pytorch.org/docs/stable/

6. Scikit-learn Documentation — StratifiedShuffleSplit. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
