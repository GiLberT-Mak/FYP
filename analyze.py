"""
analyze.py — SNN Efficiency Analysis

Quantifies the low-power and real-time advantages of the trained SNN
by measuring:

  • Spike sparsity and firing rate per layer
  • Synaptic Operations (SynOps) vs equivalent ANN MACs
  • Theoretical energy consumption (45nm CMOS model)
  • Per-sample inference latency vs real-time budget

Outputs (all saved to Result/):
  • spike_activity_<patient>.png    — bar chart of neuron firing rates
  • energy_comparison_<patient>.png — SNN vs ANN energy (log scale)
  • efficiency_<patient>.csv        — all metrics in tabular form
"""

import torch
import os
import argparse
import sys
import time
import csv

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from config import (device, MODEL_DIR, DATA_FOLDER, NUM_OUTPUTS,
                    BATCH_SIZE, RESULT_DIR, NUM_STEPS, NUM_INPUTS, HIDDEN_SIZE)
from model import TunedSNN
from dataset import SingleFileLoader

sys.stdout.reconfigure(encoding='utf-8')

# ── Energy constants (45nm CMOS process, Horowitz 2014) ──────────────────────
# MAC (multiply-accumulate) — required by every ANN neuron every timestep
# ADD (spike-triggered add) — only triggered when a pre-synaptic neuron fires
E_MAC_PJ       = 4.6    # pJ per multiply-accumulate operation
E_ADD_PJ       = 0.9    # pJ per spike-triggered add operation
SAMPLE_RATE_HZ = 2000   # NinaProDB acquisition rate in Hz


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_spike_activity(firing_pct, base_name):
    """Bar chart of neuron firing rate (%) per layer."""
    labels = ['LIF₁\n(512 neurons)', 'LIF₂\n(512 neurons)',
              'LIF₃\n(256 neurons)', 'LIF_out\n(18 neurons)']
    rates  = [firing_pct[k] for k in ['lif1', 'lif2', 'lif3', 'lif_out']]
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rates, color=colors, edgecolor='white',
                  linewidth=1.2, width=0.5, zorder=3)
    ax.bar_label(bars, fmt='%.1f%%', padding=5, fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.4 + 2)
    ax.set_ylabel('Neuron Firing Rate (%)', fontsize=12)
    ax.set_title(f'Spike Activity per Layer — {base_name}', fontsize=13, pad=12)
    ax.axhline(y=50, color='gray', ls='--', lw=1.2, alpha=0.6, label='50 % baseline', zorder=2)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, f"spike_activity_{base_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_energy_comparison(energy_ann_nj, energy_snn_nj, energy_ratio, base_name):
    """Log-scale bar chart comparing ANN vs SNN estimated energy."""
    fig, ax = plt.subplots(figsize=(7, 5))
    categories = ['Equivalent ANN\n(dense ReLU)', 'This SNN\n(sparse spikes)']
    energies   = [energy_ann_nj, energy_snn_nj]
    colors     = ['#EF5350', '#42A5F5']

    bars = ax.bar(categories, energies, color=colors, width=0.38,
                  edgecolor='white', linewidth=1.2, zorder=3)
    ax.bar_label(bars, labels=[f'{v:,.1f} nJ' for v in energies],
                 padding=6, fontsize=11, fontweight='bold')

    ax.set_yscale('log')
    ax.set_ylabel('Estimated Energy per Inference (nJ, log scale)', fontsize=11)
    ax.set_title(f'SNN vs ANN Energy Comparison\n{base_name}', fontsize=13, pad=12)

    # Annotation arrow
    ax.annotate(
        f'{energy_ratio:.0f}× more\nefficient',
        xy=(1, energy_snn_nj * 2),
        xytext=(0.55, energy_ann_nj * 0.4),
        fontsize=12, color='darkgreen', fontweight='bold', ha='center',
        arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2.0)
    )

    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.4, zorder=0)
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()

    path = os.path.join(RESULT_DIR, f"energy_comparison_{base_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyze(target_file):
    W = 62
    print(f"\n{'='*W}")
    print(f"  SNN Efficiency Analysis — {target_file}")
    print(f"{'='*W}")

    os.makedirs(RESULT_DIR, exist_ok=True)
    base_name  = os.path.splitext(target_file)[0]
    model_path = os.path.join(MODEL_DIR, f"snn_nina_trained_{base_name}.pth")

    if not os.path.exists(model_path):
        print(f"  Model not found: {model_path}")
        print(f"  Run train.py --file {target_file} first.")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    net = TunedSNN().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    net.eval()

    # ── Register spike-counting hooks ─────────────────────────────────────────
    # The model loops over timesteps internally, so each hook fires once per
    # timestep. Output of snn.Leaky is a tuple: (spk [B, neurons], mem [B, neurons])
    spike_totals   = dict(lif1=0.0, lif2=0.0, lif3=0.0, lif_out=0.0)
    spike_possible = dict(lif1=0.0, lif2=0.0, lif3=0.0, lif_out=0.0)

    def make_hook(name):
        def hook(module, inp, output):
            spk, _ = output
            spike_totals[name]   += spk.sum().item()
            spike_possible[name] += spk.numel()
        return hook

    handles = [
        net.lif1.register_forward_hook(make_hook('lif1')),
        net.lif2.register_forward_hook(make_hook('lif2')),
        net.lif3.register_forward_hook(make_hook('lif3')),
        net.lif_out.register_forward_hook(make_hook('lif_out')),
    ]

    # ── Load test data ────────────────────────────────────────────────────────
    try:
        dataset = SingleFileLoader(DATA_FOLDER, target_file)
    except FileNotFoundError as e:
        print(e)
        for h in handles: h.remove()
        return

    if len(dataset) == 0:
        print("  No valid samples found in test split.")
        for h in handles: h.remove()
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── GPU warm-up (prevents cold-start latency inflation) ───────────────────
    warm_data, _ = next(iter(loader))
    with torch.no_grad():
        _ = net(warm_data.to(device).permute(1, 0, 2))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    # Reset accumulators — warm-up spikes don't count
    for k in spike_totals:
        spike_totals[k]   = 0.0
        spike_possible[k] = 0.0

    # ── Timed inference ───────────────────────────────────────────────────────
    n_samples_total    = 0
    per_sample_lat_ms  = []

    with torch.no_grad():
        for data, _ in loader:
            batch_size = data.size(0)
            n_samples_total += batch_size
            data = data.to(device).permute(1, 0, 2)   # [T, B, C]

            t0 = time.perf_counter()
            _  = net(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            per_sample_lat_ms.append(elapsed_ms / batch_size)

    for h in handles:
        h.remove()

    if n_samples_total == 0:
        print("  No samples were processed.")
        return

    # ── Spike statistics ──────────────────────────────────────────────────────
    # avg_spikes[k] = mean total spikes produced by layer k for one input sample
    #   (summed over all T=100 timesteps and all neurons)
    avg_spikes   = {k: spike_totals[k]   / n_samples_total   for k in spike_totals}
    firing_pct   = {k: spike_totals[k]   / spike_possible[k] * 100
                    if spike_possible[k] > 0 else 0.0         for k in spike_totals}
    sparsity_pct = {k: 100.0 - firing_pct[k]                 for k in firing_pct}

    # ── SynOps model ──────────────────────────────────────────────────────────
    # A spike from layer k propagates through the next FC layer and triggers
    # one ADD per downstream connection:
    #   lif1 → fc2  (HIDDEN_SIZE=512 connections per spike)
    #   lif2 → fc3  (HIDDEN_SIZE//2=256 connections per spike)
    #   lif3 → fc_out (NUM_OUTPUTS=18 connections per spike)
    #   lif_out is terminal — classification only, no further synaptic events
    layer_fanout = {
        'lif1':    HIDDEN_SIZE,
        'lif2':    HIDDEN_SIZE // 2,
        'lif3':    NUM_OUTPUTS,
        'lif_out': 0,
    }
    synops_per_sample = {k: avg_spikes[k] * layer_fanout[k] for k in layer_fanout}

    # ── Energy model ──────────────────────────────────────────────────────────
    # Input layer: continuous EMG → FC1 always requires full MACs
    mac_input = NUM_STEPS * NUM_INPUTS * HIDDEN_SIZE          # 100×10×512 = 512,000

    # Spike-driven layers: only SynOps (ADDs, much cheaper)
    snn_synops = sum(synops_per_sample[k] for k in ['lif1', 'lif2', 'lif3'])

    # Equivalent dense ANN: every weight active every timestep
    ann_macs = (
        NUM_STEPS * NUM_INPUTS       * HIDDEN_SIZE          +  # fc1
        NUM_STEPS * HIDDEN_SIZE      * HIDDEN_SIZE          +  # fc2
        NUM_STEPS * HIDDEN_SIZE      * (HIDDEN_SIZE // 2)   +  # fc3
        NUM_STEPS * (HIDDEN_SIZE//2) * NUM_OUTPUTS             # fc_out
    )

    energy_ann_nj = ann_macs * E_MAC_PJ / 1000
    energy_snn_nj = (mac_input * E_MAC_PJ + snn_synops * E_ADD_PJ) / 1000
    energy_ratio  = energy_ann_nj / energy_snn_nj if energy_snn_nj > 0 else float('inf')
    op_reduction  = ann_macs / (mac_input + snn_synops) if (mac_input + snn_synops) > 0 else 0

    # ── Latency ───────────────────────────────────────────────────────────────
    window_ms       = NUM_STEPS / SAMPLE_RATE_HZ * 1000     # 50 ms
    avg_lat_ms      = float(np.mean(per_sample_lat_ms))
    realtime_factor = window_ms / avg_lat_ms if avg_lat_ms > 0 else float('inf')
    is_realtime     = avg_lat_ms <= window_ms

    # ── Print formatted report ────────────────────────────────────────────────
    def row(label, value, unit=''):
        print(f"  {label:<37}  {value:>14}  {unit}")

    print(f"\n  --- Real-Time Analysis ---")
    row("Signal window (real time)", f"{window_ms:.1f}", "ms")
    row("Total test samples",        f"{n_samples_total}", "")
    row("Avg inference latency",     f"{avg_lat_ms:.3f}", f"ms / sample  [{device.type.upper()}]")
    rt_str = f"YES  ({realtime_factor:.1f}x headroom)" if is_realtime else f"NO  ({realtime_factor:.2f}x)"
    row("Real-time capable",         rt_str, "")

    print(f"\n  --- Layer-wise Spike Statistics ---")
    print(f"  {'Layer':<10}  {'Sparsity':>10}  {'Firing Rate':>12}  "
          f"{'Avg spikes':>12}  {'SynOps':>12}")
    print(f"  {'-'*62}")
    for k, lbl in [('lif1','LIF₁'), ('lif2','LIF₂'), ('lif3','LIF₃'), ('lif_out','LIF_out')]:
        so = f"{synops_per_sample[k]:>12,.0f}" if layer_fanout[k] > 0 else f"{'—':>12}"
        print(f"  {lbl:<10}  {sparsity_pct[k]:>9.1f}%  {firing_pct[k]:>11.1f}%"
              f"  {avg_spikes[k]:>12,.0f}  {so}")

    print(f"\n  --- Operations per Inference Sample ---")
    print(f"  {'Metric':<38}  {'Value':>14}")
    print(f"  {'-'*54}")
    row("Equivalent ANN MACs",                  f"{ann_macs:,.0f}", "")
    row("SNN: input-layer MACs  (unavoidable)",  f"{mac_input:,.0f}", "")
    row("SNN: spike SynOps  (data-dependent)",   f"{snn_synops:,.0f}", "")
    row("Total SNN operations",                  f"{mac_input + snn_synops:,.0f}", "")
    row("Operation count reduction",             f"{op_reduction:.1f}x", "")

    print(f"\n  --- Estimated Energy per Inference Sample ---")
    print(f"  (45nm CMOS — MAC={E_MAC_PJ} pJ, ADD={E_ADD_PJ} pJ)")
    print(f"  {'Metric':<38}  {'Value':>14}")
    print(f"  {'-'*54}")
    row("Equivalent ANN energy",  f"{energy_ann_nj:,.1f}", "nJ")
    row("This SNN energy",        f"{energy_snn_nj:,.1f}", "nJ")
    row("Energy savings",         f"{energy_ratio:.1f}x", "more efficient")

    print(f"\n{'='*W}\n")

    # ── Save charts ───────────────────────────────────────────────────────────
    p1 = plot_spike_activity(firing_pct, base_name)
    p2 = plot_energy_comparison(energy_ann_nj, energy_snn_nj, energy_ratio, base_name)
    print(f"  Spike activity chart    → {p1}")
    print(f"  Energy comparison chart → {p2}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULT_DIR, f"efficiency_{base_name}.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value', 'unit'])
        w.writerow(['window_duration_ms',      f'{window_ms:.1f}',         'ms'])
        w.writerow(['avg_inference_ms',        f'{avg_lat_ms:.3f}',        'ms'])
        w.writerow(['realtime_headroom_x',     f'{realtime_factor:.2f}',   'x'])
        w.writerow(['is_realtime',             str(is_realtime),           ''])
        w.writerow(['ann_macs_per_sample',     f'{ann_macs:.0f}',          'MACs'])
        w.writerow(['snn_input_macs',          f'{mac_input:.0f}',         'MACs'])
        w.writerow(['snn_synops',              f'{snn_synops:.0f}',        'SynOps'])
        w.writerow(['operation_reduction_x',   f'{op_reduction:.2f}',      'x'])
        w.writerow(['ann_energy_nj',           f'{energy_ann_nj:.2f}',     'nJ'])
        w.writerow(['snn_energy_nj',           f'{energy_snn_nj:.2f}',     'nJ'])
        w.writerow(['energy_savings_x',        f'{energy_ratio:.2f}',      'x'])
        for k in ['lif1', 'lif2', 'lif3', 'lif_out']:
            w.writerow([f'firing_rate_{k}_%',  f'{firing_pct[k]:.2f}',    '%'])
            w.writerow([f'sparsity_{k}_%',     f'{sparsity_pct[k]:.2f}',  '%'])
            w.writerow([f'avg_spikes_{k}',     f'{avg_spikes[k]:.1f}',    'spikes/sample'])
    print(f"  Efficiency CSV          → {csv_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quantify SNN energy efficiency and real-time capability')
    parser.add_argument('--file', type=str, default='S1_A1_E2.mat',
                        help='Target .mat file in Data/ (default: S1_A1_E2.mat)')
    args = parser.parse_args()
    analyze(args.file)
