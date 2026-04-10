import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import itertools
import time
import os
import csv
import sys

from config import device, DATA_FOLDER, BATCH_SIZE, NUM_OUTPUTS
from model import TunedSNN
from train import stratified_split, compute_class_weights
from dataset import LoadDataset

sys.stdout.reconfigure(encoding='utf-8')

# Search space
WINDOWS   = [40, 50, 60, 70]
DENSITIES = [
    (512, 256, 256),
    (512, 256, 128),
    (512, 128, 128),
    (256, 256, 128),
    (256, 128, 128)
]

TARGET_FILE = "S1_A1_E2.mat"
MAX_EPOCHS  = 25

RESULT_CSV = os.path.join('Result', 'Summary', 'hyperparam_search_results.csv')
os.makedirs(os.path.dirname(RESULT_CSV), exist_ok=True)

def measure_latency(net, window_size):
    """Measures the average latency (in ms) of the forward pass."""
    net.eval()
    # Create dummy batch of size 1 for single inference timing
    dummy_input = torch.randn(window_size, 1, 10).to(device)
    
    # Warmup
    for _ in range(10):                 
        _ = net(dummy_input)

    if device.type == 'cuda': torch.cuda.synchronize()
    if device.type == 'mps': torch.mps.synchronize()

    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = net(dummy_input)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    if device.type == 'mps': torch.mps.synchronize()
            
    end_time = time.perf_counter()
    return ((end_time - start_time) / 100) * 1000  # Latency in ms


def run_experiment(window, density, train_loader, val_loader, class_weights):
    print(f"\n{'='*50}\nTesting Window = {window} | Density = {density}\n{'='*50}")
    
    net = TunedSNN(layer_sizes=density).to(device)
    loss_fn   = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    best_val_acc = 0.0
    
    for epoch in range(MAX_EPOCHS):
        net.train()
        for data, targets in train_loader:
            data, targets = data.to(device), targets.to(device)
            # Permute to [Time, Batch, Ch] and SLICE to window size!
            data = data.permute(1, 0, 2)[:window, :, :]
            
            optimizer.zero_grad()
            mem_rec = net(data)
            loss    = loss_fn(mem_rec.mean(dim=0), targets)
            loss.backward()
            optimizer.step()

        # Validate
        net.eval()
        val_acc, val_batches = 0.0, 0
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.permute(1, 0, 2)[:window, :, :]

                mem_rec = net(data)
                _, pred = torch.max(mem_rec.mean(dim=0), 1)
                acc     = (pred == targets).float().mean() * 100
                
                val_acc += acc.item()
                val_batches += 1
                
        avg_val_acc = val_acc / val_batches
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            
        print(f"  Ep {epoch:2d}/{MAX_EPOCHS} - Val Acc: {avg_val_acc:.2f}% (Best: {best_val_acc:.2f}%)", end='\r')

    print()
    latency = measure_latency(net, window)
    print(f"  Result: Best Acc = {best_val_acc:.2f}%, Latency = {latency:.3f} ms")
    return best_val_acc, latency

def main():
    # ── 1. Load the single target file completely into memory for speed
    full_dataset = LoadDataset(DATA_FOLDER, is_training=True, target_filename=TARGET_FILE, augment=False)
    # Stratified Split 80/20
    train_dataset, val_dataset = stratified_split(full_dataset, val_ratio=0.2)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    class_weights = compute_class_weights(full_dataset, NUM_OUTPUTS, device)
    
    # ── 2. Setup CSV Logs
    file_exists = os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Window_Size', 'Density_Layers', 'Peak_Val_Acc_%', 'Inference_Latency_ms'])
    
    # ── 3. Run Grid Options
    combinations = list(itertools.product(WINDOWS, DENSITIES))
    print(f"\nStarting Grid Search: {len(combinations)} total combinations on {device}\n")
    
    for window, density in combinations:
        best_acc, latency = run_experiment(window, density, train_loader, val_loader, class_weights)
        
        # Log to file row-by-row in case of crash
        with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([window, f"{density[0]}-{density[1]}-{density[2]}", f"{best_acc:.2f}", f"{latency:.4f}"])

    print(f"\nGrid search completed! Results saved to {RESULT_CSV}")

if __name__ == "__main__":
    main()
