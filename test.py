import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import scipy.io
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
# Add this line right after your imports:
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CONFIG
# ==========================================
# Automatic path detection
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(CURRENT_DIR, 'Test_Data') 
# Points to the model trained by train.py
MODEL_PATH = os.path.join(CURRENT_DIR, 'snn_nina_trained.pth') 

# Change this to a specific file you want to test
TARGET_FILE = "S21_A1_E2.mat" 

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

NUM_INPUTS = 10
NUM_OUTPUTS = 18
HIDDEN_SIZE = 512
# Note: Input steps in training was 150. Testing usually requires matching dimensions
# or valid sliding windows.
RAW_STEPS = 150 

# ==========================================
# MODEL DEFINITION (MUST MATCH TRAIN.PY)
# ==========================================
class TunedSNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.threshold = 0.7
        self.beta = 0.95

        self.fc1 = nn.Linear(NUM_INPUTS, HIDDEN_SIZE)
        self.bn1 = nn.BatchNorm1d(HIDDEN_SIZE)
        self.lif1 = snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=spike_grad)
        self.drop1 = nn.Dropout(0.25)

        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bn2 = nn.BatchNorm1d(HIDDEN_SIZE)
        self.lif2 = snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=spike_grad)
        self.drop2 = nn.Dropout(0.25)

        self.fc_out = nn.Linear(HIDDEN_SIZE, NUM_OUTPUTS)
        self.lif_out = snn.Leaky(beta=self.beta, threshold=self.threshold, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_out_rec = []

        for step in range(x.size(0)):
            cur1 = self.bn1(self.fc1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            # No dropout during eval
            
            cur2 = self.bn2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            
            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)

# ==========================================
# LOADER
# ==========================================
class SingleFileLoader:
    def __init__(self, folder_path, target_filename):
        self.samples = []
        full_path = os.path.join(folder_path, target_filename)

        if not os.path.exists(full_path):
            print(f"❌ ERROR: File not found at {full_path}")
            # Try finding any mat file
            all_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
            if all_files:
                print(f"   Found these files instead: {all_files[:3]}...")
            return

        print(f"🧐 Loading Test File: {target_filename}")

        try:
            mat = scipy.io.loadmat(full_path)
            raw_data = mat['emg']
            raw_labels = mat['restimulus']
            
            # Preprocessing (Matched to Training as closely as possible)
            # Training used: raw_emg / p99 * 5.0
            p99 = np.percentile(np.abs(raw_data), 99) + 1e-6
            norm_data = raw_data.astype(np.float32) / p99 * 5.0
            
            total_len = raw_labels.shape[0]
            stride = 150 # Non-overlapping windows

            for i in range(0, total_len - RAW_STEPS, stride):
                label_window = raw_labels[i : i + RAW_STEPS]
                vals, counts = np.unique(label_window, return_counts=True)
                label = int(vals[np.argmax(counts)])

                if label < NUM_OUTPUTS:
                    snippet = norm_data[i : i + RAW_STEPS, :]
                    self.samples.append((snippet, label))

            print(f"✅ Samples Loaded: {len(self.samples)}")

        except Exception as e:
            print(f"❌ Error reading file: {e}")

    def get_data(self):
        inputs = [s[0] for s in self.samples]
        targets = [s[1] for s in self.samples]
        if not inputs: return None, None
        return (torch.tensor(np.array(inputs), dtype=torch.float).to(device),
                torch.tensor(np.array(targets), dtype=torch.long).to(device))

# ==========================================
# TESTER
# ==========================================
def run_test():
    net = TunedSNN().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ Trained model not found! Run train.py first.")
        return

    net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    net.eval()
    print("🤖 Model Loaded.")

    loader = SingleFileLoader(DATA_FOLDER, TARGET_FILE)
    inputs, targets = loader.get_data()
    
    if inputs is None: return

    print(f"🚀 Running Inference...")
    
    batch_size = 128
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch_in = inputs[i : i + batch_size]
            batch_target = targets[i : i + batch_size]
            
            # Permute to (Time, Batch, Inputs)
            batch_in = batch_in.permute(1, 0, 2)
            
            spk_out = net(batch_in)
            _, pred = torch.max(spk_out.sum(dim=0), 1)

            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch_target.cpu().numpy())

    # Accuracy
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = np.mean(all_preds == all_targets) * 100
    print(f"📊 Accuracy: {acc:.2f}%")

    # Plot
    cm = confusion_matrix(all_targets, all_preds, labels=range(NUM_OUTPUTS))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(NUM_OUTPUTS))
    fig, ax = plt.subplots(figsize=(10,10))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f"Results for {TARGET_FILE}")
    plt.show()

if __name__ == "__main__":
    run_test()