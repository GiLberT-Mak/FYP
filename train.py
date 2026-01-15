import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import scipy.io
import os
import glob
from torch.utils.data import Dataset, DataLoader
import sys
# Add this line right after your imports:
sys.stdout.reconfigure(encoding='utf-8')

# ==========================================
# CONFIGURATION
# ==========================================
# Checks for GPU (CUDA for Windows/Linux, MPS for Mac M1/M2)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"⚙️ Running on: {device}")

# AUTOMATIC PATH SETUP
# This sets the path to the folder where this script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(CURRENT_DIR, 'Data') 
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, 'snn_nina_trained.pth')

NUM_INPUTS = 10
NUM_OUTPUTS = 18
NUM_STEPS = 100
BATCH_SIZE = 128
HIDDEN_SIZE = 512

# ==========================================
# DATASET CLASS
# ==========================================
class NinaEfficientDataset(Dataset):
    def __init__(self, folder_path):
        self.files = glob.glob(os.path.join(folder_path, "*.mat"))
        self.raw_data_list = []
        self.raw_labels_list = []
        self.indices = []

        print(f"📂 Analyzing Dataset in: {folder_path}")
        
        if not self.files:
            print("❌ No .mat files found! Check your Data folder.")

        for file_id, f_path in enumerate(self.files):
            try:
                mat = scipy.io.loadmat(f_path)
                raw_emg = mat['emg'].astype(np.float32)
                raw_labels = mat['restimulus']

                # Global Normalization
                p99 = np.percentile(np.abs(raw_emg), 99) + 1e-6
                raw_emg = raw_emg / p99 * 5.0

                self.raw_data_list.append(raw_emg)
                self.raw_labels_list.append(raw_labels)

                total_len = raw_emg.shape[0]
                stride = 50

                for i in range(0, total_len - NUM_STEPS, stride):
                    lbl_win = raw_labels[i : i+NUM_STEPS]
                    vals, counts = np.unique(lbl_win, return_counts=True)
                    label = int(vals[np.argmax(counts)])

                    if label >= NUM_OUTPUTS: continue
                    if label == 0 and np.random.rand() > 0.15: continue

                    self.indices.append((file_id, i, label))

            except Exception as e:
                print(f"   Error reading {f_path}: {e}")

        print(f"✅ Dataset Ready. Total Windows: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        file_id, start_t, label = self.indices[idx]
        data = self.raw_data_list[file_id][start_t : start_t + NUM_STEPS]
        data_tensor = torch.tensor(data, dtype=torch.float)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return data_tensor, label_tensor

# ==========================================
# MODEL
# ==========================================
class TunedSNN(nn.Module):
    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.threshold = 0.7
        self.beta = 0.90

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
            spk1 = self.drop1(spk1)

            cur2 = self.bn2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.drop2(spk2)

            cur_out = self.fc_out(spk2)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)

# ==========================================
# TRAINING LOOP
# ==========================================
def train_turbo():
    dataset = NinaEfficientDataset(DATA_FOLDER)
    if len(dataset) == 0: return

    # Windows usually prefers num_workers=0 or 2. If it hangs, set to 0.
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    net = TunedSNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Use GradScaler only if CUDA is available
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    print(f"\n🚀 Starting Training...")

    for epoch in range(101): # Reduced epochs for testing, increase to 101 later
        net.train()
        batch_loss = 0
        batch_acc = 0
        total_batches = 0

        for i, (data, targets) in enumerate(train_loader):
            data = data.to(device)
            targets = targets.to(device)
            data = data.permute(1, 0, 2) # [Time, Batch, Channels]

            optimizer.zero_grad()

            if use_amp:
                with torch.amp.autocast('cuda'):
                    spk_rec = net(data)
                    loss = loss_fn(spk_rec.sum(dim=0), targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard training for CPU/MPS
                spk_rec = net(data)
                loss = loss_fn(spk_rec.sum(dim=0), targets)
                loss.backward()
                optimizer.step()

            # Stats
            _, pred = torch.max(spk_rec.sum(dim=0), 1)
            acc = (pred == targets).float().mean() * 100
            batch_loss += loss.item()
            batch_acc += acc.item()
            total_batches += 1

            if i % 10 == 0:
                print(f"   [Ep {epoch}] Batch {i} | Loss: {loss.item():.4f}", end='\r')

        scheduler.step()
        
        if total_batches > 0:
            avg_loss = batch_loss / total_batches
            avg_acc = batch_acc / total_batches
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_acc:.2f}%")

            # Save model
            if epoch % 5 == 0:
                torch.save(net.state_dict(), MODEL_SAVE_PATH)
                print(f"💾 Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_turbo()