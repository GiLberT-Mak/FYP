import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from config import NUM_INPUTS, NUM_OUTPUTS, HIDDEN_SIZE, SLOPE, THRESHOLD, BETA


class TunedSNN(nn.Module):
    """
    Three-layer Spiking Neural Network with:
      - Learnable membrane-decay (beta) and firing threshold per LIF layer
      - BatchNorm + Dropout regularisation on every hidden layer
      - Fast-sigmoid surrogate gradient for backprop through spikes
    """

    def __init__(self):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=SLOPE)

        # ── Layer 1 ──────────────────────────────────────────────
        self.fc1  = nn.Linear(NUM_INPUTS, HIDDEN_SIZE)
        self.bn1  = nn.BatchNorm1d(HIDDEN_SIZE)
        self.lif1 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)
        self.drop1 = nn.Dropout(0.25)

        # ── Layer 2 ──────────────────────────────────────────────
        self.fc2  = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.bn2  = nn.BatchNorm1d(HIDDEN_SIZE)
        self.lif2 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)
        self.drop2 = nn.Dropout(0.25)

        # ── Layer 3 (new) ─────────────────────────────────────────
        self.fc3  = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.bn3  = nn.BatchNorm1d(HIDDEN_SIZE // 2)
        self.lif3 = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                               spike_grad=spike_grad,
                               learn_beta=True, learn_threshold=True)
        self.drop3 = nn.Dropout(0.25)

        # ── Output layer ─────────────────────────────────────────
        self.fc_out  = nn.Linear(HIDDEN_SIZE // 2, NUM_OUTPUTS)
        self.lif_out = snn.Leaky(beta=BETA, threshold=THRESHOLD,
                                  spike_grad=spike_grad,
                                  learn_beta=True, learn_threshold=True)

    def forward(self, x):
        """
        Args:
            x: [Time, Batch, Channels]
        Returns:
            spk_out_rec: [Time, Batch, NUM_OUTPUTS]  (spike train at output layer)
        """
        mem1    = self.lif1.init_leaky()
        mem2    = self.lif2.init_leaky()
        mem3    = self.lif3.init_leaky()
        mem_out = self.lif_out.init_leaky()
        spk_out_rec = []

        for step in range(x.size(0)):
            # Layer 1
            cur1 = self.bn1(self.fc1(x[step]))
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.drop1(spk1)

            # Layer 2
            cur2 = self.bn2(self.fc2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.drop2(spk2)

            # Layer 3
            cur3 = self.bn3(self.fc3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3 = self.drop3(spk3)

            # Output
            cur_out = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            spk_out_rec.append(spk_out)

        return torch.stack(spk_out_rec, dim=0)
