import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_INPUTS, NUM_OUTPUTS

class MirrorCNN(nn.Module):
    """
    1D-CNN that mirror the (256, 256, 128) architecture of the TunedSNN.
    Used as a baseline for accuracy vs. energy efficiency comparison.
    """
    def __init__(self, layer_sizes=(256, 256, 128)):
        super().__init__()
        
        # Block 1 [B, 10, 50] -> [B, 256, 25]
        self.conv1 = nn.Conv1d(NUM_INPUTS, layer_sizes[0], kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(layer_sizes[0])
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Block 2 [B, 256, 25] -> [B, 256, 12]
        self.conv2 = nn.Conv1d(layer_sizes[0], layer_sizes[1], kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(layer_sizes[1])
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Block 3 [B, 256, 12] -> [B, 128, 6]
        self.conv3 = nn.Conv1d(layer_sizes[1], layer_sizes[2], kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(layer_sizes[2])
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        self.dropout = nn.Dropout(0.25)
        
        # Output layer
        # Flattened size: filters (128) * timesteps after 3 maxpools (6) = 768
        self.fc_out = nn.Linear(layer_sizes[2] * 6, NUM_OUTPUTS)

    def forward(self, x):
        """
        Args:
            x: [Batch, Time, Channels] -> needs permute for Conv1d
        """
        # x is [B, T, C] -> Conv1d expects [B, C, T]
        x = x.permute(0, 2, 1)
        
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc_out(x)
        
        return x
