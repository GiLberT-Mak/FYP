import os
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER  = os.path.join(CURRENT_DIR, 'Data')
MODEL_DIR   = os.path.join(CURRENT_DIR, 'Trained_SNN')
RESULT_DIR         = os.path.join(CURRENT_DIR, 'Result')
TRAIN_RECORD_DIR   = os.path.join(RESULT_DIR, 'Training-Records')
CM_DIR             = os.path.join(RESULT_DIR, 'Confusion-Matrices')
RASTER_DIR         = os.path.join(RESULT_DIR, 'Spike-Rasters')
EFFICIENCY_DIR     = os.path.join(RESULT_DIR, 'Efficiency-Metrics')
SUMMARY_DIR        = os.path.join(RESULT_DIR, 'Summary')
CACHE_DIR          = os.path.join(CURRENT_DIR, '.cache')   # Preprocessed .npz cache

NUM_INPUTS  = 10
NUM_OUTPUTS = 18
NUM_STEPS   = 50    # Standardized window size (timesteps)
BATCH_SIZE  = 2048
TEST_BATCH_SIZE = 1
HIDDEN_SIZE = 512

# SNN neuron parameters
SLOPE     = 25
THRESHOLD = 0.7
BETA      = 0.90   # Initial membrane decay (learnable per-layer)

# Training hyper-parameters
NUM_EPOCHS              = 101
EARLY_STOPPING_PATIENCE = 15   # Stop if val acc doesn't improve for this many epochs

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
