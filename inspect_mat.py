import scipy.io
import glob
import numpy as np
import os

files = glob.glob('Data/*.mat')
if not files:
    print("No .mat files found in Data/")
else:
    f_path = files[0]
    print(f"Reading {f_path}")
    mat = scipy.io.loadmat(f_path)
    raw_labels = mat['restimulus'].flatten()
    print("Label shape:", raw_labels.shape)
    
    # Find transitions
    changes = np.diff(raw_labels)
    starts = np.where(changes != 0)[0] + 1
    
    from collections import defaultdict
    blocks = defaultdict(list)
    
    last_idx = 0
    current_label = raw_labels[0]
    
    for start in starts:
        if current_label != 0:
            blocks[current_label].append((last_idx, start))
        current_label = raw_labels[start]
        last_idx = start
        
    if current_label != 0:
        blocks[current_label].append((last_idx, len(raw_labels)))
        
    print("Trials per label:")
    for k in sorted(blocks.keys()):
        print(f"Label {k}: {len(blocks[k])} trials")
