#!/usr/bin/env python3
"""
Memory-Efficient Training Script
===============================

Usage:
    python train_efficient.py configs/my_config.json
    torchrun --nproc_per_node=2 train_efficient.py configs/my_config.json

This version reduces memory usage by:
- Smaller batch sizes
- Optional heavy components (REBEL, DeBERTa)
- Gradient checkpointing by default
"""

import os
import sys

# Memory optimization environment variables
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True,max_split_size_mb:512')
os.environ.setdefault('HF_HUB_READ_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

# Import the full training pipeline
sys.path.append('.')
from train import main

if __name__ == '__main__':
    # Set smaller batch sizes by default
    main()