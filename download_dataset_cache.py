#!/usr/bin/env python3
"""
Pre-download HuggingFace dataset subsets to local cache to avoid rate limits during training.
Downloads small streaming subsets that will be cached locally.
"""

import os
from datasets import load_dataset
import time

def download_dataset_subset(dataset_name, config_name=None, split='train', max_examples=1000):
    """Download a small subset of dataset to populate local cache."""
    print(f"\nDownloading subset of {dataset_name}...")
    
    try:
        # Load in streaming mode with take() to limit size
        load_params = {
            'path': dataset_name,
            'split': split,
            'streaming': True,
        }
        
        if config_name:
            load_params['name'] = config_name
            
        dataset = load_dataset(**load_params)
        
        # Cache a small subset by iterating through it
        count = 0
        for item in dataset:
            count += 1
            if count >= max_examples:
                break
                
        print(f"✓ Cached {count} examples from {dataset_name}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to cache {dataset_name}: {e}")
        return False

def main():
    """Download subsets of Phase 1 datasets to populate cache."""
    
    # Your Phase 1 datasets (excluding problematic ones)
    datasets_to_cache = [
        ('wikimedia/wikipedia', '20231101.en'),
        ('rojagtap/bookcorpus', None),
        ('HuggingFaceFW/fineweb-edu', None),
        # Skip the problematic ones - fallbacks will handle them
        # ('HuggingFaceFW/fineweb', None),  # This one has script issues
        # ('allenai/peS2o', 'abstract'),    # This one is deprecated
    ]
    
    print("🔄 Pre-downloading dataset subsets to local cache...")
    print("This will help avoid rate limits during training.")
    
    successful = 0
    total = len(datasets_to_cache)
    
    for dataset_name, config_name in datasets_to_cache:
        success = download_dataset_subset(dataset_name, config_name)
        if success:
            successful += 1
        
        # Add delay to avoid rate limiting during download
        time.sleep(5)
    
    print(f"\n✓ Successfully cached {successful}/{total} datasets")
    print("Cache location:", os.getenv('HF_DATASETS_CACHE', '~/.cache/huggingface/datasets/'))
    print("Training should now use cached data and avoid rate limits.")

if __name__ == "__main__":
    main()