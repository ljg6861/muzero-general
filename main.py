#!/usr/bin/env python3
"""
MuZero-General: Benchmark and Evaluation
========================================

Main entry point for running benchmarks and evaluations.
For training, use train.py instead.
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set environment defaults
os.environ.setdefault('HF_HUB_READ_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def select_config():
    """Interactive config selection"""
    configs_dir = Path('configs')
    cfg_files = sorted([p.name for p in configs_dir.glob('*.json')])
    
    if not cfg_files:
        raise RuntimeError("No config files found in configs/*.json")
    
    print("\n📋 Available configurations:")
    for i, cfg_file in enumerate(cfg_files, 1):
        # Load config to show description
        try:
            with open(configs_dir / cfg_file) as f:
                cfg = json.load(f)
                desc = cfg.get('description', 'No description')
                print(f"  {i}. {cfg_file} - {desc}")
        except:
            print(f"  {i}. {cfg_file}")
    
    while True:
        try:
            choice = input(f"\nSelect config (1-{len(cfg_files)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(cfg_files):
                return cfg_files[idx]
            else:
                print(f"Please enter a number between 1 and {len(cfg_files)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)


def main():
    print("🚀 MuZero-General: Benchmark & Evaluation")
    print("=" * 40)
    print("\nNOTE: For training, use 'train.py' instead:")
    print("  python train.py configs/my_config.json")
    print("  torchrun --nproc_per_node=2 train.py configs/my_config.json")
    print()
    
    # Select config
    chosen_config = select_config()
    config_path = Path('configs') / chosen_config
    
    # Load config
    with open(config_path) as f:
        config = json.load(f)
    
    # Check if this is a benchmark config
    if 'benchmark' in config and config['benchmark'].get('mode') == 'evaluate':
        print(f"\n🎯 BENCHMARK MODE: {config.get('name', 'unknown')}")
        print(f"📊 Running evaluation on: {[d['name'] for d in config['benchmark']['datasets']]}")
        
        # Import and run benchmark
        try:
            from benchmark_openqa import OpenQABenchmark
            
            # Default model path or from config
            model_path = config['benchmark'].get('model_path', 'checkpoints/lm_current.pth')
            
            benchmark = OpenQABenchmark(config_path, model_path)
            results = benchmark.run_benchmark()
            
            print(f"\n🏆 Benchmark completed! Results saved to {config['benchmark']['evaluation']['output_file']}")
            
        except ImportError:
            print("❌ Benchmark module not available")
            sys.exit(1)
    
    else:
        print(f"\n📋 Selected config: {chosen_config}")
        print("This appears to be a training config.")
        print("\nTo train with this config, run:")
        print(f"  python train.py {config_path}")
        print("or for multi-GPU:")
        print(f"  torchrun --nproc_per_node=2 train.py {config_path}")


if __name__ == '__main__':
    main()