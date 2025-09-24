#!/usr/bin/env python3
"""
Fixed Phase A Launcher
=====================
Complete launcher using token-budget trainer with proper data loading and schedule math.
Addresses all identified issues with the previous implementation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
import logging
from datetime import datetime
from typing import Optional

from baseline_lm import create_baseline_model_with_tokenizer
from token_budget_trainer import TokenBudgetTrainer, TrainingConfig
from data_registry import DataRegistry, DataConfig


def validate_environment():
    """Validate that the environment is ready for training."""
    
    print("Validating Training Environment")
    print("=" * 40)
    
    # Check CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA available: {gpu_name}")
        print(f"  GPU memory: {gpu_memory:.1f} GB")
    else:
        print("⚠ CUDA not available - training will use CPU (very slow)")
    
    # Check packages
    required_packages = ['torch', 'transformers', 'datasets', 'numpy', 'matplotlib', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    if missing_packages:
        print(f"\nInstall missing packages: pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ Environment validation complete")
    return True


def test_data_sources():
    """Test data source connectivity."""
    
    print("\nTesting Data Source Connectivity")
    print("=" * 40)
    
    from transformers import AutoTokenizer
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test registry
    registry = DataRegistry()
    
    # Test each source individually
    test_sources = ['c4', 'wikipedia', 'openwebtext', 'bookcorpus']
    working_sources = []
    
    for source_name in test_sources:
        print(f"Testing {source_name}...")
        
        config = DataConfig(
            train_tokens=100_000,  # Very small for testing
            seq_length=256,
            data_sources=[source_name],
            allow_fallback_synthetic=False
        )
        
        try:
            sources = registry.get_working_sources([source_name])
            if sources:
                working_sources.append(source_name)
                print(f"  ✓ {source_name} working")
            else:
                print(f"  ✗ {source_name} failed")
        except Exception as e:
            print(f"  ✗ {source_name} error: {e}")
    
    print(f"\nWorking sources: {working_sources}")
    return working_sources


def create_training_config(working_sources: list, mode: str = 'test') -> TrainingConfig:
    """Create training configuration based on mode."""
    
    if mode == 'test':
        # Quick test configuration
        config = TrainingConfig(
            train_tokens=10_000_000,        # 10M tokens (~5-10 minutes)
            eval_tokens=1_000_000,          # 1M tokens for eval
            warmup_tokens=1_000_000,        # 1M tokens warmup
            
            micro_batch_size=1,
            gradient_accumulation_steps=512,  # 512K tokens/step
            seq_length=512,                 # Shorter for speed
            
            eval_every_tokens=2_000_000,    # Eval every 2M tokens
            save_every_tokens=5_000_000,    # Save every 5M tokens
            
            data_sources=working_sources or ['c4'],
            allow_fallback_synthetic=True   # Allow fallback for testing
        )
    
    elif mode == 'small':
        # Small scale training
        config = TrainingConfig(
            train_tokens=100_000_000,       # 100M tokens (~30-60 minutes)
            eval_tokens=5_000_000,          # 5M tokens for eval
            warmup_tokens=10_000_000,       # 10M tokens warmup
            
            micro_batch_size=1,
            gradient_accumulation_steps=1024,  # 1M tokens/step
            seq_length=1024,
            
            eval_every_tokens=20_000_000,   # Eval every 20M tokens
            save_every_tokens=50_000_000,   # Save every 50M tokens
            
            data_sources=working_sources or ['c4'],
            allow_fallback_synthetic=len(working_sources) == 0
        )
    
    elif mode == 'full':
        # Full Phase A specification
        config = TrainingConfig(
            train_tokens=2_000_000_000,     # 2B tokens (Phase A spec)
            eval_tokens=10_000_000,         # 10M tokens for eval
            warmup_tokens=100_000_000,      # 100M tokens warmup (5% of total)
            
            micro_batch_size=1,
            gradient_accumulation_steps=1465,  # 1.5M tokens/step (Phase A spec)
            seq_length=1024,
            
            eval_every_tokens=100_000_000,  # Eval every 100M tokens
            save_every_tokens=500_000_000,  # Save every 500M tokens
            
            data_sources=working_sources or ['c4'],
            allow_fallback_synthetic=False  # No fallback for serious training
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return config


def run_training(config: TrainingConfig, save_dir: str):
    """Run training with the given configuration."""
    
    print(f"\nCreating Phase A Model")
    print("=" * 30)
    
    # Create model
    model, tokenizer, _ = create_baseline_model_with_tokenizer()
    
    print(f"Model: {config.num_layers}L × {config.hidden_size}d")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Create trainer
    trainer = TokenBudgetTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir=save_dir
    )
    
    # Run training
    results = trainer.train()
    
    return trainer, results


def analyze_results(trainer, results):
    """Analyze and report training results."""
    
    print(f"\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    
    print(f"Training completed:")
    print(f"  Steps: {results['global_step']:,}")
    print(f"  Tokens processed: {results['tokens_processed']:,}")
    print(f"  Best PPL: {results['best_ppl']:.2f}")
    print(f"  Final PPL: {results['final_ppl']:.2f}")
    
    # Assess quality
    if results['best_ppl'] < 30:
        status = "🎉 EXCELLENT"
        description = "Outstanding performance, ready for Phase B"
    elif results['best_ppl'] < 50:
        status = "✓ GOOD"
        description = "Solid baseline, can proceed to Phase B"
    elif results['best_ppl'] < 100:
        status = "⚠ ACCEPTABLE"
        description = "Basic competence, consider more training"
    else:
        status = "✗ NEEDS WORK"
        description = "Insufficient quality, need more data/training"
    
    print(f"\nPerformance Assessment:")
    print(f"  Status: {status}")
    print(f"  Description: {description}")
    
    # Generate final samples
    print(f"\nFinal Text Generation:")
    try:
        samples = trainer.generate_samples(3)
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample}")
    except Exception as e:
        print(f"  Generation failed: {e}")
    
    # Training recommendations
    schedule_params = trainer.schedule_params
    effective_batch = schedule_params['global_batch_tokens']
    
    print(f"\nTraining Analysis:")
    print(f"  Effective batch: {effective_batch:,} tokens/step")
    if effective_batch >= 1_000_000:
        print(f"  ✓ Meets Phase A batch requirement (≥1M tokens/step)")
    else:
        print(f"  ⚠ Below Phase A batch requirement (<1M tokens/step)")
    
    print(f"  Total training tokens: {results['tokens_processed']:,}")
    if results['tokens_processed'] >= 1_000_000_000:
        print(f"  ✓ Substantial training data (≥1B tokens)")
    else:
        print(f"  ⚠ Limited training data (<1B tokens)")
    
    return results['best_ppl'] < 100


def main():
    """Main launcher function."""
    
    print("Fixed Phase A: Baseline Language Model Training")
    print("=" * 70)
    print("Addresses all identified issues:")
    print("• Robust data loading with HF streaming (no local scripts)")
    print("• Token-budget training (no len() dependency)")
    print("• Fixed scheduler math (no LR=0 artifacts)")
    print("• Proper batch sizes (≥1M tokens/step)")
    print("• Real dataset connectivity")
    print("")
    
    # Validate environment
    if not validate_environment():
        print("\n✗ Environment validation failed")
        return
    
    # Test data sources
    working_sources = test_data_sources()
    
    if not working_sources:
        print("\n🚨 WARNING: No real data sources working!")
        print("Training will use synthetic fallback data.")
        print("This is for testing only - not suitable for real training.")
    
    # Select training mode
    print(f"\nTraining Mode Selection:")
    print(f"  1. Test mode (10M tokens, ~10 minutes)")
    print(f"  2. Small mode (100M tokens, ~1 hour)")
    print(f"  3. Full mode (2B tokens, Phase A spec, several hours)")
    
    try:
        choice = input("Select mode (1, 2, or 3): ").strip()
        if choice == "1":
            mode = "test"
            print("Selected: Test mode")
        elif choice == "2":
            mode = "small"
            print("Selected: Small mode")
        elif choice == "3":
            mode = "full"
            print("Selected: Full mode")
        else:
            print("Invalid choice, defaulting to test mode")
            mode = "test"
    except (KeyboardInterrupt, EOFError):
        print("\nDefaulting to test mode")
        mode = "test"
    
    # Create configuration
    config = create_training_config(working_sources, mode)
    
    # Setup save directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"results/fixed_phase_a_{mode}_{timestamp}"
    
    print(f"\nStarting {mode} training...")
    print(f"Save directory: {save_dir}")
    
    try:
        trainer, results = run_training(config, save_dir)
        success = analyze_results(trainer, results)
        
        print(f"\n" + "=" * 60)
        if success:
            print("🎉 Training completed successfully!")
            if mode == "full":
                print("Phase A baseline is ready for Phase B cognitive enhancements.")
            else:
                print("Run full mode for complete Phase A training.")
        else:
            print("⚠ Training completed with issues.")
            print("Consider adjusting configuration or training longer.")
        
        print(f"\nResults saved to: {save_dir}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()