#!/usr/bin/env python3
"""
Simple Phase A Test
==================
Quick test to verify Phase A training works with the simple data loader.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from baseline_lm import create_baseline_model_with_tokenizer
from phase_a_training import PhaseATrainer
from simple_data import create_simple_data_loader


def quick_phase_a_test():
    """Run a very quick Phase A test to verify everything works."""
    
    print("Quick Phase A Test")
    print("=" * 30)
    
    # Create model and tokenizer
    print("Creating model...")
    model, tokenizer, config = create_baseline_model_with_tokenizer()
    
    # Adjust config for quick test
    config.micro_batch_size = 1
    config.gradient_accumulation_steps = 2  # Very small for testing
    
    print(f"Model: {config.num_layers}L × {config.hidden_size}d")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer with minimal settings
    trainer = PhaseATrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir="results/quick_test"
    )
    
    # Override data loaders with simple ones
    print("\nCreating simple data loaders...")
    trainer.train_loader = create_simple_data_loader(
        tokenizer=tokenizer,
        batch_size=1,
        seq_length=512,  # Shorter for speed
        num_samples=100   # Very small for testing
    )
    
    trainer.eval_loader = create_simple_data_loader(
        tokenizer=tokenizer,
        batch_size=1,
        seq_length=512,
        num_samples=20
    )
    
    # Run minimal training
    print("\nRunning minimal training...")
    history = trainer.train(
        num_epochs=1,
        eval_every_steps=10,
        save_every_steps=1000,  # No saves for this test
        max_steps=15  # Just 15 steps
    )
    
    print(f"\nTest Results:")
    print(f"  Steps completed: {trainer.global_step}")
    print(f"  Final PPL: {trainer.best_ppl:.2f}")
    print(f"  Training worked: {'✓' if trainer.global_step > 0 else '✗'}")
    
    # Test generation
    print(f"\nSample generation:")
    samples = trainer.generate_samples(2)
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. {sample[:100]}...")
    
    return trainer.global_step > 0


if __name__ == "__main__":
    success = quick_phase_a_test()
    if success:
        print("\n🎉 Phase A test PASSED!")
        print("The system is working correctly.")
    else:
        print("\n❌ Phase A test FAILED!")
        print("Check the error messages above.")