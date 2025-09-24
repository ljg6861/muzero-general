#!/usr/bin/env python3
"""
Phase A Complete Training Launcher
=================================
Complete implementation of Phase A: Baseline LM to Competence

Specifications Implemented:
✓ Model: 6L × 384d × 6 heads with proven tokenizer (DialoGPT-medium)
✓ Data: Scales to 50-100M tokens using cleaned Wikipedia and other sources  
✓ Schedule: AdamW, lr=2e-4, 3% warmup, cosine decay
✓ Batching: 1.5M tokens/step effective batch size via gradient accumulation
✓ Training: Until PPL plateaus with stable text generation

This establishes the solid LM foundation required before cognitive enhancements.
"""

import torch
import os
import sys
from datetime import datetime
from typing import Optional

# Add model directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from baseline_lm import create_baseline_model_with_tokenizer
from phase_a_training import PhaseATrainer
from test_phase_a_config import calculate_optimal_batch_config


def validate_environment():
    """Validate that the environment is ready for Phase A training."""
    
    print("Validating Training Environment")
    print("=" * 40)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ CUDA available: {gpu_name}")
        print(f"  GPU memory: {gpu_memory:.1f} GB")
        
        if gpu_memory < 4:
            print("  ⚠ Warning: Limited GPU memory may require smaller batches")
        
    else:
        print("⚠ CUDA not available - training will use CPU (very slow)")
    
    # Check Python packages
    required_packages = ['torch', 'transformers', 'numpy', 'matplotlib', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    if missing_packages:
        print(f"\nPlease install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ Environment validation complete")
    return True


def create_optimized_phase_a_config():
    """Create optimized Phase A configuration."""
    
    print("\nCreating Optimized Phase A Configuration")
    print("=" * 45)
    
    # Create baseline model to get config
    model, tokenizer, config = create_baseline_model_with_tokenizer()
    
    # Calculate optimal batch configuration
    optimal_batch = calculate_optimal_batch_config(
        target_tokens_per_step=1_500_000,  # 1.5M tokens (within 1-2M spec)
        seq_length=config.max_seq_length,
        available_memory_gb=8.0
    )
    
    # Update config with optimal settings
    config.micro_batch_size = optimal_batch['micro_batch_size']
    config.gradient_accumulation_steps = optimal_batch['gradient_accumulation_steps']
    
    print(f"\n✓ Phase A Configuration Optimized:")
    print(f"  Model: {config.num_layers}L × {config.hidden_size}d × {config.num_attention_heads}h")
    print(f"  Vocabulary: {config.vocab_size:,} tokens")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Tokenizer: {config.tokenizer_name}")
    print(f"  Micro batch: {config.micro_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {optimal_batch['actual_tokens_per_step']:,} tokens/step")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup: {config.warmup_ratio*100:.1f}%")
    print(f"  Schedule: {config.lr_schedule}")
    
    return model, tokenizer, config


def run_phase_a_training(
    model,
    tokenizer, 
    config,
    quick_test: bool = False,
    max_steps: Optional[int] = None
):
    """Run Phase A training to establish baseline competence."""
    
    print(f"\nStarting Phase A Training")
    print("=" * 30)
    
    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f"results/phase_a_baseline_{timestamp}"
    
    # Create trainer
    trainer = PhaseATrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir=save_dir
    )
    
    # Training parameters
    if quick_test:
        print("Running quick test training...")
        num_epochs = 1
        eval_every = 50
        save_every = 100
        max_steps = max_steps or 200
    else:
        print("Running full Phase A training...")
        num_epochs = 5
        eval_every = 500
        save_every = 1000
        max_steps = max_steps or 20000
    
    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Max steps: {max_steps}")
    print(f"  Eval every: {eval_every} steps")
    print(f"  Save every: {save_every} steps")
    print(f"  Save directory: {save_dir}")
    
    # Start training
    history = trainer.train(
        num_epochs=num_epochs,
        eval_every_steps=eval_every,
        save_every_steps=save_every,
        max_steps=max_steps
    )
    
    return trainer, history


def analyze_results(trainer, history):
    """Analyze and report Phase A training results."""
    
    print(f"\n" + "=" * 60)
    print("PHASE A TRAINING RESULTS")
    print("=" * 60)
    
    # Training statistics
    print(f"Training Statistics:")
    print(f"  Total steps: {trainer.global_step:,}")
    print(f"  Best perplexity: {trainer.best_ppl:.2f}")
    print(f"  Plateau count: {trainer.plateau_count}/{trainer.max_plateau_count}")
    print(f"  Convergence: {'✓ ACHIEVED' if trainer.plateau_count >= trainer.max_plateau_count else '⚠ IN PROGRESS'}")
    
    # Performance assessment
    if trainer.best_ppl < 30:
        status = "🎉 EXCELLENT"
        description = "Outstanding baseline LM performance"
    elif trainer.best_ppl < 50:
        status = "✓ GOOD"
        description = "Solid baseline LM foundation established"
    elif trainer.best_ppl < 100:
        status = "⚠ ACCEPTABLE"
        description = "Basic competence achieved, could improve with more training"
    else:
        status = "✗ NEEDS WORK"
        description = "Insufficient competence, check data quality and training setup"
    
    print(f"\nPerformance Assessment:")
    print(f"  Status: {status}")
    print(f"  Description: {description}")
    
    # Generate final samples
    print(f"\nFinal Text Generation Samples:")
    final_samples = trainer.generate_samples(3)
    for i, sample in enumerate(final_samples, 1):
        print(f"  {i}. {sample}")
    
    # Readiness for Phase B
    ready_for_phase_b = (trainer.best_ppl < 100 and 
                        trainer.plateau_count >= trainer.max_plateau_count)
    
    print(f"\nPhase B Readiness:")
    if ready_for_phase_b:
        print(f"  ✓ READY for cognitive enhancement integration")
        print(f"  Baseline LM has achieved sufficient competence")
        print(f"  Can proceed to Phase B: Cognitive enhancements")
    else:
        print(f"  ⚠ MORE TRAINING NEEDED")
        print(f"  Continue Phase A training until better convergence")
        print(f"  Target: PPL < 50 with stable plateau")
    
    return ready_for_phase_b


def main():
    """Main Phase A training launcher."""
    
    print("Phase A: Baseline Language Model to Competence")
    print("=" * 70)
    print("Complete implementation of Phase A specifications:")
    print("• Model: 6L × 384d × 6 heads with proven tokenizer")
    print("• Data: 50-100M tokens from cleaned sources")
    print("• Training: AdamW, lr=2e-4, 3% warmup, cosine decay")
    print("• Batching: 1-2M tokens/step effective batch size")
    print("• Goal: PPL plateau with stable text generation")
    print("")
    
    # Validate environment
    if not validate_environment():
        print("\n✗ Environment validation failed")
        return
    
    # Create optimized configuration
    try:
        model, tokenizer, config = create_optimized_phase_a_config()
    except Exception as e:
        print(f"\n✗ Configuration failed: {e}")
        return
    
    # Ask user for training mode
    print(f"\nTraining Mode Selection:")
    print(f"  1. Quick test (200 steps, ~5 minutes)")
    print(f"  2. Full training (up to 20k steps, several hours)")
    
    try:
        mode = input("Select mode (1 or 2): ").strip()
        if mode == "1":
            quick_test = True
            print("Selected: Quick test mode")
        elif mode == "2":
            quick_test = False
            print("Selected: Full training mode")
        else:
            print("Invalid selection, defaulting to quick test")
            quick_test = True
    except (KeyboardInterrupt, EOFError):
        print("\nDefaulting to quick test mode")
        quick_test = True
    
    # Run training
    try:
        trainer, history = run_phase_a_training(
            model=model,
            tokenizer=tokenizer,
            config=config,
            quick_test=quick_test
        )
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        return
    
    # Analyze results
    ready_for_phase_b = analyze_results(trainer, history)
    
    # Final recommendations
    print(f"\n" + "=" * 60)
    print("PHASE A COMPLETION")
    print("=" * 60)
    
    if ready_for_phase_b:
        print("🎉 Phase A SUCCESSFUL!")
        print("")
        print("Next Steps:")
        print("  1. Phase A baseline LM competence established")
        print("  2. Ready to implement Phase B: Cognitive enhancements")
        print("  3. Can now safely add cognitive losses without interference")
        print("  4. Baseline provides strong foundation for advanced capabilities")
        
    else:
        print("⚠ Phase A requires more work")
        print("")
        print("Recommendations:")
        print("  1. Continue training with full mode for better convergence")
        print("  2. Consider increasing training data if available")
        print("  3. Monitor generation quality and perplexity trends")
        print("  4. Only proceed to Phase B when baseline is solid")
    
    print(f"\nPhase A training logs and checkpoints saved to:")
    print(f"  {trainer.save_dir}")


if __name__ == "__main__":
    main()