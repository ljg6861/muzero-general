#!/usr/bin/env python3
"""
Phase A Batch Size Optimization and Testing
==========================================
Optimizes gradient accumulation to achieve Phase A specification:
- Effective batch size of 1-2M tokens/step
- Tests model performance and memory usage
- Validates training setup before full training
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Dict, Tuple
import psutil
import os

from baseline_lm import create_baseline_model_with_tokenizer, BaselineLMConfig
from phase_a_training import PhaseATrainer


def calculate_optimal_batch_config(
    target_tokens_per_step: int = 1_500_000,  # 1.5M tokens target
    seq_length: int = 1024,
    available_memory_gb: float = 8.0
) -> Dict[str, int]:
    """Calculate optimal batch configuration for target tokens/step."""
    
    print("Calculating Optimal Batch Configuration")
    print("=" * 50)
    print(f"Target: {target_tokens_per_step:,} tokens/step")
    print(f"Sequence length: {seq_length}")
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    # Start with constraints
    max_micro_batch = 16  # Conservative limit for memory
    min_micro_batch = 1
    
    best_config = None
    best_efficiency = 0
    
    configs = []
    
    for micro_batch in range(min_micro_batch, max_micro_batch + 1):
        # Calculate required gradient accumulation steps
        required_grad_accum = math.ceil(target_tokens_per_step / (micro_batch * seq_length))
        
        # Actual tokens per step with this config
        actual_tokens = micro_batch * required_grad_accum * seq_length
        
        # Efficiency score (prefer closer to target, lower memory usage)
        target_ratio = min(actual_tokens / target_tokens_per_step, 
                          target_tokens_per_step / actual_tokens)
        memory_efficiency = 1.0 / micro_batch  # Smaller batches = better memory efficiency
        efficiency = target_ratio * 0.8 + memory_efficiency * 0.2
        
        config = {
            'micro_batch_size': micro_batch,
            'gradient_accumulation_steps': required_grad_accum,
            'actual_tokens_per_step': actual_tokens,
            'efficiency_score': efficiency
        }
        
        configs.append(config)
        
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_config = config
    
    # Display options
    print("\nConfiguration Options:")
    print("-" * 80)
    print(f"{'Micro Batch':<12} {'Grad Accum':<11} {'Tokens/Step':<12} {'Efficiency':<10} {'Status'}")
    print("-" * 80)
    
    for config in configs:
        status = "✓ BEST" if config == best_config else ""
        if config['actual_tokens_per_step'] > 2_000_000:
            status = "⚠ HIGH"
        elif config['actual_tokens_per_step'] < 1_000_000:
            status = "⚠ LOW"
        
        print(f"{config['micro_batch_size']:<12} {config['gradient_accumulation_steps']:<11} "
              f"{config['actual_tokens_per_step']:<12,} {config['efficiency_score']:<10.3f} {status}")
    
    print("-" * 80)
    print(f"\nRecommended Configuration:")
    print(f"  Micro batch size: {best_config['micro_batch_size']}")
    print(f"  Gradient accumulation: {best_config['gradient_accumulation_steps']}")
    print(f"  Effective tokens/step: {best_config['actual_tokens_per_step']:,}")
    print(f"  Efficiency score: {best_config['efficiency_score']:.3f}")
    
    return best_config


def test_memory_usage(model, tokenizer, batch_config: Dict[str, int]) -> Dict[str, float]:
    """Test memory usage with the proposed batch configuration."""
    
    print("\nTesting Memory Usage")
    print("=" * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create test batch
    micro_batch_size = batch_config['micro_batch_size']
    seq_length = 1024
    vocab_size = len(tokenizer)
    
    # Clear memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Test forward + backward pass
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    start_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    for step in range(batch_config['gradient_accumulation_steps']):
        # Create batch
        input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device=device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs['loss'] / batch_config['gradient_accumulation_steps']
        
        # Backward pass
        loss.backward()
        
        # Memory check
        current_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        
        if step == 0:
            print(f"  After first forward+backward: {current_memory:.2f} GB")
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Final memory
    final_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    peak_memory = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    memory_stats = {
        'start_memory_gb': start_memory,
        'final_memory_gb': final_memory,
        'peak_memory_gb': peak_memory,
        'memory_increase_gb': final_memory - start_memory
    }
    
    print(f"  Peak memory usage: {peak_memory:.2f} GB")
    print(f"  Memory increase: {memory_stats['memory_increase_gb']:.2f} GB")
    
    # Memory efficiency check
    if peak_memory < 6.0:
        print("  ✓ Memory usage looks good")
    elif peak_memory < 10.0:
        print("  ⚠ High memory usage but manageable")
    else:
        print("  ✗ Memory usage too high")
    
    return memory_stats


def test_training_throughput(model, tokenizer, batch_config: Dict[str, int]) -> Dict[str, float]:
    """Test training throughput with the batch configuration."""
    
    print("\nTesting Training Throughput")
    print("=" * 35)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    micro_batch_size = batch_config['micro_batch_size']
    grad_accum_steps = batch_config['gradient_accumulation_steps']
    seq_length = 1024
    vocab_size = len(tokenizer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # Warm up
    print("  Warming up...")
    for _ in range(3):
        input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device=device)
        outputs = model(input_ids=input_ids, labels=input_ids)
        loss = outputs['loss']
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Actual timing
    print("  Measuring throughput...")
    num_test_steps = 5
    total_tokens = 0
    start_time = time.time()
    
    for step in range(num_test_steps):
        # Gradient accumulation loop
        for acc_step in range(grad_accum_steps):
            input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device=device)
            
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs['loss'] / grad_accum_steps
            loss.backward()
            
            total_tokens += micro_batch_size * seq_length
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # Calculate metrics
    throughput_stats = {
        'total_tokens': total_tokens,
        'elapsed_time': elapsed,
        'tokens_per_second': total_tokens / elapsed,
        'steps_per_second': num_test_steps / elapsed,
        'effective_batch_size': batch_config['actual_tokens_per_step']
    }
    
    print(f"  Total tokens processed: {total_tokens:,}")
    print(f"  Elapsed time: {elapsed:.2f} seconds")
    print(f"  Throughput: {throughput_stats['tokens_per_second']:,.0f} tokens/sec")
    print(f"  Steps per second: {throughput_stats['steps_per_second']:.2f}")
    
    # Efficiency assessment
    if throughput_stats['tokens_per_second'] > 10000:
        print("  ✓ Good throughput")
    elif throughput_stats['tokens_per_second'] > 5000:
        print("  ⚠ Moderate throughput")
    else:
        print("  ✗ Low throughput - consider optimization")
    
    return throughput_stats


def test_model_convergence(model, tokenizer, batch_config: Dict[str, int]) -> Dict[str, float]:
    """Test that model can learn with this batch configuration."""
    
    print("\nTesting Model Convergence")
    print("=" * 30)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    micro_batch_size = batch_config['micro_batch_size']
    grad_accum_steps = batch_config['gradient_accumulation_steps']
    seq_length = 128  # Shorter for faster testing
    vocab_size = len(tokenizer)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    # Create repeated batch (should overfit quickly)
    fixed_input_ids = torch.randint(0, vocab_size, (micro_batch_size, seq_length), device=device)
    
    losses = []
    perplexities = []
    
    print("  Training on fixed batch to test convergence...")
    
    for step in range(20):  # 20 optimization steps
        optimizer.zero_grad()
        
        # Gradient accumulation
        total_loss = 0
        for acc_step in range(grad_accum_steps):
            outputs = model(input_ids=fixed_input_ids, labels=fixed_input_ids)
            loss = outputs['loss'] / grad_accum_steps
            loss.backward()
            total_loss += loss.item()
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record metrics
        losses.append(total_loss)
        perplexity = math.exp(total_loss)
        perplexities.append(perplexity)
        
        if step % 5 == 0:
            print(f"    Step {step:2d}: Loss={total_loss:.4f}, PPL={perplexity:.2f}")
    
    # Analyze convergence
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    
    convergence_stats = {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_reduction_percent': loss_reduction,
        'initial_ppl': perplexities[0],
        'final_ppl': perplexities[-1]
    }
    
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.1f}%")
    
    if loss_reduction > 50:
        print("  ✓ Model converges well")
    elif loss_reduction > 20:
        print("  ⚠ Model converges slowly")
    else:
        print("  ✗ Model not converging - check configuration")
    
    return convergence_stats


def main():
    """Run Phase A batch optimization and testing."""
    
    print("Phase A: Batch Size Optimization and Testing")
    print("=" * 60)
    
    # Create model
    print("Creating baseline model...")
    model, tokenizer, config = create_baseline_model_with_tokenizer()
    
    # Calculate optimal batch configuration
    optimal_config = calculate_optimal_batch_config(
        target_tokens_per_step=1_500_000,  # 1.5M tokens (within 1-2M range)
        seq_length=config.max_seq_length,
        available_memory_gb=8.0
    )
    
    # Update model config with optimal settings
    config.micro_batch_size = optimal_config['micro_batch_size']
    config.gradient_accumulation_steps = optimal_config['gradient_accumulation_steps']
    
    print(f"\nUpdated configuration:")
    print(f"  Micro batch size: {config.micro_batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {optimal_config['actual_tokens_per_step']:,} tokens/step")
    
    # Run tests
    try:
        # Test 1: Memory usage
        memory_stats = test_memory_usage(model, tokenizer, optimal_config)
        
        # Test 2: Training throughput
        throughput_stats = test_training_throughput(model, tokenizer, optimal_config)
        
        # Test 3: Model convergence
        convergence_stats = test_model_convergence(model, tokenizer, optimal_config)
        
        # Summary
        print("\n" + "=" * 60)
        print("PHASE A OPTIMIZATION SUMMARY")
        print("=" * 60)
        
        print(f"✓ Batch Configuration:")
        print(f"  Micro batch: {optimal_config['micro_batch_size']}")
        print(f"  Grad accumulation: {optimal_config['gradient_accumulation_steps']}")
        print(f"  Effective tokens/step: {optimal_config['actual_tokens_per_step']:,}")
        print(f"  Target compliance: {'✓ MEETS SPEC' if 1_000_000 <= optimal_config['actual_tokens_per_step'] <= 2_000_000 else '⚠ OUTSIDE SPEC'}")
        
        print(f"\n✓ Memory Usage:")
        print(f"  Peak memory: {memory_stats['peak_memory_gb']:.2f} GB")
        print(f"  Status: {'✓ EFFICIENT' if memory_stats['peak_memory_gb'] < 8 else '⚠ HIGH'}")
        
        print(f"\n✓ Training Performance:")
        print(f"  Throughput: {throughput_stats['tokens_per_second']:,.0f} tokens/sec")
        print(f"  Steps/sec: {throughput_stats['steps_per_second']:.2f}")
        
        print(f"\n✓ Model Convergence:")
        print(f"  Loss reduction: {convergence_stats['loss_reduction_percent']:.1f}%")
        print(f"  Status: {'✓ CONVERGES' if convergence_stats['loss_reduction_percent'] > 20 else '⚠ SLOW'}")
        
        # Final assessment
        meets_spec = (1_000_000 <= optimal_config['actual_tokens_per_step'] <= 2_000_000)
        good_memory = memory_stats['peak_memory_gb'] < 10
        good_convergence = convergence_stats['loss_reduction_percent'] > 20
        
        if meets_spec and good_memory and good_convergence:
            print(f"\n🎉 SUCCESS: Phase A configuration is READY for training!")
            print(f"   Meets 1-2M tokens/step specification")
            print(f"   Efficient memory usage")
            print(f"   Model converges properly")
            print(f"\n   Ready to run: python model/phase_a_training.py")
        else:
            print(f"\n⚠ ISSUES DETECTED:")
            if not meets_spec:
                print(f"   - Batch size outside 1-2M token spec")
            if not good_memory:
                print(f"   - High memory usage")
            if not good_convergence:
                print(f"   - Poor convergence")
            print(f"\n   Consider adjusting configuration before full training")
    
    except Exception as e:
        print(f"\n✗ Testing failed: {e}")
        print("This may be due to insufficient GPU memory or other hardware limitations.")
        print("Consider reducing batch sizes or using CPU for testing.")


if __name__ == "__main__":
    main()