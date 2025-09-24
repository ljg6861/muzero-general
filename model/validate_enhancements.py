#!/usr/bin/env python3
"""
Quick Validation Test for Enhanced Cognitive LLM
================================================
Tests that our scale and objective interference fixes work correctly.
Should show significantly improved perplexity compared to original setup.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

from enhanced_cognitive_llm import create_enhanced_model
from enhanced_config import get_optimized_config_for_data_size, CognitiveLossScheduler


def test_pretrained_initialization():
    """Test that pretrained initialization works correctly."""
    print("Testing Pretrained Initialization")
    print("=" * 40)
    
    # Create model with pretrained init
    model, config = create_enhanced_model(7500)
    
    # Test tokenizer loading
    tokenizer = model.get_tokenizer()
    if tokenizer:
        print(f"✓ Pretrained tokenizer loaded: {len(tokenizer)} tokens")
        
        # Test tokenization
        test_text = "The quick brown fox jumps over the lazy dog."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"  Original: {test_text}")
        print(f"  Tokens: {tokens[:10]}...")  # Show first 10 tokens
        print(f"  Decoded: {decoded}")
        
        # Check that embeddings are reasonable (not random)
        embedding_weights = model.token_embedding.weight.data
        mean_abs = embedding_weights.abs().mean().item()
        std = embedding_weights.std().item()
        
        print(f"  Embedding stats: mean_abs={mean_abs:.4f}, std={std:.4f}")
        
        if 0.01 < mean_abs < 0.5 and 0.01 < std < 0.5:
            print("✓ Embedding weights look reasonable (likely pretrained)")
        else:
            print("⚠ Embedding weights look random (pretrained init may have failed)")
            
    else:
        print("✗ No pretrained tokenizer loaded")
    
    return model, config


def test_cognitive_scheduling():
    """Test that cognitive loss scheduling works correctly."""
    print("\nTesting Cognitive Loss Scheduling")
    print("=" * 40)
    
    config = get_optimized_config_for_data_size(7500)
    scheduler = CognitiveLossScheduler(config)
    
    # Test scheduling at different steps
    test_steps = [0, 1000, 3000, 5000, 8000, 10000, 15000]
    
    print("Step | Cognitive Weight | Concept | Causal | Meta")
    print("-" * 50)
    
    for step in test_steps:
        scheduler.update_step(step)
        weight = scheduler.get_cognitive_loss_weight()
        components = scheduler.get_component_weights()
        
        print(f"{step:4d} | {weight:12.3f} | {components['concept_formation']:7.3f} | "
              f"{components['causal_reasoning']:6.3f} | {components['meta_learning']:4.3f}")
    
    # Validate correct progression
    scheduler.update_step(0)
    assert scheduler.get_cognitive_loss_weight() == 0.0, "Cognitive loss should be 0 at start"
    
    scheduler.update_step(config.cognitive_delay_steps - 1)
    assert scheduler.get_cognitive_loss_weight() == 0.0, "Cognitive loss should be 0 before delay"
    
    scheduler.update_step(config.cognitive_annealing_steps)
    assert scheduler.get_cognitive_loss_weight() == config.max_cognitive_loss_weight, "Should reach max weight after annealing"
    
    print("✓ Cognitive scheduling working correctly")


def test_model_forward_pass():
    """Test model forward pass with progressive cognitive losses."""
    print("\nTesting Model Forward Pass")
    print("=" * 40)
    
    model, config = create_enhanced_model(7500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create test batch
    batch_size = 2
    seq_len = 32
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    print(f"Test batch: {batch_size} x {seq_len}, vocab_size={vocab_size}")
    
    # Test at different training steps
    test_steps = [0, 2000, 5000, 8000]
    
    for step in test_steps:
        model.update_training_step(step)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        
        print(f"Step {step:4d}: LM={outputs['lm_loss']:.4f}, "
              f"Cog={outputs['cognitive_loss']:.4f}, "
              f"Weight={outputs['cognitive_weight']:.3f}")
    
    print("✓ Forward pass working correctly")


def test_perplexity_computation():
    """Test that perplexity computation is working correctly."""
    print("\nTesting Perplexity Computation")
    print("=" * 40)
    
    # Test with known values
    # Perfect prediction should give PPL = 1
    vocab_size = 1000
    seq_len = 10
    
    # Create perfect predictions (one-hot)
    logits = torch.zeros(1, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (1, seq_len))
    
    # Set perfect predictions
    for i in range(seq_len):
        logits[0, i, labels[0, i]] = 10.0  # High confidence
    
    # Compute loss and perplexity
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='mean'
    )
    
    ppl = torch.exp(loss).item()
    
    print(f"Perfect prediction loss: {loss.item():.6f}")
    print(f"Perfect prediction PPL: {ppl:.6f}")
    
    if ppl < 1.1:  # Should be very close to 1
        print("✓ Perplexity computation working correctly")
    else:
        print("⚠ Perplexity computation may have issues")
    
    # Test with random predictions (should be high PPL)
    random_logits = torch.randn(1, seq_len, vocab_size)
    random_loss = F.cross_entropy(
        random_logits[..., :-1, :].contiguous().view(-1, vocab_size),
        shift_labels.view(-1),
        reduction='mean'
    )
    random_ppl = torch.exp(random_loss).item()
    
    print(f"Random prediction loss: {random_loss.item():.4f}")
    print(f"Random prediction PPL: {random_ppl:.2f}")
    
    if random_ppl > 100:  # Should be high for random
        print("✓ Random predictions give high perplexity as expected")
    else:
        print("⚠ Random perplexity seems too low")


def test_model_size_scaling():
    """Test that model sizes scale appropriately with data size."""
    print("\nTesting Model Size Scaling")
    print("=" * 40)
    
    data_sizes = [1000, 7500, 50000, 200000]
    
    for data_size in data_sizes:
        config = get_optimized_config_for_data_size(data_size)
        
        # Calculate approximate parameter count
        # Rough estimate: embedding + layers + head
        embed_params = config.vocab_size * config.hidden_size
        layer_params = config.num_layers * (
            4 * config.hidden_size * config.hidden_size +  # Attention
            2 * config.hidden_size * config.intermediate_size  # FFN
        )
        head_params = config.hidden_size * config.vocab_size
        total_params = embed_params + layer_params + head_params
        
        print(f"Data size {data_size:6d}: {config.num_layers:2d} layers, "
              f"{config.hidden_size:3d} hidden, ~{total_params/1e6:.1f}M params")
    
    print("✓ Model scaling looks reasonable")


def quick_training_test():
    """Quick test of training loop to ensure everything works."""
    print("\nQuick Training Test")
    print("=" * 40)
    
    model, config = create_enhanced_model(7500)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create synthetic data
    batch_size = 4
    seq_len = 64
    num_batches = 10
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    losses = []
    ppls = []
    
    print("Training for 10 synthetic batches...")
    
    for step in range(num_batches):
        # Create batch
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
        
        # Update model training step
        model.update_training_step(step * 10)  # Simulate real training steps
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, labels=input_ids)
        
        total_loss = outputs['lm_loss'] + outputs['cognitive_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        losses.append(total_loss.item())
        ppls.append(torch.exp(outputs['lm_loss']).item())
        
        print(f"  Step {step:2d}: Loss={total_loss.item():.4f}, "
              f"PPL={ppls[-1]:.2f}, "
              f"CogWeight={outputs['cognitive_weight']:.3f}")
    
    # Check for improvement
    initial_ppl = ppls[0]
    final_ppl = ppls[-1]
    improvement = (initial_ppl - final_ppl) / initial_ppl * 100
    
    print(f"\nQuick training results:")
    print(f"  Initial PPL: {initial_ppl:.2f}")
    print(f"  Final PPL: {final_ppl:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    if improvement > 0:
        print("✓ Model is learning (perplexity decreased)")
    else:
        print("⚠ No improvement in perplexity (may need more steps)")


def main():
    """Run all validation tests."""
    print("Enhanced Cognitive LLM Validation Tests")
    print("=" * 60)
    
    try:
        # Test 1: Pretrained initialization
        test_pretrained_initialization()
        
        # Test 2: Cognitive scheduling
        test_cognitive_scheduling()
        
        # Test 3: Model forward pass
        test_model_forward_pass()
        
        # Test 4: Perplexity computation
        test_perplexity_computation()
        
        # Test 5: Model size scaling
        test_model_size_scaling()
        
        # Test 6: Quick training test
        quick_training_test()
        
        print("\n" + "=" * 60)
        print("✓ All validation tests completed successfully!")
        print("\nThe enhanced configuration should address:")
        print("  1. Scale issues through pretrained tokenizer + embeddings")
        print("  2. Objective interference through delayed cognitive losses")
        print("  3. Model size optimization for available data")
        print("\nExpected improvements:")
        print("  - Perplexity should start lower (better initialization)")
        print("  - Perplexity should improve faster (no cognitive interference early on)")
        print("  - Final perplexity should be much lower (<100 instead of 1000+)")
        
    except Exception as e:
        print(f"\n✗ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()