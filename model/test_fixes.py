#!/usr/bin/env python3
"""
Simple verification test for fixed Phase A components
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import torch
import logging
from baseline_lm import create_baseline_model_with_tokenizer
from token_budget_trainer import TokenBudgetTrainer, TrainingConfig

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_fixed_components():
    """Test that the core fixes work correctly."""
    
    print("Testing Fixed Phase A Components")
    print("=" * 40)
    
    # 1. Test model creation
    print("1. Testing model creation...")
    model, tokenizer, _ = create_baseline_model_with_tokenizer()
    print(f"   ✓ Model created: {model.__class__.__name__}")
    
    # 2. Test configuration
    print("2. Testing token-budget configuration...")
    config = TrainingConfig(
        train_tokens=1_000_000,      # 1M tokens (very small)
        eval_tokens=100_000,         # 100K tokens
        warmup_tokens=100_000,       # 100K tokens warmup
        micro_batch_size=1,
        gradient_accumulation_steps=32,  # Small for testing
        seq_length=256,              # Short for speed
        allow_fallback_synthetic=True
    )
    
    # Check schedule computation
    schedule_params = config.compute_schedule_params()
    print(f"   ✓ Schedule computed:")
    print(f"     Global batch: {schedule_params['global_batch_tokens']:,} tokens/step")
    print(f"     Total steps: {schedule_params['total_steps']:,}")
    print(f"     Warmup steps: {schedule_params['warmup_steps']:,}")
    
    # Verify no LR=0 artifacts
    assert schedule_params['warmup_steps'] >= 1, "Warmup steps should be ≥1"
    assert schedule_params['total_steps'] > schedule_params['warmup_steps'], "Total steps should be > warmup steps"
    print("   ✓ No LR=0 artifacts")
    
    # 3. Test trainer creation
    print("3. Testing trainer creation...")
    trainer = TokenBudgetTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir="results/test_verification"
    )
    print("   ✓ Trainer created successfully")
    
    # 4. Test single forward pass
    print("4. Testing forward pass...")
    # Create a simple batch
    input_text = "The future of artificial intelligence is"
    input_ids = tokenizer.encode(input_text, return_tensors='pt', max_length=256, truncation=True, padding='max_length')
    
    batch = {
        'input_ids': input_ids,
        'labels': input_ids.clone(),
        'attention_mask': torch.ones_like(input_ids)
    }
    
    # Test training step
    model.train()
    try:
        loss = trainer.train_step(batch)
        print(f"   ✓ Training step successful, loss: {loss:.4f}")
    except Exception as e:
        print(f"   ✗ Training step failed: {e}")
        return False
    
    # 5. Test evaluation
    print("5. Testing evaluation...")
    try:
        # Create a simple eval loader for testing
        eval_data = [batch for _ in range(5)]  # 5 copies for testing
        
        class SimpleEvalLoader:
            def __init__(self, data):
                self.data = data
            def __iter__(self):
                return iter(self.data)
        
        trainer.eval_loader = SimpleEvalLoader(eval_data)
        
        eval_results = trainer.evaluate()
        print(f"   ✓ Evaluation successful:")
        print(f"     Eval loss: {eval_results['eval_loss']:.4f}")
        print(f"     Eval PPL: {eval_results['eval_ppl']:.2f}")
        print(f"     Eval tokens: {eval_results['eval_tokens']:,}")
        
    except Exception as e:
        print(f"   ✗ Evaluation failed: {e}")
        return False
    
    # 6. Test scheduler
    print("6. Testing LR scheduler...")
    try:
        initial_lr = trainer.scheduler.get_last_lr()[0]
        print(f"   Initial LR: {initial_lr:.2e}")
        
        # Test a few steps
        for step in range(5):
            lr = trainer.scheduler.step()
            print(f"   Step {step+1} LR: {lr:.2e}")
            
        # Verify LR doesn't go to 0 immediately
        final_lr = trainer.scheduler.get_last_lr()[0]
        assert final_lr > 0, "LR should not be 0"
        print("   ✓ Scheduler working correctly")
        
    except Exception as e:
        print(f"   ✗ Scheduler test failed: {e}")
        return False
    
    # 7. Test generation
    print("7. Testing text generation...")
    try:
        samples = trainer.generate_samples(1)
        print(f"   ✓ Generation successful:")
        print(f"     Sample: {samples[0][:100]}...")
        
    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        return False
    
    print("\n🎉 All core components working correctly!")
    print("\nKey fixes verified:")
    print("  ✓ Token-budget configuration (no len() dependency)")
    print("  ✓ Proper scheduler math (no LR=0 artifacts)")
    print("  ✓ Robust model forward pass")
    print("  ✓ Working evaluation loop")
    print("  ✓ Functional text generation")
    
    return True

if __name__ == "__main__":
    success = test_fixed_components()
    if success:
        print("\n✅ All tests passed - fixed components are working!")
    else:
        print("\n❌ Some tests failed - need more fixes")
    
    exit(0 if success else 1)