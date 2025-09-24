#!/usr/bin/env python3
"""
Ultra-Efficient Cognitive LLM Launcher
=====================================
Optimized training for limited data (~7500 articles).
"""

import torch
from enhanced_training import EnhancedCognitiveTrainer
from ultra_efficient_config import create_ultra_efficient_config

def main():
    print("Ultra-Efficient Cognitive LLM Training")
    print("=" * 50)
    
    # Create ultra-efficient configuration
    config = create_ultra_efficient_config()
    
    print(f"Configuration: {config.num_layers} layers, {config.hidden_size} hidden")
    print(f"Expected parameters: ~{19.3:}M (vs {95.2:}M original)")
    print(f"Cognitive scheduling: {config.cognitive_delay_steps} → {config.cognitive_annealing_steps}")
    
    # Create trainer with ultra-efficient config
    trainer = EnhancedCognitiveTrainer(
        config=config,
        save_dir="results/ultra_efficient_cognitive"
    )
    
    # Train with aggressive schedule for quick results
    history = trainer.train(
        num_epochs=3,      # Fewer epochs, more steps per epoch
        eval_every=100,    # Frequent evaluation
        save_every=200     # Frequent saves
    )
    
    # Report results
    final_ppl = history['perplexity'][-1]
    print(f"\nFinal Results:")
    print(f"  Perplexity: {final_ppl:.1f}")
    
    if final_ppl < 60:
        print("✓ SUCCESS: Good perplexity for limited data!")
    elif final_ppl < 150:
        print("⚠ DECENT: Reasonable learning, could improve with more training")
    else:
        print("✗ NEEDS WORK: Still too high, check data quality")

if __name__ == "__main__":
    main()
