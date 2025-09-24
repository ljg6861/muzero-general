#!/usr/bin/env python3
"""
Enhanced Training Script with Scale and Objective Interference Fixes
===================================================================
Integrates all improvements:
1. Pretrained tokenizer + embeddings for better scale
2. Progressive cognitive loss scheduling to prevent interference
3. Proper model sizing for available data
4. Trustworthy evaluation with perplexity monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from enhanced_cognitive_llm import EnhancedCognitiveLLM, create_enhanced_model
from enhanced_config import (
    EnhancedCognitiveLLMConfig, 
    get_optimized_config_for_data_size,
    CognitiveLossScheduler
)
from wikipedia_data import WikipediaDataProcessor, WikipediaDataset
from trustworthy_training import TrustworthyLossFunction


class EnhancedCognitiveTrainer:
    """Enhanced trainer addressing scale and objective interference issues."""
    
    def __init__(self, 
                 config: EnhancedCognitiveLLMConfig,
                 save_dir: str = "results/enhanced_cognitive_llm"):
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize model
        self.model, _ = create_enhanced_model(7500)  # Optimized for current data size
        self.model.to(self.device)
        
        # Get tokenizer (pretrained or None)
        self.tokenizer = self.model.get_tokenizer()
        
        # Initialize trustworthy loss function
        self.loss_fn = TrustworthyLossFunction()
        
        # Setup optimizer with weight decay exclusions
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.global_step = 0
        self.best_ppl = float('inf')
        self.training_history = {
            'lm_loss': [],
            'cognitive_loss': [],
            'total_loss': [],
            'perplexity': [],
            'cognitive_weight': [],
            'component_weights': {
                'concept_formation': [],
                'causal_reasoning': [],
                'meta_learning': []
            }
        }
        
        print(f"Enhanced Cognitive Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  Using pretrained tokenizer: {self.tokenizer is not None}")
        print(f"  Save directory: {save_dir}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with proper weight decay exclusions."""
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            # Check if parameter should be excluded from weight decay
            exclude = False
            for exclude_name in self.config.exclude_from_decay:
                if exclude_name in name:
                    exclude = True
                    break
            
            if exclude:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # Create parameter groups
        param_groups = [
            {
                'params': decay_params,
                'weight_decay': self.config.weight_decay,
                'lr': self.config.learning_rate
            },
            {
                'params': no_decay_params,
                'weight_decay': 0.0,
                'lr': self.config.learning_rate
            }
        ]
        
        # Add cognitive parameters with different learning rate if they exist
        cognitive_params = []
        for name, param in self.model.named_parameters():
            if 'cognitive' in name.lower() and param.requires_grad:
                cognitive_params.append(param)
        
        if cognitive_params:
            param_groups.append({
                'params': cognitive_params,
                'weight_decay': self.config.weight_decay * 0.5,  # Lower weight decay for cognitive
                'lr': self.config.cognitive_learning_rate
            })
        
        optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
        
        print(f"Optimizer created:")
        print(f"  Decay params: {len(decay_params)} parameters")
        print(f"  No decay params: {len(no_decay_params)} parameters")
        print(f"  Cognitive params: {len(cognitive_params)} parameters")
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            else:
                # Cosine annealing after warmup
                progress = (step - self.config.warmup_steps) / max(1, num_training_steps - self.config.warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def load_data(self, data_config=None) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare data with enhanced tokenization."""
        
        # Use pretrained tokenizer if available
        if self.tokenizer:
            print("Using pretrained tokenizer for data processing")
            
            # Create data processor with pretrained tokenizer
            processor = WikipediaDataProcessor(
                data_config or {
                    'cache_dir': 'data/wikipedia_cache',
                    'num_articles': 7500,
                    'min_article_length': 500,
                    'max_article_length': 5000,
                    'categories': [
                        'Science', 'Technology', 'History', 'Geography',
                        'Mathematics', 'Physics', 'Biology', 'Philosophy'
                    ]
                }
            )
            
            # Override tokenizer in processor
            processor.tokenizer = self.tokenizer
            
        else:
            print("Using standard tokenizer for data processing")
            processor = WikipediaDataProcessor(data_config)
        
        # Process articles
        processed_articles = processor.process_articles()
        print(f"Processed {len(processed_articles)} articles")
        
        # Create datasets
        train_dataset = WikipediaDataset(
            processed_articles[:int(0.9 * len(processed_articles))],
            processor.tokenizer,
            max_length=self.config.max_seq_length
        )
        
        val_dataset = WikipediaDataset(
            processed_articles[int(0.9 * len(processed_articles)):],
            processor.tokenizer,
            max_length=self.config.max_seq_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Data loaded:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step with enhanced loss computation."""
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Update model's training step for cognitive scheduling
        self.model.update_training_step(self.global_step)
        
        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,  # Autoregressive training
            enable_cognitive=True
        )
        
        # Enhanced loss computation with trustworthy evaluation
        loss_details = self.loss_fn.compute_detailed_loss(
            logits=outputs['logits'],
            labels=input_ids,
            attention_mask=attention_mask
        )
        
        # Combine with cognitive loss
        total_loss = loss_details['loss_mean'] + outputs['cognitive_loss']
        
        # Backward pass
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update global step
        self.global_step += 1
        
        # Compute perplexity
        perplexity = torch.exp(loss_details['loss_mean']).item()
        
        return {
            'lm_loss': outputs['lm_loss'].item(),
            'cognitive_loss': outputs['cognitive_loss'].item(),
            'total_loss': total_loss.item(),
            'perplexity': perplexity,
            'cognitive_weight': outputs['cognitive_weight'],
            'component_weights': outputs['component_weights'],
            'n_tokens': loss_details['n_tokens'],
            'seq_len_mean': loss_details['seq_len_mean']
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        
        self.model.eval()
        total_lm_loss = 0.0
        total_cognitive_loss = 0.0
        total_tokens = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass (with current cognitive state)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                    enable_cognitive=True
                )
                
                # Compute detailed loss
                loss_details = self.loss_fn.compute_detailed_loss(
                    logits=outputs['logits'],
                    labels=input_ids,
                    attention_mask=attention_mask
                )
                
                total_lm_loss += loss_details['loss_sum'].item()
                total_cognitive_loss += outputs['cognitive_loss'].item() * loss_details['n_tokens']
                total_tokens += loss_details['n_tokens']
                num_batches += 1
        
        # Compute averages
        avg_lm_loss = total_lm_loss / total_tokens
        avg_cognitive_loss = total_cognitive_loss / total_tokens
        perplexity = math.exp(avg_lm_loss)
        
        return {
            'lm_loss': avg_lm_loss,
            'cognitive_loss': avg_cognitive_loss,
            'total_loss': avg_lm_loss + avg_cognitive_loss,
            'perplexity': perplexity,
            'n_tokens': total_tokens,
            'n_batches': num_batches
        }
    
    def train(self, 
              num_epochs: int = 10,
              eval_every: int = 500,
              save_every: int = 1000,
              data_config: Optional[Dict] = None) -> Dict[str, List[float]]:
        """Train the enhanced cognitive model."""
        
        print(f"\nStarting Enhanced Cognitive Training:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Eval every: {eval_every} steps")
        print(f"  Save every: {save_every} steps")
        
        # Load data
        train_loader, val_loader = self.load_data(data_config)
        
        # Create scheduler
        num_training_steps = num_epochs * len(train_loader)
        scheduler = self._create_scheduler(num_training_steps)
        
        # Training loop
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            epoch_losses = []
            pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            
            for batch_idx, batch in enumerate(pbar):
                
                # Training step
                step_results = self.train_step(batch)
                epoch_losses.append(step_results)
                
                # Update scheduler
                scheduler.step()
                
                # Update progress bar
                pbar.set_postfix({
                    'LM Loss': f"{step_results['lm_loss']:.4f}",
                    'Cog Loss': f"{step_results['cognitive_loss']:.4f}",
                    'PPL': f"{step_results['perplexity']:.2f}",
                    'Cog Weight': f"{step_results['cognitive_weight']:.3f}"
                })
                
                # Record training history
                self.training_history['lm_loss'].append(step_results['lm_loss'])
                self.training_history['cognitive_loss'].append(step_results['cognitive_loss'])
                self.training_history['total_loss'].append(step_results['total_loss'])
                self.training_history['perplexity'].append(step_results['perplexity'])
                self.training_history['cognitive_weight'].append(step_results['cognitive_weight'])
                
                for component, weight in step_results['component_weights'].items():
                    self.training_history['component_weights'][component].append(weight)
                
                # Evaluation
                if self.global_step % eval_every == 0:
                    print(f"\nEvaluation at step {self.global_step}:")
                    val_results = self.validate(val_loader)
                    
                    print(f"  LM Loss: {val_results['lm_loss']:.4f}")
                    print(f"  Cognitive Loss: {val_results['cognitive_loss']:.4f}")
                    print(f"  Total Loss: {val_results['total_loss']:.4f}")
                    print(f"  Perplexity: {val_results['perplexity']:.2f}")
                    print(f"  Tokens evaluated: {val_results['n_tokens']:,}")
                    
                    # Save best model
                    if val_results['perplexity'] < self.best_ppl:
                        self.best_ppl = val_results['perplexity']
                        self.save_checkpoint(f"best_model_ppl_{self.best_ppl:.2f}.pt")
                        print(f"  ✓ New best perplexity: {self.best_ppl:.2f}")
                
                # Save checkpoint
                if self.global_step % save_every == 0:
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                    self.save_training_plots()
        
        print(f"\nTraining completed!")
        print(f"Best perplexity: {self.best_ppl:.2f}")
        
        # Final evaluation
        final_results = self.validate(val_loader)
        print(f"Final perplexity: {final_results['perplexity']:.2f}")
        
        # Save final model and plots
        self.save_checkpoint("final_model.pt")
        self.save_training_plots()
        
        return self.training_history
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'best_ppl': self.best_ppl,
            'training_history': self.training_history
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def save_training_plots(self):
        """Save training progress plots."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Cognitive LLM Training Progress', fontsize=16)
        
        steps = list(range(len(self.training_history['lm_loss'])))
        
        # LM Loss
        axes[0, 0].plot(steps, self.training_history['lm_loss'], label='LM Loss', alpha=0.7)
        axes[0, 0].set_title('Language Modeling Loss')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cognitive Loss
        axes[0, 1].plot(steps, self.training_history['cognitive_loss'], label='Cognitive Loss', color='orange', alpha=0.7)
        axes[0, 1].set_title('Cognitive Loss (Progressive)')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Perplexity (log scale)
        axes[0, 2].plot(steps, self.training_history['perplexity'], label='Perplexity', color='red', alpha=0.7)
        axes[0, 2].set_title('Perplexity (Lower is Better)')
        axes[0, 2].set_xlabel('Step')
        axes[0, 2].set_ylabel('Perplexity')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Cognitive Weight Schedule
        axes[1, 0].plot(steps, self.training_history['cognitive_weight'], label='Cognitive Weight', color='green', alpha=0.7)
        axes[1, 0].set_title('Cognitive Loss Weight Schedule')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Component Weights
        for component, weights in self.training_history['component_weights'].items():
            if weights:  # Only plot if we have data
                axes[1, 1].plot(steps[:len(weights)], weights, label=component.replace('_', ' ').title(), alpha=0.7)
        axes[1, 1].set_title('Cognitive Component Weights')
        axes[1, 1].set_xlabel('Step')
        axes[1, 1].set_ylabel('Weight')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Total Loss
        axes[1, 2].plot(steps, self.training_history['total_loss'], label='Total Loss', color='purple', alpha=0.7)
        axes[1, 2].set_title('Total Loss (LM + Cognitive)')
        axes[1, 2].set_xlabel('Step')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved: {plot_path}")


def main():
    """Main training function."""
    
    # Create enhanced configuration for current data size
    config = get_optimized_config_for_data_size(7500)
    
    print("Enhanced Cognitive LLM Training")
    print("=" * 50)
    print(f"Configuration optimized for ~7500 articles:")
    print(f"  Model: {config.num_layers} layers, {config.hidden_size} hidden")
    print(f"  Cognitive scheduling: delay={config.cognitive_delay_steps}, anneal={config.cognitive_annealing_steps}")
    print(f"  Max cognitive weight: {config.max_cognitive_loss_weight}")
    print(f"  Pretrained init: {config.use_pretrained_init} ({config.pretrained_model_name})")
    
    # Create trainer
    trainer = EnhancedCognitiveTrainer(
        config=config,
        save_dir=f"results/enhanced_cognitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Train model
    history = trainer.train(
        num_epochs=5,
        eval_every=200,
        save_every=500
    )
    
    print("\nTraining Summary:")
    print(f"  Final LM loss: {history['lm_loss'][-1]:.4f}")
    print(f"  Final cognitive loss: {history['cognitive_loss'][-1]:.4f}")
    print(f"  Final perplexity: {history['perplexity'][-1]:.2f}")
    print(f"  Best perplexity: {trainer.best_ppl:.2f}")
    
    # Show improvement analysis
    initial_ppl = history['perplexity'][0]
    final_ppl = history['perplexity'][-1]
    improvement = (initial_ppl - final_ppl) / initial_ppl * 100
    
    print(f"\nPerplexity Improvement:")
    print(f"  Initial: {initial_ppl:.2f}")
    print(f"  Final: {final_ppl:.2f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    if final_ppl < 100:
        print("✓ SUCCESS: Perplexity below 100 - model learned basic language patterns!")
    elif final_ppl < 1000:
        print("⚠ PARTIAL: Perplexity below 1000 - some learning but needs more training")
    else:
        print("✗ NEEDS WORK: Perplexity still very high - check data quality and model size")


if __name__ == "__main__":
    main()