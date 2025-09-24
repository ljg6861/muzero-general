#!/usr/bin/env python3
"""
Phase A Training Script: Baseline LM to Competence
=================================================
Implements Phase A training specifications:
- AdamW optimizer, lr=2e-4, 3% warmup, cosine decay
- Gradient accumulation for 1-2M tokens/step effective batch
- Train until PPL plateaus with stable generation
- Focus on establishing solid LM foundation before cognitive enhancements
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import gc

from baseline_lm import BaselineLanguageModel, create_baseline_model_with_tokenizer, BaselineLMConfig
from large_scale_data import LargeScaleDataConfig, create_large_scale_dataloader


class PhaseATrainer:
    """Phase A trainer implementing baseline LM training to competence."""
    
    def __init__(self, 
                 model: BaselineLanguageModel,
                 tokenizer,
                 config: BaselineLMConfig,
                 save_dir: str = "results/phase_a_baseline"):
        
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.save_dir = save_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer (Phase A: AdamW)
        self.optimizer = self._create_optimizer()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_ppl = float('inf')
        self.plateau_count = 0
        self.max_plateau_count = 5  # Stop after 5 evaluations without improvement
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_ppl': [],
            'eval_loss': [],
            'eval_ppl': [],
            'learning_rate': [],
            'throughput': [],  # tokens/sec
            'memory_usage': []  # GB
        }
        
        # Calculate effective batch size in tokens
        self.effective_batch_tokens = (
            config.micro_batch_size * 
            config.gradient_accumulation_steps * 
            config.max_seq_length
        )
        
        print(f"Phase A Trainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Effective batch size: {self.effective_batch_tokens:,} tokens/step")
        print(f"  Target: 1-2M tokens/step (current: {self.effective_batch_tokens/1e6:.1f}M)")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with proper weight decay exclusions."""
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Exclude biases and norms from weight decay
            if any(nd in name for nd in ["bias", "norm", "embed_tokens"]):
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
        
        # Phase A: AdamW optimizer
        optimizer = optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),  # Standard for LM training
            eps=1e-8,
            weight_decay=self.config.weight_decay
        )
        
        print(f"AdamW optimizer created:")
        print(f"  Decay params: {len(decay_params)}")
        print(f"  No decay params: {len(no_decay_params)}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Weight decay: {self.config.weight_decay}")
        
        return optimizer
    
    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler: 3% warmup + cosine decay."""
        
        warmup_steps = int(num_training_steps * self.config.warmup_ratio)  # 3% warmup
        
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / max(1, warmup_steps)
            else:
                # Cosine annealing
                progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        print(f"Learning rate scheduler created:")
        print(f"  Warmup steps: {warmup_steps:,} ({self.config.warmup_ratio*100:.1f}%)")
        print(f"  Total steps: {num_training_steps:,}")
        print(f"  Schedule: 3% warmup + cosine decay")
        
        return scheduler
    
    def _calculate_memory_usage(self) -> float:
        """Calculate current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
        return 0.0
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform a single training step."""
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch.get('attention_mask'),
            labels=batch['labels']
        )
        
        loss = outputs['loss']
        
        # Scale loss by gradient accumulation steps
        scaled_loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        # Calculate metrics
        perplexity = outputs['perplexity'].item()
        
        return {
            'loss': loss.item(),
            'perplexity': perplexity,
            'scaled_loss': scaled_loss.item()
        }
    
    def evaluate(self, eval_dataloader: DataLoader, max_eval_steps: int = 100) -> Dict[str, float]:
        """Evaluate the model."""
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        num_steps = 0
        
        with torch.no_grad():
            for step, batch in enumerate(eval_dataloader):
                if step >= max_eval_steps:
                    break
                
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                
                # Count tokens (exclude padding)
                if 'attention_mask' in batch:
                    num_tokens = batch['attention_mask'].sum().item()
                else:
                    num_tokens = batch['input_ids'].numel()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
                num_steps += 1
        
        # Calculate averages
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        avg_ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_ppl': avg_ppl,
            'eval_tokens': total_tokens,
            'eval_steps': num_steps
        }
    
    def generate_samples(self, num_samples: int = 3) -> List[str]:
        """Generate text samples to assess quality."""
        
        prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists have discovered"
        ]
        
        samples = []
        self.model.eval()
        
        with torch.no_grad():
            for prompt in prompts[:num_samples]:
                # Encode prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Generate
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=50,
                    temperature=0.8,
                    top_k=50,
                    top_p=0.9,
                    do_sample=True,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                samples.append(generated_text)
        
        self.model.train()
        return samples
    
    def train(self, 
              num_epochs: int = 3,
              eval_every_steps: int = 1000,
              save_every_steps: int = 2000,
              max_steps: Optional[int] = None) -> Dict[str, List[float]]:
        """Train the baseline model to competence."""
        
        print(f"\nStarting Phase A Training: Baseline LM to Competence")
        print("=" * 60)
        print(f"Target: Train until PPL plateaus with stable generation")
        print(f"Max epochs: {num_epochs}")
        print(f"Eval every: {eval_every_steps} steps")
        print(f"Save every: {save_every_steps} steps")
        if max_steps:
            print(f"Max steps: {max_steps}")
        
        # Load training data
        print("\nLoading large-scale training data...")
        data_config = LargeScaleDataConfig(
            target_tokens=100_000_000,  # 100M tokens
            min_tokens=50_000_000,      # 50M minimum
            max_seq_length=self.config.max_seq_length,
            cache_dir=os.path.join(self.save_dir, "data_cache")
        )
        
        try:
            train_dataloader = create_large_scale_dataloader(
                config=data_config,
                tokenizer=self.tokenizer,
                batch_size=self.config.micro_batch_size,
                num_workers=2
            )
            
            # Create a smaller eval dataset (subset of training data)
            eval_config = LargeScaleDataConfig(
                target_tokens=1_000_000,  # 1M tokens for eval
                max_seq_length=self.config.max_seq_length,
                cache_dir=os.path.join(self.save_dir, "eval_cache")
            )
            
            eval_dataloader = create_large_scale_dataloader(
                config=eval_config,
                tokenizer=self.tokenizer,
                batch_size=self.config.micro_batch_size,
                num_workers=1
            )
            
            # Check if data loaders have actual data
            train_has_data = len(train_dataloader.dataset) > 0
            eval_has_data = len(eval_dataloader.dataset) > 0
            
            if not train_has_data or not eval_has_data:
                raise ValueError("Data loaders are empty - no data sources available")
            
        except Exception as e:
            print(f"⚠ Failed to load large-scale data: {e}")
            print("Using simple fallback data loader...")
            
            from simple_data import create_simple_data_loader
            
            # Create training data loader
            train_dataloader = create_simple_data_loader(
                tokenizer=self.tokenizer,
                batch_size=self.config.micro_batch_size,
                seq_length=self.config.max_seq_length,
                num_samples=20000  # 20K samples for training
            )
            
            # Create evaluation data loader  
            eval_dataloader = create_simple_data_loader(
                tokenizer=self.tokenizer,
                batch_size=self.config.micro_batch_size,
                seq_length=self.config.max_seq_length,
                num_samples=1000   # 1K samples for evaluation
            )
        
        # Estimate total training steps
        estimated_steps_per_epoch = 10000  # Rough estimate
        total_training_steps = num_epochs * estimated_steps_per_epoch
        if max_steps:
            total_training_steps = min(total_training_steps, max_steps)
        
        # Create scheduler
        scheduler = self._create_scheduler(total_training_steps)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            epoch_losses = []
            epoch_ppls = []
            tokens_processed = 0
            
            pbar = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}")
            
            for step, batch in enumerate(pbar):
                
                # Check max steps
                if max_steps and self.global_step >= max_steps:
                    print(f"Reached max steps: {max_steps}")
                    break
                
                # Training step
                step_results = self.train_step(batch)
                epoch_losses.append(step_results['loss'])
                epoch_ppls.append(step_results['perplexity'])
                
                # Count tokens processed
                batch_tokens = batch['input_ids'].numel()
                tokens_processed += batch_tokens
                
                # Gradient accumulation
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    
                    # Record metrics
                    current_lr = scheduler.get_last_lr()[0]
                    self.history['learning_rate'].append(current_lr)
                    
                    # Calculate throughput
                    elapsed = time.time() - start_time
                    throughput = tokens_processed / elapsed if elapsed > 0 else 0
                    self.history['throughput'].append(throughput)
                    
                    # Memory usage
                    memory_gb = self._calculate_memory_usage()
                    self.history['memory_usage'].append(memory_gb)
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{step_results['loss']:.4f}",
                    'PPL': f"{step_results['perplexity']:.1f}",
                    'LR': f"{scheduler.get_last_lr()[0]:.2e}",
                    'Mem': f"{self._calculate_memory_usage():.1f}GB"
                })
                
                # Evaluation
                if (self.global_step > 0 and 
                    self.global_step % eval_every_steps == 0):
                    
                    print(f"\nEvaluation at step {self.global_step}")
                    eval_results = self.evaluate(eval_dataloader)
                    
                    print(f"  Eval loss: {eval_results['eval_loss']:.4f}")
                    print(f"  Eval PPL: {eval_results['eval_ppl']:.2f}")
                    print(f"  Eval tokens: {eval_results['eval_tokens']:,}")
                    
                    # Record eval metrics
                    self.history['eval_loss'].append(eval_results['eval_loss'])
                    self.history['eval_ppl'].append(eval_results['eval_ppl'])
                    
                    # Check for improvement
                    if eval_results['eval_ppl'] < self.best_ppl:
                        self.best_ppl = eval_results['eval_ppl']
                        self.plateau_count = 0
                        self.save_checkpoint("best_model.pt")
                        print(f"  ✓ New best PPL: {self.best_ppl:.2f}")
                    else:
                        self.plateau_count += 1
                        print(f"  No improvement ({self.plateau_count}/{self.max_plateau_count})")
                    
                    # Generate samples to check quality
                    print("  Sample generations:")
                    samples = self.generate_samples(2)
                    for i, sample in enumerate(samples):
                        print(f"    {i+1}: {sample}")
                    
                    # Check for plateau (convergence)
                    if self.plateau_count >= self.max_plateau_count:
                        print(f"  ✓ PPL plateau reached after {self.plateau_count} evaluations")
                        print(f"  ✓ Baseline LM achieved competence!")
                        return self.history
                
                # Save checkpoint
                if (self.global_step > 0 and 
                    self.global_step % save_every_steps == 0):
                    self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                
                # Memory cleanup
                if step % 100 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            # Epoch summary
            avg_loss = np.mean(epoch_losses)
            avg_ppl = np.mean(epoch_ppls)
            self.history['train_loss'].append(avg_loss)
            self.history['train_ppl'].append(avg_ppl)
            
            print(f"Epoch {epoch+1} complete:")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Average PPL: {avg_ppl:.2f}")
            print(f"  Tokens processed: {tokens_processed:,}")
            
            self.epoch += 1
        
        print(f"\nTraining completed!")
        print(f"Best PPL achieved: {self.best_ppl:.2f}")
        
        # Final evaluation and samples
        print("\nFinal evaluation...")
        final_eval = self.evaluate(eval_dataloader)
        print(f"Final PPL: {final_eval['eval_ppl']:.2f}")
        
        print("\nFinal sample generations:")
        final_samples = self.generate_samples(3)
        for i, sample in enumerate(final_samples):
            print(f"  {i+1}: {sample}")
        
        # Save final model
        self.save_checkpoint("final_model.pt")
        self.save_training_plots()
        
        return self.history
    
    def _create_synthetic_dataloader(self) -> DataLoader:
        """Create synthetic dataloader for testing."""
        print("Creating synthetic training data...")
        
        # Generate synthetic text data
        synthetic_texts = [
            "The quick brown fox jumps over the lazy dog. This is a sample text for training.",
            "Artificial intelligence is transforming the world in many different ways.",
            "Machine learning models require large amounts of data to train effectively.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning networks can learn complex patterns from data automatically."
        ] * 1000  # Repeat to create more data
        
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, texts, tokenizer, max_length):
                self.texts = texts
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = self.texts[idx]
                tokens = self.tokenizer.encode(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding='max_length'
                )
                input_ids = torch.tensor(tokens, dtype=torch.long)
                return {
                    'input_ids': input_ids,
                    'labels': input_ids.clone(),
                    'attention_mask': torch.ones_like(input_ids)
                }
        
        dataset = SyntheticDataset(synthetic_texts, self.tokenizer, self.config.max_seq_length)
        return DataLoader(dataset, batch_size=self.config.micro_batch_size, shuffle=True)
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_ppl': self.best_ppl,
            'plateau_count': self.plateau_count,
            'history': self.history,
            'tokenizer_name': self.config.tokenizer_name
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def save_training_plots(self):
        """Save training progress plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase A: Baseline LM Training Progress', fontsize=16)
        
        # Training loss
        if self.history['train_loss']:
            axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Training perplexity
        if self.history['train_ppl']:
            axes[0, 1].plot(self.history['train_ppl'], label='Train PPL')
            axes[0, 1].set_title('Training Perplexity')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('PPL')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Evaluation perplexity
        if self.history['eval_ppl']:
            eval_steps = [i * 1000 for i in range(len(self.history['eval_ppl']))]
            axes[0, 2].plot(eval_steps, self.history['eval_ppl'], 'o-', label='Eval PPL')
            axes[0, 2].set_title('Evaluation Perplexity')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('PPL')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate
        if self.history['learning_rate']:
            axes[1, 0].plot(self.history['learning_rate'], label='Learning Rate')
            axes[1, 0].set_title('Learning Rate Schedule')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('LR')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Throughput
        if self.history['throughput']:
            axes[1, 1].plot(self.history['throughput'], label='Throughput')
            axes[1, 1].set_title('Training Throughput')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Tokens/sec')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Memory usage
        if self.history['memory_usage']:
            axes[1, 2].plot(self.history['memory_usage'], label='Memory Usage')
            axes[1, 2].set_title('GPU Memory Usage')
            axes[1, 2].set_xlabel('Step')
            axes[1, 2].set_ylabel('GB')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(self.save_dir, 'training_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training plots saved: {plot_path}")


def main():
    """Main Phase A training function."""
    
    print("Phase A: Training Baseline Language Model to Competence")
    print("=" * 70)
    
    # Create baseline model with proven tokenizer
    model, tokenizer, config = create_baseline_model_with_tokenizer()
    
    print(f"\nPhase A Configuration:")
    print(f"  Model: {config.num_layers}L × {config.hidden_size}d × {config.num_attention_heads}h")
    print(f"  Tokenizer: {config.tokenizer_name}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Warmup ratio: {config.warmup_ratio * 100:.1f}%")
    print(f"  Schedule: {config.lr_schedule}")
    print(f"  Optimizer: {config.optimizer}")
    
    # Create trainer
    trainer = PhaseATrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir=f"results/phase_a_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Train to competence
    print("\nStarting training to establish baseline competence...")
    history = trainer.train(
        num_epochs=5,        # Multiple epochs if needed
        eval_every_steps=500, # Frequent evaluation
        save_every_steps=1000,
        max_steps=10000      # Limit for demonstration
    )
    
    print("\nPhase A Training Summary:")
    print(f"  Best perplexity achieved: {trainer.best_ppl:.2f}")
    print(f"  Total training steps: {trainer.global_step}")
    print(f"  Convergence status: {'✓ Converged' if trainer.plateau_count >= trainer.max_plateau_count else '⚠ More training needed'}")
    
    if trainer.best_ppl < 50:
        print("✓ SUCCESS: Baseline LM achieved good competence!")
        print("  Ready for Phase B: Cognitive enhancement integration")
    elif trainer.best_ppl < 100:
        print("⚠ PARTIAL: Baseline LM showing learning but needs more training")
    else:
        print("✗ NEEDS WORK: Baseline LM not yet competent, check data/config")
    
    print("\nPhase A complete. Baseline LM foundation established.")


if __name__ == "__main__":
    main()