"""
Token-Budget Trainer for Phase A
===============================
Trainer that computes steps from token budgets rather than dataset length.
Fixes scheduler math and LR=0 artifacts.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import PreTrainedTokenizer
import os
import time
import logging
import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from data_registry import DataRegistry, DataConfig


@dataclass
class TrainingConfig:
    """Training configuration with token-budget approach."""
    
    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 384
    num_layers: int = 6
    num_attention_heads: int = 6
    max_seq_length: int = 1024
    
    # Token budgets (the fundamental units)
    train_tokens: int = 2_000_000_000     # 2B tokens for training
    eval_tokens: int = 10_000_000         # 10M tokens for eval
    warmup_tokens: int = 100_000_000      # 100M tokens for warmup
    
    # Batch configuration
    micro_batch_size: int = 1             # Start small for memory
    gradient_accumulation_steps: int = 1024  # Will be computed
    seq_length: int = 1024                # Fixed sequence length
    
    # Optimizer settings
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Scheduler settings
    lr_scheduler: str = 'cosine'
    cosine_min_lr_ratio: float = 0.1      # LR floor at 10% of peak
    
    # Data settings
    data_sources: List[str] = None
    allow_fallback_synthetic: bool = False
    
    # Checkpointing
    eval_every_tokens: int = 100_000_000   # Eval every 100M tokens
    save_every_tokens: int = 500_000_000   # Save every 500M tokens
    
    def __post_init__(self):
        if self.data_sources is None:
            self.data_sources = ['wikipedia', 'openwebtext', 'c4']
    
    def compute_schedule_params(self) -> Dict[str, int]:
        """Compute scheduler parameters from token budgets."""
        
        # Global effective batch size in tokens
        global_batch_tokens = (
            self.micro_batch_size * 
            self.seq_length * 
            self.gradient_accumulation_steps
        )
        
        # Compute steps from tokens
        warmup_steps = max(1, math.ceil(self.warmup_tokens / global_batch_tokens))
        total_steps = math.ceil(self.train_tokens / global_batch_tokens)
        eval_every_steps = max(1, math.ceil(self.eval_every_tokens / global_batch_tokens))
        save_every_steps = max(1, math.ceil(self.save_every_tokens / global_batch_tokens))
        
        return {
            'global_batch_tokens': global_batch_tokens,
            'warmup_steps': warmup_steps,
            'total_steps': total_steps,
            'eval_every_steps': eval_every_steps,
            'save_every_steps': save_every_steps
        }


class TokenBudgetTrainer:
    """Trainer using token budgets instead of epoch/dataset length."""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: TrainingConfig,
        save_dir: str = "results/token_budget_training"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.save_dir = save_dir
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Compute schedule parameters
        self.schedule_params = config.compute_schedule_params()
        
        # Training state
        self.global_step = 0
        self.tokens_processed = 0
        self.best_ppl = float('inf')
        self.history = {
            'train_loss': [],
            'train_ppl': [],
            'eval_loss': [],
            'eval_ppl': [],
            'learning_rate': [],
            'tokens_processed': []
        }
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(save_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        # Log configuration
        self._log_config()
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create data loaders
        self.train_loader, self.eval_loader = self._create_data_loaders()
        
    def _log_config(self):
        """Log training configuration."""
        
        self.logger.info("Token-Budget Trainer Configuration")
        self.logger.info("=" * 50)
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Model: {self.config.num_layers}L × {self.config.hidden_size}d")
        self.logger.info(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")
        self.logger.info(f"Device: {self.device}")
        
        # Token budgets
        self.logger.info(f"\nToken Budgets:")
        self.logger.info(f"  Training: {self.config.train_tokens:,} tokens")
        self.logger.info(f"  Warmup: {self.config.warmup_tokens:,} tokens") 
        self.logger.info(f"  Evaluation: {self.config.eval_tokens:,} tokens")
        
        # Schedule parameters
        params = self.schedule_params
        self.logger.info(f"\nSchedule Parameters:")
        self.logger.info(f"  Global batch: {params['global_batch_tokens']:,} tokens/step")
        self.logger.info(f"  Total steps: {params['total_steps']:,}")
        self.logger.info(f"  Warmup steps: {params['warmup_steps']:,}")
        self.logger.info(f"  Eval every: {params['eval_every_steps']:,} steps")
        self.logger.info(f"  Save every: {params['save_every_steps']:,} steps")
        
        # Batch breakdown
        self.logger.info(f"\nBatch Configuration:")
        self.logger.info(f"  Micro batch size: {self.config.micro_batch_size}")
        self.logger.info(f"  Sequence length: {self.config.seq_length}")
        self.logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        self.logger.info(f"  Effective batch: {params['global_batch_tokens']:,} tokens")
        
        # Check if batch size meets Phase A spec
        if params['global_batch_tokens'] >= 1_000_000:
            self.logger.info(f"  ✓ Meets Phase A spec (≥1M tokens/step)")
        else:
            self.logger.warning(f"  ⚠ Below Phase A spec (<1M tokens/step)")
        
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Don't apply weight decay to biases and layer norms
                if 'bias' in name or 'norm' in name or 'ln' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps
        )
        
        self.logger.info(f"AdamW optimizer created:")
        self.logger.info(f"  Decay params: {len(decay_params)}")
        self.logger.info(f"  No decay params: {len(no_decay_params)}")
        self.logger.info(f"  Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  Weight decay: {self.config.weight_decay}")
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler with proper cosine decay."""
        
        params = self.schedule_params
        total_steps = params['total_steps']
        warmup_steps = params['warmup_steps']
        
        # Calculate minimum LR (floor)
        min_lr = self.config.learning_rate * self.config.cosine_min_lr_ratio
        
        class WarmupCosineScheduler:
            def __init__(self, optimizer, warmup_steps, total_steps, max_lr, min_lr):
                self.optimizer = optimizer
                self.warmup_steps = warmup_steps
                self.total_steps = total_steps
                self.max_lr = max_lr
                self.min_lr = min_lr
                self.step_count = 0
            
            def step(self):
                self.step_count += 1
                
                if self.step_count <= self.warmup_steps:
                    # Linear warmup
                    lr = self.max_lr * (self.step_count / self.warmup_steps)
                else:
                    # Cosine decay with floor
                    progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
                    progress = min(progress, 1.0)  # Clamp to 1.0
                    
                    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
                    lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_factor
                
                # Update optimizer
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                return lr
            
            def get_last_lr(self):
                return [group['lr'] for group in self.optimizer.param_groups]
        
        scheduler = WarmupCosineScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            max_lr=self.config.learning_rate,
            min_lr=min_lr
        )
        
        self.logger.info(f"Learning rate scheduler created:")
        self.logger.info(f"  Type: Warmup + Cosine decay with floor")
        self.logger.info(f"  Warmup steps: {warmup_steps:,}")
        self.logger.info(f"  Total steps: {total_steps:,}")
        self.logger.info(f"  Max LR: {self.config.learning_rate}")
        self.logger.info(f"  Min LR: {min_lr} ({self.config.cosine_min_lr_ratio*100:.1f}% of max)")
        
        return scheduler
    
    def _create_data_loaders(self):
        """Create data loaders using DataRegistry."""
        
        # Create data config
        data_config = DataConfig(
            train_tokens=self.config.train_tokens,
            eval_tokens=self.config.eval_tokens,
            seq_length=self.config.seq_length,
            data_sources=self.config.data_sources,
            allow_fallback_synthetic=self.config.allow_fallback_synthetic
        )
        
        # Create registry
        registry = DataRegistry()
        
        # Create loaders
        train_loader = registry.create_dataloader(
            tokenizer=self.tokenizer,
            config=data_config,
            batch_size=self.config.micro_batch_size,
            is_eval=False,
            use_sized=False  # Use streaming for large-scale training
        )
        
        eval_loader = registry.create_dataloader(
            tokenizer=self.tokenizer,
            config=data_config,
            batch_size=self.config.micro_batch_size,
            is_eval=True,
            use_sized=False
        )
        
        return train_loader, eval_loader
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        # Handle different output formats
        if hasattr(outputs, 'loss'):
            loss = outputs.loss
        elif isinstance(outputs, dict) and 'loss' in outputs:
            loss = outputs['loss']
        else:
            # Manual loss computation
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            labels = batch['labels']
            
            # Shift logits and labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model on eval set."""
        
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in self.eval_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                # Handle different output formats
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    # Manual loss computation
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                    labels = batch['labels']
                    
                    # Shift logits and labels for language modeling
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Compute cross entropy loss
                    loss_fct = nn.CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                # Count actual tokens (not padding)
                valid_tokens = (batch['labels'] != -100).sum().item()
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
        
        self.model.train()
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        return {
            'eval_loss': avg_loss,
            'eval_ppl': ppl,
            'eval_tokens': total_tokens
        }
    
    def generate_samples(self, num_samples: int = 3) -> List[str]:
        """Generate text samples for evaluation."""
        
        self.model.eval()
        samples = []
        
        prompts = [
            "The future of artificial intelligence",
            "In a world where technology",
            "Scientists have discovered"
        ]
        
        with torch.no_grad():
            for prompt in prompts[:num_samples]:
                # Tokenize prompt
                input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
                
                # Simple greedy generation
                generated_ids = input_ids.clone()
                max_new_tokens = 50
                
                for _ in range(max_new_tokens):
                    # Get model output
                    outputs = self.model(generated_ids)
                    
                    # Get logits for last token
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                    elif isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                    else:
                        break  # Can't generate
                    
                    # Get next token (greedy)
                    next_token_logits = logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Add to sequence
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Stop if too long
                    if generated_ids.shape[1] > self.config.seq_length:
                        break
                
                # Decode
                generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                samples.append(generated_text)
        
        self.model.train()
        return samples
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
            'global_step': self.global_step,
            'tokens_processed': self.tokens_processed,
            'best_ppl': self.best_ppl,
            'history': self.history
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def train(self) -> Dict[str, Any]:
        """Run full training with token budget."""
        
        self.logger.info("\nStarting Token-Budget Training")
        self.logger.info("=" * 50)
        
        params = self.schedule_params
        start_time = time.time()
        
        # Training loop
        self.model.train()
        accumulated_loss = 0.0
        
        pbar = tqdm(
            total=params['total_steps'],
            desc="Training",
            unit="step"
        )
        
        for batch in self.train_loader:
            # Training step
            step_loss = self.train_step(batch)
            accumulated_loss += step_loss
            
            # Count tokens processed
            valid_tokens = (batch['labels'] != -100).sum().item()
            self.tokens_processed += valid_tokens
            
            # Check if we should update (gradient accumulation)
            if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                # Scheduler step
                current_lr = self.scheduler.step()
                
                # Record metrics
                avg_loss = accumulated_loss / self.config.gradient_accumulation_steps
                ppl = math.exp(avg_loss) if avg_loss < 10 else float('inf')
                
                self.history['train_loss'].append(avg_loss)
                self.history['train_ppl'].append(ppl)
                self.history['learning_rate'].append(current_lr)
                self.history['tokens_processed'].append(self.tokens_processed)
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'PPL': f'{ppl:.1f}',
                    'LR': f'{current_lr:.2e}',
                    'Tokens': f'{self.tokens_processed:,}'
                })
                
                # Reset accumulation
                accumulated_loss = 0.0
                self.global_step += 1
                
                # Evaluation
                if self.global_step % params['eval_every_steps'] == 0:
                    eval_results = self.evaluate()
                    
                    self.history['eval_loss'].append(eval_results['eval_loss'])
                    self.history['eval_ppl'].append(eval_results['eval_ppl'])
                    
                    self.logger.info(f"\nEvaluation at step {self.global_step}:")
                    self.logger.info(f"  Eval loss: {eval_results['eval_loss']:.4f}")
                    self.logger.info(f"  Eval PPL: {eval_results['eval_ppl']:.2f}")
                    self.logger.info(f"  Eval tokens: {eval_results['eval_tokens']:,}")
                    
                    # Save best model
                    if eval_results['eval_ppl'] < self.best_ppl:
                        self.best_ppl = eval_results['eval_ppl']
                        best_path = os.path.join(self.save_dir, 'best_model.pt')
                        self.save_checkpoint(best_path)
                        self.logger.info(f"  ✓ New best PPL: {self.best_ppl:.2f}")
                        
                        # Generate samples
                        samples = self.generate_samples(2)
                        self.logger.info(f"  Sample generations:")
                        for i, sample in enumerate(samples, 1):
                            self.logger.info(f"    {i}. {sample}")
                
                # Save checkpoint
                if self.global_step % params['save_every_steps'] == 0:
                    checkpoint_path = os.path.join(self.save_dir, f'checkpoint_step_{self.global_step}.pt')
                    self.save_checkpoint(checkpoint_path)
                
                # Check if training complete
                if self.global_step >= params['total_steps']:
                    break
        
        pbar.close()
        
        # Final evaluation
        final_eval = self.evaluate()
        self.logger.info(f"\nTraining completed!")
        self.logger.info(f"Steps: {self.global_step:,}")
        self.logger.info(f"Tokens processed: {self.tokens_processed:,}")
        self.logger.info(f"Best PPL: {self.best_ppl:.2f}")
        self.logger.info(f"Final PPL: {final_eval['eval_ppl']:.2f}")
        
        # Save final model
        final_path = os.path.join(self.save_dir, 'final_model.pt')
        self.save_checkpoint(final_path)
        
        # Save training plots
        self._save_plots()
        
        return {
            'global_step': self.global_step,
            'tokens_processed': self.tokens_processed,
            'best_ppl': self.best_ppl,
            'final_ppl': final_eval['eval_ppl'],
            'history': self.history
        }
    
    def _save_plots(self):
        """Save training progress plots."""
        
        if not self.history['train_loss']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss plot
        axes[0,0].plot(self.history['train_loss'], label='Train Loss')
        if self.history['eval_loss']:
            eval_steps = [i * self.schedule_params['eval_every_steps'] 
                         for i in range(len(self.history['eval_loss']))]
            axes[0,0].plot(eval_steps, self.history['eval_loss'], label='Eval Loss', marker='o')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Training Loss')
        axes[0,0].legend()
        
        # PPL plot
        axes[0,1].plot(self.history['train_ppl'], label='Train PPL')
        if self.history['eval_ppl']:
            axes[0,1].plot(eval_steps, self.history['eval_ppl'], label='Eval PPL', marker='o')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Perplexity')
        axes[0,1].set_title('Perplexity')
        axes[0,1].legend()
        axes[0,1].set_yscale('log')
        
        # Learning rate
        axes[1,0].plot(self.history['learning_rate'])
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Learning Rate')
        axes[1,0].set_title('Learning Rate Schedule')
        
        # Tokens processed
        axes[1,1].plot(self.history['tokens_processed'])
        axes[1,1].set_xlabel('Step')
        axes[1,1].set_ylabel('Tokens Processed')
        axes[1,1].set_title('Token Throughput')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), dpi=150)
        plt.close()
        
        self.logger.info(f"Training plots saved: {self.save_dir}/training_progress.png")


# Example usage
if __name__ == "__main__":
    from baseline_lm import create_baseline_model_with_tokenizer
    
    # Create model
    model, tokenizer, _ = create_baseline_model_with_tokenizer()
    
    # Create config for testing
    config = TrainingConfig(
        train_tokens=10_000_000,      # 10M tokens for testing
        eval_tokens=1_000_000,        # 1M tokens for eval
        warmup_tokens=1_000_000,      # 1M tokens for warmup
        micro_batch_size=1,
        gradient_accumulation_steps=64,  # Smaller for testing
        seq_length=512,               # Shorter for testing
        allow_fallback_synthetic=True  # Allow fallback for testing
    )
    
    # Create trainer
    trainer = TokenBudgetTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        save_dir="results/token_budget_test"
    )
    
    # Run training
    results = trainer.train()
    
    print(f"\n✓ Token-budget training completed!")
    print(f"Steps: {results['global_step']:,}")
    print(f"Tokens: {results['tokens_processed']:,}")
    print(f"Best PPL: {results['best_ppl']:.2f}")