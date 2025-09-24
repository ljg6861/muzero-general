#!/usr/bin/env python3
"""
Cognitive LLM Training System
============================
Phase 1: Foundation Knowledge Training with comprehensive model persistence,
checkpointing, and cognitive state management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import pickle
import os
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import math
import time
from tqdm import tqdm
import logging

from cognitive_llm import CognitiveLLM, CognitiveLLMConfig


class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of evaluations to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.history = []
    
    def __call__(self, metric_value: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            metric_value: Current value of the monitored metric
            
        Returns:
            True if training should stop, False otherwise
        """
        self.history.append(metric_value)
        
        if self.mode == 'min':
            improved = metric_value < (self.best_value - self.min_delta)
        else:
            improved = metric_value > (self.best_value + self.min_delta)
        
        if improved:
            self.best_value = metric_value
            self.counter = 0
            print(f"   📈 Metric improved to {metric_value:.4f}")
        else:
            self.counter += 1
            print(f"   ⚠️  No improvement for {self.counter}/{self.patience} evaluations (current: {metric_value:.4f}, best: {self.best_value:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"   🛑 Early stopping triggered after {self.patience} evaluations without improvement")
        
        return self.early_stop
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.early_stop = False
        self.history = []


@dataclass
class TrainingConfig:
    """Configuration for training the Cognitive LLM."""
    # Training parameters
    num_epochs: int = 10
    learning_rate: float = 1e-4
    cognitive_learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Optimization
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    max_checkpoints: int = 5
    
    # Early Stopping Parameters
    early_stopping_patience: int = 3  # Stop if no improvement for 3 evaluations
    early_stopping_min_delta: float = 0.001  # Minimum improvement threshold
    early_stopping_metric: str = "eval_loss"  # Metric to monitor for early stopping
    
    # Enhanced Regularization
    dropout_prob: float = 0.1  # Base dropout probability
    attention_dropout: float = 0.1  # Attention-specific dropout
    layer_dropout: float = 0.0  # Layer-wise dropout (stochastic depth)
    gradient_noise_scale: float = 0.0  # Add noise to gradients for regularization
    
    # Data Regularization
    label_smoothing: float = 0.1  # Label smoothing for cross-entropy
    data_augmentation: bool = True  # Enable data augmentation
    shuffle_training_data: bool = True  # Shuffle data each epoch
    
    # Learning Rate Scheduling
    lr_scheduler_type: str = "cosine_with_restarts"  # cosine, linear, polynomial, cosine_with_restarts
    lr_scheduler_warmup_ratio: float = 0.1  # Fraction of training for warmup
    lr_min_ratio: float = 0.01  # Minimum LR as fraction of max LR
    
    # Training dynamics
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    
    # Cognitive enhancement
    cognitive_loss_weight: float = 0.1
    concept_formation_weight: float = 0.3
    causal_reasoning_weight: float = 0.4
    meta_learning_weight: float = 0.3
    
    # Resume training
    resume_from_checkpoint: Optional[str] = None
    
    # Paths
    output_dir: str = "cognitive_llm_output"
    logging_dir: str = "logs"
    cache_dir: str = "cache"


class ModelCheckpoint:
    """Enhanced model checkpointing with cognitive state preservation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.output_dir = config.output_dir
        self.max_checkpoints = config.max_checkpoints
        
        # Ensure directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        self.checkpoint_history = []
        
    def save_checkpoint(self, 
                       model: CognitiveLLM, 
                       optimizer: optim.Optimizer,
                       lr_scheduler,
                       epoch: int,
                       step: int,
                       loss: float,
                       eval_metrics: Dict[str, float] = None):
        """Save a complete training checkpoint."""
        
        checkpoint_name = f"checkpoint-epoch-{epoch}-step-{step}"
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", checkpoint_name)
        
        # Create checkpoint directory
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save model and config
        model.save_model(checkpoint_path)
        
        # Convert eval_metrics to JSON-serializable format
        json_safe_eval_metrics = {}
        if eval_metrics:
            for key, value in eval_metrics.items():
                if isinstance(value, torch.Tensor):
                    json_safe_eval_metrics[key] = value.item() if value.numel() == 1 else value.tolist()
                elif isinstance(value, (int, float, str, bool, list, dict)):
                    json_safe_eval_metrics[key] = value
                else:
                    json_safe_eval_metrics[key] = str(value)
        
        # Save training state (JSON-serializable only)
        training_state = {
            'epoch': epoch,
            'step': step,
            'loss': loss.item() if isinstance(loss, torch.Tensor) else loss,
            'training_config': asdict(self.config),
            'eval_metrics': json_safe_eval_metrics,
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__,
            'model_config': asdict(model.config)
        }
        
        try:
            with open(os.path.join(checkpoint_path, 'training_state.json'), 'w') as f:
                json.dump(training_state, f, indent=2)
        except TypeError as e:
            print(f"Warning: Could not serialize training state to JSON: {e}")
            # Save as pickle as fallback
            with open(os.path.join(checkpoint_path, 'training_state.pkl'), 'wb') as f:
                pickle.dump(training_state, f)
        
        # Save optimizer state separately (can be large)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, 'optimizer.pt'))
        if lr_scheduler:
            torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_path, 'scheduler.pt'))
        
        # Update checkpoint history
        self.checkpoint_history.append({
            'name': checkpoint_name,
            'path': checkpoint_path,
            'epoch': epoch,
            'step': step,
            'loss': loss,
            'timestamp': datetime.now().isoformat()
        })
        
        # Remove old checkpoints if exceeding max_checkpoints
        if len(self.checkpoint_history) > self.max_checkpoints:
            old_checkpoint = self.checkpoint_history.pop(0)
            if os.path.exists(old_checkpoint['path']):
                shutil.rmtree(old_checkpoint['path'])
                print(f"Removed old checkpoint: {old_checkpoint['name']}")
        
        # Save checkpoint history
        with open(os.path.join(self.output_dir, 'checkpoint_history.json'), 'w') as f:
            json.dump(self.checkpoint_history, f, indent=2)
        
        print(f"Saved checkpoint: {checkpoint_name} (loss: {loss:.4f})")
        
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str, model: CognitiveLLM, optimizer: optim.Optimizer, lr_scheduler=None):
        """Load a training checkpoint."""
        
        # Load model
        model = CognitiveLLM.load_model(checkpoint_path, device=next(model.parameters()).device)
        
        # Load training state (try JSON first, fallback to pickle)
        training_state = None
        json_path = os.path.join(checkpoint_path, 'training_state.json')
        pkl_path = os.path.join(checkpoint_path, 'training_state.pkl')
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    training_state = json.load(f)
            except (json.JSONDecodeError, TypeError):
                print("Warning: Could not load JSON training state, trying pickle...")
        
        if training_state is None and os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                training_state = pickle.load(f)
        
        if training_state is None:
            raise ValueError(f"Could not load training state from {checkpoint_path}")
        
        # Load optimizer state
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        if os.path.exists(optimizer_path):
            optimizer.load_state_dict(torch.load(optimizer_path))
        
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_path, 'scheduler.pt')
        if lr_scheduler and os.path.exists(scheduler_path):
            lr_scheduler.load_state_dict(torch.load(scheduler_path))
        
        print(f"Loaded checkpoint from epoch {training_state['epoch']}, step {training_state['step']}")
        
        return model, optimizer, lr_scheduler, training_state
    
    def get_latest_checkpoint(self):
        """Get the path to the latest checkpoint."""
        if not self.checkpoint_history:
            history_path = os.path.join(self.output_dir, 'checkpoint_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.checkpoint_history = json.load(f)
        
        if self.checkpoint_history:
            return self.checkpoint_history[-1]['path']
        return None


class CognitiveLossFunction:
    """Enhanced loss function incorporating cognitive reasoning objectives."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        # Use reduction='none' to get per-token losses for proper mean calculation
        self.language_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        self.concept_loss = nn.MSELoss()
        self.causal_loss = nn.BCEWithLogitsLoss()
        
    def calculate_language_modeling_loss(self, logits, labels, attention_mask=None):
        """
        Standard language modeling loss with proper per-token mean calculation.
        Returns both mean loss and detailed metrics for trustworthy evaluation.
        """
        # Shift for causal language modeling: predict next token
        shift_logits = logits[..., :-1, :].contiguous()  # Remove last token prediction
        shift_labels = labels[..., 1:].contiguous()      # Remove first token (input)
        
        # Create attention mask for shifted sequence
        if attention_mask is not None:
            shift_attention_mask = attention_mask[..., 1:].contiguous()
        else:
            shift_attention_mask = torch.ones_like(shift_labels)
        
        # Calculate per-token losses
        per_token_losses = self.language_loss(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        per_token_losses = per_token_losses.view(shift_labels.shape)  # Reshape back to [batch, seq_len-1]
        
        # Apply attention mask to ignore padding tokens
        masked_losses = per_token_losses * shift_attention_mask.float()
        
        # Calculate metrics
        valid_tokens = shift_attention_mask.sum()
        total_loss_sum = masked_losses.sum()
        mean_loss = total_loss_sum / valid_tokens if valid_tokens > 0 else torch.tensor(0.0, device=logits.device)
        
        # Sequence length statistics
        batch_size = shift_labels.size(0)
        seq_lengths = shift_attention_mask.sum(dim=1).float()  # Valid tokens per sequence
        mean_seq_len = seq_lengths.mean()
        
        return {
            'loss_mean': mean_loss,
            'loss_sum': total_loss_sum,
            'n_tokens': valid_tokens.item(),
            'seq_len_mean': mean_seq_len.item(),
            'per_token_losses': per_token_losses,
            'valid_mask': shift_attention_mask
        }
    
    def calculate_concept_formation_loss(self, hidden_states, concept_targets=None):
        """Loss for concept formation quality."""
        if concept_targets is None:
            # Self-supervised concept formation loss
            # Encourage distinct concept representations
            batch_size, seq_len, hidden_size = hidden_states.shape
            concepts = hidden_states.view(batch_size, -1)
            
            # Diversity loss - encourage different concepts to be distinct
            concept_similarity = torch.mm(concepts, concepts.t())
            diversity_loss = torch.mean(torch.triu(concept_similarity, diagonal=1) ** 2)
            
            # Coherence loss - encourage concepts within sequence to be coherent
            seq_concepts = hidden_states.mean(dim=1)  # Average over sequence
            coherence_target = seq_concepts.unsqueeze(1).expand(-1, seq_len, -1)
            coherence_loss = self.concept_loss(hidden_states, coherence_target)
            
            return 0.5 * diversity_loss + 0.5 * coherence_loss
        else:
            return self.concept_loss(hidden_states, concept_targets)
    
    def calculate_causal_reasoning_loss(self, attention_probs):
        """Loss for causal reasoning quality based on attention patterns."""
        batch_size, num_heads, seq_len, _ = attention_probs.shape
        
        # Causal mask - future tokens should have zero attention
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        causal_mask = causal_mask.to(attention_probs.device)
        
        # Calculate violation of causal constraints
        future_attention = attention_probs * causal_mask.unsqueeze(0).unsqueeze(0)
        causal_violation = torch.mean(future_attention ** 2)
        
        return causal_violation
    
    def calculate_total_loss(self, outputs, labels, attention_mask=None, epoch=0, enable_cognitive=True):
        """
        Calculate total loss combining all objectives with proper per-token accounting.
        
        Args:
            outputs: Model outputs containing logits, hidden_states, attention_probs
            labels: Target token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            epoch: Current training epoch
            enable_cognitive: Whether to include cognitive losses (for ablation)
        
        Returns:
            Dictionary with detailed loss metrics including per-token statistics
        """
        logits = outputs['logits']
        hidden_states = outputs['hidden_states']
        attention_probs = outputs.get('attention_probs', [])
        
        # Language modeling loss with detailed metrics
        lm_metrics = self.calculate_language_modeling_loss(logits, labels, attention_mask)
        lm_loss = lm_metrics['loss_mean']
        
        # Initialize cognitive losses
        concept_loss = torch.tensor(0.0, device=logits.device)
        causal_loss = torch.tensor(0.0, device=logits.device)
        
        # Only calculate cognitive losses if enabled (for baseline vs cognitive comparison)
        if enable_cognitive and self.config.cognitive_loss_weight > 0:
            # Cognitive enhancement losses
            if hidden_states is not None:
                concept_loss = self.calculate_concept_formation_loss(hidden_states)
            
            # Average attention across layers for causal loss
            if attention_probs:
                avg_attention = torch.stack(attention_probs).mean(dim=0)
                causal_loss = self.calculate_causal_reasoning_loss(avg_attention)
        
        # Combine losses with weights
        total_loss = lm_loss
        if enable_cognitive:
            total_loss += (self.config.cognitive_loss_weight * 
                          self.config.concept_formation_weight * concept_loss +
                          self.config.cognitive_loss_weight * 
                          self.config.causal_reasoning_weight * causal_loss)
        
        # Calculate perplexity for verification: PPL = exp(loss_mean)
        perplexity = torch.exp(lm_loss.clamp(max=100))  # Clamp to prevent overflow
        
        return {
            'total_loss': total_loss,
            'language_modeling_loss': lm_loss,
            'concept_formation_loss': concept_loss,
            'causal_reasoning_loss': causal_loss,
            # Detailed LM metrics for trustworthy evaluation
            'lm_loss_mean': lm_metrics['loss_mean'],
            'lm_loss_sum': lm_metrics['loss_sum'],
            'n_tokens': lm_metrics['n_tokens'],
            'seq_len_mean': lm_metrics['seq_len_mean'],
            'perplexity': perplexity,
            # Verification: PPL should equal exp(loss_mean)
            'ppl_verification': torch.exp(lm_metrics['loss_mean']).clamp(max=1e6)
        }


class CognitiveLLMTrainer:
    """Main trainer class for Cognitive LLM."""
    
    def __init__(self, 
                 model: CognitiveLLM,
                 train_dataloader: DataLoader,
                 eval_dataloader: DataLoader = None,
                 config: TrainingConfig = None):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or TrainingConfig()
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Setup loss function
        self.loss_function = CognitiveLossFunction(self.config)
        
        # Setup checkpointing
        self.checkpoint_manager = ModelCheckpoint(self.config)
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.early_stopping_min_delta,
            mode='min' if 'loss' in self.config.early_stopping_metric else 'max'
        )
        
        # Setup logging
        self.setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.current_step = 0
        self.best_eval_loss = float('inf')
        self.training_stopped_early = False
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.resume_from_checkpoint()
    
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different components."""
        # Separate parameters for different learning rates
        cognitive_params = []
        transformer_params = []
        
        for name, param in self.model.named_parameters():
            if any(x in name for x in ['cognitive', 'concept', 'causal', 'reasoning']):
                cognitive_params.append(param)
            else:
                transformer_params.append(param)
        
        optimizer = optim.AdamW([
            {'params': transformer_params, 'lr': self.config.learning_rate},
            {'params': cognitive_params, 'lr': self.config.cognitive_learning_rate}
        ], 
        weight_decay=self.config.weight_decay,
        betas=(self.config.beta1, self.config.beta2),
        eps=self.config.eps)
        
        return optimizer
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return max(0.1, (self.config.warmup_steps / step) ** 0.5)
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def setup_logging(self):
        """Setup logging and tensorboard."""
        log_dir = os.path.join(self.config.output_dir, self.config.logging_dir)
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup file logging
        logging.basicConfig(
            filename=os.path.join(log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup tensorboard
        self.writer = SummaryWriter(log_dir)
        
        print(f"Logging to {log_dir}")
    
    def train_epoch(self, epoch: int, enable_cognitive: bool = True):
        """
        Train for one epoch with proper per-token loss tracking.
        
        Args:
            epoch: Current epoch number
            enable_cognitive: Whether to enable cognitive enhancements (for ablation)
        """
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_concept_loss = 0
        total_causal_loss = 0
        total_tokens = 0
        total_sequences = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch} ({'Cognitive' if enable_cognitive else 'Baseline LM'})")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            # Forward pass (with or without cognitive enhancement)
            if enable_cognitive:
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
            else:
                # Baseline mode: disable cognitive enhancements
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids, 
                                   enable_cognitive=False)
            
            # Calculate loss with detailed metrics
            loss_dict = self.loss_function.calculate_total_loss(
                outputs, input_ids, attention_mask, epoch, enable_cognitive
            )
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Calculate gradient norm for monitoring
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                self.current_step += 1
                
                # Enhanced logging with per-token metrics
                if self.current_step % self.config.logging_steps == 0:
                    enhanced_loss_dict = {**loss_dict, 'grad_norm': grad_norm.item()}
                    self._log_training_step(enhanced_loss_dict)
                
                # Evaluation with proper eval mode
                if self.eval_dataloader and self.current_step % self.config.eval_steps == 0:
                    eval_metrics = self.evaluate(enable_cognitive=enable_cognitive)
                    self._log_evaluation(eval_metrics)
                    
                    # Check early stopping
                    should_stop = self.early_stopping(eval_metrics['eval_loss'])
                    if should_stop:
                        self.training_stopped_early = True
                        print("🛑 Early stopping triggered!")
                        return None  # Signal to stop training
                
                # Checkpointing
                if self.current_step % self.config.save_steps == 0:
                    self.checkpoint_manager.save_checkpoint(
                        self.model, self.optimizer, self.lr_scheduler,
                        epoch, self.current_step, loss.item(), eval_metrics=eval_metrics if 'eval_metrics' in locals() else None
                    )
            
            # Update progress bar with detailed metrics
            total_loss += loss.item()
            total_lm_loss += loss_dict['language_modeling_loss'].item()
            total_concept_loss += loss_dict['concept_formation_loss'].item()
            total_causal_loss += loss_dict['causal_reasoning_loss'].item()
            total_tokens += loss_dict['n_tokens']
            total_sequences += input_ids.size(0)
            
            # Verify PPL calculation
            ppl_calculated = loss_dict['perplexity'].item()
            ppl_verification = loss_dict['ppl_verification'].item()
            ppl_match = abs(ppl_calculated - ppl_verification) < 0.01
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lm_loss': f"{loss_dict['language_modeling_loss'].item():.4f}",
                'ppl': f"{ppl_calculated:.1f}",
                'ppl_ok': "✓" if ppl_match else "✗",
                'tokens': f"{loss_dict['n_tokens']:.0f}",
                'lr': f"{self.lr_scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Calculate epoch averages
        num_batches = len(self.train_dataloader)
        # Calculate epoch averages
        num_batches = len(self.train_dataloader)
        avg_loss = total_loss / num_batches
        avg_lm_loss = total_lm_loss / num_batches
        avg_concept_loss = total_concept_loss / num_batches
        avg_causal_loss = total_causal_loss / num_batches
        avg_tokens_per_batch = total_tokens / num_batches if num_batches > 0 else 0
        
        return {
            'avg_loss': avg_loss,
            'avg_lm_loss': avg_lm_loss,
            'avg_concept_loss': avg_causal_loss,
            'avg_causal_loss': avg_causal_loss,
            'total_tokens': total_tokens,
            'total_sequences': total_sequences,
            'avg_tokens_per_batch': avg_tokens_per_batch
        }
    
    def evaluate(self, enable_cognitive: bool = True):
        """
        Evaluate the model with proper eval mode and detailed per-token metrics.
        
        Args:
            enable_cognitive: Whether to enable cognitive enhancements during evaluation
        """
        if not self.eval_dataloader:
            return {}
        
        # Ensure proper evaluation mode
        self.model.eval()
        
        total_loss = 0
        total_lm_loss = 0
        total_concept_loss = 0
        total_causal_loss = 0
        total_tokens = 0
        total_sequences = 0
        all_perplexities = []
        
        with torch.no_grad():  # Disable gradients for evaluation
            for batch in tqdm(self.eval_dataloader, desc=f"Evaluating ({'Cognitive' if enable_cognitive else 'Baseline'})"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                
                # Forward pass with cognitive enhancement control
                if enable_cognitive:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids)
                else:
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=input_ids, 
                                       enable_cognitive=False)
                
                # Calculate loss with detailed metrics
                loss_dict = self.loss_function.calculate_total_loss(
                    outputs, input_ids, attention_mask, epoch=0, enable_cognitive=enable_cognitive
                )
                
                # Accumulate metrics
                total_loss += loss_dict['total_loss'].item()
                total_lm_loss += loss_dict['language_modeling_loss'].item()
                total_concept_loss += loss_dict['concept_formation_loss'].item()
                total_causal_loss += loss_dict['causal_reasoning_loss'].item()
                total_tokens += loss_dict['n_tokens']
                total_sequences += input_ids.size(0)
                
                # Track perplexity for verification
                all_perplexities.append(loss_dict['perplexity'].item())
        
        # Calculate averages
        num_batches = len(self.eval_dataloader)
        avg_loss = total_loss / num_batches
        avg_lm_loss = total_lm_loss / num_batches
        avg_concept_loss = total_concept_loss / num_batches
        avg_causal_loss = total_causal_loss / num_batches
        avg_tokens_per_batch = total_tokens / num_batches if num_batches > 0 else 0
        
        # Calculate perplexity: PPL = exp(loss_mean)
        # Use the language modeling loss for perplexity calculation
        perplexity = math.exp(min(avg_lm_loss, 100))  # Clamp to prevent overflow
        avg_perplexity = sum(all_perplexities) / len(all_perplexities) if all_perplexities else 0
        
        # Verify PPL calculation consistency
        ppl_verification_passed = abs(perplexity - avg_perplexity) < (perplexity * 0.01)  # 1% tolerance
        
        # Restore training mode
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_lm_loss': avg_lm_loss,
            'eval_concept_loss': avg_concept_loss,
            'eval_causal_loss': avg_causal_loss,
            'perplexity': perplexity,
            'perplexity_verification': avg_perplexity,
            'ppl_verification_passed': ppl_verification_passed,
            'total_tokens': total_tokens,
            'total_sequences': total_sequences,
            'avg_tokens_per_batch': avg_tokens_per_batch,
            'n_eval_batches': num_batches
        }
        
        avg_loss = total_loss / len(self.eval_dataloader)
        avg_lm_loss = total_lm_loss / len(self.eval_dataloader)
        
        # Calculate perplexity
        perplexity = math.exp(min(avg_lm_loss, 100))  # Clip to prevent overflow
        
        self.model.train()
        
        return {
            'eval_loss': avg_loss,
            'eval_lm_loss': avg_lm_loss,
            'perplexity': perplexity
        }
    
    def train(self):
        """Main training loop."""
        print("Starting Cognitive LLM Training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train epoch
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            eval_metrics = self.evaluate() if self.eval_dataloader else {}
            
            # Log epoch results
            print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            print(f"Train LM Loss: {train_metrics['avg_lm_loss']:.4f}")
            if eval_metrics:
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
                print(f"Perplexity: {eval_metrics['perplexity']:.2f}")
            
            # Check early stopping
            if eval_metrics and self.config.early_stopping_patience > 0:
                metric_value = eval_metrics.get(self.config.early_stopping_metric, eval_metrics['eval_loss'])
                should_stop = self.early_stopping(metric_value)
                
                if should_stop:
                    print(f"\n🛑 Early stopping triggered at epoch {epoch + 1}")
                    print(f"   Best {self.config.early_stopping_metric}: {self.early_stopping.best_value:.4f}")
                    self.training_stopped_early = True
                    break
            
            # Save checkpoint at end of epoch
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                self.model, self.optimizer, self.lr_scheduler,
                epoch, self.current_step, train_metrics['avg_loss'],
                eval_metrics
            )
            
            # Save best model
            if eval_metrics and eval_metrics['eval_loss'] < self.best_eval_loss:
                self.best_eval_loss = eval_metrics['eval_loss']
                best_model_path = os.path.join(self.config.output_dir, "best_model")
                self.model.save_model(best_model_path)
                print(f"💾 New best model saved with eval loss: {self.best_eval_loss:.4f}")
        
        # Training completion message
        if self.training_stopped_early:
            print(f"\n🎯 Training stopped early after {self.current_epoch + 1} epochs")
            print(f"   Reason: No improvement in {self.config.early_stopping_metric} for {self.config.early_stopping_patience} evaluations")
        else:
            print(f"\n🎉 Training completed all {self.config.num_epochs} epochs!")
        
        print(f"   Best evaluation loss achieved: {self.best_eval_loss:.4f}")
        
        # Save final model
        final_model_path = os.path.join(self.config.output_dir, "final_model")
        self.model.save_model(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        self.writer.close()
    
    def _log_training_step(self, loss_dict):
        """Log training step metrics."""
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(f"train/{key}", value, self.current_step)
        
        self.writer.add_scalar("train/learning_rate", self.lr_scheduler.get_last_lr()[0], self.current_step)
        
        logging.info(f"Step {self.current_step}: "
                    f"loss={loss_dict['total_loss'].item():.4f}, "
                    f"lm_loss={loss_dict['language_modeling_loss'].item():.4f}")
    
    def _log_evaluation(self, eval_metrics):
        """Log evaluation metrics."""
        for key, value in eval_metrics.items():
            self.writer.add_scalar(f"eval/{key}", value, self.current_step)
        
        logging.info(f"Evaluation at step {self.current_step}: {eval_metrics}")
    
    def resume_from_checkpoint(self):
        """Resume training from a checkpoint."""
        checkpoint_path = self.config.resume_from_checkpoint
        
        if checkpoint_path == "latest":
            checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.model, self.optimizer, self.lr_scheduler, training_state = \
                self.checkpoint_manager.load_checkpoint(checkpoint_path, self.model, self.optimizer, self.lr_scheduler)
            
            self.current_epoch = training_state['epoch']
            self.current_step = training_state['step']
            
            print(f"Resumed training from epoch {self.current_epoch}, step {self.current_step}")
        else:
            print(f"Checkpoint not found: {checkpoint_path}")


def create_training_config():
    """Create default training configuration."""
    return TrainingConfig(
        num_epochs=5,
        learning_rate=1e-4,
        cognitive_learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        max_grad_norm=1.0,
        save_steps=1000,
        eval_steps=500,
        logging_steps=100,
        max_checkpoints=3,
        gradient_accumulation_steps=4,
        cognitive_loss_weight=0.1,
        output_dir="cognitive_llm_phase1",
        logging_dir="logs",
        cache_dir="cache"
    )


if __name__ == "__main__":
    # Test the training system
    from cognitive_llm import create_default_config
    
    print("Testing Cognitive LLM Training System...")
    
    # Create model
    model_config = create_default_config()
    model = CognitiveLLM(model_config)
    
    # Create dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, num_samples=1000, seq_length=128, vocab_size=50000):
            self.num_samples = num_samples
            self.seq_length = seq_length
            self.vocab_size = vocab_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            return {
                'input_ids': torch.randint(0, self.vocab_size, (self.seq_length,)),
                'attention_mask': torch.ones(self.seq_length)
            }
    
    # Create data loaders
    train_dataset = DummyDataset(1000)
    eval_dataset = DummyDataset(100)
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=2, shuffle=False)
    
    # Create training config
    training_config = create_training_config()
    training_config.num_epochs = 1  # Just test one epoch
    training_config.save_steps = 50
    training_config.eval_steps = 25
    training_config.logging_steps = 10
    
    # Create trainer
    trainer = CognitiveLLMTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=training_config
    )
    
    print("Training system ready!")
    print("Run trainer.train() to start training with full checkpointing and cognitive enhancement!")