#!/usr/bin/env python3
"""
Baseline-Continue: Pure LM CE Training with Plateau Controller
============================================================
Control curve: Continue training the LM with pure CE loss and plateau detection.
This serves as the baseline for comparing against MuZero-adapted models.
"""

import os
import sys
import time
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler
import torch.distributed as dist
import socket
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import multiprocessing
import torch.multiprocessing as mp
import json
import random
import numpy as np
from datetime import datetime
sys.path.append('model')
sys.path.append('.')

from data_registry import DataRegistry, DataConfig

# Performance optimizations
os.environ.setdefault('HF_HUB_READ_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

def setup_ddp():
    """Setup distributed data parallel if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank % torch.cuda.device_count()))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return True, rank, world_size, device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return False, 0, 1, device

def set_seed(seed):
    """Set random seed for reproducibility - optimized for performance"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _auto_ddp_entry(local_rank, world_size):
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    main()

class FlashAttentionLayer(nn.Module):
    """Custom transformer layer using Flash attention when available"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.gradient_checkpointing = gradient_checkpointing
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def _forward_impl(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        x2, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask, 
            need_weights=False,
            is_causal=False
        )
        x = self.norm1(x + self.dropout1(x2))
        
        # Feed forward block
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        
        return x
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, src, src_mask, src_key_padding_mask)
        else:
            return self._forward_impl(src, src_mask, src_key_padding_mask)

class SimpleLM(nn.Module):
    """Simple transformer language model for baseline continuation"""
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, max_seq_len=512, gradient_checkpointing=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.tok_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            FlashAttentionLayer(hidden_size, num_heads, 
                              dim_feedforward=hidden_size * 4, 
                              dropout=0.1,
                              gradient_checkpointing=gradient_checkpointing)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Register causal mask buffer
        causal_mask_full = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask_full', causal_mask_full, persistent=False)
        
        # Register position ids buffer
        pos_ids_full = torch.arange(max_seq_len, dtype=torch.long)
        self.register_buffer('pos_ids_full', pos_ids_full, persistent=False)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def encode(self, input_ids, pad_token_id=None):
        """Encode input tokens to hidden states"""
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        pos_ids = self.pos_ids_full[:seq_len].unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        tok_emb = self.tok_embedding(input_ids)
        pos_emb = self.pos_embedding(pos_ids)
        x = tok_emb + pos_emb
        
        # Create causal mask for this sequence length
        causal_mask = self.causal_mask_full[:seq_len, :seq_len]
        
        # Create padding mask if pad_token_id is provided
        key_padding_mask = None
        if pad_token_id is not None:
            key_padding_mask = (input_ids == pad_token_id)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        return x
    
    def forward(self, input_ids, pad_token_id=None):
        """Full forward pass with logits"""
        hidden_states = self.encode(input_ids, pad_token_id)
        logits = self.lm_head(hidden_states)
        return logits

    def extend_max_seq_len(self, new_max_len: int):
        """Extend max sequence length (for context extension later)"""
        if new_max_len <= self.max_seq_len:
            return
        device = self.pos_embedding.weight.device
        # Extend pos embedding
        old_pe = self.pos_embedding
        new_pe = nn.Embedding(new_max_len, self.hidden_size).to(device)
        torch.nn.init.normal_(new_pe.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            new_pe.weight[: self.max_seq_len].copy_(old_pe.weight)
        self.pos_embedding = new_pe
        # Update buffers
        causal_mask_full = torch.triu(torch.ones(new_max_len, new_max_len, dtype=torch.bool, device=device), diagonal=1)
        self.register_buffer('causal_mask_full', causal_mask_full, persistent=False)
        pos_ids_full = torch.arange(new_max_len, dtype=torch.long, device=device)
        self.register_buffer('pos_ids_full', pos_ids_full, persistent=False)
        self.max_seq_len = new_max_len

def compute_loss_with_proper_scaling(logits, input_ids, tokenizer, reduction='mean'):
    """Compute cross-entropy efficiently over non-pad tokens.
    Uses ignore_index path in fused CE to avoid Python-side masking overhead.
    Returns (loss, num_valid_tokens). For reduction='sum', loss is the summed CE over valid tokens.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Flatten for fused CE
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    # Use ignore_index to skip pads in the kernel directly
    has_pad = hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None
    ignore_index = tokenizer.pad_token_id if has_pad else -100

    loss_sum = F.cross_entropy(
        flat_logits,
        flat_labels,
        reduction='sum',
        ignore_index=ignore_index
    )

    if has_pad:
        num_valid_tokens = (flat_labels != ignore_index).sum().item()
    else:
        num_valid_tokens = flat_labels.numel()

    if reduction == 'mean':
        return loss_sum / max(1, num_valid_tokens), num_valid_tokens
    elif reduction == 'sum':
        return loss_sum, num_valid_tokens
    else:
        # Rarely used, but keep for completeness
        # Recompute elementwise if needed
        loss_per_token = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction='none',
            ignore_index=ignore_index
        )
        return loss_per_token, num_valid_tokens

def _build_ngram_map(sequence: torch.Tensor, n: int):
    mapping = {}
    if sequence.numel() < n:
        return mapping
    seq_list = sequence.tolist()
    for i in range(len(seq_list) - n + 1):
        prefix = tuple(seq_list[i:i + n - 1])
        nxt = seq_list[i + n - 1]
        s = mapping.get(prefix)
        if s is None:
            s = set()
            mapping[prefix] = s
        s.add(nxt)
    return mapping

def generate_samples(model, tokenizer, device, prompts=None, max_new_tokens=40, temperature=0.9, top_p=0.95, no_repeat_ngram_size=3):
    """Generate text samples for evaluation"""
    model.eval()
    prompts = prompts or ["The future of AI is", "Scientists discovered"]
    with torch.no_grad():
        print("🎯 Sample generations:")
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            for _ in range(max_new_tokens):
                base_model = model.module if hasattr(model, 'module') else model
                hs = base_model.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
                logits = base_model.lm_head(hs)
                next_logits = logits[:, -1, :] / max(1e-5, temperature)
                # N-gram blocking
                if no_repeat_ngram_size and no_repeat_ngram_size > 1:
                    nmap = _build_ngram_map(input_ids[0], no_repeat_ngram_size)
                    if input_ids.shape[1] >= no_repeat_ngram_size - 1:
                        prefix = tuple(input_ids[0, -(no_repeat_ngram_size - 1):].tolist())
                        blocked = nmap.get(prefix, set())
                        if blocked:
                            next_logits[:, list(blocked)] = float('-inf')
                # Top-p nucleus sampling
                probs = F.softmax(next_logits, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative > top_p
                mask[..., 0] = 0
                filtered_probs = sorted_probs.masked_fill(mask, 0)
                probs_final = torch.zeros_like(probs).scatter(1, sorted_idx, filtered_probs)
                probs_final = probs_final / probs_final.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                next_token = torch.multinomial(probs_final, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if next_token.item() == tokenizer.eos_token_id:
                    break
            generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            print(f"   '{prompt}' → '{generated}'")
    model.train()

def evaluate_on_dataset(model, tokenizer, device, registry, data_config, per_process_batch):
    """Evaluate CE and PPL on streaming eval data"""
    model.eval()
    loader = registry.create_dataloader(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=per_process_batch,
        is_eval=True,
        use_sized=False,
        num_workers=0,
        prefetch_factor=None,
        persistent_workers=False
    )
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            
            base_model = model.module if hasattr(model, 'module') else model
            hidden_states = base_model.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
            logits = base_model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            
            loss_flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                mask = (shift_labels.view(-1) != tokenizer.pad_token_id)
                loss_flat = loss_flat * mask.float()
                valid_tokens = mask.sum().item()
            else:
                valid_tokens = loss_flat.numel()
            
            if valid_tokens > 0:
                total_loss += loss_flat.sum().item()
                total_tokens += valid_tokens
            
            if total_tokens >= data_config.eval_tokens:
                break
    
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        ppl = math.exp(avg_loss)
        return avg_loss, ppl
    else:
        return None, None

class PlateauController:
    """Manages plateau detection and controlled changes"""
    def __init__(self, window_size=10_000_000, improvement_threshold=0.03, max_changes=3):
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.max_changes = max_changes
        
        # State variables
        self.ce_history = []
        self.consecutive_windows = 0
        self.changes_made = 0
        self.tokens_at_last_change = 0
        self.adjustments = []
    
    def check_plateau(self, tokens_processed, eval_ce):
        """Check for plateau and return True if controlled change should be made"""
        self.ce_history.append((tokens_processed, eval_ce))
        
        if len(self.ce_history) < 2:
            return False, 0.0
        
        # Find evaluations that are at least window_size tokens apart
        current_tokens = tokens_processed
        windows_to_check = []
        
        for i in range(len(self.ce_history) - 1, -1, -1):
            eval_tokens, eval_ce_val = self.ce_history[i]
            if current_tokens - eval_tokens >= self.window_size:
                windows_to_check.append((eval_tokens, eval_ce_val))
                if len(windows_to_check) >= 2:
                    break
        
        if len(windows_to_check) < 2:
            return False, 0.0
        
        # Calculate improvement (positive = better, since lower CE is better)
        prev_ce = windows_to_check[0][1]
        curr_ce = eval_ce
        improvement = prev_ce - curr_ce
        
        if improvement < self.improvement_threshold:
            self.consecutive_windows += 1
            
            if self.consecutive_windows >= 2 and self.changes_made < self.max_changes:
                return True, improvement
        else:
            self.consecutive_windows = 0
        
        return False, improvement
    
    def make_controlled_change(self, optimizer, tokens_processed, change_type="lr_reduction"):
        """Make a controlled change and record it"""
        if change_type == "lr_reduction":
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.5
                self.adjustments.append((tokens_processed, "lr_reduction", old_lr, param_group['lr']))
                print(f"   📉 Learning rate reduced: {old_lr:.2e} → {param_group['lr']:.2e}")
        
        self.changes_made += 1
        self.tokens_at_last_change = tokens_processed
        self.consecutive_windows = 0
    
    def get_state(self):
        """Get current state for saving"""
        return {
            'window_size': self.window_size,
            'improvement_threshold': self.improvement_threshold,
            'max_changes': self.max_changes,
            'ce_history': self.ce_history,
            'consecutive_windows': self.consecutive_windows,
            'changes_made': self.changes_made,
            'tokens_at_last_change': self.tokens_at_last_change,
            'adjustments': self.adjustments
        }
    
    def load_state(self, state_dict):
        """Load state from saved checkpoint"""
        self.window_size = state_dict['window_size']
        self.improvement_threshold = state_dict['improvement_threshold']
        self.max_changes = state_dict['max_changes']
        self.ce_history = state_dict['ce_history']
        self.consecutive_windows = state_dict['consecutive_windows']
        self.changes_made = state_dict['changes_made']
        self.tokens_at_last_change = state_dict['tokens_at_last_change']
        self.adjustments = state_dict['adjustments']

def load_baseline_checkpoint(checkpoint_path, model, optimizer, scaler, plateau_controller):
    """Load checkpoint and return the token count, step, eval_ce, eval_ppl."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=model.device if hasattr(model, 'device') else 'cpu', weights_only=False)
    
    # Handle DataParallel module prefix
    state_dict = checkpoint['model_state_dict']
    
    # Check if we need to remove 'module.' prefix
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())
    
    if model_keys[0].startswith('module.') and not checkpoint_keys[0].startswith('module.'):
        # Model has module prefix, checkpoint doesn't - add prefix
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}
    elif not model_keys[0].startswith('module.') and checkpoint_keys[0].startswith('module.'):
        # Model doesn't have module prefix, checkpoint does - remove prefix
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # Safely handle positional embedding length mismatches (sequence length changes)
    try:
        # Access current model pos embedding weight to determine target shape
        current_pos_key = 'module.pos_embedding.weight' if next(iter(model.state_dict().keys())).startswith('module.') else 'pos_embedding.weight'
        target_shape = model.state_dict()[current_pos_key].shape  # (max_seq_len, hidden_size)
        source_key = 'pos_embedding.weight' if current_pos_key == 'pos_embedding.weight' else 'module.pos_embedding.weight'
        # After prefix normalization above, pos key should match current_pos_key
        src_key = current_pos_key
        if src_key in state_dict:
            ckpt_pe = state_dict[src_key]
            if ckpt_pe.shape != target_shape:
                # Create new weight starting from the model's current PE, then copy overlap from checkpoint
                model_pe = model.state_dict()[current_pos_key]
                new_pe = model_pe.clone()
                min_len = min(ckpt_pe.shape[0], target_shape[0])
                with torch.no_grad():
                    new_pe[:min_len].copy_(ckpt_pe[:min_len].to(new_pe.device))
                state_dict[src_key] = new_pe.to(ckpt_pe.device)
                print(f"✓ Adjusted pos_embedding from {tuple(ckpt_pe.shape)} to {tuple(target_shape)} (copied first {min_len} rows)")
    except Exception as e:
        # If anything goes wrong, drop the key to avoid size mismatch; model keeps its current PE
        pos_key = 'module.pos_embedding.weight' if any(k.startswith('module.') for k in model.state_dict().keys()) else 'pos_embedding.weight'
        if pos_key in state_dict:
            del state_dict[pos_key]
            print(f"⚠️  Dropped '{pos_key}' from checkpoint due to mismatch: {e}")
    
    # Load with strict=False to tolerate any benign key differences
    model.load_state_dict(state_dict, strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scaler state if available
    if 'scaler_state_dict' in checkpoint and scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Load plateau controller state if available
    if 'plateau_controller_state' in checkpoint and plateau_controller is not None:
        state = checkpoint['plateau_controller_state']
        plateau_controller.ce_history = state.get('ce_history', [])
        plateau_controller.consecutive_windows = state.get('consecutive_windows', 0)
    
    tokens_trained = checkpoint.get('tokens_processed', 0)  # Note: 'tokens_processed' in checkpoint
    step = checkpoint.get('step', 0)
    eval_ce = checkpoint.get('eval_ce', float('inf'))
    eval_ppl = checkpoint.get('eval_ppl', float('inf'))
    
    print(f"Loaded checkpoint: {tokens_trained:,} tokens trained, step {step}")
    print(f"  Last eval: CE={eval_ce:.4f}, PPL={eval_ppl:.2f}")
    return tokens_trained, step, eval_ce, eval_ppl

def save_baseline_checkpoint(model, optimizer, scaler, plateau_controller, tokens_processed, 
                           step, eval_ce, eval_ppl, seed, args, save_dir="baseline_continue_checkpoints"):
    """Save baseline training checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokens_m = tokens_processed // 1_000_000
    checkpoint_name = f"baseline_continue_{tokens_m}M_tokens_{timestamp}"
    checkpoint_path = os.path.join(save_dir, f"{checkpoint_name}.pt")
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    checkpoint = {
        'timestamp': timestamp,
        'tokens_processed': tokens_processed,
        'step': step,
        'seed': seed,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'plateau_controller_state': plateau_controller.get_state(),
        'eval_ce': eval_ce,
        'eval_ppl': eval_ppl,
        'model_config': {
            'vocab_size': model_to_save.vocab_size,
            'hidden_size': model_to_save.hidden_size,
            'num_layers': len(model_to_save.layers),
            'max_seq_len': model_to_save.max_seq_len,
        },
        'training_args': vars(args),
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    torch.save(checkpoint, checkpoint_path)

    # Generation-based aliases: {gen_prefix}_{gen_id}-{n}.pt and latest pointer
    def _update_generation_aliases(gen_prefix: str, gen_id: str, new_path: str):
        try:
            # Maintain a simple index file per generation id
            index_file = os.path.join(save_dir, f"{gen_prefix}_{gen_id}_index.txt")
            try:
                with open(index_file, 'r') as f:
                    idx = int(f.read().strip())
            except Exception:
                idx = 0
            idx += 1
            with open(index_file, 'w') as f:
                f.write(str(idx))

            # Symlink name like benchmark_gen1-3.pt
            gen_name = f"{gen_prefix}_{gen_id}-{idx}.pt"
            gen_path = os.path.join(save_dir, gen_name)
            # Use relative path for portability
            rel = os.path.relpath(new_path, start=save_dir)
            # If a symlink/file with same name exists, remove it first
            if os.path.islink(gen_path) or os.path.isfile(gen_path):
                os.remove(gen_path)
            os.symlink(rel, gen_path)

            # Update latest pointer for this generation
            latest = os.path.join(save_dir, f"{gen_prefix}_{gen_id}_latest.pt")
            if os.path.islink(latest) or os.path.isfile(latest):
                os.remove(latest)
            os.symlink(rel, latest)
        except Exception:
            # Best-effort; ignore failures
            pass

    _update_generation_aliases(getattr(args, 'gen_prefix', 'benchmark'), getattr(args, 'gen_id', 'gen1'), checkpoint_path)
    
    # Save summary
    summary_path = os.path.join(save_dir, f"{checkpoint_name}_summary.json")
    summary = {
        'checkpoint_name': checkpoint_name,
        'branch': 'baseline-continue',
        'timestamp': timestamp,
        'tokens_processed': tokens_processed,
        'step': step,
        'eval_ce': eval_ce,
        'eval_ppl': eval_ppl,
        'plateau_changes_made': plateau_controller.changes_made,
        'current_lr': optimizer.param_groups[0]['lr'],
        'training_args': vars(args)
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Baseline-continue checkpoint saved:")
    print(f"   Path: {checkpoint_path}")
    print(f"   Tokens: {tokens_processed:,} ({tokens_m}M)")
    print(f"   Eval CE: {eval_ce:.4f}, PPL: {eval_ppl:.2f}")
    print(f"   Generation alias written: {getattr(args, 'gen_prefix', 'benchmark')}_{getattr(args, 'gen_id', 'gen1')}_latest.pt")
    
    return checkpoint_path, summary_path

def main():
    parser = argparse.ArgumentParser(description="Baseline-Continue: Pure LM CE Training with Plateau Controller")
    
    # Training configuration
    parser.add_argument("--train_tokens", type=int, default=50_000_000, help="Total training tokens")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length")
    parser.add_argument("--eval_tokens", type=int, default=1_500_000, help="Evaluation token budget")
    
    # Model configuration
    parser.add_argument("--hidden_size", type=int, default=256, help="Model hidden size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    
    # Performance knobs
    parser.add_argument("--per_gpu_batch_size", type=int, default=128, help="Per-GPU batch size")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers per process")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="DataLoader prefetch factor")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    
    # Optimization
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--fused_adamw", action="store_true", help="Use fused AdamW if available")
    
    # Plateau rule configuration
    parser.add_argument("--plateau_window", type=int, default=10_000_000, help="Plateau detection window (tokens)")
    parser.add_argument("--plateau_threshold", type=float, default=0.03, help="Plateau improvement threshold")
    parser.add_argument("--plateau_max_changes", type=int, default=3, help="Maximum plateau-triggered changes")
    
    # Checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="baseline_continue_checkpoints", help="Checkpoint directory")
    parser.add_argument("--load_checkpoint", type=str, default=None, help="Load from checkpoint to continue training")
    # Generation naming
    parser.add_argument("--gen_prefix", type=str, default="benchmark", help="Prefix for generation aliases (e.g., 'benchmark')")
    parser.add_argument("--gen_id", type=str, default="gen1", help="Generation id (e.g., 'gen1')")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Multi-GPU
    parser.add_argument("--no_auto_ddp", action="store_true", help="Disable automatic multi-GPU DDP")
    
    # Logging
    parser.add_argument("--log_json", type=str, default=None, help="Write final metrics to JSON file")
    
    args, _ = parser.parse_known_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Auto DDP setup
    if 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 and not args.no_auto_ddp:
            world_size = torch.cuda.device_count()
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", 0))
                _, free_port = s.getsockname()
            os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
            os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(free_port))
            
            print(f"🔁 Auto-launching DDP across {world_size} GPUs")
            mp.spawn(_auto_ddp_entry, nprocs=world_size, args=(world_size,))
            return
    
    # Setup DDP
    use_ddp, rank, world_size, device = setup_ddp()
    
    # Enable performance optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # Prefer fast SDPA kernels (Flash or Mem-Efficient) over math fallback
        try:
            from torch.backends.cuda import sdp_kernel
            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass
        if rank == 0:
            print(f"✓ Enabled TF32 and cuDNN optimizations")
    
    # Print configuration
    if rank == 0:
        print("🔥 Baseline-Continue: Pure LM CE Training")
        print(f"   Device: {device}")
        print(f"   Seed: {args.seed}")
        print(f"   Target tokens: {args.train_tokens:,}")
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"   Available GPUs: {num_gpus}")
            if use_ddp:
                print(f"   🚀 Using DDP with {world_size} processes!")
    
    # Setup data
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    data_config = DataConfig(
        train_tokens=args.train_tokens,
        seq_length=args.seq_length,
        data_sources=['wikipedia'],
        allow_fallback_synthetic=False
    )
    data_config.eval_tokens = args.eval_tokens
    
    registry = DataRegistry()
    
    # DataLoader setup
    base_batch_size = args.per_gpu_batch_size
    if not use_ddp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
    
    # Use a few workers per process to overlap tokenization
    num_workers_dl = 4 if args.num_workers is None else max(0, args.num_workers)
    
    train_loader = registry.create_dataloader(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=base_batch_size,
        use_sized=False,
        num_workers=num_workers_dl,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(num_workers_dl > 0)
    )
    
    # Create model
    model = SimpleLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_length,
        gradient_checkpointing=args.grad_ckpt
    )
    
    model = model.to(device)
    
    # Setup DDP
    if use_ddp:
        model = DDP(model, device_ids=[rank])
        if rank == 0:
            print(f"🚀 Wrapped model with DDP")
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model: {num_params:,} parameters")
    
    # Setup optimizer and scaler
    optim_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    # Prefer fused AdamW when available even if flag not passed
    use_fused = False
    try:
        # PyTorch exposes a kwarg 'fused' on supported builds/GPUs
        if args.fused_adamw:
            use_fused = True
        else:
            # Auto-enable if the optimizer supports it and CUDA is available
            test = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
            # If creation succeeds, discard and recreate below properly
            del test
            use_fused = True
    except Exception:
        use_fused = False

    if use_fused:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
        if rank == 0:
            print("✓ Using fused AdamW optimizer")
    else:
        # foreach tends to be faster on many GPUs for large parameter sets
        optimizer = torch.optim.AdamW(model.parameters(), foreach=True, **optim_kwargs)
    
    scaler = GradScaler('cuda')
    
    # Setup plateau controller
    plateau_controller = PlateauController(
        window_size=args.plateau_window,
        improvement_threshold=args.plateau_threshold,
        max_changes=args.plateau_max_changes
    )
    
    # Load checkpoint if provided (all ranks load to keep optimizer/scaler in sync)
    start_tokens = 0
    start_step = 0
    last_eval_ce = float('inf')
    last_eval_ppl = float('inf')
    if args.load_checkpoint:
        st, ss, lec, lpp = load_baseline_checkpoint(
            args.load_checkpoint, model, optimizer, scaler, plateau_controller
        )
        start_tokens, start_step, last_eval_ce, last_eval_ppl = st, ss, lec, lpp
        if rank == 0:
            print(f"✓ Resumed from checkpoint: {start_tokens:,} tokens, step {start_step}")
            print(f"  Last eval: CE={last_eval_ce:.4f}, PPL={last_eval_ppl:.2f}")
    
    # Training variables
    accumulation_steps = max(1, args.accumulation_steps)
    step = start_step
    tokens_processed = start_tokens
    new_tokens_processed = 0  # Track only new tokens since checkpoint for speed calculation
    accumulated_loss = 0
    accumulated_tokens = 0
    
    start_time = time.time()
    last_log_time = start_time
    
    if rank == 0:
        print(f"🔥 Baseline-Continue Training Started...")
        print(f"   🎛️  Plateau rule: eval CE improvement < {args.plateau_threshold} per {args.plateau_window:,} tokens")
        print(f"   💾 Checkpoints: every {args.checkpoint_every} steps to {args.checkpoint_dir}/")
    
    model.train()
    
    # AMP dtype selection (optimize for RTX 30xx → fp16; Hopper+ → bf16)
    if torch.cuda.is_available():
        cc_major, _ = torch.cuda.get_device_capability()
        amp_dtype = torch.float16 if cc_major == 8 else (torch.bfloat16 if cc_major >= 9 else torch.float16)
    else:
        amp_dtype = torch.float16
    
    # Training loop
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        
        batch_size_actual, seq_len = input_ids.shape
        tokens_in_batch = batch_size_actual * seq_len
        tokens_processed += tokens_in_batch
        new_tokens_processed += tokens_in_batch  # Track new tokens for speed calculation
        
        # Forward pass
        with autocast('cuda', dtype=amp_dtype):
            base_model = model.module if hasattr(model, 'module') else model
            hidden_states = base_model.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
            logits = base_model.lm_head(hidden_states)
            
            loss_sum, num_valid_tokens = compute_loss_with_proper_scaling(
                logits, input_ids, tokenizer, reduction='sum'
            )
            
            loss = loss_sum / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        accumulated_loss += loss_sum.item()
        accumulated_tokens += num_valid_tokens
        
        # Optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            step += 1
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Logging
            if step % 10 == 0 and rank == 0:
                current_time = time.time()
                avg_loss = accumulated_loss / accumulated_tokens if accumulated_tokens > 0 else 0
                tokens_per_sec = new_tokens_processed / (current_time - start_time) if current_time > start_time else 0
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                
                print(f"Step {step:,} | LR: {optimizer.param_groups[0]['lr']:.2e} | Loss: {avg_loss:.3f} | "
                      f"Tokens: {tokens_processed:,} | New: {new_tokens_processed:,} | Tokens/sec: {tokens_per_sec:.0f} | GPU: {gpu_mem:.1f}GB")
                
                if math.isnan(avg_loss):
                    print("⚠️  NaN loss detected - stopping training")
                    break
                
                last_log_time = current_time
            
            # Evaluation and plateau detection
            if step % 500 == 0 and step > 0 and rank == 0:
                print(f"\n📊 Running evaluation at step {step}...")
                eval_model = model.module if hasattr(model, 'module') else model
                eval_ce, eval_ppl = evaluate_on_dataset(
                    eval_model, tokenizer, device, registry, data_config, per_process_batch=max(1, base_batch_size//2)
                )
                
                if eval_ce is not None:
                    print(f"   eval_ce: {eval_ce:.4f} | eval_ppl: {eval_ppl:.2f}")
                    
                    # Plateau detection
                    should_change, improvement = plateau_controller.check_plateau(tokens_processed, eval_ce)
                    print(f"   🔍 Plateau check: CE improvement: {improvement:.4f}")
                    
                    if should_change:
                        print(f"   🎛️  PLATEAU RULE TRIGGERED! Making controlled change #{plateau_controller.changes_made + 1}")
                        plateau_controller.make_controlled_change(optimizer, tokens_processed)
                
                generate_samples(eval_model, tokenizer, device)
                print()
            
            # Save checkpoint
            if step % args.checkpoint_every == 0 and rank == 0:
                eval_model = model.module if hasattr(model, 'module') else model
                eval_ce, eval_ppl = evaluate_on_dataset(
                    eval_model, tokenizer, device, registry, data_config, per_process_batch=max(1, base_batch_size//4)
                )
                
                if eval_ce is None:
                    eval_ce, eval_ppl = 0.0, 0.0
                
                save_baseline_checkpoint(
                    model, optimizer, scaler, plateau_controller, tokens_processed,
                    step, eval_ce, eval_ppl, args.seed, args, args.checkpoint_dir
                )
            
            accumulated_loss = 0
            accumulated_tokens = 0
        
        # Stop when we hit token target
        if tokens_processed >= args.train_tokens:
            break
    
    # Final checkpoint and evaluation
    if rank == 0:
        total_time = time.time() - start_time
        final_tokens_per_sec = tokens_processed / total_time if total_time > 0 else 0
        
        print(f"🎉 Baseline-Continue training complete!")
        print(f"   Steps: {step:,}, Tokens: {tokens_processed:,}")
        print(f"   Time: {total_time:.1f}s, Tokens/sec: {final_tokens_per_sec:.0f}")
        
        # Final evaluation
        print(f"\n📊 Final evaluation:")
        eval_model = model.module if hasattr(model, 'module') else model
        eval_ce, eval_ppl = evaluate_on_dataset(eval_model, tokenizer, device, registry, data_config, per_process_batch=max(1, base_batch_size//2))
        if eval_ce is not None:
            print(f"   eval_ce: {eval_ce:.4f} | eval_ppl: {eval_ppl:.2f}")
        else:
            eval_ce, eval_ppl = 0.0, 0.0
        
        generate_samples(eval_model, tokenizer, device)
        
        # Save final checkpoint
        final_checkpoint_path, final_summary_path = save_baseline_checkpoint(
            model, optimizer, scaler, plateau_controller, tokens_processed,
            step, eval_ce, eval_ppl, args.seed, args, args.checkpoint_dir
        )
        
        # Optional JSON log
        if args.log_json:
            out = {
                "branch": "baseline-continue",
                "steps": step,
                "tokens": tokens_processed,
                "tokens_per_sec": final_tokens_per_sec,
                "eval_ce": eval_ce,
                "eval_ppl": eval_ppl,
                "seed": args.seed,
                "plateau_changes": plateau_controller.changes_made,
                "plateau_adjustments": plateau_controller.adjustments,
                "final_lr": optimizer.param_groups[0]['lr'],
                "checkpoint_path": final_checkpoint_path,
                "summary_path": final_summary_path,
                "config": {
                    "seq_length": args.seq_length,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "accumulation_steps": accumulation_steps,
                    "per_gpu_batch_size": args.per_gpu_batch_size
                }
            }
            try:
                with open(args.log_json, 'w') as f:
                    json.dump(out, f, indent=2)
                print(f"✓ Metrics written to {args.log_json}")
            except Exception as e:
                print(f"⚠️  Failed to write metrics JSON: {e}")
    
    # Attempt to gracefully shutdown dataloader workers to avoid finalizer crashes
    try:
        if 'train_loader' in locals():
            it = getattr(train_loader, '_iterator', None)
            if it is not None:
                it._shutdown_workers()  # best-effort internal cleanup
    except Exception:
        pass

    # Synchronize CUDA work before exit
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass

    # Clean up DDP
    if use_ddp:
        try:
            dist.barrier()
        except Exception:
            pass
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    # Hard-exit to bypass Python finalization issues seen with some C extensions (e.g., pyarrow, HF streaming)
    import os as _os
    _os._exit(0)

if __name__ == "__main__":
    main()