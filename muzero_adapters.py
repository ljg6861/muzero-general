#!/usr/bin/env python3
"""
MuZero-Adapters: Frozen LM + Policy/Value/Dynamics Adapters
==========================================================
Load LM checkpoint, freeze LM weights, add small adapters:
- f(h) → (π, v) policy/value heads
- g(h, a) → h' dynamics (tiny GRU/MLP)
- Router to enable planning only on reasoning-type prompts
- Train adapters on narrow tasks with verifiers (math/short QA)
- Keep small LM CE term (0.1×) to avoid drift
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
import re
sys.path.append('model')
sys.path.append('.')

from model.data_registry import DataRegistry, DataConfig
from model.math_tasks import create_math_dataloader
from model.instruction_data import create_instruction_dataloader

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
    """Custom transformer layer (frozen in adapters mode)"""
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
    """Simple transformer language model (frozen base)"""
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

    def freeze_lm_weights(self):
        """Freeze all LM parameters"""
        for param in self.parameters():
            param.requires_grad = False
        print("🧊 Frozen all LM weights")
    
    def unfreeze_lm_weights(self):
        """Unfreeze all LM parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("🔥 Unfrozen all LM weights")

class ReasoningRouter(nn.Module):
    """Uncertainty-based router to trigger planning on hard tokens (domain-agnostic).
    Combines pooled hidden state with uncertainty features from next-token distribution.
    """
    def __init__(self, hidden_size, vocab_size, use_patterns: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_patterns = use_patterns
        
        # Classifier over [pooled_hidden, entropy, margin]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + 2, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 2)  # [no_planning, planning]
        )
        
        # Optional legacy patterns (disabled by default)
        self.reasoning_patterns = [
            r'\b(?:explain|because|therefore|since|why|how)\b',
            r'\?\s*$'
        ]
        self.reasoning_regex = re.compile('|'.join(self.reasoning_patterns), re.IGNORECASE)
    
    def bootstrap_reasoning_labels(self, input_text):
        if not self.use_patterns:
            # No bootstrap when disabled
            if isinstance(input_text, list):
                return [0 for _ in input_text]
            return 0
        if isinstance(input_text, list):
            return [1 if self.reasoning_regex.search(text) else 0 for text in input_text]
        else:
            return 1 if self.reasoning_regex.search(input_text) else 0
    
    def _uncertainty_features(self, last_logits: torch.Tensor):
        # last_logits: [batch, vocab]
        probs = F.softmax(last_logits, dim=-1)
        entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1, keepdim=True)  # [batch,1]
        top2 = torch.topk(probs, k=2, dim=-1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).unsqueeze(-1)  # [batch,1]
        return entropy, margin
    
    def forward(self, hidden_states, attention_mask=None, policy_logits: torch.Tensor | None = None):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: [batch, seq_len]
            policy_logits: [batch, seq_len, vocab] (optional) to compute entropy/margin
        Returns:
            planning_logits: [batch, 2]
        """
        # Pool over sequence
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(hidden_states).float()
            pooled = (hidden_states * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Uncertainty from last token distribution if provided
        if policy_logits is not None:
            last_logits = policy_logits[:, -1, :]
            entropy, margin = self._uncertainty_features(last_logits)
        else:
            # Default zeros if not available
            zeros = torch.zeros(pooled.size(0), 1, device=pooled.device, dtype=pooled.dtype)
            entropy, margin = zeros, zeros
        
        feats = torch.cat([pooled, entropy, margin], dim=-1)
        planning_logits = self.classifier(feats)
        return planning_logits

class PolicyValueHead(nn.Module):
    """Policy and Value heads: f(h) → (π, v)"""
    def __init__(self, hidden_size, vocab_size, action_space_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # For language models, actions are next tokens, so action_space = vocab_size
        self.action_space_size = action_space_size or vocab_size
        
        # Policy head: outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, self.action_space_size)
        )
        
        # Value head: outputs scalar value
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size] or [batch, hidden_size]
        
        Returns:
            policy_logits: [batch, seq_len, action_space_size] or [batch, action_space_size]
            values: [batch, seq_len, 1] or [batch, 1]
        """
        policy_logits = self.policy_head(hidden_states)
        values = self.value_head(hidden_states)
        
        return policy_logits, values

class DynamicsModel(nn.Module):
    """Dynamics model: g(h, a) → h'"""
    def __init__(self, hidden_size, action_space_size, use_gru=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_space_size = action_space_size
        self.use_gru = use_gru
        
        # Action embedding
        self.action_embedding = nn.Embedding(action_space_size, hidden_size // 4)
        
        if use_gru:
            # GRU-based dynamics
            self.dynamics = nn.GRU(
                input_size=hidden_size + hidden_size // 4,  # state + action
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
        else:
            # MLP-based dynamics
            self.dynamics = nn.Sequential(
                nn.Linear(hidden_size + hidden_size // 4, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size)
            )
    
    def forward(self, hidden_state, action):
        """
        Args:
            hidden_state: [batch, hidden_size] - current state
            action: [batch] - action taken (token IDs)
        
        Returns:
            next_hidden_state: [batch, hidden_size] - predicted next state
        """
        batch_size = hidden_state.shape[0]
        
        # Embed action
        action_emb = self.action_embedding(action)  # [batch, hidden_size // 4]
        
        # Concatenate state and action
        state_action = torch.cat([hidden_state, action_emb], dim=-1)  # [batch, hidden_size + hidden_size // 4]
        
        if self.use_gru:
            # GRU expects [batch, seq_len=1, input_size]
            state_action = state_action.unsqueeze(1)
            next_state, _ = self.dynamics(state_action)
            next_state = next_state.squeeze(1)  # [batch, hidden_size]
        else:
            # MLP
            next_state = self.dynamics(state_action)
        
        return next_state

class MuZeroLM(nn.Module):
    """MuZero-adapted Language Model with frozen LM + trainable adapters"""
    def __init__(self, base_lm, lm_ce_weight=0.1, enable_planning_threshold=0.5):
        super().__init__()
        self.base_lm = base_lm
        self.lm_ce_weight = lm_ce_weight  # Weight for LM CE loss to prevent drift
        self.enable_planning_threshold = enable_planning_threshold
        # Heuristic thresholds used at inference if router is not trained
        self.entropy_plan_threshold = 3.0  # increase to plan only on high uncertainty
        self.margin_plan_threshold = 0.05  # plan when top1-top2 is narrow
        
        hidden_size = base_lm.hidden_size
        vocab_size = base_lm.vocab_size
        
        # Freeze the base LM
        self.base_lm.freeze_lm_weights()
        
        # Add adapters
        self.reasoning_router = ReasoningRouter(hidden_size, vocab_size)
        self.policy_value_head = PolicyValueHead(hidden_size, vocab_size)
        self.dynamics_model = DynamicsModel(hidden_size, vocab_size, use_gru=True)
        
    def forward(self, input_ids, pad_token_id=None, return_planning_logits=True, input_text=None,
                compute_aux: bool = False, plan_top_k: int = 5):
        """
        Args:
            input_ids: [batch, seq_len]
            pad_token_id: padding token ID
            return_planning_logits: whether to compute planning logits
            input_text: list of strings for bootstrap reasoning labels
        
        Returns:
            outputs: dict with 'lm_logits', 'policy_logits', 'values', 'planning_logits'
        """
        batch_size, seq_len = input_ids.shape
        
        # Get hidden states from frozen LM
        with torch.no_grad():
            hidden_states = self.base_lm.encode(input_ids, pad_token_id)  # [batch, seq_len, hidden_size]
        
        # LM logits (for CE loss)
        lm_logits = self.base_lm.lm_head(hidden_states)  # [batch, seq_len, vocab_size]
        
        # Policy and value outputs
        policy_logits, values = self.policy_value_head(hidden_states)  # [batch, seq_len, vocab_size], [batch, seq_len, 1]
        
        outputs = {
            'lm_logits': lm_logits,
            'policy_logits': policy_logits,
            'values': values,
            'hidden_states': hidden_states
        }
        
        # Planning router
        if return_planning_logits:
            attention_mask = None
            if pad_token_id is not None:
                attention_mask = (input_ids != pad_token_id).float()
            
            planning_logits = self.reasoning_router(hidden_states, attention_mask, policy_logits=policy_logits)  # [batch, 2]
            outputs['planning_logits'] = planning_logits
            # also expose planning probability for logging
            outputs['plan_prob'] = F.softmax(planning_logits, dim=-1)[:, 1]
            
            # Bootstrap reasoning labels if input_text provided
            if input_text is not None:
                bootstrap_labels = self.reasoning_router.bootstrap_reasoning_labels(input_text)
                outputs['bootstrap_reasoning_labels'] = torch.tensor(bootstrap_labels, device=input_ids.device)
        
        # Auxiliary computations inside forward so DDP tracks parameter usage
        if compute_aux:
            # Dynamics consistency predictions across sequence
            B, T, H = hidden_states.shape
            if T > 1:
                next_h_target = hidden_states[:, 1:, :].contiguous()
                preds = []
                for t in range(T - 1):
                    a_t = input_ids[:, t + 1]
                    preds.append(self.dynamics_model(hidden_states[:, t, :], a_t))
                dyn_pred_next = torch.stack(preds, dim=1)  # [B, T-1, H]
                outputs['next_h_target'] = next_h_target
                outputs['dyn_pred_next'] = dyn_pred_next
            else:
                outputs['next_h_target'] = hidden_states[:, :0, :]
                outputs['dyn_pred_next'] = hidden_states[:, :0, :]

            # One-step policy improvement at last step
            last_h = hidden_states[:, -1, :]
            last_logits = policy_logits[:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            k = min(plan_top_k, probs.size(-1))
            topk_probs, topk_ids = torch.topk(probs, k=k, dim=-1)
            values_est = []
            for j in range(k):
                a = topk_ids[:, j]
                s1 = self.dynamics_model(last_h, a)
                _, v1 = self.policy_value_head(s1)
                values_est.append(v1.squeeze(-1))
            values_est = torch.stack(values_est, dim=-1)  # [B, k]
            improved = F.softmax(values_est, dim=-1)
            sel_probs = probs.gather(1, topk_ids)
            sel_probs = sel_probs / sel_probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            outputs['pi_topk_ids'] = topk_ids
            outputs['pi_sel_probs'] = sel_probs
            outputs['pi_improved_probs'] = improved

            # Two-step dynamics predictions for added consistency
            if hidden_states.size(1) > 2:
                two_step_preds = []
                for t in range(hidden_states.size(1) - 2):
                    a1 = input_ids[:, t + 1]
                    a2 = input_ids[:, t + 2]
                    s1 = self.dynamics_model(hidden_states[:, t, :], a1)
                    s2 = self.dynamics_model(s1, a2)
                    two_step_preds.append(s2)
                outputs['dyn_two_step_pred'] = torch.stack(two_step_preds, dim=1)
                outputs['dyn_two_step_target'] = hidden_states[:, 2:, :].contiguous()
            else:
                outputs['dyn_two_step_pred'] = hidden_states[:, :0, :]
                outputs['dyn_two_step_target'] = hidden_states[:, :0, :]

        return outputs
    
    def planning_forward(self, hidden_state, action_sequence, max_depth=3):
        """
        Perform planning using the dynamics model
        
        Args:
            hidden_state: [batch, hidden_size] - current state
            action_sequence: [batch, max_depth] - sequence of actions to simulate
            max_depth: maximum planning depth
        
        Returns:
            predicted_values: [batch, max_depth] - predicted values at each step
            predicted_states: [batch, max_depth, hidden_size] - predicted states
        """
        batch_size = hidden_state.shape[0]
        current_state = hidden_state
        
        predicted_values = []
        predicted_states = []
        
        for step in range(max_depth):
            if step < action_sequence.shape[1]:
                action = action_sequence[:, step]  # [batch]
                
                # Predict next state
                next_state = self.dynamics_model(current_state, action)  # [batch, hidden_size]
                
                # Predict value at next state
                _, values = self.policy_value_head(next_state)  # [batch, 1]
                
                predicted_values.append(values.squeeze(-1))  # [batch]
                predicted_states.append(next_state)
                
                current_state = next_state
            else:
                break
        
        if predicted_values:
            predicted_values = torch.stack(predicted_values, dim=1)  # [batch, steps]
            predicted_states = torch.stack(predicted_states, dim=1)  # [batch, steps, hidden_size]
        else:
            predicted_values = torch.zeros(batch_size, 0, device=hidden_state.device)
            predicted_states = torch.zeros(batch_size, 0, hidden_state.shape[-1], device=hidden_state.device)
        
        return predicted_values, predicted_states
    
    def count_adapter_parameters(self):
        """Count trainable adapter parameters"""
        adapter_params = 0
        total_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                adapter_params += param.numel()
        
        return adapter_params, total_params

    @torch.no_grad()
    def plan_rerank_next_token(
        self,
        input_ids: torch.Tensor,
        tokenizer,
        top_k: int = 5,
        depth: int = 1,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ):
        """One-step planning: re-rank top-k next tokens using dynamics+value.
        Args:
            input_ids: [batch, seq_len]
        Returns:
            next_token_ids [batch, 1], scores [batch, top_k]
        """
        device = input_ids.device
        # Encode and get last hidden state
        hidden_states = self.base_lm.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
        last_h = hidden_states[:, -1, :]
        # Use base LM for candidate list (more fluent)
        lm_logits = self.base_lm.lm_head(hidden_states)
        logits = lm_logits[:, -1, :]
        if temperature and temperature != 1.0:
            logits = logits / max(1e-6, temperature)
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
        # Score each candidate via dynamics -> value, with simple repetition-aware penalties
        batch = input_ids.size(0)
        scores = torch.empty(batch, topk_ids.size(1), device=device)
        # Precompute presence of tokens in history for simple penalty
        hist = input_ids
        # Build set of existing n-grams if needed
        def violates_ngram(seq_row: torch.Tensor, next_tok: int, n: int) -> bool:
            if n <= 0:
                return False
            if seq_row.size(0) < n - 1:
                return False
            last_n_1 = seq_row[-(n - 1):].tolist() if n > 1 else []
            # collect existing ngrams
            existing = set()
            lst = seq_row.tolist()
            for i in range(len(lst) - n + 1):
                existing.add(tuple(lst[i:i+n]))
            cand = tuple((last_n_1 + [next_tok])[-n:]) if n > 0 else (next_tok,)
            return cand in existing
        for j in range(topk_ids.size(1)):
            a = topk_ids[:, j]
            state = last_h
            for _ in range(max(1, depth)):
                state = self.dynamics_model(state, a)
            # value of state
            _, v = self.policy_value_head(state)
            raw = v.squeeze(-1)
            # Apply penalties
            if repetition_penalty and repetition_penalty > 1.0:
                # presence penalty: if candidate token already in sequence, subtract a small offset
                # map repetition_penalty -> offset ~= log(repetition_penalty)
                offset = float(torch.log(torch.tensor(repetition_penalty)))
                present = torch.zeros(batch, device=device)
                for b in range(batch):
                    if int(a[b].item()) in set(hist[b].tolist()):
                        present[b] = 1.0
                raw = raw - offset * present
            if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                penalty = torch.zeros(batch, device=device)
                for b in range(batch):
                    if violates_ngram(hist[b], int(a[b].item()), no_repeat_ngram_size):
                        penalty[b] = 1.0
                # large penalty to effectively avoid
                raw = raw - 1000.0 * penalty
            scores[:, j] = raw
        # pick best
        best_idx = torch.argmax(scores, dim=-1, keepdim=True)
        best_tokens = topk_ids.gather(1, best_idx)
        return best_tokens, scores

    @torch.no_grad()
    def generate_with_planning(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 32,
        plan_top_k: int = 5,
        plan_depth: int = 1,
        temperature: float = 0.8,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        disable_planning: bool = False,
        sample_top_k: int = 0,
    ):
        """Generate text; use router to decide planning vs LM sampling at each step.
        Args:
            top_p: nucleus sampling cutoff (0..1); if set, apply during LM fallback sampling.
            repetition_penalty: >1.0 to discourage repeats; applied to logits prior to sampling.
            no_repeat_ngram_size: if >0, prevent generating any n-gram that already occurred.
        """
        self.eval()
        tokens = tokenizer(prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(next(self.parameters()).device)

        def apply_repetition_penalty_to_logits(logits: torch.Tensor, seq: torch.Tensor, penalty: float):
            if penalty is None or penalty == 1.0:
                return logits
            # Simple scheme: penalize all previously used tokens
            unique_tokens = torch.unique(seq)
            logits[..., unique_tokens] = logits[..., unique_tokens] / penalty
            return logits

        def filter_no_repeat_ngram(probs: torch.Tensor, seq: torch.Tensor, next_token_candidates: torch.Tensor, n: int):
            if n <= 0 or seq.size(1) < n - 1:
                return probs
            # Build set of existing n-grams
            ngrams = set()
            seq_list = seq[0].tolist()
            for i in range(len(seq_list) - n + 1):
                ngrams.add(tuple(seq_list[i:i+n]))
            # For each candidate, if it forms a repeated n-gram, zero its prob
            if next_token_candidates is None:
                return probs
            last_n_1 = seq[0, - (n - 1):].tolist() if n > 1 else []
            for idx in range(next_token_candidates.size(1)):
                t = int(next_token_candidates[0, idx].item())
                cand_ngram = tuple((last_n_1 + [t])[-n:]) if n > 0 else (t,)
                if cand_ngram in ngrams:
                    probs[0, t] = 0.0
            # renormalize
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            return probs
        for _ in range(max_new_tokens):
            # Decide whether to plan
            with torch.no_grad():
                hs = self.base_lm.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
                attn_mask = (input_ids != tokenizer.pad_token_id).float() if tokenizer.pad_token_id is not None else None
                pol_logits, _ = self.policy_value_head(hs)
                # Compute heuristic uncertainty metrics
                last_logits = pol_logits[:, -1, :]
                # Apply repetition penalty (policy path for uncertainty calc)
                last_logits = apply_repetition_penalty_to_logits(last_logits.clone(), input_ids, repetition_penalty)
                probs = F.softmax(last_logits, dim=-1)
                entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
                top2 = torch.topk(probs, k=2, dim=-1).values
                margin = (top2[:, 0] - top2[:, 1])
                # Try router if trained; fallback to heuristic
                do_plan = False
                if not disable_planning:
                    try:
                        plan_logits = self.reasoning_router(hs, attn_mask, policy_logits=pol_logits)
                        plan_prob = F.softmax(plan_logits, dim=-1)[:, 1]
                        do_plan = bool(plan_prob.item() >= self.enable_planning_threshold)
                    except Exception:
                        do_plan = False
                    # Heuristic OR: if highly uncertain, force planning
                    if float(entropy.item()) >= self.entropy_plan_threshold and float(margin.item()) <= self.margin_plan_threshold:
                        do_plan = True
            if do_plan and not disable_planning:
                next_token, _ = self.plan_rerank_next_token(
                    input_ids, tokenizer,
                    top_k=plan_top_k, depth=plan_depth, temperature=temperature,
                    repetition_penalty=repetition_penalty,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                )
            else:
                # fall back to LM sampling for fluency
                lm_logits = self.base_lm.lm_head(hs)
                logits = lm_logits[:, -1, :]
                logits = apply_repetition_penalty_to_logits(logits, input_ids, repetition_penalty)
                logits = logits / max(1e-6, temperature)
                probs = F.softmax(logits, dim=-1)
                # optional top-k
                if sample_top_k and sample_top_k > 0:
                    topk = min(sample_top_k, probs.size(-1))
                    topk_probs, topk_idx = torch.topk(probs, k=topk, dim=-1)
                    mask = torch.ones_like(probs)
                    mask.scatter_(1, topk_idx, 0.0)
                    probs = probs.masked_fill(mask.bool(), 0.0)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                if top_p is not None and 0.0 < top_p < 1.0:
                    # nucleus sampling
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > top_p
                    # keep first token that exceeds threshold as well
                    mask[..., 1:] = mask[..., :-1]
                    mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                    # map back
                    probs.zero_().scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                if no_repeat_ngram_size and no_repeat_ngram_size > 0:
                    # zero out any candidate that causes ngram repeat before sampling
                    topk_ids = torch.topk(probs, k=min(256, probs.size(-1)), dim=-1).indices  # limit to 256 for speed
                    probs = filter_no_repeat_ngram(probs, input_ids, topk_ids, no_repeat_ngram_size)
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if tokenizer.eos_token_id is not None and int(next_token.item()) == tokenizer.eos_token_id:
                break
        return tokenizer.decode(input_ids[0], skip_special_tokens=True)

def compute_muzero_loss(outputs, input_ids, tokenizer, targets=None, planning_targets=None, bootstrap_reasoning=True):
    """
    Compute combined MuZero loss:
    - Small LM CE loss (0.1×) to prevent drift
    - Policy loss (imitation of LM policy)
    - Value loss (predict future rewards)
    - Planning router loss (reasoning detection)
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape
    
    # 1. LM CE Loss (small weight to prevent drift)
    lm_logits = outputs['lm_logits']
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    
    lm_loss_flat = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction='none'
    )
    
    # Mask padding
    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        mask = (shift_labels.view(-1) != tokenizer.pad_token_id)
        lm_loss_flat = lm_loss_flat * mask.float()
        valid_tokens = mask.sum().item()
    else:
        valid_tokens = lm_loss_flat.numel()
    
    lm_ce_loss = lm_loss_flat.sum() / max(1, valid_tokens)
    
    # 2. Policy Loss (imitate LM policy) with uncertainty weighting
    policy_logits = outputs['policy_logits']
    shift_policy_logits = policy_logits[:, :-1, :].contiguous()
    
    # Use LM logits as soft targets for policy (knowledge distillation), memory-stable
    with torch.no_grad():
        log_teacher = F.log_softmax(shift_logits, dim=-1)  # teacher in log-space to avoid extra allocations
        # Teacher entropy per token to focus on hard tokens
        teacher_probs = log_teacher.exp()
        entropy_per_tok = -(teacher_probs * log_teacher).sum(dim=-1)  # [batch, T-1]
        # Normalize to mean≈1 for stability
        ent_mean = entropy_per_tok.mean()
        ent_std = entropy_per_tok.std(unbiased=False).clamp_min(1e-6)
        ent_weights = (entropy_per_tok - ent_mean) / ent_std
        ent_weights = (ent_weights * 0.25 + 1.0).clamp(0.25, 2.0)  # shrink range
    
    policy_loss_flat = F.kl_div(
        F.log_softmax(shift_policy_logits.view(-1, shift_policy_logits.size(-1)), dim=-1),
        log_teacher.view(-1, log_teacher.size(-1)),
        reduction='none',
        log_target=True
    ).sum(dim=-1)
    # Reshape to [batch, T-1] and apply entropy weights
    policy_loss_tok = policy_loss_flat.view(shift_policy_logits.size(0), -1)
    policy_loss_tok = policy_loss_tok * ent_weights

    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
        policy_loss_tok = policy_loss_tok * mask.view(shift_policy_logits.size(0), -1).float()
    
    policy_loss = policy_loss_tok.sum() / max(1, valid_tokens)
    
    # 3. Value Loss (self-supervised: predict future ease of continuation)
    values = outputs['values']
    # Use last W tokens CE to construct target, z-normalized across batch
    with torch.no_grad():
        per_tok_ce = lm_loss_flat.view(batch_size, seq_len - 1)
        W = min(8, per_tok_ce.size(1))
        tail_ce = per_tok_ce[:, -W:]
        cont_cost = tail_ce.mean(dim=-1, keepdim=True)  # [batch,1]
        mu, sigma = cont_cost.mean(), cont_cost.std(unbiased=False).clamp_min(1e-6)
        z = (cont_cost - mu) / sigma
        value_targets = (-z).clamp(-2.0, 2.0)  # higher is better
    
    # Value loss on last position
    value_preds = values[:, -1, :]  # [batch, 1]
    value_loss = F.mse_loss(value_preds, value_targets)
    
    # 4. Planning Router Loss (reasoning detection)
    planning_loss = torch.tensor(0.0, device=device)
    if 'planning_logits' in outputs:
        planning_logits = outputs['planning_logits']  # [batch, 2]
        if 'bootstrap_reasoning_labels' in outputs:
            reasoning_labels = outputs['bootstrap_reasoning_labels']  # [batch]
            planning_loss = F.cross_entropy(planning_logits, reasoning_labels)
        else:
            # Unsupervised pseudo-labels using uncertainty heuristics
            last_logits = outputs['policy_logits'][:, -1, :]
            probs = F.softmax(last_logits, dim=-1)
            entropy = -(probs * (probs.clamp_min(1e-9)).log()).sum(dim=-1)  # [batch]
            top2 = torch.topk(probs, k=2, dim=-1).values
            margin = (top2[:, 0] - top2[:, 1])  # [batch]
            # Heuristics: plan when high entropy and small margin
            entropy_thresh = 3.0
            margin_thresh = 0.05
            pseudo = ((entropy >= entropy_thresh) & (margin <= margin_thresh)).long()
            if pseudo.numel() > 0:
                planning_loss = F.cross_entropy(planning_logits, pseudo)
    
    # Combine losses
    total_loss = (
        0.1 * lm_ce_loss +      # Small weight to prevent drift
        1.0 * policy_loss +     # Main policy learning
        0.5 * value_loss +      # Value prediction (self-supervised)
        0.2 * planning_loss     # Reasoning detection
    )
    
    loss_dict = {
        'total_loss': total_loss,
        'lm_ce_loss': lm_ce_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'planning_loss': planning_loss,
        'valid_tokens': valid_tokens
    }
    # Log mean plan probability if available
    if 'plan_prob' in outputs:
        try:
            loss_dict['plan_prob_mean'] = outputs['plan_prob'].mean()
        except Exception:
            pass
    
    return total_loss, loss_dict

def load_lm_checkpoint_for_adapters(checkpoint_path, lm_model):
    """Load LM checkpoint for adapter training"""
    print(f"🔄 Loading LM checkpoint for adapters: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load only model state
    model_to_load = lm_model.module if hasattr(lm_model, 'module') else lm_model

    # Prepare state_dict and align DataParallel prefixes
    state_dict = checkpoint['model_state_dict']
    model_keys = list(model_to_load.state_dict().keys())
    ckpt_keys = list(state_dict.keys())
    if model_keys and ckpt_keys:
        if model_keys[0].startswith('module.') and not ckpt_keys[0].startswith('module.'):
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        elif not model_keys[0].startswith('module.') and ckpt_keys[0].startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Handle positional embedding length mismatch
    try:
        current_pos_key = 'module.pos_embedding.weight' if any(k.startswith('module.') for k in model_to_load.state_dict().keys()) else 'pos_embedding.weight'
        target_shape = model_to_load.state_dict()[current_pos_key].shape
        src_key = current_pos_key
        if src_key in state_dict:
            ckpt_pe = state_dict[src_key]
            if ckpt_pe.shape != target_shape:
                model_pe = model_to_load.state_dict()[current_pos_key]
                new_pe = model_pe.clone()
                min_len = min(ckpt_pe.shape[0], target_shape[0])
                with torch.no_grad():
                    new_pe[:min_len].copy_(ckpt_pe[:min_len].to(new_pe.device))
                state_dict[src_key] = new_pe.to(ckpt_pe.device)
                print(f"✓ Adjusted LM pos_embedding from {tuple(ckpt_pe.shape)} to {tuple(target_shape)} (copied first {min_len} rows)")
    except Exception as e:
        pos_key = 'module.pos_embedding.weight' if any(k.startswith('module.') for k in model_to_load.state_dict().keys()) else 'pos_embedding.weight'
        if pos_key in state_dict:
            del state_dict[pos_key]
            print(f"⚠️  Dropped '{pos_key}' from LM checkpoint due to mismatch: {e}")

    # Load with strict=False to tolerate benign differences
    model_to_load.load_state_dict(state_dict, strict=False)
    
    print(f"✓ Loaded LM weights from checkpoint (tokens: {checkpoint.get('tokens_processed', 0):,})")
    return checkpoint.get('tokens_processed', 0), checkpoint.get('eval_ce', 0.0)

def save_muzero_checkpoint(model, optimizer, scaler, tokens_processed, step, eval_metrics,
                          seed, args, save_dir="muzero_adapters_checkpoints"):
    """Save MuZero adapters checkpoint"""
    os.makedirs(save_dir, exist_ok=True)

    # Determine save strategy: explicit name, latest-only, or timestamped
    gen_prefix = getattr(args, 'gen_prefix', 'benchmark')
    gen_id = getattr(args, 'gen_id', 'gen1')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tokens_m = tokens_processed // 1_000_000
    if getattr(args, 'save_name', None):
        base_name = str(args.save_name).rstrip('.pt')
        checkpoint_name = base_name
        checkpoint_path = os.path.join(save_dir, f"{base_name}.pt")
        summary_path = os.path.join(save_dir, f"{base_name}_summary.json")
    elif getattr(args, 'save_latest_only', True):
        base_name = f"{gen_prefix}_{gen_id}_latest"
        checkpoint_name = base_name
        checkpoint_path = os.path.join(save_dir, f"{base_name}.pt")
        summary_path = os.path.join(save_dir, f"{base_name}_summary.json")
    else:
        checkpoint_name = f"muzero_adapters_{tokens_m}M_tokens_{timestamp}"
        checkpoint_path = os.path.join(save_dir, f"{checkpoint_name}.pt")
        summary_path = os.path.join(save_dir, f"{checkpoint_name}_summary.json")
    
    model_to_save = model.module if hasattr(model, 'module') else model
    
    # Count adapter parameters
    adapter_params, total_params = model_to_save.count_adapter_parameters()
    
    checkpoint = {
        'timestamp': timestamp,
        'tokens_processed': tokens_processed,
        'step': step,
        'seed': seed,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'eval_metrics': eval_metrics,
        'adapter_params': adapter_params,
        'total_params': total_params,
        'training_args': vars(args),
        'random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_random_state': torch.get_rng_state(),
        'cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Save summary
    summary = {
        'checkpoint_name': checkpoint_name,
        'branch': 'muzero-adapters',
        'timestamp': timestamp,
        'tokens_processed': tokens_processed,
        'step': step,
        'eval_metrics': eval_metrics,
        'adapter_params': adapter_params,
        'total_params': total_params,
        'adapter_percentage': f"{100 * adapter_params / total_params:.2f}%",
        'current_lr': optimizer.param_groups[0]['lr'],
        'training_args': vars(args)
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 MuZero-adapters checkpoint saved:")
    print(f"   Path: {checkpoint_path}")
    print(f"   Tokens: {tokens_processed:,} ({tokens_m}M)")
    print(f"   Adapter params: {adapter_params:,} ({100 * adapter_params / total_params:.2f}%)")
    if getattr(args, 'save_name', None):
        print(f"   Name: {args.save_name}.pt (explicit)")
    elif getattr(args, 'save_latest_only', True):
        print(f"   Name: {gen_prefix}_{gen_id}_latest.pt (latest-only)")
    else:
        print(f"   Name: {checkpoint_name}.pt (timestamped)")
    
    return checkpoint_path, summary_path

def main():
    parser = argparse.ArgumentParser(description="MuZero-Adapters: Frozen LM + Policy/Value/Dynamics Adapters")
    
    # Training configuration
    parser.add_argument("--train_tokens", type=int, default=20_000_000, help="Total adapter training tokens")
    parser.add_argument("--seq_length", type=int, default=256, help="Sequence length")
    parser.add_argument("--eval_tokens", type=int, default=500_000, help="Evaluation token budget")
    
    # Model configuration (must match LM checkpoint)
    parser.add_argument("--hidden_size", type=int, default=256, help="Model hidden size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    
    # Performance knobs
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Per-GPU batch size (smaller for adapters)")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers per process")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor")
    parser.add_argument("--grad_ckpt", action="store_true", help="Enable gradient checkpointing")
    
    # Adapter-specific optimization
    parser.add_argument("--adapter_lr", type=float, default=1e-3, help="Adapter learning rate (higher than LM)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--fused_adamw", action="store_true", help="Use fused AdamW if available")
    
    # MuZero-specific
    parser.add_argument("--lm_ce_weight", type=float, default=0.1, help="Weight for LM CE loss to prevent drift")
    parser.add_argument("--enable_planning_threshold", type=float, default=0.5, help="Threshold for enabling planning")
    parser.add_argument("--dynamics_gru", action="store_true", help="Use GRU for dynamics model (default: MLP)")
    # Checkpoint naming
    parser.add_argument("--save_name", type=str, default=None, help="Explicit checkpoint base name (no auto timestamp)")
    parser.add_argument("--save_latest_only", action="store_true", help="Save/overwrite only {gen_prefix}_{gen_id}_latest.pt (default)")
    parser.set_defaults(save_latest_only=True)
    # Data and resume options
    parser.add_argument("--data_sources", type=str, default="wikipedia",
                        help="Comma-separated list of data sources (e.g., 'wikipedia,openwebtext,c4,bookcorpus')")
    parser.add_argument("--load_muzero_checkpoint", type=str, default=None,
                        help="Path to a previous MuZero-adapters checkpoint to resume weights from (.pt)")
    # DDP options
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", 
                        help="Set find_unused_parameters=True for DDP (default False). Enable only if some adapter params are truly unused in some iterations.")

    # Math tasks mixing
    parser.add_argument("--use_math_tasks", action="store_true", help="Mix in synthetic math QA tasks for planning/value supervision")
    parser.add_argument("--math_mix_prob", type=float, default=0.5, help="Probability of using a math batch vs text batch")
    parser.add_argument("--math_corrupt_prob", type=float, default=0.15, help="Probability to corrupt answers for negative value supervision")
    # Instruction tuning
    parser.add_argument("--use_instruction_tuning", action="store_true", help="Mix in instruction/QA batches with answer-token supervision")
    parser.add_argument("--instr_mix_prob", type=float, default=0.5, help="Probability of using instruction batch vs text batch when enabled")
    
    # Checkpointing
    parser.add_argument("--checkpoint_every", type=int, default=1000, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="muzero_adapters_checkpoints", help="Checkpoint directory")
    parser.add_argument("--load_lm_checkpoint", type=str, required=True, help="LM checkpoint to load and freeze")
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
        if rank == 0:
            print(f"✓ Enabled TF32 and cuDNN optimizations")
    
    # Print configuration
    if rank == 0:
        print("🧊 MuZero-Adapters: Frozen LM + Trainable Adapters")
        print(f"   Device: {device}")
        print(f"   Seed: {args.seed}")
        print(f"   Target tokens: {args.train_tokens:,}")
        print(f"   LM checkpoint: {args.load_lm_checkpoint}")
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"   Available GPUs: {num_gpus}")
            if use_ddp:
                print(f"   🚀 Using DDP with {world_size} processes!")
    
    # Setup data
    tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Parse data sources
    ds_list = [s.strip() for s in str(args.data_sources).split(',') if s.strip()]
    data_config = DataConfig(
        train_tokens=args.train_tokens,
        seq_length=args.seq_length,
        data_sources=ds_list,
        allow_fallback_synthetic=False
    )
    data_config.eval_tokens = args.eval_tokens
    
    registry = DataRegistry()
    
    # DataLoader setup
    base_batch_size = args.per_gpu_batch_size
    if not use_ddp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
    
    num_workers_dl = 2 if args.num_workers is None else max(0, args.num_workers)
    
    train_loader = registry.create_dataloader(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=base_batch_size,
        use_sized=False,
        num_workers=num_workers_dl,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=(num_workers_dl > 0)
    )

    # Optional math tasks dataloader
    math_loader = None
    if args.use_math_tasks:
        math_loader = create_math_dataloader(
            tokenizer=tokenizer,
            seq_length=args.seq_length,
            batch_size=base_batch_size,
            num_workers=max(0, num_workers_dl // 2),
            corrupt_prob=args.math_corrupt_prob
        )
        if rank == 0:
            print(f"✓ Math tasks enabled (mix_prob={args.math_mix_prob:.2f}, corrupt_prob={args.math_corrupt_prob:.2f})")
    # Optional instruction tuning dataloader
    instr_loader = None
    if args.use_instruction_tuning:
        instr_loader = create_instruction_dataloader(
            tokenizer=tokenizer,
            registry=registry,
            config=data_config,
            batch_size=base_batch_size,
            num_workers=max(0, num_workers_dl // 2)
        )
        if rank == 0:
            print(f"✓ Instruction tuning enabled (mix_prob={args.instr_mix_prob:.2f})")
    
    # Create base LM
    base_lm = SimpleLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        max_seq_len=args.seq_length,
        gradient_checkpointing=args.grad_ckpt
    )
    
    # Load LM checkpoint (all ranks for consistency)
    if args.load_lm_checkpoint:
        lm_tokens_trained, lm_eval_ce = load_lm_checkpoint_for_adapters(args.load_lm_checkpoint, base_lm)
        if rank == 0:
            print(f"✓ Loaded LM: {lm_tokens_trained:,} tokens trained, eval_ce: {lm_eval_ce:.4f}")
    
    # Create MuZero model with adapters
    model = MuZeroLM(
        base_lm=base_lm,
        lm_ce_weight=args.lm_ce_weight,
        enable_planning_threshold=args.enable_planning_threshold
    )
    
    model = model.to(device)
    
    # Optionally resume full MuZero-adapters weights from a prior checkpoint (before DDP wrap)
    if args.load_muzero_checkpoint:
        try:
            if rank == 0:
                print(f"🔄 Resuming MuZero-adapters from: {args.load_muzero_checkpoint}")
            state = torch.load(args.load_muzero_checkpoint, map_location='cpu', weights_only=False)
            sd = state.get('model_state_dict', state)
            model_sd = model.state_dict()
            needs_module = any(k.startswith('module.') for k in model_sd.keys())
            has_module = any(k.startswith('module.') for k in sd.keys())
            if needs_module and not has_module:
                sd = {f'module.{k}': v for k, v in sd.items()}
            elif has_module and not needs_module:
                sd = {k.replace('module.', ''): v for k, v in sd.items()}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if rank == 0:
                if missing:
                    print(f"[warn] Missing keys: {len(missing)} (showing up to 8): {missing[:8]}")
                if unexpected:
                    print(f"[warn] Unexpected keys: {len(unexpected)} (showing up to 8): {unexpected[:8]}")
                print("✓ MuZero-adapters weights loaded")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Failed to load MuZero-adapters checkpoint: {e}")
    
    # Setup DDP
    if use_ddp:
        # Use find_unused_parameters=False by default for performance; enable via flag if needed
        model = DDP(model, device_ids=[rank], find_unused_parameters=args.ddp_find_unused_parameters)
        if rank == 0:
            print(f"🚀 Wrapped model with DDP")
    
    if rank == 0:
        # Count parameters
        model_to_check = model.module if hasattr(model, 'module') else model
        adapter_params, total_params = model_to_check.count_adapter_parameters()
        print(f"✓ Model: {total_params:,} total parameters")
        print(f"  🧊 Frozen LM: {total_params - adapter_params:,} parameters")
        print(f"  🔥 Trainable adapters: {adapter_params:,} parameters ({100 * adapter_params / total_params:.2f}%)")
    
    # Setup optimizer (only adapter parameters)
    adapter_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            adapter_params.append(param)
    
    optim_kwargs = dict(lr=args.adapter_lr, weight_decay=args.weight_decay, betas=(0.9, 0.95), eps=1e-8)
    if args.fused_adamw:
        optimizer = torch.optim.AdamW(adapter_params, fused=True, **optim_kwargs)
        if rank == 0:
            print("✓ Using fused AdamW optimizer for adapters")
    else:
        optimizer = torch.optim.AdamW(adapter_params, **optim_kwargs)
    
    scaler = GradScaler('cuda')
    
    # Training variables
    accumulation_steps = max(1, args.accumulation_steps)
    step = 0
    tokens_processed = 0
    accumulated_losses = {}
    accumulated_tokens = 0
    accumulated_batches = 0
    
    start_time = time.time()
    last_log_time = start_time
    
    if rank == 0:
        print(f"🔥 MuZero-Adapters Training Started...")
        print(f"   🧊 LM weights: FROZEN")
        print(f"   🔥 Adapter weights: TRAINABLE")
        print(f"   💾 Checkpoints: every {args.checkpoint_every} steps to {args.checkpoint_dir}/")
    
    model.train()
    
    # Training loop
    # Iterators for streaming loaders
    text_iter = iter(train_loader)
    math_iter = iter(math_loader) if math_loader is not None else None
    instr_iter = iter(instr_loader) if instr_loader is not None else None
    batch_idx = -1
    while True:
        # Choose source
        use_math = math_iter is not None and (random.random() < args.math_mix_prob)
        use_instr = (not use_math) and (instr_iter is not None) and (random.random() < args.instr_mix_prob)
        try:
            if use_math:
                batch = next(math_iter)
            elif use_instr:
                batch = next(instr_iter)
            else:
                batch = next(text_iter)
        except StopIteration:
            # Re-create iterators on exhaustion
            if use_math and math_iter is not None:
                math_iter = iter(math_loader)
                batch = next(math_iter)
            elif use_instr and instr_iter is not None:
                instr_iter = iter(instr_loader)
                batch = next(instr_iter)
            else:
                text_iter = iter(train_loader)
                batch = next(text_iter)
        batch_idx += 1

        input_ids = batch['input_ids'].to(device, non_blocking=True)
        
        batch_size_actual, seq_len = input_ids.shape
        tokens_in_batch = batch_size_actual * seq_len
        tokens_processed += tokens_in_batch
        
        # Provide raw text for router: for math use provided, else skip
        input_text = None
        if args.use_math_tasks and 'raw_text' in batch:
            # batch['raw_text'] is a list[str] of length batch
            input_text = list(batch['raw_text']) if isinstance(batch['raw_text'], list) else batch['raw_text']
        
        # Forward pass
        with autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16):
            outputs = model(input_ids, pad_token_id=tokenizer.pad_token_id, input_text=input_text, compute_aux=True, plan_top_k=5)
            
            total_loss, loss_dict = compute_muzero_loss(
                outputs, input_ids, tokenizer, bootstrap_reasoning=True
            )

            # If instruction tuning: compute focused CE on answer tokens only
            if args.use_instruction_tuning and 'labels' in batch:
                labels = batch['labels'].to(device)
                logits = outputs['policy_logits']  # [B, T, V]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_flat = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                    reduction='mean'
                )
                total_loss = total_loss + 0.8 * loss_flat
                loss_dict['instr_loss'] = loss_flat

            # Dynamics consistency: g(h_t, a_t) ≈ h_{t+1} (computed inside forward)
            dyn_pred_next = outputs['dyn_pred_next']
            next_h_target = outputs['next_h_target']
            dyn_loss = F.mse_loss(dyn_pred_next, next_h_target)
            total_loss = total_loss + 0.1 * dyn_loss
            loss_dict['dyn_consistency'] = dyn_loss

            # Two-step consistency (small weight)
            dyn2_pred = outputs['dyn_two_step_pred']
            dyn2_tgt = outputs['dyn_two_step_target']
            if dyn2_pred.numel() > 0:
                dyn2_loss = F.mse_loss(dyn2_pred, dyn2_tgt)
                total_loss = total_loss + 0.05 * dyn2_loss
                loss_dict['dyn2_consistency'] = dyn2_loss

            # One-step policy improvement via value (computed inside forward)
            sel_probs = outputs['pi_sel_probs']
            improved = outputs['pi_improved_probs']
            imp_loss = F.kl_div((sel_probs + 1e-9).log(), (improved + 1e-9).log(), reduction='batchmean', log_target=True)
            total_loss = total_loss + 0.05 * imp_loss
            loss_dict['policy_improve'] = imp_loss

            # Add value supervision on math batches when correctness is available
            if args.use_math_tasks and 'correctness' in batch:
                # We supervise the last position value toward +1 for correct, -1 for incorrect
                corr = batch['correctness'].to(device).view(-1, 1)
                # map {0,1} -> {-1,+1}
                target_val = 2.0 * corr - 1.0
                value_preds = outputs['values'][:, -1, :]
                value_loss = F.mse_loss(value_preds, target_val)
                # Mix into total loss with moderate weight
                total_loss = total_loss + 0.5 * value_loss
                loss_dict['value_supervision_loss'] = value_loss
            
            loss = total_loss / accumulation_steps
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Accumulate losses
        for key, value in loss_dict.items():
            if key not in accumulated_losses:
                accumulated_losses[key] = 0
            accumulated_losses[key] += value.item() if hasattr(value, 'item') else value
        accumulated_tokens += loss_dict['valid_tokens']
        accumulated_batches += 1
        
        # Optimizer step
        if (batch_idx + 1) % accumulation_steps == 0:
            step += 1
            
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Logging
            if step % 10 == 0 and rank == 0:
                current_time = time.time()
                tokens_per_sec = tokens_processed / (current_time - start_time) if current_time > start_time else 0
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                
                # Average losses
                denom = max(1, accumulated_batches)
                avg_losses = {}
                for k, v in accumulated_losses.items():
                    if k == 'valid_tokens':
                        continue
                    # Token-normalize only CE-like per-token losses
                    if k in ('lm_ce_loss', 'policy_loss') and accumulated_tokens > 0:
                        avg_losses[k] = v / accumulated_tokens
                    else:
                        avg_losses[k] = v / denom
                print(
                    f"Step {step:,} | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                    f"Total: {avg_losses.get('total_loss', 0):.4f} | "
                    f"LM: {avg_losses.get('lm_ce_loss', 0):.4f} | "
                    f"Policy: {avg_losses.get('policy_loss', 0):.4f} | "
                    f"Value: {avg_losses.get('value_loss', 0):.4f} | "
                    f"Plan: {avg_losses.get('planning_loss', 0):.4f} | "
                    f"Dyn: {avg_losses.get('dyn_consistency', 0):.4f} | "
                    f"Dyn2: {avg_losses.get('dyn2_consistency', 0):.4f} | "
                    f"Improve: {avg_losses.get('policy_improve', 0):.4f} | "
                    f"PlanProb: {avg_losses.get('plan_prob_mean', 0):.4f}"
                )
                print(
                    f"       Tokens: {tokens_processed:,} | Tokens/sec: {tokens_per_sec:.0f} | GPU: {gpu_mem:.1f}GB"
                )
                
                if math.isnan(avg_losses.get('total_loss', 0)):
                    print("⚠️  NaN loss detected - stopping training")
                    break
                
                last_log_time = current_time
            
            # Save checkpoint
            if step % args.checkpoint_every == 0 and rank == 0:
                # Quick eval metrics (simplified)
                eval_metrics = {
                    'step': step,
                    'tokens_processed': tokens_processed,
                    'avg_total_loss': accumulated_losses.get('total_loss', 0) / accumulated_tokens if accumulated_tokens > 0 else 0,
                    'avg_lm_ce_loss': accumulated_losses.get('lm_ce_loss', 0) / accumulated_tokens if accumulated_tokens > 0 else 0,
                    'avg_policy_loss': accumulated_losses.get('policy_loss', 0) / accumulated_tokens if accumulated_tokens > 0 else 0,
                }
                
                save_muzero_checkpoint(
                    model, optimizer, scaler, tokens_processed, step, eval_metrics,
                    args.seed, args, args.checkpoint_dir
                )
            
            # Reset accumulators
            accumulated_losses = {}
            accumulated_tokens = 0
            accumulated_batches = 0
        
        # Stop when we hit token target
        if tokens_processed >= args.train_tokens:
            break
    
    # Final checkpoint and summary
    if rank == 0:
        total_time = time.time() - start_time
        final_tokens_per_sec = tokens_processed / total_time if total_time > 0 else 0
        
        print(f"🎉 MuZero-Adapters training complete!")
        print(f"   Steps: {step:,}, Tokens: {tokens_processed:,}")
        print(f"   Time: {total_time:.1f}s, Tokens/sec: {final_tokens_per_sec:.0f}")
        
        # Final evaluation metrics
        final_eval_metrics = {
            'step': step,
            'tokens_processed': tokens_processed,
            'final_tokens_per_sec': final_tokens_per_sec,
            'total_training_time': total_time
        }
        
        # Save final checkpoint
        final_checkpoint_path, final_summary_path = save_muzero_checkpoint(
            model, optimizer, scaler, tokens_processed, step, final_eval_metrics,
            args.seed, args, args.checkpoint_dir
        )
        
        # Optional JSON log
        if args.log_json:
            model_to_check = model.module if hasattr(model, 'module') else model
            adapter_params, total_params = model_to_check.count_adapter_parameters()
            
            out = {
                "branch": "muzero-adapters",
                "steps": step,
                "tokens": tokens_processed,
                "tokens_per_sec": final_tokens_per_sec,
                "seed": args.seed,
                "adapter_params": adapter_params,
                "total_params": total_params,
                "adapter_percentage": 100 * adapter_params / total_params,
                "final_lr": optimizer.param_groups[0]['lr'],
                "checkpoint_path": final_checkpoint_path,
                "summary_path": final_summary_path,
                "lm_checkpoint_loaded": args.load_lm_checkpoint,
                "config": {
                    "seq_length": args.seq_length,
                    "hidden_size": args.hidden_size,
                    "num_layers": args.num_layers,
                    "lm_ce_weight": args.lm_ce_weight,
                    "adapter_lr": args.adapter_lr
                }
            }
            try:
                with open(args.log_json, 'w') as f:
                    json.dump(out, f, indent=2)
                print(f"✓ Metrics written to {args.log_json}")
            except Exception as e:
                print(f"⚠️  Failed to write metrics JSON: {e}")
    
    # Attempt to gracefully shutdown dataloader workers and CUDA before exit
    try:
        if 'train_loader' in locals():
            it = getattr(train_loader, '_iterator', None)
            if it is not None:
                it._shutdown_workers()
    except Exception:
        pass

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

    import os as _os
    _os._exit(0)

if __name__ == "__main__":
    main()