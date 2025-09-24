#!/usr/bin/env python3
"""
MuZero-General: Simple Fast Training
===================================
Just run this. No runtime parameters. Trains fast.

DO NOT ADD RUNTIME PARAMETERS!!
"""

import os
import sys
import time
import math
import argparse
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Set
from collections import Counter, deque
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
sys.path.append('model')
sys.path.append('.')

from model.data_registry import DataRegistry, DataConfig
from model.hybrid_memory import HybridMemoryEngine
from pathlib import Path
import subprocess
import glob

# Improve resilience of HF streaming: increase read timeout and enable faster transfer
os.environ.setdefault('HF_HUB_READ_TIMEOUT', '60')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
# Reduce CUDA fragmentation by default
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# No CUDA graphs or compile hooks – keep baseline fast path

# Top-level entry for auto DDP spawn (must be picklable)
def _auto_ddp_entry(local_rank, world_size):
    os.environ['RANK'] = str(local_rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(local_rank)
    # Re-enter main as a DDP worker
    main()

class FlashAttentionLayer(nn.Module):
    """Custom transformer layer using Flash attention when available"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.gradient_checkpointing = gradient_checkpointing
        
        # Multi-head attention - use standard MHA without SDPA backend enforcement
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
        # Self attention block (standard MHA)
        x = src
        x2, _ = self.self_attn(
            x, x, x, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask, 
            need_weights=False,
            # Use explicit mask for broad PyTorch compatibility
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
    """Simple transformer language model for fast training with Flash/SDPA attention"""
    def __init__(self, vocab_size, hidden_size=512, num_layers=6, num_heads=8, max_seq_len=512, gradient_checkpointing=False,
                 enable_policy=False, enable_value=False, enable_dynamics=False, enable_span_proj=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        self.enable_policy = enable_policy
        self.enable_value = enable_value
        self.enable_dynamics = enable_dynamics
        self.enable_span_proj = enable_span_proj

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        # Cache a full-size causal mask and position ids; slice per forward
        causal_mask_full = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask_full', causal_mask_full, persistent=False)
        pos_ids_full = torch.arange(max_seq_len, dtype=torch.long)
        self.register_buffer('pos_ids_full', pos_ids_full, persistent=False)

        # Use custom Flash attention layers instead of nn.TransformerEncoder
        self.layers = nn.ModuleList([
            FlashAttentionLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                gradient_checkpointing=gradient_checkpointing
            ) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Auxiliary heads
        if self.enable_policy:
            # Predict distribution over next token (reuse vocab size)
            self.policy_head = nn.Linear(hidden_size, vocab_size)
        if self.enable_value:
            # Scalar future value (expected future negative log likelihood proxy)
            self.value_head = nn.Linear(hidden_size, 1)
        if self.enable_dynamics:
            # Latent transition: concatenate root hidden state with aggregated token embeddings (2H → H)
            # (Earlier version incorrectly depended on vocab_size causing shape mismatch.)
            self.dynamics_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        if self.enable_span_proj:
            # Projection for contrastive span representations
            self.span_projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def create_causal_mask(self, seq_len, device):
        """Get cached causal mask for given seq_len"""
        # Buffer lives on same device as module after .to(device)
        return self.causal_mask_full[:seq_len, :seq_len]
    
    def create_key_padding_mask(self, input_ids, pad_token_id):
        """Create key padding mask to ignore pad tokens"""
        return input_ids == pad_token_id
    
    def encode(self, input_ids, pad_token_id=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings computed inside compiled graph; avoid cloning here to prevent
        # treating them as graph outputs that can be overwritten across runs.
        token_embeddings = self.embedding(input_ids)
        position_ids = self.pos_ids_full[:seq_len].unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.pos_embedding(position_ids)
        
        hidden_states = token_embeddings + position_embeddings
        
        # Build key padding mask to ignore pad tokens during attention
        key_padding_mask = None
        if pad_token_id is not None:
            key_padding_mask = (input_ids == pad_token_id)
        
        # Build explicit causal mask (upper triangular True = masked)
        causal_mask = self.create_causal_mask(seq_len, device)
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        return hidden_states

    def forward(self, input_ids, pad_token_id=None, policy_positions: torch.Tensor | None = None, dynamics_tokens: torch.Tensor | None = None):
        """Forward producing logits plus optional auxiliary outputs.
        policy_positions: indices at which to compute policy/value (e.g., last token per sequence).
        dynamics_tokens: optional one-hot or token ids for predicting next latent (N-step rollout can feed back)."""
        hidden_states = self.encode(input_ids, pad_token_id)
        logits = self.lm_head(hidden_states)
        aux = {}
        # Policy / Value heads use final position unless specified
        if self.enable_policy or self.enable_value:
            if policy_positions is None:
                policy_positions = torch.full((hidden_states.size(0),), hidden_states.size(1)-1, device=hidden_states.device, dtype=torch.long)
            gather_h = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), policy_positions]
            if self.enable_policy:
                aux['policy_logits'] = self.policy_head(gather_h)
            if self.enable_value:
                aux['value'] = self.value_head(gather_h).squeeze(-1)
        if self.enable_dynamics and dynamics_tokens is not None:
            # dynamics_tokens expected shape (B, K) token ids for next positions (teacher forcing small window)
            # Efficient embedding lookup instead of one-hot → avoids huge (vocab) dimension and fixes shape.
            dyn_in = dynamics_tokens.clamp_min(0)
            token_embed = self.embedding(dyn_in)  # (B,K,H)
            token_agg = token_embed.mean(dim=1)   # (B,H)
            root_h = hidden_states[:, -1, :]      # (B,H)
            dyn_inp = torch.cat([root_h, token_agg], dim=-1)  # (B,2H)
            next_latent = self.dynamics_fc(dyn_inp)
            aux['dynamics_next'] = next_latent
        if self.enable_span_proj:
            aux['span_proj'] = self.span_projector(hidden_states)
        return logits, aux

    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize token embedding and output head to match tokenizer size."""
        if new_vocab_size == self.vocab_size:
            return
        old_vocab = self.vocab_size
        self.vocab_size = new_vocab_size
        # Resize input embedding
        old_embed = self.embedding
        new_embed = nn.Embedding(new_vocab_size, self.hidden_size)
        # Init new
        torch.nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
        # Copy old
        num_to_copy = min(old_vocab, new_vocab_size)
        with torch.no_grad():
            new_embed.weight[:num_to_copy].copy_(old_embed.weight[:num_to_copy])
        self.embedding = new_embed
        # Resize output head
        old_head = self.lm_head
        new_head = nn.Linear(self.hidden_size, new_vocab_size)
        torch.nn.init.normal_(new_head.weight, mean=0.0, std=0.02)
        if old_head.bias is not None:
            torch.nn.init.zeros_(new_head.bias)
        with torch.no_grad():
            new_head.weight[:num_to_copy].copy_(old_head.weight[:num_to_copy])
            if old_head.bias is not None and new_head.bias is not None:
                new_head.bias[:num_to_copy].copy_(old_head.bias[:num_to_copy])
        self.lm_head = new_head

    def extend_max_seq_len(self, new_max_len: int):
        """Increase max sequence length (positional embeddings and masks)."""
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


@dataclass(frozen=True)
class FactCandidate:
    """Simple fact representation extracted from raw text."""
    head: str
    relation: str
    tail: str
    sentence: str
    expected_type: Optional[str] = None


_FACT_PATTERNS: List[Tuple[str, re.Pattern, Optional[str]]] = [
    (
        'is_a',
        re.compile(r"(?P<head>[A-Z][A-Za-z0-9\-\' ]{2,})\s+(?:is|was|are|were)\s+(?:an|a|the)\s+(?P<tail>[A-Za-z0-9\-\' ]{2,})", re.IGNORECASE),
        'category'
    ),
    (
        'born_in',
        re.compile(r"(?P<head>[A-Z][A-Za-z0-9\-\' ]{2,})\s+(?:was|is)\s+born\s+in\s+(?P<tail>[A-Z][A-Za-z0-9\-\' ]{2,})", re.IGNORECASE),
        'location'
    ),
    (
        'capital_of',
        re.compile(r"(?P<head>[A-Z][A-Za-z0-9\-\' ]{2,})\s+(?:is|was)\s+the\s+capital\s+of\s+(?P<tail>[A-Z][A-Za-z0-9\-\' ]{2,})", re.IGNORECASE),
        'location'
    ),
    (
        'located_in',
        re.compile(r"(?P<head>[A-Z][A-Za-z0-9\-\' ]{2,})\s+(?:is|lies|is located|is situated)\s+in\s+(?P<tail>[A-Z][A-Za-z0-9\-\' ]{2,})", re.IGNORECASE),
        'location'
    ),
    (
        'collaborated_with',
        re.compile(r"(?P<head>[A-Z][A-Za-z0-9\-\' ]{2,})\s+(?:collaborated|worked)\s+with\s+(?P<tail>[A-Z][A-Za-z0-9\-\' ]{2,})", re.IGNORECASE),
        'person'
    ),
    (
        'field',
        re.compile(r"(?P<head>[A-Z][A-Za-z0-9\-\' ]{2,})\s+(?:studied|specialized|worked)\s+in\s+(?P<tail>[A-Za-z0-9\-\' ]{2,})", re.IGNORECASE),
        'discipline'
    ),
]


def _normalize_fact_component(component: str) -> str:
    comp = re.sub(r"\s+", " ", component).strip(" \t\n\r\f\v.,:;\"'()[]{}")
    lowered = comp.lower()
    for article in ("the ", "a ", "an "):
        if lowered.startswith(article):
            comp = comp[len(article):].strip()
            break
    return comp


def extract_fact_candidates(text: str) -> List[FactCandidate]:
    """Pull lightweight fact triples from text using regex heuristics."""
    if not text:
        return []
    candidates: List[FactCandidate] = []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    for sentence in sentences:
        sent = sentence.strip()
        if len(sent) < 16:
            continue
        for relation, pattern, expected_type in _FACT_PATTERNS:
            for match in pattern.finditer(sent):
                head = _normalize_fact_component(match.group('head'))
                tail = _normalize_fact_component(match.group('tail'))
                if not head or not tail:
                    continue
                if len(head) > 64 or len(tail) > 64:
                    continue
                candidates.append(FactCandidate(head=head, relation=relation, tail=tail, sentence=sent, expected_type=expected_type))
    return candidates


class _FactDeduper:
    """Bounded set to avoid re-ingesting duplicate facts each step."""

    def __init__(self, max_size: int = 8192):
        self.max_size = max_size
        self._cache: Set[Tuple[str, str, str]] = set()
        self._queue: deque = deque()

    def add(self, key: Tuple[str, str, str]) -> bool:
        if key in self._cache:
            return False
        self._cache.add(key)
        self._queue.append(key)
        if len(self._queue) > self.max_size:
            old = self._queue.popleft()
            self._cache.discard(old)
        return True


def update_hybrid_memory_from_batch(
    hybrid_memory: Optional[HybridMemoryEngine],
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    deduper: Optional[_FactDeduper],
    max_decode: int = 2
) -> List[FactCandidate]:
    """Decode a few sequences from the batch, harvest facts, and ingest into memory."""
    if hybrid_memory is None or deduper is None or max_decode <= 0:
        return []
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    num_samples = min(max_decode, input_ids.size(0))
    cpu_ids = input_ids[:num_samples].detach().cpu()
    texts = tokenizer.batch_decode(cpu_ids, skip_special_tokens=True)
    new_facts: List[FactCandidate] = []
    for text in texts:
        for fact in extract_fact_candidates(text):
            key = (fact.head.lower(), fact.relation, fact.tail.lower())
            if not deduper.add(key):
                continue
            evidence = None
            if hybrid_memory.passage_index.embeddings is None:
                evidence = [fact.sentence]
            hybrid_memory.ingest_fact(fact.head, fact.relation, fact.tail, evidence)
            new_facts.append(fact)
    if new_facts and len(hybrid_memory.graph.edge_heads) < 32:
        hybrid_memory.graph._rebuild_offsets()
    return new_facts

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
    """Generate text with top-p and n-gram blocking to reduce repetition."""
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
    """Evaluate CE and PPL on streaming eval data and assert CE↔PPL parity."""
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
    total_via_mean = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            
            base_model = model.module if hasattr(model, 'module') else model
            hidden_states = base_model.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
            logits = base_model.lm_head(hidden_states)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            # Sum over tokens (mask pads explicitly)
            loss_flat = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss_tok = loss_flat.view(shift_labels.shape)
            pad_mask = (shift_labels != tokenizer.pad_token_id)
            valid = pad_mask.sum().item()
            if valid == 0:
                continue
            total_loss += (loss_tok * pad_mask.float()).sum().item()
            total_tokens += valid
            # Mean(ignore_index) path for parity
            mean_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
                reduction='mean'
            )
            total_via_mean += mean_loss.item() * valid
    if total_tokens == 0:
        return None, None
    ce = total_loss / total_tokens
    ppl = math.exp(ce)
    # Parity asserts
    if total_via_mean > 0:
        ce_via_mean = total_via_mean / total_tokens
        assert abs(ce - ce_via_mean) < 1e-3, f"CE parity mismatch: ce={ce:.6f} via_mean={ce_via_mean:.6f}"
    assert abs(math.log(ppl) - ce) < 1e-6, "PPL/CE parity mismatch"
    model.train()
    return ce, ppl

def setup_ddp():
    """Setup Distributed Data Parallel if available"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        # Check if we have enough GPUs
        num_gpus = torch.cuda.device_count()
        if rank >= num_gpus:
            raise RuntimeError(
                f"Not enough GPUs for DDP! Requested rank {rank} but only {num_gpus} GPUs available. "
                f"Use --nproc_per_node={num_gpus} instead of {world_size}"
            )
        
        # Initialize the process group
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        
        # Set device for this process
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        
        return True, rank, world_size, device
    else:
        return False, 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_loss_with_proper_scaling(logits, input_ids, tokenizer, reduction='sum'):
    """Compute loss with proper scaling over non-pad tokens (standard LM)."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    pad_mask = (shift_labels != tokenizer.pad_token_id)
    if reduction == 'sum':
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction='none'
        )
        loss = loss.view(shift_labels.shape)
        loss = (loss * pad_mask.float()).sum()
        num_tokens = pad_mask.sum().item()
        return loss, num_tokens
    else:
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
            reduction='mean'
        )
        return loss, None

# UL2/infilling removed – standard LM objective only

def _rank0_select_config(all_configs: list[str]) -> str:
    """Show an interactive menu on rank 0 to select a config filename from configs/.
    Returns the selected filename (basename)."""
    print("\nSelect a training preset (configs/*.json):")
    for i, name in enumerate(all_configs, start=1):
        print(f"  [{i}] {name}")
    default_idx = 1
    choice = input(f"Enter choice [default {default_idx}]: ").strip()
    if not choice:
        idx = default_idx
    else:
        try:
            idx = int(choice)
        except Exception:
            idx = default_idx
    idx = max(1, min(len(all_configs), idx))
    return all_configs[idx - 1]

def _ddp_broadcast_string(s: str) -> str:
    """Broadcast a UTF-8 string from rank 0 to all ranks (DDP)."""
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return s
    rank = torch.distributed.get_rank()
    # First broadcast length
    if rank == 0:
        data = s.encode('utf-8')
        dev = torch.device('cuda', torch.cuda.current_device())
        n = torch.tensor([len(data)], dtype=torch.int64, device=dev)
    else:
        data = None
        dev = torch.device('cuda', torch.cuda.current_device())
        n = torch.tensor([0], dtype=torch.int64, device=dev)
    torch.distributed.broadcast(n, src=0)
    # Then broadcast payload
    if rank != 0:
        buf = torch.empty(int(n.item()), dtype=torch.uint8, device=dev)
    else:
        buf = torch.tensor(list(data), dtype=torch.uint8, device=dev)
    torch.distributed.broadcast(buf, src=0)
    if rank != 0:
        s = bytes(buf.tolist()).decode('utf-8')
    return s

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _rotate_checkpoints(save_path: Path, tmp_path: Path | None = None):
    """Maintain rolling three checkpoints: current, prev, prev2.
    save_path points to checkpoints/lm_current.pth to be written; we rotate prev files first.
    """
    base_dir = save_path.parent
    current = base_dir / 'lm_current.pth'
    prev = base_dir / 'lm_prev.pth'
    prev2 = base_dir / 'lm_prev2.pth'
    # Rotate: prev -> prev2, current -> prev
    if prev.exists():
        try:
            prev2.unlink(missing_ok=True)
        except Exception:
            pass
        prev.rename(prev2)
    if current.exists():
        current.rename(prev)
    # Move tmp to current if provided; otherwise assume caller will write
    if tmp_path is not None and tmp_path.exists():
        tmp_path.rename(current)

def _find_latest_checkpoint() -> str | None:
    """Find the current rolling checkpoint first, then legacy locations."""
    ck_current = Path('checkpoints') / 'lm_current.pth'
    if ck_current.exists():
        return str(ck_current)
    candidates = []
    candidates.extend(sorted(glob.glob('baseline_continue_checkpoints/*.pt'), key=lambda p: Path(p).stat().st_mtime, reverse=True))
    candidates.extend(sorted(glob.glob('baseline_checkpoints/*.pt'), key=lambda p: Path(p).stat().st_mtime, reverse=True))
    if Path('simple_lm_trained.pth').exists():
        candidates.append('simple_lm_trained.pth')
    return candidates[0] if candidates else None

def main():
    print("🚀 MuZero-General: Simple Fast Training")
    print("=" * 40)
    
    # DO NOT ADD RUNTIME PARAMETERS!! All knobs come from a selected config in configs/*.json
    # Minimal parser retained only to optionally disable auto DDP via env if ever needed, but unused.
    class _Args: pass
    args = _Args()
    args.no_auto_ddp = False

    # If the script isn't launched under torchrun (no RANK/WORLD_SIZE), but multiple GPUs are
    # available and the user didn't disable it, automatically launch multi-process DDP so
    # `python main.py --compile` will fully utilize all GPUs.
    # Remove auto-spawn; require explicit torchrun to manage processes
    
    # Setup DDP if available
    use_ddp, rank, world_size, device = setup_ddp()
    
    # Enable performance optimizations
    if torch.cuda.is_available():
        # Enable TF32 for faster training on A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Enable channels-last memory format for performance (convolutions)
        torch.backends.cudnn.benchmark = True
        
        print(f"✓ Enabled TF32 and cuDNN optimizations")
    
    # Only print on main process for DDP
    if rank == 0:
        print(f"Device: {device}")
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Available GPUs: {num_gpus}")
            for i in range(num_gpus):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")
            
            if use_ddp:
                print(f"🚀 Using DDP with {world_size} processes!")
            elif num_gpus > 1:
                print(f"ℹ️  Multiple GPUs detected. Launch with torchrun for multi-GPU. Running on single GPU 0 by default.")
        
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    
    # Use bf16 instead of fp16 for better numerical stability
    scaler = GradScaler('cuda')
    
    # Load configs and select preset (rank 0 prompts, then broadcast)
    configs_dir = Path('configs')
    cfg_files = sorted([p.name for p in configs_dir.glob('*.json')])
    if not cfg_files:
        raise RuntimeError("No config files found in configs/*.json")

    # Rank 0: choose; Others: wait for broadcast
    if use_ddp and dist.get_rank() == 0:
        chosen = _rank0_select_config(cfg_files)
        with open(configs_dir / chosen, 'r') as f:
            cfg_text = f.read()
    elif not use_ddp:
        chosen = _rank0_select_config(cfg_files)
        with open(configs_dir / chosen, 'r') as f:
            cfg_text = f.read()
    else:
        cfg_text = ""
    cfg_text = _ddp_broadcast_string(cfg_text)
    cfg = json.loads(cfg_text)

    # Echo chosen config and policy (always train latest with rolling snapshots)
    if rank == 0:
        print(f"\n✓ Training policy: always continue from latest and keep (current, prev, prev2)")
        print(f"✓ Using config: {cfg.get('name','unknown')} ({cfg.get('description','')})")
        print("--- Defaults ---")
        print(json.dumps(cfg, indent=2))
        print("Note: Training token target is fixed to 100,000,000 (ignoring preset).")

    # Fast data setup (tokenizer)
    # Use smaller vocab tokenizer to reduce logits memory unless config overrides later
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    # Fast training config - REAL DATA ONLY
    # Build DataConfig from chosen cfg (override train_tokens to fixed 100M)
    dc = cfg.get('data', {})
    tc = cfg.get('train', {})
    mc = cfg.get('model', {})
    aux_cfg = cfg.get('auxiliary', {})
    hm_cfg = cfg.get('hybrid_memory', {})
    data_config = DataConfig(
        train_tokens=100_000_000,
        eval_tokens=int(tc.get('eval_tokens', 1_500_000)),
        seq_length=int(mc.get('seq_length', 256)),
        min_text_length=int(dc.get('min_text_length', 50)),
        data_sources=list(dc.get('sources', ['wikipedia'])),
        custom_sources=list(dc.get('custom_sources', [])),
        mix_strategy=str(dc.get('mix_strategy', 'round_robin')),
        source_weights=dc.get('source_weights'),
        allow_fallback_synthetic=False
    )
    # Proactively drop known failing HF script sources (e.g. legacy OpenWebText script) to prevent empty sources & repeated errors
    removed = []
    filtered_sources = []
    for s in data_config.data_sources:
        if isinstance(s, str) and 'openwebtext' in s.lower():
            removed.append(s)
        else:
            filtered_sources.append(s)
    if removed and rank == 0:
        print(f"⚠️  Removing unsupported sources: {removed} (HF script deprecation)")
    data_config.data_sources = filtered_sources
    # Auxiliary feature flags (all default off)
    enable_policy = bool(aux_cfg.get('enable_policy', False))
    enable_value = bool(aux_cfg.get('enable_value', False))
    enable_dynamics = bool(aux_cfg.get('enable_dynamics', False))
    enable_span_proj = bool(aux_cfg.get('enable_span_contrastive', False))
    aux_loss_weights = {
        'policy_kl': float(aux_cfg.get('policy_kl_weight', 0.05)),
        'value_mse': float(aux_cfg.get('value_weight', 0.05)),
        'dynamics': float(aux_cfg.get('dynamics_weight', 0.05)),
        'span_contrastive': float(aux_cfg.get('span_contrastive_weight', 0.05)),
    }
    span_cfg = {
        'mask_fraction': float(aux_cfg.get('span_mask_fraction', 0.15)),
        'temperature': float(aux_cfg.get('span_contrastive_temperature', 0.07)),
        'proj_norm': bool(aux_cfg.get('span_l2_normalize', True)),
    }
    curriculum_cfg = {
        'enable_adaptive_sources': bool(aux_cfg.get('enable_adaptive_sources', False)),
        'adapt_interval_tokens': int(aux_cfg.get('adaptive_interval_tokens', 5_000_000)),
        'min_weight': float(aux_cfg.get('adaptive_min_weight', 0.5)),
        'max_weight': float(aux_cfg.get('adaptive_max_weight', 3.0)),
        'smoothing': float(aux_cfg.get('adaptive_smoothing', 0.9)),
    }
    # Hybrid memory configuration
    enable_hybrid_memory = bool(hm_cfg.get('enable', False))
    hm_entity_dim = int(hm_cfg.get('entity_dim', 128))
    hm_passage_dim = int(hm_cfg.get('passage_dim', 384))
    hm_decode_per_batch = int(hm_cfg.get('decode_per_batch', 2))
    hm_log_interval = int(hm_cfg.get('log_interval_steps', 200))
    hm_dedupe_window = int(hm_cfg.get('dedupe_window', 8192))
    
    registry = DataRegistry()
    
    # Scale batch size based on number of GPUs - REALISTIC for speed  
    # Fast default: larger per-GPU batch size; adjust down if OOM
    base_batch_size = int(tc.get('per_gpu_batch_size', 128))
    # Auto batch size tuning: reduce until it fits estimated memory budget if previous OOM or large model
    # Heuristic: estimated activation memory ~ batch * seq_len * hidden * 2 bytes * num_layers * 3 (attn+ff) / 1e9 GB
    auto_tune = bool(tc.get('auto_tune_batch', True))
    if auto_tune:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device)
            free_gb = free_mem / 1e9
            hidden_est = int(mc.get('hidden_size', 256))
            layers_est = int(mc.get('num_layers', 3))
            seq_est = data_config.seq_length
            vocab_est = 0
            try:
                vocab_est = tokenizer.vocab_size
            except Exception:
                pass
            # Rough GB per sample (activations + logits buffer). Logits dominate for large vocab.
            gb_per_sample = (seq_est * hidden_est * layers_est * 3 * 2) / 1e9
            gb_per_sample += (seq_est * vocab_est * 2) / 1e9  # logits bf16
            if gb_per_sample > 0:
                max_samples_fit = max(1, int((free_gb * 0.55) / gb_per_sample))  # leave headroom
                if base_batch_size > max_samples_fit:
                    if rank == 0:
                        print(f"⚙️  Auto-tune batch: {base_batch_size} → {max_samples_fit} (memory heuristic; free {free_gb:.1f} GB)")
                    base_batch_size = max_samples_fit
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Auto batch size heuristic failed: {e}")
    # If not using DDP, default to single GPU 0 to avoid DataParallel + compile issues
    if not use_ddp and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
    num_gpus = world_size if use_ddp else 1
    batch_size = base_batch_size * max(1, num_gpus)  # Total batch size across GPUs
    
    # Create distributed sampler if using DDP
    sampler = None
    if use_ddp:
        print(f"✓ Using DistributedSampler for DDP training")
    
    # DataLoader performance knobs (tuned for multi-GPU)
    # num_workers: roughly 2-4 per GPU is a good start
    # prefetch_factor: number of batches prefetched per worker
    # persistent_workers: keep workers alive between epochs (useful as we stream)
    # Determine DataLoader workers per-process
    # Streaming from web endpoints benefits from lower concurrency
    num_workers_dl = int(tc.get('num_workers', 2))
    # Clamp workers for single-shard iterable datasets to silence HF warning
    try:
        preview_ds = registry.create_dataloader(
            tokenizer=tokenizer,
            config=data_config,
            batch_size=1,
            use_sized=False,
            num_workers=0,
            prefetch_factor=2,
            persistent_workers=False
        )
        # If dataset has attribute num_shards and it's 1, clamp workers
        ds_inner = preview_ds.dataset
        shards = getattr(ds_inner, 'num_shards', 1)
        if shards == 1 and num_workers_dl > 1:
            if rank == 0:
                print(f"⚙️  Clamping DataLoader workers {num_workers_dl} → 1 (single shard)")
            num_workers_dl = 1
    except Exception:
        pass

    # Proceed with training: always start from the latest 'current' if available
    train_loader = registry.create_dataloader(
        tokenizer=tokenizer,
        config=data_config,
        batch_size=base_batch_size,  # Per-process batch size
        use_sized=False,
        num_workers=num_workers_dl,
        prefetch_factor=int(tc.get('prefetch_factor', 2)),
        persistent_workers=(num_workers_dl > 0)
    )
    
    if rank == 0:
        print(f"✓ Data loaded: {data_config.train_tokens:,} tokens target")
    
    # Simple transformer model - optimized for speed and stability
    model = SimpleLM(
        vocab_size=tokenizer.vocab_size,
        hidden_size=int(mc.get('hidden_size', 256)),
        num_layers=int(mc.get('num_layers', 3)),
        num_heads=int(mc.get('num_heads', 4)),
        max_seq_len=data_config.seq_length,
        gradient_checkpointing=bool(tc.get('grad_checkpointing', False)),
        enable_policy=enable_policy,
        enable_value=enable_value,
        enable_dynamics=enable_dynamics,
        enable_span_proj=enable_span_proj
    )
    
    # Try to resume from the rolling current checkpoint
    ckpt = _find_latest_checkpoint()
    if rank == 0:
        print(f"Resume checkpoint: {ckpt if ckpt else 'NONE'}")
    if ckpt and Path(ckpt).exists():
        try:
            # PyTorch 2.6 defaults to weights_only=True; allow full load for our trusted local checkpoints
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            # Extend pos embedding if needed
            if 'pos_embedding.weight' in state and state['pos_embedding.weight'].shape[0] < data_config.seq_length:
                model.extend_max_seq_len(data_config.seq_length)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if rank == 0 and (missing or unexpected):
                print(f"ℹ️  Loaded with missing={len(missing)}, unexpected={len(unexpected)}")
        except Exception as e:
            if rank == 0:
                print(f"⚠️  Failed to load checkpoint '{ckpt}': {e}")
    
    model = model.to(device)

    # Lightweight forward self-test (single tiny batch) to catch dimension issues early
    if rank == 0 and enable_dynamics:
        try:
            with torch.no_grad():
                test_ids = torch.randint(0, tokenizer.vocab_size, (2, min(8, data_config.seq_length)), device=device)
                dyn_tokens = test_ids[:, -4:-1] if test_ids.size(1) > 4 else None
                _logits, _aux = model(test_ids, pad_token_id=tokenizer.pad_token_id, dynamics_tokens=dyn_tokens)
                if enable_dynamics and 'dynamics_next' not in _aux:
                    print("⚠️  Self-test: dynamics_next missing from aux output")
                else:
                    print("✓ Self-test forward (dynamics) passed: logits", _logits.shape)
        except Exception as e:
            print(f"❌ Self-test forward failed: {e}")
            raise
    
    # Multi-GPU support - prefer DDP; no DataParallel fallback
    if use_ddp:
        model = DDP(model, device_ids=[rank])
        if rank == 0:
            print(f"🚀 Wrapped model with DDP for {world_size} processes")
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ Model: {num_params:,} parameters")

    # Initialize hybrid memory (entity-relation graph + passage index)
    hybrid_memory = None
    fact_deduper = None
    fact_relation_counter: Counter[str] = Counter()
    total_facts_ingested = 0
    if enable_hybrid_memory:
        if rank == 0:
            print("🧠 Initializing Hybrid Memory (ER graph + passage index)...")
        hybrid_memory = HybridMemoryEngine(entity_dim=hm_entity_dim, passage_dim=hm_passage_dim, use_faiss=True)
        fact_deduper = _FactDeduper(max_size=hm_dedupe_window)
        # Demo ingestion (placeholder facts); in future tie to dataset scanning
        if rank == 0:
            demo_facts = [
                ("Ada Lovelace", "field", "Mathematics", ["Ada Lovelace contributed foundational ideas to computing."]),
                ("Ada Lovelace", "collaborated_with", "Charles Babbage", ["She collaborated closely with Charles Babbage on the Analytical Engine."]),
                ("Charles Babbage", "designed", "Analytical Engine", ["The Analytical Engine design prefigured modern computers."])
            ]
            for h, r, t, ev in demo_facts:
                hybrid_memory.ingest_fact(h, r, t, ev)
            q = hybrid_memory.query("Ada Lovelace", topk_neighbors=5, topk_passages=3)
            print("🔎 Hybrid memory demo (Ada Lovelace):", q)
    
    # Fast optimizer setup with gradient clipping
    optim_kwargs = dict(lr=float(tc.get('lr', 5e-4)), weight_decay=float(tc.get('weight_decay', 0.01)), betas=(0.9, 0.95), eps=1e-8)
    if bool(tc.get('fused_adamw', False)):
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optim_kwargs)
        if rank == 0:
            print("✓ Using fused AdamW optimizer")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optim_kwargs)
    
    # Gradient accumulation for effective batch size
    accumulation_steps = max(1, int(tc.get('accumulation_steps', 4)))
    
    # Train fast
    model.train()
    step = 0
    tokens_processed = 0
    accumulated_loss = 0
    accumulated_tokens = 0
    recent_ce_log = []  # for simple trend tracking
    # Adaptive source weighting state
    source_token_loss_sum = []
    source_token_count = []
    last_adapt_tokens = 0
    if curriculum_cfg['enable_adaptive_sources']:
        ds_obj = train_loader.dataset
        if hasattr(ds_obj, 'data_sources'):
            nsrc = len(ds_obj.data_sources)
            source_token_loss_sum = [0.0] * nsrc
            source_token_count = [0] * nsrc
    
    # Plateau rule: eval CE improvement < 0.03 per 10M tokens for two consecutive windows → make one controlled change
    plateau_window_size = 10_000_000  # 10M tokens per window
    plateau_ce_history = []  # Store (tokens_at_eval, eval_ce) tuples
    plateau_improvement_threshold = 0.03
    plateau_consecutive_windows = 0
    plateau_controlled_changes_made = 0
    
    # Wall-clock timing for accurate tokens/sec
    start_time = time.time()
    last_log_time = start_time
    tokens_processed_last_log = 0
    
    if rank == 0:
        print("🔥 Training started...")
        print(f"   Processes: {world_size if use_ddp else 1}")
        print(f"   GPUs: {num_gpus}")
        print(f"   Batch size per GPU: {base_batch_size}")
        print(f"   Total batch size: {batch_size}")
        print(f"   Effective batch size: {batch_size * accumulation_steps}")
        print(f"   Target tokens: {data_config.train_tokens:,}")
        print(f"   🎛️  Plateau rule: eval CE improvement < {plateau_improvement_threshold} per {plateau_window_size:,} tokens for 2 consecutive windows → controlled change")
        print("   Starting first batch...")
    
    # OOM adaptive state
    oom_retries = 0
    max_oom_retries = 4
    reduced_batch_sizes = set()
    while True:
        try:
            for batch_idx, batch in enumerate(train_loader):
                # Mark step boundary to keep CUDA Graph captures isolated per iteration
                if rank == 0 and batch_idx == 0:
                    print(f"   Batch {batch_idx} loaded, processing...")
                cpu_input_ids = batch['input_ids']
                newly_ingested: List[FactCandidate] = []
                if enable_hybrid_memory and hybrid_memory is not None:
                    newly_ingested = update_hybrid_memory_from_batch(
                        hybrid_memory,
                        tokenizer,
                        cpu_input_ids,
                        fact_deduper,
                        max_decode=hm_decode_per_batch
                    )
                    if newly_ingested:
                        total_facts_ingested += len(newly_ingested)
                        for fact in newly_ingested:
                            fact_relation_counter[fact.relation] += 1
                        if rank == 0 and (step == 0 or (hm_log_interval > 0 and (step % hm_log_interval) == 0)):
                            preview = ', '.join(f"{f.head} --{f.relation}→ {f.tail}" for f in newly_ingested[:3])
                            print(f"   🧠 Hybrid memory ingested {len(newly_ingested)} fact(s); sample: {preview}")
                input_ids = cpu_input_ids.to(device, non_blocking=True)
                batch_size_actual, seq_len = input_ids.shape
                tokens_in_batch = batch_size_actual * seq_len
                tokens_processed += tokens_in_batch
                # Use bf16 autocast for better numerical stability
                with autocast('cuda', dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16):
                    base_model = model.module if hasattr(model, 'module') else model
                    dyn_tokens = None
                    if enable_dynamics and input_ids.shape[1] > 4:
                        dyn_tokens = input_ids[:, -4:-1]
                    logits, aux_out = base_model(input_ids, pad_token_id=tokenizer.pad_token_id, dynamics_tokens=dyn_tokens)
                    loss_sum, num_valid_tokens = compute_loss_with_proper_scaling(logits, input_ids, tokenizer, reduction='sum')
                    total_aux_loss = torch.zeros((), device=logits.device)
                    if enable_policy and 'policy_logits' in aux_out:
                        final_logits = logits[:, -1, :].detach()
                        teacher = F.log_softmax(final_logits, dim=-1)
                        student = F.log_softmax(aux_out['policy_logits'], dim=-1)
                        kl = F.kl_div(student, teacher.exp(), reduction='batchmean')
                        total_aux_loss = total_aux_loss + aux_loss_weights['policy_kl'] * kl
                    if enable_value and 'value' in aux_out:
                        with torch.no_grad():
                            shift_logits = logits[:, :-1, :]
                            shift_labels = input_ids[:, 1:]
                            B_, L_, V_ = shift_logits.size(0), shift_logits.size(1), shift_logits.size(2)
                            pad_mask_val = (shift_labels != tokenizer.pad_token_id)
                            est_bytes = B_ * L_ * V_ * 2
                            chunk_threshold = 3 * (1024 ** 3)
                            if est_bytes > chunk_threshold:
                                if rank == 0 and step < 5:
                                    print(f"ℹ️  Using streaming value CE (est {est_bytes/1e9:.1f} GB) step {step}")
                                ce_sum = torch.zeros(B_, device=shift_logits.device)
                                token_count = pad_mask_val.sum(dim=1).clamp_min(1)
                                for t in range(L_):
                                    if not pad_mask_val[:, t].any():
                                        continue
                                    lp_t = F.log_softmax(shift_logits[:, t, :], dim=-1)
                                    gather_t = lp_t.gather(1, shift_labels[:, t].unsqueeze(1)).squeeze(1)
                                    ce_sum += (-gather_t) * pad_mask_val[:, t]
                                ce_reduced = ce_sum / token_count
                            else:
                                log_probs = F.log_softmax(shift_logits, dim=-1)
                                gathered = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
                                ce_tok = -gathered * pad_mask_val
                                ce_reduced = ce_tok.sum(dim=1) / pad_mask_val.sum(dim=1).clamp_min(1)
                        mse = F.mse_loss(aux_out['value'], ce_reduced)
                        total_aux_loss = total_aux_loss + aux_loss_weights['value_mse'] * mse
                    if enable_dynamics and 'dynamics_next' in aux_out:
                        with torch.no_grad():
                            hs_dyn = base_model.encode(input_ids, pad_token_id=tokenizer.pad_token_id)
                            target_latent = hs_dyn[:, -3:, :].mean(dim=1)
                        dyn_loss = F.mse_loss(aux_out['dynamics_next'], target_latent)
                        total_aux_loss = total_aux_loss + aux_loss_weights['dynamics'] * dyn_loss
                    if enable_span_proj and 'span_proj' in aux_out:
                        proj = aux_out['span_proj']
                        B, L, Hh = proj.shape
                        span_len = max(1, int(L * span_cfg['mask_fraction']))
                        start_idx = torch.randint(0, max(1, L - span_len + 1), (B,), device=proj.device)
                        mask = torch.zeros((B, L), device=proj.device, dtype=torch.bool)
                        for bi in range(B):
                            mask[bi, start_idx[bi]:start_idx[bi]+span_len] = True
                        span_rep = (proj * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
                        masked_proj = proj.clone()
                        masked_proj[mask] = 0.0
                        q = masked_proj.mean(dim=1)
                        k = span_rep
                        if span_cfg['proj_norm']:
                            q = F.normalize(q, dim=-1)
                            k = F.normalize(k, dim=-1)
                        logits_con = (q @ k.t()) / span_cfg['temperature']
                        labels_con = torch.arange(B, device=proj.device)
                        con_loss = F.cross_entropy(logits_con, labels_con)
                        total_aux_loss = total_aux_loss + aux_loss_weights['span_contrastive'] * con_loss
                    loss_total = (loss_sum + total_aux_loss) / accumulation_steps
                    loss = loss_total
                scaler.scale(loss).backward()
                accumulated_loss += loss_sum.item()
                accumulated_tokens += num_valid_tokens
                if curriculum_cfg['enable_adaptive_sources'] and 'source_index' in batch:
                    src_idx_tensor = batch['source_index']
                    if isinstance(src_idx_tensor, torch.Tensor):
                        if src_idx_tensor.dim() == 0:
                            src_idx_tensor = src_idx_tensor.unsqueeze(0)
                        vals, counts = torch.unique(src_idx_tensor, return_counts=True)
                        majority = vals[counts.argmax()].item()
                        if majority < len(source_token_loss_sum):
                            source_token_loss_sum[majority] += loss_sum.item()
                            source_token_count[majority] += num_valid_tokens
                if (batch_idx + 1) % accumulation_steps == 0:
                    step += 1
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if step % 10 == 0 and rank == 0:
                        current_time = time.time()
                        time_elapsed = current_time - last_log_time
                        avg_loss = accumulated_loss / accumulated_tokens if accumulated_tokens > 0 else 0
                        tokens_per_sec = (tokens_processed - tokens_processed_last_log) / time_elapsed if time_elapsed > 0 else 0
                        print(f"   step {step:05d} | tokens {tokens_processed:,} | loss {avg_loss:.3f} | tok/s {tokens_per_sec:,.0f}")
                        last_log_time = current_time
                        tokens_processed_last_log = tokens_processed
                        accumulated_loss = 0
                        accumulated_tokens = 0
                if tokens_processed >= data_config.train_tokens:
                    break
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom_retries += 1
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"💥 CUDA OOM encountered (retry {oom_retries}/{max_oom_retries}). Adjusting configuration...")
                if oom_retries > max_oom_retries:
                    if rank == 0:
                        print("❌ Exceeded maximum OOM retries. Aborting.")
                    raise
                if base_batch_size > 1:
                    new_bs = max(1, base_batch_size // 2)
                    if new_bs != base_batch_size and rank == 0:
                        print(f"🔧 Reducing per-GPU batch size {base_batch_size} → {new_bs}")
                    base_batch_size = new_bs
                if not bool(mc.get('grad_checkpointing', False)):
                    if rank == 0:
                        print("🔁 Enabling gradient checkpointing after OOM")
                    mc['grad_checkpointing'] = True
                    base_model = model.module if hasattr(model, 'module') else model
                    for lyr in base_model.layers:
                        lyr.gradient_checkpointing = True
                train_loader = registry.create_dataloader(
                    tokenizer=tokenizer,
                    config=data_config,
                    batch_size=base_batch_size,
                    use_sized=False,
                    num_workers=num_workers_dl,
                    prefetch_factor=int(tc.get('prefetch_factor', 2)),
                    persistent_workers=(num_workers_dl > 0)
                )
                if rank == 0:
                    print("↻ Restarting batch loop after OOM adjustments")
                continue
            else:
                raise
                
                # Calculate tokens/sec using wall-clock time
                tokens_per_sec = tokens_processed / (current_time - start_time) if current_time > start_time else 0
                
                # GPU memory usage for monitoring
                gpu_mem = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_max = torch.cuda.max_memory_allocated() / 1024**3  # GB

                cur_lr = optimizer.param_groups[0]['lr']
                print(f"Step {step:,} | LR: {cur_lr:.2e} | Loss: {avg_loss:.3f} | Tokens: {tokens_processed:,} | "
                      f"Tokens/sec: {tokens_per_sec:.0f} | GPU: {gpu_mem:.1f}/{gpu_max:.1f}GB")

                # track CE trend
                recent_ce_log.append(avg_loss)
                if len(recent_ce_log) > 20:
                    recent_ce_log.pop(0)

                # Check for nan loss
                if math.isnan(avg_loss):
                    print("⚠️  NaN loss detected - stopping training")
                    break

                last_log_time = current_time
            # Adaptive source reweighting (token interval based)
            if curriculum_cfg['enable_adaptive_sources'] and (tokens_processed - last_adapt_tokens) >= curriculum_cfg['adapt_interval_tokens']:
                if sum(source_token_count) > 0 and len(source_token_count) > 0:
                    avg_losses = [source_token_loss_sum[i] / max(1, source_token_count[i]) if source_token_count[i] > 0 else 0.0 for i in range(len(source_token_count))]
                    max_loss = max(avg_losses) if avg_losses else 0.0
                    if max_loss > 0:
                        normalized = [l / max_loss for l in avg_losses]
                        inverted = [1.0 + (1.0 - n) for n in normalized]
                        min_w = curriculum_cfg['min_weight']; max_w = curriculum_cfg['max_weight']
                        new_w = [min(max(v, min_w), max_w) for v in inverted]
                        mean_w = sum(new_w)/len(new_w)
                        new_w = [w/mean_w for w in new_w]
                        ds_obj = train_loader.dataset
                        if getattr(ds_obj, 'dynamic_weights', None):
                            sm = curriculum_cfg['smoothing']
                            new_w = [sm*old + (1-sm)*nw for old, nw in zip(ds_obj.dynamic_weights, new_w)]
                        ds_obj.update_dynamic_weights(new_w)
                        if rank == 0:
                            print(f"↺ Adaptive source weights: {[f'{w:.2f}' for w in new_w]}")
                last_adapt_tokens = tokens_processed
            
            # Run evaluation every 500 steps (less frequent for speed)
            if step % 500 == 0 and step > 0 and rank == 0:
                print(f"\n📊 Running evaluation at step {step}...")
                eval_model = model.module if hasattr(model, 'module') else model
                eval_ce, eval_ppl = evaluate_on_dataset(
                    eval_model, tokenizer, device, registry, data_config, per_process_batch=max(1, base_batch_size//2)
                )
                if eval_ce is not None:
                    train_ce_recent = accumulated_loss / max(1, accumulated_tokens)
                    gap = eval_ce - train_ce_recent
                    print(f"   eval_ce: {eval_ce:.4f} | eval_ppl: {eval_ppl:.2f} | train→eval gap: {gap:+.4f}")
                    
                    # Plateau rule: Track CE improvement per 10M token windows
                    plateau_ce_history.append((tokens_processed, eval_ce))
                    
                    # Check if we have enough history to evaluate plateau (need at least 2 windows)
                    if len(plateau_ce_history) >= 2:
                        # Find evaluations that are at least 10M tokens apart
                        current_tokens = tokens_processed
                        windows_to_check = []
                        
                        for i in range(len(plateau_ce_history) - 1, -1, -1):
                            eval_tokens, eval_ce_val = plateau_ce_history[i]
                            if current_tokens - eval_tokens >= plateau_window_size:
                                windows_to_check.append((eval_tokens, eval_ce_val))
                                if len(windows_to_check) >= 2:
                                    break
                        
                        # Check improvement between consecutive 10M token windows
                        if len(windows_to_check) >= 2:
                            # Most recent window vs previous window (reverse order due to how we collected)
                            prev_ce = windows_to_check[0][1]  # Earlier evaluation
                            curr_ce = eval_ce  # Current evaluation
                            improvement = prev_ce - curr_ce  # Positive = improvement (lower is better)
                            
                            print(f"   🔍 Plateau check: CE improvement over 10M tokens: {improvement:.4f}")
                            
                            if improvement < plateau_improvement_threshold:
                                plateau_consecutive_windows += 1
                                print(f"   ⚠️  Plateau detected! Consecutive low-improvement windows: {plateau_consecutive_windows}")
                                
                                # Trigger controlled change after 2 consecutive low-improvement windows
                                if plateau_consecutive_windows >= 2 and plateau_controlled_changes_made < 3:  # Limit changes
                                    plateau_controlled_changes_made += 1
                                    print(f"   🎛️  PLATEAU RULE TRIGGERED! Making controlled change #{plateau_controlled_changes_made}")
                                    
                                    # Controlled change: Reduce learning rate by 50%
                                    for param_group in optimizer.param_groups:
                                        old_lr = param_group['lr']
                                        param_group['lr'] *= 0.5
                                        print(f"   📉 Learning rate reduced: {old_lr:.2e} → {param_group['lr']:.2e}")
                                    
                                    # Reset consecutive window counter after making a change
                                    plateau_consecutive_windows = 0
                            else:
                                # Reset counter if we see good improvement
                                plateau_consecutive_windows = 0
                                print(f"   ✅ Good improvement detected, plateau counter reset")
                    
                else:
                    print("   Eval loader produced 0 tokens; skipping metrics")
                generate_samples(eval_model, tokenizer, device)
                print()
            
            accumulated_loss = 0
            accumulated_tokens = 0
        
        # No dynamic sequence length switching
            # Completed for-loop without OOM
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                oom_retries += 1
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"💥 CUDA OOM encountered (retry {oom_retries}/{max_oom_retries}). Adjusting configuration...")
                if oom_retries > max_oom_retries:
                    if rank == 0:
                        print("❌ Exceeded maximum OOM retries. Aborting.")
                    raise
                # Halve per-GPU batch size
                if base_batch_size > 1:
                    new_bs = max(1, base_batch_size // 2)
                    if new_bs != base_batch_size and rank == 0:
                        print(f"🔧 Reducing per-GPU batch size {base_batch_size} → {new_bs}")
                    base_batch_size = new_bs
                # Enable gradient checkpointing after first OOM
                if not bool(mc.get('grad_checkpointing', False)):
                    if rank == 0:
                        print("🔁 Enabling gradient checkpointing after OOM")
                    mc['grad_checkpointing'] = True
                    # Re-wrap layers for checkpointing
                    base_model = model.module if hasattr(model, 'module') else model
                    for lyr in base_model.layers:
                        lyr.gradient_checkpointing = True
                # Rebuild dataloader with new batch size
                train_loader = registry.create_dataloader(
                    tokenizer=tokenizer,
                    config=data_config,
                    batch_size=base_batch_size,
                    use_sized=False,
                    num_workers=num_workers_dl,
                    prefetch_factor=int(tc.get('prefetch_factor', 2)),
                    persistent_workers=(num_workers_dl > 0)
                )
                if rank == 0:
                    print("↻ Restarting batch loop after OOM adjustments")
                continue  # Retry while True
            else:
                raise
    
    if rank == 0:
        total_time = time.time() - start_time
        final_tokens_per_sec = tokens_processed / total_time if total_time > 0 else 0
        
        print(f"🎉 Training complete! {step:,} steps, {tokens_processed:,} tokens")
        print(f"⏱️  Total time: {total_time:.1f}s, Final tokens/sec: {final_tokens_per_sec:.0f}")

        if enable_hybrid_memory and hybrid_memory is not None:
            num_entities = len(hybrid_memory.graph.entities)
            num_relations = len(hybrid_memory.graph.relations)
            num_edges = len(hybrid_memory.graph.edge_heads)
            print("\n🧠 Hybrid memory summary:")
            print(f"   Entities: {num_entities} | Relations: {num_relations} | Facts stored: {num_edges}")
            if total_facts_ingested:
                top_rel = fact_relation_counter.most_common(5)
                rel_desc = ', '.join(f"{name}:{count}" for name, count in top_rel)
                print(f"   New facts ingested during training: {total_facts_ingested} ({rel_desc})")
            sample_entity = hybrid_memory.graph.entities[0].name if hybrid_memory.graph.entities else None
            if sample_entity:
                sample_query = hybrid_memory.query(sample_entity, topk_neighbors=3, topk_passages=2)
                print(f"   Sample query [{sample_entity}]: {sample_query}")

        # Final evaluation
        print(f"\n📊 Final evaluation:")
        eval_model = model.module if hasattr(model, 'module') else model
        eval_ce, eval_ppl = evaluate_on_dataset(eval_model, tokenizer, device, registry, data_config, per_process_batch=max(1, base_batch_size//2))
        if eval_ce is not None:
                    print(f"   eval_ce: {eval_ce:.4f} | eval_ppl: {eval_ppl:.2f}")
        generate_samples(eval_model, tokenizer, device)
        
    # Save model (handle DataParallel/DDP wrapper) with rolling snapshots
    _ensure_dir(Path('checkpoints'))
    model_to_save = model.module if hasattr(model, 'module') else model
    tmp_path = Path('checkpoints') / 'lm_tmp.pth'
    torch.save(model_to_save.state_dict(), tmp_path)
    _rotate_checkpoints(Path('checkpoints') / 'lm_current.pth', tmp_path)
    # Also keep legacy single-file for compatibility
    torch.save(model_to_save.state_dict(), 'simple_lm_trained.pth')
    print("✓ Model saved to checkpoints/lm_current.pth (rotated prev, prev2) and simple_lm_trained.pth")
        # Optional JSON log disabled to avoid runtime params; could write fixed filename if desired.
    
    # Clean up DDP
    if use_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
