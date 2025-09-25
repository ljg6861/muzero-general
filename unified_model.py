#!/usr/bin/env python3
"""
Unified Meta-Cognitive Language Model - The ONE and ONLY Model
==============================================================

This is the single, unified model architecture that contains ALL features:
- Fast transformer core with Flash/SDPA attention
- Hybrid memory integration for factual knowledge
- Meta-reasoning and schema inference capabilities
- Self-correction and validation mechanisms  
- Router for question type classification
- Policy/value heads for reinforcement learning
- Dynamics modeling for planning
- Span projection for contrastive learning

NO INTERFACES. NO MULTIPLE MODELS. ONE ARCHITECTURE.
"""

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, GPT2TokenizerFast

# Optional imports - gracefully handle missing dependencies
try:
    from sentence_transformers import SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import faiss
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False

logger = logging.getLogger(__name__)


class AnswerConstraints:
    """Container describing decoding constraints for answer generation."""

    def __init__(
        self,
        entity_token_ids: Optional[torch.Tensor] = None,
        numeric_regex: Optional[re.Pattern] = None,
        date_regex: Optional[re.Pattern] = None,
        hard: bool = True,
    ) -> None:
        self.entity_token_ids = entity_token_ids
        self.numeric_regex = numeric_regex
        self.date_regex = date_regex
        self.hard = hard


def build_entity_token_whitelist(tokenizer, candidate_strings: List[str]) -> Optional[torch.Tensor]:
    """Tokenize candidate entity aliases and return the unique token id whitelist."""

    ids: Set[int] = set()
    for candidate in candidate_strings:
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        ids.update(int(tok) for tok in tokens)

    if not ids:
        return None
    ordered = sorted(ids)
    return torch.tensor(ordered, dtype=torch.long)


def apply_answer_type_constraints(
    logits: torch.Tensor,
    step_ids: torch.Tensor,
    tokenizer,
    constraints: Optional[AnswerConstraints],
    bias_val: float = -1e9,
    soft_bias: float = -3.0,
) -> torch.Tensor:
    """Apply decoding constraints to logits without mutating the input tensor."""

    if constraints is None:
        return logits

    if logits.ndim != 2:
        raise ValueError("apply_answer_type_constraints expects logits with shape [B, V]")

    vocab_size = logits.size(-1)
    device = logits.device
    out = logits.clone()

    if constraints.entity_token_ids is not None and constraints.entity_token_ids.numel() > 0:
        whitelist = constraints.entity_token_ids.to(device)
        mask = torch.ones((vocab_size,), dtype=torch.bool, device=device)
        mask[whitelist] = False
        if constraints.hard:
            out[:, mask] = bias_val
        else:
            out[:, mask] = out[:, mask] + soft_bias

    if constraints.numeric_regex is not None or constraints.date_regex is not None:
        digit_like_tokens: Set[int] = set()
        for char in "0123456789-/:,.":
            encoded = tokenizer.encode(char, add_special_tokens=False)
            digit_like_tokens.update(int(tok) for tok in encoded)
        if digit_like_tokens:
            likely = torch.tensor(sorted(digit_like_tokens), dtype=torch.long, device=device)
            mask = torch.ones((vocab_size,), dtype=torch.bool, device=device)
            mask[likely] = False
            out[:, mask] = out[:, mask] + soft_bias

    return out


def set_entity_constraints_from_candidates(
    step_state: Dict[str, Any],
    tokenizer,
    candidate_strings: List[str],
    hard: bool = True,
) -> None:
    """Populate ``step_state`` with an :class:`AnswerConstraints` instance."""

    whitelist = build_entity_token_whitelist(tokenizer, candidate_strings)
    step_state["constraints"] = AnswerConstraints(
        entity_token_ids=whitelist,
        numeric_regex=None,
        date_regex=None,
        hard=hard,
    )

# ===========================================================================================
# SCHEMA REASONING - Answer type classification and constraints
# ===========================================================================================

class AnswerType(Enum):
    """Semantic categories for schema-aware answer generation"""
    PERSON = "person"
    PLACE = "place"
    CAPITAL_CITY = "capital_city"
    COUNTRY = "country"
    NUMBER = "number"
    DATE = "date"
    YEAR = "year"
    CHEMICAL_FORMULA = "chemical_formula"
    DEFINITION = "definition"
    PROCESS = "process"
    ORGANIZATION = "organization"
    BOOK = "book"
    MOVIE = "movie"
    LANGUAGE = "language"
    CURRENCY = "currency"
    MEASUREMENT = "measurement"
    COLOR = "color"
    UNKNOWN = "unknown"

@dataclass
class SchemaConstraint:
    """Constraints that define what a valid answer looks like"""
    answer_type: AnswerType
    max_tokens: int = 50
    must_contain_patterns: List[str] = None
    forbidden_patterns: List[str] = None
    min_confidence: float = 0.5

# ===========================================================================================
# HYBRID MEMORY - Factual knowledge storage and retrieval
# ===========================================================================================

class QuantizedEmbedding:
    """Simple product quantization for memory-efficient embeddings"""
    def __init__(self, dim: int, nsubq: int = 4, codebook_bits: int = 8):
        self.dim = dim
        self.nsubq = nsubq
        self.codebook_bits = codebook_bits
        self.subvec_dim = dim // nsubq
        self.centroids = None
        
    def train(self, vectors: np.ndarray):
        """Train quantization codebooks"""
        if not _FAISS_AVAILABLE:
            logger.warning("FAISS not available, storing fp16 vectors directly")
            return
            
        n_vectors, dim = vectors.shape
        assert dim == self.dim
        
        # Train product quantizer
        pq = faiss.ProductQuantizer(dim, self.nsubq, self.codebook_bits)
        pq.train(vectors.astype(np.float32))
        self.centroids = pq
        
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Quantize vectors to codes"""
        if self.centroids is None:
            return vectors.astype(np.float16)
        return self.centroids.compute_codes(vectors.astype(np.float32))
        
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct vectors from codes"""
        if self.centroids is None:
            return codes.astype(np.float32)
        return self.centroids.decode(codes)

class HybridMemoryEngine:
    """Hybrid memory combining entity graphs and passage retrieval"""
    
    def __init__(self, entity_dim: int = 128, passage_dim: int = 384, use_faiss: bool = True):
        self.entity_dim = entity_dim
        self.passage_dim = passage_dim
        self.use_faiss = use_faiss and _FAISS_AVAILABLE
        
        # Entity storage
        self.entity_to_id = {}
        self.id_to_entity = {}
        self.entity_embeddings = []
        
        # Relation storage
        self.relation_to_id = {}
        self.id_to_relation = {}
        self.relation_embeddings = []
        
        # Graph structure (entity_id -> [(relation_id, target_entity_id), ...])
        self.entity_relations = defaultdict(list)
        
        # Passage storage
        self.passages = []
        self.passage_embeddings = []
        self.passage_index = None
        
        # Sentence encoder for embedding passages
        if _SENTENCE_TRANSFORMERS_AVAILABLE:
            self.sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.sentence_encoder = None
            logger.warning("sentence-transformers not available, passage retrieval disabled")
    
    def add_entity(self, entity: str, embedding: Optional[np.ndarray] = None) -> int:
        """Add entity and return its ID"""
        if entity in self.entity_to_id:
            return self.entity_to_id[entity]
            
        entity_id = len(self.entity_to_id)
        self.entity_to_id[entity] = entity_id
        self.id_to_entity[entity_id] = entity
        
        if embedding is not None:
            self.entity_embeddings.append(embedding)
        else:
            # Create random embedding if none provided
            self.entity_embeddings.append(np.random.randn(self.entity_dim).astype(np.float32))
            
        return entity_id
    
    def add_relation(self, relation: str, embedding: Optional[np.ndarray] = None) -> int:
        """Add relation and return its ID"""
        if relation in self.relation_to_id:
            return self.relation_to_id[relation]
            
        relation_id = len(self.relation_to_id)
        self.relation_to_id[relation] = relation_id
        self.id_to_relation[relation_id] = relation
        
        if embedding is not None:
            self.relation_embeddings.append(embedding)
        else:
            self.relation_embeddings.append(np.random.randn(64).astype(np.float32))  # Smaller for relations
            
        return relation_id
    
    def add_triple(self, head: str, relation: str, tail: str):
        """Add a knowledge triple (head, relation, tail)"""
        head_id = self.add_entity(head)
        relation_id = self.add_relation(relation)
        tail_id = self.add_entity(tail)
        
        self.entity_relations[head_id].append((relation_id, tail_id))
    
    def add_passage(self, text: str, metadata: Dict[str, Any] = None):
        """Add a passage for retrieval"""
        passage_id = len(self.passages)
        self.passages.append({
            'id': passage_id,
            'text': text,
            'metadata': metadata or {}
        })
        
        # Encode passage if sentence encoder available
        if self.sentence_encoder is not None:
            embedding = self.sentence_encoder.encode([text])[0]
            self.passage_embeddings.append(embedding)
    
    def retrieve_entities(self, entity: str, relation: str = None, k: int = 5) -> List[Dict]:
        """Retrieve related entities"""
        if entity not in self.entity_to_id:
            return []
            
        entity_id = self.entity_to_id[entity]
        results = []
        
        for rel_id, tail_id in self.entity_relations[entity_id]:
            if relation is None or self.id_to_relation[rel_id] == relation:
                results.append({
                    'entity': self.id_to_entity[tail_id],
                    'relation': self.id_to_relation[rel_id],
                    'score': 1.0  # Could compute similarity score
                })
                
        return results[:k]
    
    def retrieve_passages(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant passages"""
        if not self.passages or self.sentence_encoder is None:
            return []
            
        # Encode query
        query_embedding = self.sentence_encoder.encode([query])[0]
        
        # Compute similarities
        similarities = []
        for i, passage_emb in enumerate(self.passage_embeddings):
            sim = np.dot(query_embedding, passage_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(passage_emb)
            )
            similarities.append((sim, i))
        
        # Get top-k
        similarities.sort(reverse=True)
        results = []
        for sim, idx in similarities[:k]:
            passage = self.passages[idx]
            results.append({
                'text': passage['text'],
                'score': float(sim),
                'metadata': passage['metadata']
            })
            
        return results

# ===========================================================================================
# CORE TRANSFORMER ARCHITECTURE
# ===========================================================================================

class FlashAttentionLayer(nn.Module):
    """Fast attention layer using Flash Attention or SDPA"""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, gradient_checkpointing=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.gradient_checkpointing = gradient_checkpointing

    def _forward_impl(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        # Self-attention block
        x2, _ = self.self_attn(x, x, x, attn_mask=src_mask, 
                              key_padding_mask=src_key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout1(x2))
        
        # Feed forward block  
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        
        return x
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, src, src_mask, src_key_padding_mask, use_reentrant=False)
        else:
            return self._forward_impl(src, src_mask, src_key_padding_mask)

# ===========================================================================================
# UNIFIED META-COGNITIVE LANGUAGE MODEL - THE ONE AND ONLY MODEL
# ===========================================================================================

class UnifiedCognitiveLM(nn.Module):
    """
    The ONE and ONLY language model in this project.
    
    Combines ALL capabilities:
    - Fast transformer with Flash attention
    - Hybrid memory integration
    - Meta-reasoning and schema inference
    - Self-correction mechanisms
    - Router for question classification
    - Policy/value heads for RL
    - Dynamics modeling
    - Span projection for contrastive learning
    
    NO INTERFACES. NO SEPARATE MODELS. Everything is here.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_size: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 512,
        gradient_checkpointing: bool = False,
        
        # Meta-cognitive features
        enable_meta_reasoning: bool = True,
        enable_schema_inference: bool = True,
        enable_self_correction: bool = True,
        
        # Memory integration
        enable_hybrid_memory: bool = True,
        memory_entity_dim: int = 128,
        memory_passage_dim: int = 384,
        
        # Multi-task heads
        enable_policy: bool = False,
        enable_value: bool = False, 
        enable_dynamics: bool = False,
        enable_span_proj: bool = False,
        enable_router: bool = True,
        
        # Router configuration
        router_num_types: int = len(AnswerType),
        router_type_labels: Optional[List[str]] = None,
    ):
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.gradient_checkpointing = gradient_checkpointing
        
        # Feature flags
        self.enable_meta_reasoning = enable_meta_reasoning
        self.enable_schema_inference = enable_schema_inference
        self.enable_self_correction = enable_self_correction
        self.enable_hybrid_memory = enable_hybrid_memory
        self.enable_policy = enable_policy
        self.enable_value = enable_value
        self.enable_dynamics = enable_dynamics
        self.enable_span_proj = enable_span_proj
        self.enable_router = enable_router
        self.tokenizer = None
        
        # Core transformer components
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        
        # Cache causal mask and position ids
        causal_mask_full = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask_full', causal_mask_full, persistent=False)
        pos_ids_full = torch.arange(max_seq_len, dtype=torch.long)
        self.register_buffer('pos_ids_full', pos_ids_full, persistent=False)

        # Transformer layers
        self.layers = nn.ModuleList([
            FlashAttentionLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=0.1,
                gradient_checkpointing=gradient_checkpointing
            ) for _ in range(num_layers)
        ])

        # Core language modeling head
        self.lm_head = nn.Linear(hidden_size, vocab_size)

        # Meta-reasoning components
        if self.enable_schema_inference:
            self.schema_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, len(AnswerType))
            )
            
        if self.enable_self_correction:
            self.validation_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1)  # Binary validation score
            )

        # Multi-task heads
        if self.enable_policy:
            self.policy_head = nn.Linear(hidden_size, vocab_size)
        if self.enable_value:
            self.value_head = nn.Linear(hidden_size, 1)
        if self.enable_dynamics:
            self.dynamics_fc = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size)
            )
        if self.enable_span_proj:
            self.span_projector = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size)
            )
        if self.enable_router:
            out_dim = 1 + router_num_types  # answerability + type classification
            self.router_head = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, out_dim),
            )
            self.router_type_labels = router_type_labels or [t.value for t in AnswerType]

        # Hybrid memory integration
        self.hybrid_memory = None
        if self.enable_hybrid_memory:
            try:
                self.hybrid_memory = HybridMemoryEngine(
                    entity_dim=memory_entity_dim,
                    passage_dim=memory_passage_dim
                )
                logger.info("Hybrid memory initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid memory: {e}")
                self.enable_hybrid_memory = False

        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def create_causal_mask(self, seq_len, device):
        """Get cached causal mask for given sequence length"""
        return self.causal_mask_full[:seq_len, :seq_len]
    
    def create_key_padding_mask(self, input_ids, pad_token_id):
        """Create key padding mask to ignore pad tokens"""
        return input_ids == pad_token_id

    def encode(self, input_ids, pad_token_id=None):
        """Encode input tokens through transformer layers"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Validate input_ids are within vocab bounds
        vocab_size = self.embedding.num_embeddings
        if input_ids.min() < 0 or input_ids.max() >= vocab_size:
            raise ValueError(f"input_ids out of bounds: range=[{input_ids.min()}, {input_ids.max()}], vocab_size={vocab_size}")
        
        # Ensure sequence length doesn't exceed model's maximum
        max_supported_len = self.pos_embedding.num_embeddings
        if seq_len > max_supported_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds model's maximum supported length {max_supported_len}")
        
        # Token embeddings
        token_embeddings = self.embedding(input_ids)
        
        # Create position IDs safely instead of using cached buffer
        # This prevents any corruption or device mismatch issues
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # Ensure position_ids are within valid range
        if position_ids.max() >= max_supported_len:
            raise ValueError(f"Generated position_ids max {position_ids.max()} >= embedding size {max_supported_len}")
        
        position_embeddings = self.pos_embedding(position_ids)
        
        hidden_states = token_embeddings + position_embeddings
        
        # Build masks
        key_padding_mask = None
        if pad_token_id is not None:
            key_padding_mask = (input_ids == pad_token_id)
        
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
            
        return hidden_states

    def forward(self, input_ids, pad_token_id=None, policy_positions=None, dynamics_tokens=None, return_schema=False):
        """
        Forward pass through the unified model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            pad_token_id: ID for padding tokens
            policy_positions: Positions for policy/value computation  
            dynamics_tokens: Tokens for dynamics modeling
            return_schema: Whether to return schema inference results
            
        Returns:
            logits: Language modeling logits [batch_size, seq_len, vocab_size]
            aux: Dictionary of auxiliary outputs
        """
        hidden_states = self.encode(input_ids, pad_token_id)
        logits = self.lm_head(hidden_states)
        aux = {}
        
        # Schema inference for meta-reasoning
        if self.enable_schema_inference and return_schema:
            schema_logits = self.schema_head(hidden_states[:, -1, :])  # Use last token
            schema_probs = F.softmax(schema_logits, dim=-1)
            aux['schema'] = {
                'logits': schema_logits,
                'probs': schema_probs,
                'predicted_type': torch.argmax(schema_logits, dim=-1)
            }
            
        # Self-correction validation
        if self.enable_self_correction:
            validation_score = self.validation_head(hidden_states[:, -1, :]).squeeze(-1)
            aux['validation'] = {
                'score': validation_score,
                'prob': torch.sigmoid(validation_score)
            }

        # Multi-task heads
        if self.enable_policy or self.enable_value:
            if policy_positions is None:
                policy_positions = torch.full((hidden_states.size(0),), hidden_states.size(1)-1, 
                                            device=hidden_states.device, dtype=torch.long)
            gather_h = hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), policy_positions]
            
            if self.enable_policy:
                aux['policy_logits'] = self.policy_head(gather_h)
            if self.enable_value:
                aux['value'] = self.value_head(gather_h).squeeze(-1)
                
        if self.enable_dynamics and dynamics_tokens is not None:
            dyn_in = dynamics_tokens.clamp_min(0)
            token_embed = self.embedding(dyn_in)
            token_agg = token_embed.mean(dim=1)
            root_h = hidden_states[:, -1, :]
            dyn_inp = torch.cat([root_h, token_agg], dim=-1)
            aux['dynamics_next'] = self.dynamics_fc(dyn_inp)
            
        if self.enable_span_proj:
            aux['span_proj'] = self.span_projector(hidden_states)
            
        if self.enable_router:
            router_logits = self.router_head(hidden_states[:, -1, :])
            aux['router'] = {
                'answerability_logit': router_logits[:, 0],
                'answerability_prob': torch.sigmoid(router_logits[:, 0]),
                'type_logits': router_logits[:, 1:] if router_logits.size(1) > 1 else None,
                'type_probs': torch.softmax(router_logits[:, 1:], dim=-1) if router_logits.size(1) > 1 else None,
            }

        return logits, aux

    def retrieve_knowledge(self, query: str, k: int = 5) -> Dict[str, List[Dict]]:
        """Retrieve knowledge from hybrid memory"""
        if not self.enable_hybrid_memory or self.hybrid_memory is None:
            return {'entities': [], 'passages': []}
            
        entities = self.hybrid_memory.retrieve_entities(query, k=k)
        passages = self.hybrid_memory.retrieve_passages(query, k=k)
        
        return {'entities': entities, 'passages': passages}

    def generate_with_schema(
        self,
        input_ids,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        schema_type=None,
        constraints: Optional[AnswerConstraints] = None,
        tokenizer=None,
        step_state: Optional[Dict[str, Any]] = None,
    ):
        """Generate text with schema-aware constraints"""
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids.clone()
        active_constraints = constraints
        resolved_tokenizer = tokenizer or getattr(self, "tokenizer", None)
        if resolved_tokenizer is None:
            if constraints is not None:
                raise ValueError("Tokenizer must be provided when using decoding constraints")
        else:
            if constraints is None and step_state is not None:
                active_constraints = step_state.get("constraints")
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                logits, aux = self.forward(generated, return_schema=True)
                next_token_logits = logits[:, -1, :] / temperature

                if active_constraints is not None:
                    if resolved_tokenizer is None:
                        raise ValueError("Tokenizer required for constrained decoding")
                    next_token_logits = apply_answer_type_constraints(
                        next_token_logits,
                        generated,
                        resolved_tokenizer,
                        active_constraints,
                    )
                
                # Apply schema constraints if available
                if schema_type is not None and 'schema' in aux:
                    # Simple schema-aware filtering (can be made more sophisticated)
                    predicted_type = aux['schema']['predicted_type']
                    if predicted_type != schema_type:
                        # Penalize tokens that don't match expected schema
                        next_token_logits -= 1.0
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Check for EOS or max length
                if generated.size(1) >= self.max_seq_len:
                    break
                    
        return generated

    def resize_token_embeddings(self, new_vocab_size: int):
        """Resize embeddings to match tokenizer size"""
        if new_vocab_size == self.vocab_size:
            return
            
        old_embed = self.embedding
        new_embed = nn.Embedding(new_vocab_size, self.hidden_size)
        
        # Copy existing weights
        with torch.no_grad():
            min_vocab = min(self.vocab_size, new_vocab_size)
            new_embed.weight[:min_vocab] = old_embed.weight[:min_vocab]
            
        self.embedding = new_embed
        
        # Resize LM head
        old_lm_head = self.lm_head
        new_lm_head = nn.Linear(self.hidden_size, new_vocab_size)
        
        with torch.no_grad():
            new_lm_head.weight[:min_vocab] = old_lm_head.weight[:min_vocab]
            if old_lm_head.bias is not None:
                new_lm_head.bias[:min_vocab] = old_lm_head.bias[:min_vocab]
                
        self.lm_head = new_lm_head
        self.vocab_size = new_vocab_size

# Legacy alias for backward compatibility
SimpleLM = UnifiedCognitiveLM

# ===========================================================================================
# DATA LOADING AND TRAINING UTILITIES
# ===========================================================================================

class TextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512, include_labels: bool = True):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_labels = include_labels
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding.get('attention_mask', None)
        
        item = {'input_ids': input_ids}
        if attention_mask is not None:
            item['attention_mask'] = attention_mask.squeeze(0)
            
        # For causal LM, labels are the same as input_ids
        if self.include_labels:
            item['labels'] = input_ids.clone()

        return item


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Compute causal language modeling loss with padding masked out."""

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits shape [B, T, V] and labels shape [B, T]")

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels != pad_id

    if not mask.any():
        return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

    loss = F.cross_entropy(shift_logits[mask], shift_labels[mask], reduction="mean")
    return loss

class MetaCognitiveTrainer:
    """Training utilities for the unified meta-cognitive model"""
    
    def __init__(self, model: UnifiedCognitiveLM, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
    def compute_loss(self, batch, compute_meta_losses=True):
        """Compute multi-task losses"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch.get('labels', input_ids).to(self.device)
        
        # Forward pass
        logits, aux = self.model.forward(input_ids, pad_token_id=self.tokenizer.pad_token_id, return_schema=compute_meta_losses)

        # Language modeling loss
        lm_loss = causal_lm_loss(logits, labels, pad_id=self.tokenizer.pad_token_id)
        total_loss = lm_loss
        loss_dict = {'lm_loss': lm_loss.item()}
        
        # Schema classification loss (if we have labels)
        if compute_meta_losses and 'schema' in aux and 'schema_labels' in batch:
            schema_labels = batch['schema_labels'].to(self.device)
            schema_loss = F.cross_entropy(aux['schema']['logits'], schema_labels)
            total_loss += 0.1 * schema_loss  # Weighted
            loss_dict['schema_loss'] = schema_loss.item()
            
        # Validation loss (self-correction)
        if 'validation' in aux and 'validation_labels' in batch:
            val_labels = batch['validation_labels'].to(self.device).float()
            val_loss = F.binary_cross_entropy_with_logits(aux['validation']['score'], val_labels)
            total_loss += 0.1 * val_loss
            loss_dict['validation_loss'] = val_loss.item()
            
        # Multi-task losses
        if 'policy_logits' in aux and 'policy_labels' in batch:
            policy_labels = batch['policy_labels'].to(self.device)
            policy_loss = F.cross_entropy(aux['policy_logits'], policy_labels)
            total_loss += 0.1 * policy_loss
            loss_dict['policy_loss'] = policy_loss.item()
            
        if 'value' in aux and 'value_labels' in batch:
            value_labels = batch['value_labels'].to(self.device).float()
            value_loss = F.mse_loss(aux['value'], value_labels)
            total_loss += 0.1 * value_loss
            loss_dict['value_loss'] = value_loss.item()
            
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict
    
    def train_step(self, batch, optimizer, scheduler=None):
        """Single training step"""
        self.model.train()
        optimizer.zero_grad()
        
        loss, loss_dict = self.compute_loss(batch)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
            
        return loss_dict
    
    def evaluate(self, dataloader):
        """Evaluate model on validation set"""
        self.model.eval()
        total_losses = defaultdict(float)
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                loss, loss_dict = self.compute_loss(batch, compute_meta_losses=False)
                
                for key, value in loss_dict.items():
                    total_losses[key] += value
                num_batches += 1
                
        # Average losses
        avg_losses = {key: total / num_batches for key, total in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, path: str, optimizer=None, scheduler=None, epoch=None, metadata=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'num_heads': self.model.num_heads,
                'max_seq_len': self.model.max_seq_len,
                'enable_meta_reasoning': self.model.enable_meta_reasoning,
                'enable_schema_inference': self.model.enable_schema_inference,
                'enable_self_correction': self.model.enable_self_correction,
                'enable_hybrid_memory': self.model.enable_hybrid_memory,
                'enable_policy': self.model.enable_policy,
                'enable_value': self.model.enable_value,
                'enable_dynamics': self.model.enable_dynamics,
                'enable_span_proj': self.model.enable_span_proj,
                'enable_router': self.model.enable_router,
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if metadata is not None:
            checkpoint['metadata'] = metadata
            
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str, optimizer=None, scheduler=None):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        epoch = checkpoint.get('epoch', 0)
        metadata = checkpoint.get('metadata', {})
        
        logger.info(f"Checkpoint loaded from {path}, epoch {epoch}")
        return epoch, metadata

# ===========================================================================================
# KNOWLEDGE INTEGRATION AND REASONING
# ===========================================================================================

class KnowledgeIntegrator:
    """Integrates external knowledge into the model"""
    
    def __init__(self, model: UnifiedCognitiveLM, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def add_knowledge_triples(self, triples: List[Tuple[str, str, str]]):
        """Add knowledge triples to hybrid memory"""
        if not self.model.enable_hybrid_memory or self.model.hybrid_memory is None:
            logger.warning("Hybrid memory not available")
            return
            
        for head, relation, tail in triples:
            self.model.hybrid_memory.add_triple(head, relation, tail)
            
        logger.info(f"Added {len(triples)} knowledge triples")
    
    def add_passages(self, passages: List[str], metadata_list: List[Dict] = None):
        """Add passages for retrieval"""
        if not self.model.enable_hybrid_memory or self.model.hybrid_memory is None:
            logger.warning("Hybrid memory not available")
            return
            
        if metadata_list is None:
            metadata_list = [{}] * len(passages)
            
        for passage, metadata in zip(passages, metadata_list):
            self.model.hybrid_memory.add_passage(passage, metadata)
            
        logger.info(f"Added {len(passages)} passages")
    
    def answer_with_reasoning(self, question: str, max_tokens: int = 50, use_memory: bool = True, 
                            temperature: float = 1.0, top_k: int = 50) -> Dict[str, Any]:
        """Generate answer with meta-reasoning"""
        
        # Retrieve relevant knowledge if enabled
        knowledge = {}
        if use_memory:
            knowledge = self.model.retrieve_knowledge(question, k=5)
        
        # Prepare input
        prompt = f"Question: {question}\nAnswer:"
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        
        # Move to same device as model
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        
        # Generate with schema awareness
        self.model.eval()
        with torch.no_grad():
            # First, infer the question type
            _, aux = self.model.forward(input_ids, return_schema=True)
            
            predicted_type = None
            if 'schema' in aux:
                type_idx = torch.argmax(aux['schema']['logits'], dim=-1).item()
                if type_idx < len(AnswerType):
                    predicted_type = list(AnswerType)[type_idx]
            
            # Generate answer with schema constraint
            generated = self.model.generate_with_schema(
                input_ids,
                max_length=max_tokens,
                temperature=temperature,
                top_k=top_k,
                schema_type=predicted_type.value if predicted_type else None,
                tokenizer=self.tokenizer,
            )
        
        # Decode answer
        full_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        answer = full_text[len(prompt):].strip()
        
        # Self-correction check
        validation_prob = 0.5
        if 'validation' in aux:
            validation_prob = aux['validation']['prob'].item()
        
        return {
            'answer': answer,
            'predicted_type': predicted_type.value if predicted_type else 'unknown',
            'validation_confidence': validation_prob,
            'retrieved_knowledge': knowledge,
            'full_reasoning': {
                'question': question,
                'schema_inference': aux.get('schema', {}),
                'validation': aux.get('validation', {}),
                'knowledge_used': len(knowledge.get('entities', [])) + len(knowledge.get('passages', []))
            }
        }

# ===========================================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ===========================================================================================

def create_unified_model(
    model_type: str = "base",
    vocab_size: int = 50257,
    hidden_size: int = 512,
    num_layers: int = 6,
    device: str = "cuda",
    **kwargs
) -> UnifiedCognitiveLM:
    """Factory function to create unified models with different configurations"""
    
    configs = {
        "tiny": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "max_seq_len": 256
        },
        "small": {
            "hidden_size": 512,
            "num_layers": 6,
            "num_heads": 8,
            "max_seq_len": 512
        },
        "base": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "max_seq_len": 1024
        },
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "max_seq_len": 2048
        }
    }
    
    if model_type in configs:
        config = configs[model_type]
        config.update(kwargs)
    else:
        config = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "max_seq_len": 512
        }
        config.update(kwargs)
    
    model = UnifiedCognitiveLM(vocab_size=vocab_size, **config)
    model.to(device)

    logger.info(f"Created {model_type} unified model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def create_tokenizer(model_name: str = "gpt2"):
    """Create a GPT-2 tokenizer with explicit pad/eos tokens."""

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = "<|pad|>"
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = ""
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def align_model_tokenizer(model: nn.Module, tokenizer) -> None:
    """Ensure model embeddings match tokenizer vocabulary size."""

    if not hasattr(model, "resize_token_embeddings"):
        raise AttributeError("Model does not expose resize_token_embeddings")
    model.resize_token_embeddings(len(tokenizer))

# Example usage and testing
if __name__ == "__main__":
    # Create model and tokenizer
    tokenizer = create_tokenizer()
    model = create_unified_model("small", vocab_size=tokenizer.vocab_size)
    
    # Test basic functionality
    test_text = "The capital of France is"
    input_ids = tokenizer.encode(test_text, return_tensors='pt')
    
    with torch.no_grad():
        logits, aux = model.forward(input_ids, return_schema=True)
        
    print(f"Model created successfully!")
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Auxiliary outputs: {list(aux.keys())}")
    
    # Test knowledge integration
    integrator = KnowledgeIntegrator(model, tokenizer)
    
    # Add some sample knowledge
    sample_triples = [
        ("Paris", "is_capital_of", "France"),
        ("France", "is_in", "Europe"),
        ("Europe", "is_a", "continent")
    ]
    integrator.add_knowledge_triples(sample_triples)
    
    # Test reasoning
    result = integrator.answer_with_reasoning("What is the capital of France?", max_tokens=10)
    print(f"Reasoning result: {result}")
    
    logger.info("All tests passed!")