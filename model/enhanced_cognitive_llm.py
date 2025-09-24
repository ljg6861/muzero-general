#!/usr/bin/env python3
"""
Enhanced Cognitive LLM with Scale and Training Strategy Fixes
============================================================
Integrates enhanced configuration addressing:
1. Scale issues through pretrained initialization
2. Objective interference through delayed/annealed cognitive losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import pickle
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import math

from advanced_cognitive_architecture import AdvancedCognitiveArchitecture
from enhanced_config import (
    EnhancedCognitiveLLMConfig, 
    PretrainedInitializer, 
    CognitiveLossScheduler,
    get_optimized_config_for_data_size
)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, hidden_size: int, max_seq_length: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, hidden_size)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * 
                           (-math.log(10000.0) / hidden_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EnhancedCognitiveAttentionLayer(nn.Module):
    """Enhanced attention layer with progressive cognitive processing integration."""
    
    def __init__(self, config: EnhancedCognitiveLLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        assert self.hidden_size % self.num_heads == 0
        
        # Standard transformer attention
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Cognitive enhancement projections (smaller to avoid interference)
        self.concept_proj = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.causal_proj = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.meta_proj = nn.Linear(self.hidden_size, self.hidden_size // 8)
        
        # Attention and dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.hidden_dropout = nn.Dropout(config.hidden_dropout)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cognitive_scheduler: Optional[CognitiveLossScheduler] = None,
                layer_idx: Optional[int] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Standard transformer attention
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attention_dropout(attn_probs)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_probs, value)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        # Project back to hidden size
        output = self.out_proj(attn_output)
        output = self.hidden_dropout(output)
        
        # Progressive cognitive enhancement (only if enabled and scheduled)
        cognitive_outputs = {}
        
        if (cognitive_scheduler is not None and 
            layer_idx in self.config.cognitive_integration_layers):
            
            component_weights = cognitive_scheduler.get_component_weights()
            
            # Concept formation (extract concepts for knowledge representation)
            if component_weights['concept_formation'] > 0:
                concept_features = self.concept_proj(output)
                cognitive_outputs['concepts'] = concept_features * component_weights['concept_formation']
            
            # Causal reasoning (model causal relationships)
            if component_weights['causal_reasoning'] > 0:
                causal_features = self.causal_proj(output)
                cognitive_outputs['causal'] = causal_features * component_weights['causal_reasoning']
            
            # Meta-learning (learn to learn)
            if component_weights['meta_learning'] > 0:
                meta_features = self.meta_proj(output)
                cognitive_outputs['meta'] = meta_features * component_weights['meta_learning']
        
        return output, cognitive_outputs


class EnhancedTransformerLayer(nn.Module):
    """Enhanced transformer layer with progressive cognitive integration."""
    
    def __init__(self, config: EnhancedCognitiveLLMConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        
        # Attention layer
        self.attention = EnhancedCognitiveAttentionLayer(config)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout)
        )
        
        # Layer dropout (stochastic depth)
        self.layer_dropout = nn.Dropout(config.layer_dropout)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                cognitive_scheduler: Optional[CognitiveLossScheduler] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        # Pre-norm attention with residual connection
        normed_states = self.ln1(hidden_states)
        attn_output, cognitive_outputs = self.attention(
            normed_states, 
            attention_mask=attention_mask,
            cognitive_scheduler=cognitive_scheduler,
            layer_idx=self.layer_idx
        )
        
        # Apply layer dropout and residual connection
        if self.training and torch.rand(1).item() < self.config.layer_dropout:
            attn_output = torch.zeros_like(attn_output)
        
        hidden_states = hidden_states + attn_output
        
        # Pre-norm FFN with residual connection
        normed_states = self.ln2(hidden_states)
        ffn_output = self.ffn(normed_states)
        
        if self.training and torch.rand(1).item() < self.config.layer_dropout:
            ffn_output = torch.zeros_like(ffn_output)
            
        hidden_states = hidden_states + ffn_output
        
        return hidden_states, cognitive_outputs


class EnhancedCognitiveLLM(nn.Module):
    """Enhanced Cognitive Language Model with scale and training strategy fixes."""
    
    def __init__(self, config: EnhancedCognitiveLLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_size, config.max_seq_length)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(config, i) for i in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_ln = nn.LayerNorm(config.hidden_size)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Cognitive processing components (initialized but used progressively)
        if config.enable_concept_formation:
            self.cognitive_arch = AdvancedCognitiveArchitecture(
                input_dim=config.hidden_size // 4,
                hidden_dim=config.hidden_size // 2,
                num_concepts=256,  # Smaller for limited data
                num_reasoning_steps=3,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        
        # Loss scheduler for progressive training
        self.cognitive_scheduler = CognitiveLossScheduler(config)
        
        # Initialize with pretrained weights if requested
        self.pretrained_initializer = None
        if config.use_pretrained_init:
            self.pretrained_initializer = PretrainedInitializer(config)
            self._initialize_from_pretrained()
        else:
            self._initialize_weights()
    
    def _initialize_from_pretrained(self):
        """Initialize model with pretrained weights."""
        if self.pretrained_initializer.load_pretrained_components():
            # Initialize embeddings
            success = self.pretrained_initializer.initialize_embeddings(self.token_embedding)
            if success:
                print("✓ Initialized with pretrained embeddings")
            else:
                print("! Failed pretrained embedding init, using random")
                self._initialize_weights()
        else:
            print("! Pretrained loading failed, using random initialization")
            self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights randomly."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def get_tokenizer(self):
        """Get the tokenizer (pretrained or None)."""
        if self.pretrained_initializer:
            return self.pretrained_initializer.get_tokenizer()
        return None
    
    def update_training_step(self, step: int):
        """Update the training step for cognitive loss scheduling."""
        self.cognitive_scheduler.update_step(step)
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                enable_cognitive: bool = True) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Token embeddings with dropout
        hidden_states = self.token_embedding(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Add positional encoding
        hidden_states = self.pos_encoding(hidden_states)
        
        # Pass through transformer layers
        all_cognitive_outputs = []
        for layer in self.layers:
            hidden_states, cognitive_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                cognitive_scheduler=self.cognitive_scheduler if enable_cognitive else None
            )
            if cognitive_outputs:
                all_cognitive_outputs.append(cognitive_outputs)
        
        # Final layer norm
        hidden_states = self.final_ln(hidden_states)
        
        # Language modeling logits
        lm_logits = self.lm_head(hidden_states)
        
        # Compute losses
        outputs = {'logits': lm_logits}
        
        if labels is not None:
            # Language modeling loss
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            outputs['lm_loss'] = lm_loss
            
            # Progressive cognitive losses (only if enabled and scheduled)
            cognitive_loss = torch.tensor(0.0, device=device)
            cognitive_details = {}
            
            if (enable_cognitive and 
                self.config.enable_concept_formation and 
                hasattr(self, 'cognitive_arch') and
                all_cognitive_outputs):
                
                component_weights = self.cognitive_scheduler.get_component_weights()
                
                # Only compute cognitive losses if they should be active
                if any(weight > 0 for weight in component_weights.values()):
                    try:
                        # Aggregate cognitive features across layers
                        all_concepts = []
                        all_causal = []
                        all_meta = []
                        
                        for layer_outputs in all_cognitive_outputs:
                            if 'concepts' in layer_outputs:
                                all_concepts.append(layer_outputs['concepts'])
                            if 'causal' in layer_outputs:
                                all_causal.append(layer_outputs['causal'])
                            if 'meta' in layer_outputs:
                                all_meta.append(layer_outputs['meta'])
                        
                        # Compute cognitive losses if features are available
                        if all_concepts and component_weights['concept_formation'] > 0:
                            concept_features = torch.stack(all_concepts).mean(0)  # Average across layers
                            concept_loss = self.cognitive_arch.compute_concept_formation_loss(concept_features)
                            cognitive_loss += concept_loss * component_weights['concept_formation']
                            cognitive_details['concept_loss'] = concept_loss.item()
                        
                        if all_causal and component_weights['causal_reasoning'] > 0:
                            causal_features = torch.stack(all_causal).mean(0)
                            causal_loss = self.cognitive_arch.compute_causal_reasoning_loss(causal_features)
                            cognitive_loss += causal_loss * component_weights['causal_reasoning']
                            cognitive_details['causal_loss'] = causal_loss.item()
                        
                        if all_meta and component_weights['meta_learning'] > 0:
                            meta_features = torch.stack(all_meta).mean(0)
                            meta_loss = self.cognitive_arch.compute_meta_learning_loss(meta_features)
                            cognitive_loss += meta_loss * component_weights['meta_learning']
                            cognitive_details['meta_loss'] = meta_loss.item()
                            
                    except Exception as e:
                        print(f"Warning: Cognitive loss computation failed: {e}")
                        cognitive_loss = torch.tensor(0.0, device=device)
            
            outputs['cognitive_loss'] = cognitive_loss
            outputs['cognitive_details'] = cognitive_details
            outputs['total_loss'] = lm_loss + cognitive_loss
            
            # Add scheduling information
            outputs['cognitive_weight'] = self.cognitive_scheduler.get_cognitive_loss_weight()
            outputs['component_weights'] = self.cognitive_scheduler.get_component_weights()
        
        return outputs
    
    def generate(self, 
                 input_ids: torch.Tensor,
                 max_length: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(input_ids, enable_cognitive=False)  # No cognitive during generation
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token (if using pretrained tokenizer)
                tokenizer = self.get_tokenizer()
                if tokenizer and next_token.item() == tokenizer.eos_token_id:
                    break
        
        self.train()
        return input_ids


def create_enhanced_model(data_size: int = 7500) -> Tuple[EnhancedCognitiveLLM, EnhancedCognitiveLLMConfig]:
    """Create an enhanced cognitive LLM optimized for the given data size."""
    
    # Get optimized configuration
    config = get_optimized_config_for_data_size(data_size)
    
    print(f"Creating Enhanced Cognitive LLM for ~{data_size} articles:")
    print(f"  Model: {config.num_layers} layers, {config.hidden_size} hidden, {config.num_attention_heads} heads")
    print(f"  Cognitive delays: concept={config.concept_formation_delay}, causal={config.causal_reasoning_delay}, meta={config.meta_learning_delay}")
    print(f"  Max cognitive weight: {config.max_cognitive_loss_weight}")
    print(f"  Pretrained init: {config.use_pretrained_init}")
    
    # Create model
    model = EnhancedCognitiveLLM(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model, config


if __name__ == "__main__":
    # Test the enhanced model
    model, config = create_enhanced_model(7500)
    
    # Test forward pass
    batch_size = 2
    seq_len = 64
    vocab_size = config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test at different training steps
    print("\nTesting progressive cognitive loss scheduling:")
    
    for step in [0, 1000, 3000, 5000, 8000, 10000]:
        model.update_training_step(step)
        
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        
        print(f"Step {step:5d}: LM loss={outputs['lm_loss']:.4f}, "
              f"Cognitive loss={outputs['cognitive_loss']:.4f}, "
              f"Weight={outputs['cognitive_weight']:.4f}")
        
        # Show component weights
        comp_weights = outputs['component_weights']
        print(f"           Components: concept={comp_weights['concept_formation']:.3f}, "
              f"causal={comp_weights['causal_reasoning']:.3f}, "
              f"meta={comp_weights['meta_learning']:.3f}")