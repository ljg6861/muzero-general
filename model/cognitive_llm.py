#!/usr/bin/env python3
"""
Cognitive-Enhanced Language Model
=================================
Phase 1: Foundation Knowledge Training with Cognitive Architecture Integration
Combines transformer-based language modeling with our cognitive reasoning framework.
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


@dataclass
class CognitiveLLMConfig:
    """Configuration for the Cognitive-Enhanced LLM."""
    # Language Model Parameters
    vocab_size: int = 50000
    max_seq_length: int = 2048
    hidden_size: int = 768
    num_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout_prob: float = 0.1
    
    # Cognitive Enhancement Parameters
    enable_concept_formation: bool = True
    enable_causal_reasoning: bool = True
    enable_meta_learning: bool = True
    cognitive_integration_layers: List[int] = None  # Which transformer layers to enhance
    
    # Training Parameters
    learning_rate: float = 1e-4
    cognitive_learning_rate: float = 5e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    # Regularization Parameters
    dropout_prob: float = 0.1
    attention_dropout: float = 0.1
    layer_dropout: float = 0.0  # Stochastic depth
    hidden_dropout: float = 0.1  # Hidden layer dropout
    embedding_dropout: float = 0.1  # Embedding dropout
    
    def __post_init__(self):
        if self.cognitive_integration_layers is None:
            # Integrate cognitive processing at middle and later layers
            self.cognitive_integration_layers = [6, 9, 11]


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


class CognitiveAttentionLayer(nn.Module):
    """Enhanced attention layer with cognitive processing integration."""
    
    def __init__(self, config: CognitiveLLMConfig):
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
        
        # Cognitive enhancement projections
        self.concept_proj = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.causal_proj = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.reasoning_proj = nn.Linear(self.hidden_size, self.hidden_size // 2)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, hidden_states, attention_mask=None, cognitive_context=None):
        residual = hidden_states
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Standard multi-head attention
        queries = self.query(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = self.key(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = self.value(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attention_scores += attention_mask
        
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, values)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Cognitive enhancement
        if cognitive_context is not None:
            concept_features = self.concept_proj(cognitive_context.get('concepts', torch.zeros_like(hidden_states)))
            causal_features = self.causal_proj(cognitive_context.get('causal_relations', torch.zeros_like(hidden_states)))
            reasoning_features = self.reasoning_proj(cognitive_context.get('reasoning_state', torch.zeros_like(hidden_states)))
            
            # Combine cognitive features
            cognitive_enhancement = torch.cat([concept_features, causal_features, reasoning_features], dim=-1)
            context = context + cognitive_enhancement
        
        output = self.out_proj(context)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        
        return output, attention_probs


class CognitiveFeedForward(nn.Module):
    """Feed-forward network with cognitive reasoning integration."""
    
    def __init__(self, config: CognitiveLLMConfig):
        super().__init__()
        self.config = config
        
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
        # Cognitive reasoning pathways
        self.reasoning_gate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.concept_gate = nn.Linear(config.hidden_size, config.intermediate_size)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, hidden_states, cognitive_context=None):
        residual = hidden_states
        
        # Standard feed-forward
        hidden_states = F.gelu(self.dense1(hidden_states))
        
        # Cognitive reasoning integration
        if cognitive_context is not None:
            reasoning_activation = torch.sigmoid(self.reasoning_gate(residual))
            concept_activation = torch.sigmoid(self.concept_gate(residual))
            
            # Modulate activations based on cognitive state
            hidden_states = hidden_states * reasoning_activation * concept_activation
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        output = self.layer_norm(hidden_states + residual)
        return output


class CognitiveTransformerLayer(nn.Module):
    """Single transformer layer with cognitive enhancement."""
    
    def __init__(self, config: CognitiveLLMConfig):
        super().__init__()
        self.attention = CognitiveAttentionLayer(config)
        self.feed_forward = CognitiveFeedForward(config)
        
    def forward(self, hidden_states, attention_mask=None, cognitive_context=None):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, cognitive_context
        )
        layer_output = self.feed_forward(attention_output, cognitive_context)
        return layer_output, attention_probs


class CognitiveLLM(nn.Module):
    """
    Cognitive-Enhanced Language Model combining transformer architecture
    with advanced cognitive reasoning capabilities.
    """
    
    def __init__(self, config: CognitiveLLMConfig):
        super().__init__()
        self.config = config
        
        # Core transformer components
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_encoding = PositionalEncoding(config.hidden_size, config.max_seq_length)
        
        self.layers = nn.ModuleList([
            CognitiveTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Cognitive architecture integration
        self.cognitive_architecture = AdvancedCognitiveArchitecture()
        
        # Cognitive state processors
        self.concept_processor = nn.Linear(config.hidden_size, config.hidden_size)
        self.causal_processor = nn.Linear(config.hidden_size, config.hidden_size)
        self.reasoning_processor = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout_prob)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def _process_cognitive_context(self, hidden_states, layer_idx):
        """Process cognitive context for enhanced reasoning."""
        if layer_idx not in self.config.cognitive_integration_layers:
            return None
        
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Extract semantic content for cognitive processing
        semantic_content = hidden_states.mean(dim=1)  # Pool sequence representation
        
        try:
            # Process through cognitive architecture
            # Convert to numpy for cognitive processing
            content_np = semantic_content.detach().cpu().numpy()
            
            # Concept formation
            concepts = torch.zeros_like(hidden_states)
            if self.config.enable_concept_formation:
                concept_features = self.concept_processor(hidden_states)
                concepts = F.relu(concept_features)
            
            # Causal reasoning
            causal_relations = torch.zeros_like(hidden_states)
            if self.config.enable_causal_reasoning:
                causal_features = self.causal_processor(hidden_states)
                causal_relations = F.tanh(causal_features)
            
            # Meta-learning reasoning state
            reasoning_state = torch.zeros_like(hidden_states)
            if self.config.enable_meta_learning:
                reasoning_features = self.reasoning_processor(hidden_states)
                reasoning_state = F.sigmoid(reasoning_features)
            
            return {
                'concepts': concepts,
                'causal_relations': causal_relations,
                'reasoning_state': reasoning_state
            }
            
        except Exception as e:
            # Fallback if cognitive processing fails
            return None
    
    def forward(self, input_ids, attention_mask=None, labels=None, enable_cognitive=True):
        """
        Forward pass with optional cognitive enhancement control.
        
        Args:
            input_ids: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len] 
            labels: Target labels for loss calculation [batch_size, seq_len]
            enable_cognitive: Whether to enable cognitive enhancement (for ablation)
        """
        batch_size, seq_len = input_ids.shape
        
        # Ensure input_ids are within vocabulary bounds
        input_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Convert attention mask to attention bias
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Embeddings and positional encoding
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.positional_encoding(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Process through transformer layers with optional cognitive enhancement
        all_attention_probs = []
        for i, layer in enumerate(self.layers):
            # Only apply cognitive processing if enabled and this is a cognitive layer
            cognitive_context = None
            if enable_cognitive:
                cognitive_context = self._process_cognitive_context(hidden_states, i)
            
            hidden_states, attention_probs = layer(hidden_states, attention_mask, cognitive_context)
            all_attention_probs.append(attention_probs)
        
        # Final processing
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'hidden_states': hidden_states,
            'attention_probs': all_attention_probs
        }
    
    def generate(self, input_ids, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
        """Generate text with cognitive enhancement."""
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(input_ids)
                logits = outputs['logits']
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to input_ids
                input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
    
    def save_model(self, save_path: str):
        """Save model weights and configuration."""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        
        # Save configuration
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'max_seq_length': self.config.max_seq_length,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'intermediate_size': self.config.intermediate_size,
            'dropout_prob': self.config.dropout_prob,
            'enable_concept_formation': self.config.enable_concept_formation,
            'enable_causal_reasoning': self.config.enable_causal_reasoning,
            'enable_meta_learning': self.config.enable_meta_learning,
            'cognitive_integration_layers': self.config.cognitive_integration_layers
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save cognitive architecture state
        try:
            with open(os.path.join(save_path, 'cognitive_state.pkl'), 'wb') as f:
                pickle.dump({
                    'cognitive_architecture': self.cognitive_architecture,
                    'save_timestamp': datetime.now().isoformat()
                }, f)
        except Exception as e:
            print(f"Warning: Could not save cognitive architecture state: {e}")
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, load_path: str, device='cpu'):
        """Load model weights and configuration."""
        # Load configuration
        with open(os.path.join(load_path, 'config.json'), 'r') as f:
            config_dict = json.load(f)
        
        config = CognitiveLLMConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load model weights
        state_dict = torch.load(os.path.join(load_path, 'pytorch_model.bin'), map_location=device)
        model.load_state_dict(state_dict)
        
        # Load cognitive architecture state if available
        cognitive_state_path = os.path.join(load_path, 'cognitive_state.pkl')
        if os.path.exists(cognitive_state_path):
            try:
                with open(cognitive_state_path, 'rb') as f:
                    cognitive_data = pickle.load(f)
                    model.cognitive_architecture = cognitive_data['cognitive_architecture']
                    print(f"Loaded cognitive state from {cognitive_data['save_timestamp']}")
            except Exception as e:
                print(f"Warning: Could not load cognitive architecture state: {e}")
        
        model.to(device)
        print(f"Model loaded from {load_path}")
        return model


def create_default_config():
    """Create a default configuration for the Cognitive LLM."""
    return CognitiveLLMConfig(
        vocab_size=50000,
        max_seq_length=1024,
        hidden_size=512,
        num_layers=8,
        num_attention_heads=8,
        intermediate_size=2048,
        dropout_prob=0.1,
        enable_concept_formation=True,
        enable_causal_reasoning=True,
        enable_meta_learning=True,
        cognitive_integration_layers=[3, 5, 7],
        learning_rate=1e-4,
        cognitive_learning_rate=5e-5,
        batch_size=4,
        gradient_accumulation_steps=8
    )


if __name__ == "__main__":
    # Test model creation and basic functionality
    config = create_default_config()
    model = CognitiveLLM(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    print("Testing Cognitive LLM...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    outputs = model(input_ids)
    print(f"Output logits shape: {outputs['logits'].shape}")
    
    # Test save/load functionality
    save_path = "test_cognitive_llm"
    model.save_model(save_path)
    
    loaded_model = CognitiveLLM.load_model(save_path)
    print("Save/load test successful!")
    
    print("Cognitive LLM foundation ready for Phase 1 training!")