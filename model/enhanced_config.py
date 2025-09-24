#!/usr/bin/env python3
"""
Enhanced Configuration for Practical Cognitive LLM Training
==========================================================
Addresses scale and training strategy issues:
1. Uses pretrained tokenizer + embeddings from small open models
2. Implements delayed/annealed cognitive losses to prevent objective interference
3. Provides proper model sizing for available data scale
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import math


@dataclass
class EnhancedCognitiveLLMConfig:
    """Enhanced configuration addressing scale and training strategy issues."""
    
    # === SCALE SOLUTIONS ===
    # Option 1: Use pretrained tokenizer + embeddings (RECOMMENDED)
    use_pretrained_init: bool = True
    pretrained_model_name: str = "distilgpt2"  # Small, fast, good tokenizer
    # Alternative options: "gpt2", "microsoft/DialoGPT-small", "EleutherAI/gpt-neo-125M"
    
    # Option 2: Smaller model for limited data (if not using pretrained)
    vocab_size: int = 50257  # GPT-2 tokenizer size (when using pretrained)
    max_seq_length: int = 1024  # Reasonable for limited data
    hidden_size: int = 384  # Reduced from 768 for better data efficiency  
    num_layers: int = 6  # Reduced from 12 for limited data
    num_attention_heads: int = 6  # Matches hidden_size division
    intermediate_size: int = 1536  # 4 * hidden_size
    
    # === OBJECTIVE INTERFERENCE SOLUTIONS ===
    # Delayed cognitive loss introduction
    cognitive_delay_steps: int = 5000  # Don't start cognitive losses until base LM learns basics
    cognitive_annealing_steps: int = 10000  # Gradually increase cognitive loss weight
    max_cognitive_loss_weight: float = 0.1  # Keep cognitive losses small relative to LM loss
    
    # Progressive cognitive complexity
    enable_progressive_cognitive: bool = True
    concept_formation_delay: int = 2000  # Start concept formation first
    causal_reasoning_delay: int = 5000   # Add causal reasoning later
    meta_learning_delay: int = 8000      # Add meta-learning last
    
    # === TRAINING PARAMETERS ===
    learning_rate: float = 5e-4  # Higher LR for smaller model
    cognitive_learning_rate: float = 1e-4  # Lower for cognitive components
    batch_size: int = 16  # Increase for efficiency
    gradient_accumulation_steps: int = 2  # Effective batch size = 32
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # === REGULARIZATION ===
    dropout_prob: float = 0.1
    attention_dropout: float = 0.1
    layer_dropout: float = 0.05  # Light stochastic depth
    hidden_dropout: float = 0.1
    embedding_dropout: float = 0.05
    
    # Weight decay (excluding embeddings and layer norms)
    weight_decay: float = 0.01
    exclude_from_decay: List[str] = None
    
    # === COGNITIVE PARAMETERS ===
    enable_concept_formation: bool = True
    enable_causal_reasoning: bool = True  
    enable_meta_learning: bool = True
    cognitive_integration_layers: List[int] = None  # [3, 5] for 6-layer model
    
    def __post_init__(self):
        # Set cognitive integration layers based on model size
        if self.cognitive_integration_layers is None:
            if self.num_layers <= 6:
                self.cognitive_integration_layers = [3, 5]  # Middle and late layers
            elif self.num_layers <= 12:
                self.cognitive_integration_layers = [6, 9, 11]
            else:
                # For larger models, integrate at 1/3, 2/3, and final layers
                self.cognitive_integration_layers = [
                    self.num_layers // 3,
                    2 * self.num_layers // 3,
                    self.num_layers - 1
                ]
        
        # Set weight decay exclusions
        if self.exclude_from_decay is None:
            self.exclude_from_decay = [
                'bias', 'LayerNorm.weight', 'layernorm.weight', 
                'token_embedding.weight', 'position_embedding.weight'
            ]
        
        # Validate configuration
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        assert self.cognitive_delay_steps < self.cognitive_annealing_steps, \
            "cognitive_delay_steps must be less than cognitive_annealing_steps"


class PretrainedInitializer:
    """Handles initialization from pretrained models."""
    
    def __init__(self, config: EnhancedCognitiveLLMConfig):
        self.config = config
        self.tokenizer = None
        self.pretrained_model = None
        
    def load_pretrained_components(self):
        """Load tokenizer and model for initialization."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.pretrained_model_name,
                pad_token='<|endoftext|>'  # Use EOS as PAD for GPT-style models
            )
            
            self.pretrained_model = AutoModel.from_pretrained(
                self.config.pretrained_model_name
            )
            
            print(f"✓ Loaded pretrained components from {self.config.pretrained_model_name}")
            print(f"  Tokenizer vocab size: {len(self.tokenizer)}")
            print(f"  Model hidden size: {self.pretrained_model.config.hidden_size}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load pretrained components: {e}")
            print("  Falling back to random initialization")
            return False
    
    def get_tokenizer(self):
        """Get the pretrained tokenizer."""
        if self.tokenizer is None:
            self.load_pretrained_components()
        return self.tokenizer
    
    def initialize_embeddings(self, embedding_layer: nn.Embedding):
        """Initialize embedding layer with pretrained weights."""
        if self.pretrained_model is None:
            return False
            
        try:
            pretrained_embeddings = self.pretrained_model.wte.weight.data
            
            # Handle size mismatches
            pretrained_vocab_size, pretrained_hidden_size = pretrained_embeddings.shape
            target_vocab_size, target_hidden_size = embedding_layer.weight.shape
            
            # Initialize with pretrained embeddings (handling size differences)
            if pretrained_vocab_size >= target_vocab_size and pretrained_hidden_size >= target_hidden_size:
                # Pretrained is larger - truncate
                embedding_layer.weight.data = pretrained_embeddings[:target_vocab_size, :target_hidden_size]
            elif pretrained_vocab_size <= target_vocab_size and pretrained_hidden_size <= target_hidden_size:
                # Pretrained is smaller - copy and pad with random
                embedding_layer.weight.data[:pretrained_vocab_size, :pretrained_hidden_size] = pretrained_embeddings
                # Random init for the rest (using same std as pretrained)
                std = pretrained_embeddings.std().item()
                embedding_layer.weight.data[pretrained_vocab_size:].normal_(0, std)
                embedding_layer.weight.data[:, pretrained_hidden_size:].normal_(0, std)
            else:
                # Different sizes - use interpolation or projection
                print(f"  Size mismatch: pretrained {pretrained_embeddings.shape} vs target {embedding_layer.weight.shape}")
                # Simple approach: copy what fits, random init the rest
                min_vocab = min(pretrained_vocab_size, target_vocab_size)
                min_hidden = min(pretrained_hidden_size, target_hidden_size)
                embedding_layer.weight.data[:min_vocab, :min_hidden] = pretrained_embeddings[:min_vocab, :min_hidden]
            
            print(f"✓ Initialized embeddings from pretrained model")
            return True
            
        except Exception as e:
            print(f"✗ Failed to initialize embeddings: {e}")
            return False


class CognitiveLossScheduler:
    """Manages delayed and annealed introduction of cognitive losses."""
    
    def __init__(self, config: EnhancedCognitiveLLMConfig):
        self.config = config
        self.step = 0
        
    def update_step(self, step: int):
        """Update the current training step."""
        self.step = step
    
    def get_cognitive_loss_weight(self) -> float:
        """Get the current weight for cognitive losses."""
        if self.step < self.config.cognitive_delay_steps:
            return 0.0
        elif self.step < self.config.cognitive_annealing_steps:
            # Linear annealing from 0 to max_weight
            progress = (self.step - self.config.cognitive_delay_steps) / \
                      (self.config.cognitive_annealing_steps - self.config.cognitive_delay_steps)
            return progress * self.config.max_cognitive_loss_weight
        else:
            return self.config.max_cognitive_loss_weight
    
    def get_component_weights(self) -> Dict[str, float]:
        """Get weights for individual cognitive components."""
        base_weight = self.get_cognitive_loss_weight()
        
        if base_weight == 0.0:
            return {
                'concept_formation': 0.0,
                'causal_reasoning': 0.0,
                'meta_learning': 0.0
            }
        
        weights = {}
        
        # Concept formation (starts first)
        if self.step >= self.config.concept_formation_delay:
            weights['concept_formation'] = base_weight
        else:
            weights['concept_formation'] = 0.0
            
        # Causal reasoning (starts later)
        if self.step >= self.config.causal_reasoning_delay:
            weights['causal_reasoning'] = base_weight
        else:
            weights['causal_reasoning'] = 0.0
            
        # Meta-learning (starts last)
        if self.step >= self.config.meta_learning_delay:
            weights['meta_learning'] = base_weight
        else:
            weights['meta_learning'] = 0.0
            
        return weights
    
    def should_enable_component(self, component: str) -> bool:
        """Check if a cognitive component should be enabled."""
        weights = self.get_component_weights()
        return weights.get(component, 0.0) > 0.0


def get_optimized_config_for_data_size(num_articles: int) -> EnhancedCognitiveLLMConfig:
    """Get optimized configuration based on available data size."""
    
    if num_articles < 1000:
        # Very limited data - tiny model
        return EnhancedCognitiveLLMConfig(
            hidden_size=256,
            num_layers=4,
            num_attention_heads=4,
            cognitive_delay_steps=1000,
            cognitive_annealing_steps=3000,
            max_cognitive_loss_weight=0.05
        )
    elif num_articles < 10000:
        # Limited data - small model (current case ~7500 articles)
        return EnhancedCognitiveLLMConfig(
            hidden_size=384,
            num_layers=6,
            num_attention_heads=6,
            cognitive_delay_steps=3000,
            cognitive_annealing_steps=8000,
            max_cognitive_loss_weight=0.1
        )
    elif num_articles < 100000:
        # Moderate data - medium model
        return EnhancedCognitiveLLMConfig(
            hidden_size=512,
            num_layers=8,
            num_attention_heads=8,
            cognitive_delay_steps=5000,
            cognitive_annealing_steps=12000,
            max_cognitive_loss_weight=0.15
        )
    else:
        # Large data - can use larger model
        return EnhancedCognitiveLLMConfig(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            cognitive_delay_steps=8000,
            cognitive_annealing_steps=20000,
            max_cognitive_loss_weight=0.2
        )


# Example usage configurations
TINY_MODEL_CONFIG = EnhancedCognitiveLLMConfig(
    hidden_size=256,
    num_layers=4,
    num_attention_heads=4,
    max_seq_length=512,
    cognitive_delay_steps=1000,
    cognitive_annealing_steps=3000,
    max_cognitive_loss_weight=0.05
)

SMALL_MODEL_CONFIG = EnhancedCognitiveLLMConfig(
    hidden_size=384,
    num_layers=6,
    num_attention_heads=6,
    max_seq_length=1024,
    cognitive_delay_steps=3000,
    cognitive_annealing_steps=8000,
    max_cognitive_loss_weight=0.1
)

MEDIUM_MODEL_CONFIG = EnhancedCognitiveLLMConfig(
    hidden_size=512,
    num_layers=8,
    num_attention_heads=8,
    max_seq_length=1024,
    cognitive_delay_steps=5000,
    cognitive_annealing_steps=12000,
    max_cognitive_loss_weight=0.15
)


if __name__ == "__main__":
    # Test configuration
    config = get_optimized_config_for_data_size(7500)  # Current data size
    print("Optimized config for ~7500 articles:")
    print(f"  Model size: {config.num_layers} layers, {config.hidden_size} hidden")
    print(f"  Cognitive delay: {config.cognitive_delay_steps} steps")
    print(f"  Max cognitive weight: {config.max_cognitive_loss_weight}")
    print(f"  Uses pretrained init: {config.use_pretrained_init}")
    
    # Test pretrained initializer
    if config.use_pretrained_init:
        initializer = PretrainedInitializer(config)
        success = initializer.load_pretrained_components()
        if success:
            tokenizer = initializer.get_tokenizer()
            print(f"  Loaded tokenizer with {len(tokenizer)} tokens")