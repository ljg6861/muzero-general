#!/usr/bin/env python3
"""
Phase A: Baseline Language Model to Competence
==============================================
Clean implementation of 6L×384d baseline LM with proven tokenizer.
Focus: Establish solid language modeling foundation before cognitive enhancements.

Specifications:
- Model: 6 layers, 384 hidden, 6 heads
- Tokenizer: SentencePiece/BPE from proven model
- Data: 50-100M tokens (scaled Wikipedia)
- Training: AdamW, lr=2e-4, 3% warmup, cosine decay
- Batching: 1-2M tokens/step effective batch size
- Goal: PPL plateau with stable generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from transformers import AutoTokenizer


@dataclass
class BaselineLMConfig:
    """Configuration for Phase A baseline language model."""
    
    # Model architecture (Phase A specification)
    vocab_size: int = 32000  # Will be set by tokenizer
    max_seq_length: int = 1024
    hidden_size: int = 384   # Phase A: 384d
    num_layers: int = 6      # Phase A: 6L
    num_attention_heads: int = 6  # Phase A: 6 heads
    intermediate_size: int = 1536  # 4 × hidden_size
    
    # Proven tokenizer configuration
    tokenizer_name: str = "microsoft/DialoGPT-medium"  # Good SentencePiece tokenizer
    # Alternative options: "gpt2", "facebook/opt-350m", "EleutherAI/gpt-neo-125M"
    
    # Training configuration (Phase A specification)
    learning_rate: float = 2e-4      # Phase A: lr ~2e-4 for small model
    warmup_ratio: float = 0.03       # Phase A: 3% warmup
    lr_schedule: str = "cosine"      # Phase A: cosine decay
    optimizer: str = "adamw"         # Phase A: AdamW
    
    # Batching configuration (Phase A specification)
    micro_batch_size: int = 4        # Actual batch size per forward pass
    gradient_accumulation_steps: int = 128  # To reach 1-2M tokens/step
    # Effective batch = micro_batch_size × grad_accum × seq_length
    # 4 × 128 × 1024 = 524,288 tokens/step (will adjust for 1-2M target)
    
    # Regularization
    dropout_prob: float = 0.1
    attention_dropout: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data configuration
    target_tokens: int = 100_000_000  # Phase A: 50-100M tokens
    min_tokens: int = 50_000_000
    
    # Architecture details
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        # Calculate effective batch size in tokens
        effective_tokens_per_step = (self.micro_batch_size * 
                                   self.gradient_accumulation_steps * 
                                   self.max_seq_length)
        
        print(f"Effective batch size: {effective_tokens_per_step:,} tokens/step")
        
        if effective_tokens_per_step < 1_000_000:
            print(f"⚠ Warning: Effective batch size below 1M tokens/step")
        elif effective_tokens_per_step > 2_000_000:
            print(f"⚠ Warning: Effective batch size above 2M tokens/step")


class RMSNorm(nn.Module):
    """RMSNorm for better training stability."""
    
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, x):
        # RMSNorm formula: x / sqrt(mean(x^2) + eps) * weight
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding."""
    
    def __init__(self, dim: int, max_seq_length: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_length = max_seq_length
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute position encodings
        t = torch.arange(max_seq_length).float()
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
            
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        return cos, sin


def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary position embedding to query/key tensors."""
    # Split into two halves
    x1, x2 = x[..., ::2], x[..., 1::2]
    # Apply rotation
    rotated = torch.cat([-x2, x1], dim=-1)
    # Combine with cos/sin
    return (x * cos) + (rotated * sin)


class BaselineAttention(nn.Module):
    """Multi-head attention for baseline LM."""
    
    def __init__(self, config: BaselineLMConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout_prob)
        
        # Rotary position embedding
        self.rope = RotaryPositionalEmbedding(
            self.head_dim, 
            max_seq_length=config.max_seq_length
        )
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary position embedding
        cos, sin = self.rope(query, seq_len)
        query = apply_rotary_pos_emb(query, cos, sin)
        key = apply_rotary_pos_emb(key, cos, sin)
        
        # Handle past key/value for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=-2)
            value = torch.cat([past_value, value], dim=-2)
        
        present_key_value = (key, value) if use_cache else None
        
        # Compute attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply to values
        attn_output = torch.matmul(attn_weights, value)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present_key_value


class BaselineMLP(nn.Module):
    """Feed-forward network for baseline LM."""
    
    def __init__(self, config: BaselineLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_prob)
    
    def forward(self, x):
        # SwiGLU activation: gate(x) * swish(up(x))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        hidden = F.silu(gate) * up  # SiLU = Swish
        output = self.down_proj(hidden)
        return self.dropout(output)


class BaselineTransformerBlock(nn.Module):
    """Transformer block for baseline LM."""
    
    def __init__(self, config: BaselineLMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = BaselineAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = BaselineMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        residual = hidden_states
        
        # Pre-norm attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            use_cache=use_cache,
            past_key_value=past_key_value
        )
        hidden_states = residual + hidden_states
        
        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, present_key_value


class BaselineLanguageModel(nn.Module):
    """Phase A: Baseline Language Model (6L×384d)."""
    
    def __init__(self, config: BaselineLMConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            BaselineTransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final norm and output
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Tie embeddings and output weights (common practice)
        self.lm_head.weight = self.embed_tokens.weight
    
    def _init_weights(self, module):
        """Initialize weights according to GPT-style initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create causal attention mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
            attention_mask = attention_mask.view(1, 1, seq_len, seq_len)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Convert to additive mask (0 for allowed, -inf for masked)
        attention_mask = (1.0 - attention_mask) * -1e9
        
        # Token embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Apply layers
        all_key_values = () if use_cache else None
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present_key_value = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                use_cache=use_cache,
                past_key_value=past_key_value
            )
            
            if use_cache:
                all_key_values = all_key_values + (present_key_value,)
        
        # Final norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        outputs = {
            'logits': logits,
            'past_key_values': all_key_values if use_cache else None
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Shift labels for autoregressive loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            
            outputs['loss'] = loss
            outputs['perplexity'] = torch.exp(loss)
        
        return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text using the baseline model."""
        
        self.eval()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    use_cache=True,
                    past_key_values=past_key_values
                )
                
                past_key_values = outputs['past_key_values']
                logits = outputs['logits'][:, -1, :]  # Last token logits
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        
        self.train()
        return input_ids


def create_baseline_model_with_tokenizer() -> Tuple[BaselineLanguageModel, AutoTokenizer, BaselineLMConfig]:
    """Create baseline model with proven tokenizer."""
    
    print("Creating Phase A Baseline Language Model")
    print("=" * 50)
    
    # Create configuration
    config = BaselineLMConfig()
    
    # Load proven tokenizer
    print(f"Loading tokenizer: {config.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    
    # Handle tokenizer configuration
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Update config with actual vocab size
    config.vocab_size = len(tokenizer)
    
    print(f"Tokenizer loaded: {len(tokenizer)} tokens")
    print(f"Special tokens: PAD={tokenizer.pad_token_id}, EOS={tokenizer.eos_token_id}")
    
    # Create model
    model = BaselineLanguageModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created:")
    print(f"  Architecture: {config.num_layers}L × {config.hidden_size}d × {config.num_attention_heads}h")
    print(f"  Vocabulary: {config.vocab_size:,} tokens")
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  Memory (FP32): ~{total_params * 4 / 1e9:.1f} GB")
    
    return model, tokenizer, config


if __name__ == "__main__":
    # Test baseline model creation
    model, tokenizer, config = create_baseline_model_with_tokenizer()
    
    # Test forward pass
    print("\nTesting forward pass...")
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(input_ids, labels=input_ids)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {outputs['logits'].shape}")
    print(f"Loss: {outputs['loss'].item():.4f}")
    print(f"Perplexity: {outputs['perplexity'].item():.2f}")
    
    # Test generation
    print("\nTesting generation...")
    prompt = "The future of artificial intelligence"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    generated = model.generate(
        input_ids=input_ids,
        max_new_tokens=20,
        temperature=0.8,
        do_sample=True
    )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated_text}")
    
    print("\n✓ Baseline model working correctly!")