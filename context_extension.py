#!/usr/bin/env python3
"""
Context Extension: Extend seq_len 256 → 512
==========================================
After training curves flatten, extend context length using:
- RoPE (preferred): Direct extension with LR warmup
- Position interpolation: Interpolate learned position embeddings
- Switch to RoPE: Reinit only position components, keep rest
"""

import os
import sys
import torch
import torch.nn as nn
import math
import argparse
import json
from datetime import datetime
sys.path.append('model')
sys.path.append('.')

def apply_rotary_pos_emb(x, cos, sin):
    """Apply rotary positional embedding"""
    # x: [batch, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    
    # Split x into x1 and x2 (first half and second half of head_dim)
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    
    # Apply rotation
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)

class RoPEAttentionLayer(nn.Module):
    """Transformer layer with RoPE instead of learned positional embeddings"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, max_seq_len=512, gradient_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.gradient_checkpointing = gradient_checkpointing
        self.max_seq_len = max_seq_len
        
        # Multi-head attention components (manual implementation for RoPE)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # RoPE frequencies
        self.register_buffer('inv_freq', self._get_rope_frequencies())
    
    def _get_rope_frequencies(self):
        """Get RoPE frequencies"""
        # Standard RoPE frequencies
        dim = self.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        return inv_freq
    
    def _get_rope_embeddings(self, seq_len, device):
        """Get RoPE embeddings for given sequence length"""
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, head_dim//2]
        cos = torch.cos(freqs)  # [seq_len, head_dim//2]
        sin = torch.sin(freqs)  # [seq_len, head_dim//2]
        
        # Duplicate for full head_dim
        cos = torch.cat([cos, cos], dim=-1)  # [seq_len, head_dim]
        sin = torch.cat([sin, sin], dim=-1)  # [seq_len, head_dim]
        
        return cos, sin
    
    def _forward_impl(self, src, src_mask=None, src_key_padding_mask=None):
        batch_size, seq_len, d_model = src.shape
        
        # Self attention with RoPE
        x = src
        
        # Project to Q, K, V
        q = self.q_proj(x)  # [batch, seq_len, d_model]
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.nhead, self.head_dim)  # [batch, seq_len, nhead, head_dim]
        k = k.view(batch_size, seq_len, self.nhead, self.head_dim)
        v = v.view(batch_size, seq_len, self.nhead, self.head_dim)
        
        # Apply RoPE
        cos, sin = self._get_rope_embeddings(seq_len, src.device)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)
        
        # Transpose for attention: [batch, nhead, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        if src_mask is not None:
            scores = scores.masked_fill(src_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply key padding mask
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [batch, nhead, seq_len, head_dim]
        
        # Transpose back and reshape: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection
        attn_output = self.out_proj(attn_output)
        
        # First residual connection
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed forward block
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        
        return x
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward_impl, src, src_mask, src_key_padding_mask)
        else:
            return self._forward_impl(src, src_mask, src_key_padding_mask)

def extend_positional_embeddings_interpolation(model, new_max_len, method='linear'):
    """
    Extend positional embeddings using interpolation
    
    Args:
        model: Model with learned positional embeddings
        new_max_len: New maximum sequence length
        method: 'linear' or 'cosine' interpolation
    """
    if new_max_len <= model.max_seq_len:
        print(f"Model already supports seq_len {model.max_seq_len} >= {new_max_len}")
        return
    
    print(f"📏 Extending positional embeddings: {model.max_seq_len} → {new_max_len} ({method} interpolation)")
    
    device = model.pos_embedding.weight.device
    old_pe = model.pos_embedding.weight.data  # [old_max_len, hidden_size]
    old_max_len = model.max_seq_len
    hidden_size = model.hidden_size
    
    # Create new position embedding
    new_pe = nn.Embedding(new_max_len, hidden_size).to(device)
    
    with torch.no_grad():
        if method == 'linear':
            # Linear interpolation
            old_positions = torch.linspace(0, old_max_len - 1, old_max_len)
            new_positions = torch.linspace(0, old_max_len - 1, new_max_len)
            
            # Interpolate each dimension
            new_weights = torch.zeros(new_max_len, hidden_size, device=device)
            for dim in range(hidden_size):
                new_weights[:, dim] = torch.nn.functional.interpolate(
                    old_pe[:, dim].unsqueeze(0).unsqueeze(0),  # [1, 1, old_max_len]
                    size=new_max_len,
                    mode='linear',
                    align_corners=True
                ).squeeze()
            
        elif method == 'cosine':
            # Cosine interpolation (smoother)
            old_positions = torch.linspace(0, math.pi, old_max_len)
            new_positions = torch.linspace(0, math.pi, new_max_len)
            
            new_weights = torch.zeros(new_max_len, hidden_size, device=device)
            for i, new_pos in enumerate(new_positions):
                # Find closest old positions
                weights = torch.cos(torch.abs(old_positions - new_pos))
                weights = weights / weights.sum()
                
                # Weighted combination
                new_weights[i] = (old_pe * weights.unsqueeze(1)).sum(dim=0)
        
        else:
            raise ValueError(f"Unknown interpolation method: {method}")
        
        new_pe.weight.data = new_weights
    
    # Replace position embedding
    model.pos_embedding = new_pe
    
    # Update buffers
    causal_mask_full = torch.triu(torch.ones(new_max_len, new_max_len, dtype=torch.bool, device=device), diagonal=1)
    model.register_buffer('causal_mask_full', causal_mask_full, persistent=False)
    pos_ids_full = torch.arange(new_max_len, dtype=torch.long, device=device)
    model.register_buffer('pos_ids_full', pos_ids_full, persistent=False)
    model.max_seq_len = new_max_len
    
    print(f"✓ Position embeddings extended to {new_max_len}")

def convert_to_rope(model, max_seq_len=512):
    """
    Convert model from learned positional embeddings to RoPE
    
    Args:
        model: Model with learned positional embeddings
        max_seq_len: Maximum sequence length for RoPE
    """
    print(f"🔄 Converting model to RoPE (max_seq_len: {max_seq_len})")
    
    hidden_size = model.hidden_size
    num_heads = len(model.layers)  # Assuming each layer has the same number of heads
    
    # Get number of heads from first layer
    first_layer = model.layers[0]
    if hasattr(first_layer, 'self_attn'):
        num_heads = first_layer.self_attn.num_heads
    else:
        num_heads = 8  # Default fallback
    
    device = next(model.parameters()).device
    
    # Replace transformer layers with RoPE versions
    new_layers = nn.ModuleList()
    for i, old_layer in enumerate(model.layers):
        # Create new RoPE layer
        new_layer = RoPEAttentionLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            max_seq_len=max_seq_len,
            gradient_checkpointing=old_layer.gradient_checkpointing if hasattr(old_layer, 'gradient_checkpointing') else False
        ).to(device)
        
        # Copy weights from old layer (excluding attention projections which we'll reinit)
        with torch.no_grad():
            # Copy feed forward weights
            if hasattr(old_layer, 'linear1') and hasattr(new_layer, 'linear1'):
                new_layer.linear1.weight.copy_(old_layer.linear1.weight)
                if old_layer.linear1.bias is not None:
                    new_layer.linear1.bias.copy_(old_layer.linear1.bias)
            
            if hasattr(old_layer, 'linear2') and hasattr(new_layer, 'linear2'):
                new_layer.linear2.weight.copy_(old_layer.linear2.weight)
                if old_layer.linear2.bias is not None:
                    new_layer.linear2.bias.copy_(old_layer.linear2.bias)
            
            # Copy layer norms
            if hasattr(old_layer, 'norm1') and hasattr(new_layer, 'norm1'):
                new_layer.norm1.weight.copy_(old_layer.norm1.weight)
                new_layer.norm1.bias.copy_(old_layer.norm1.bias)
            
            if hasattr(old_layer, 'norm2') and hasattr(new_layer, 'norm2'):
                new_layer.norm2.weight.copy_(old_layer.norm2.weight)
                new_layer.norm2.bias.copy_(old_layer.norm2.bias)
            
            # Reinitialize attention projections for RoPE
            # (Keep existing initialization which should be reasonable)
        
        new_layers.append(new_layer)
        print(f"  ✓ Converted layer {i + 1}/{len(model.layers)} to RoPE")
    
    # Replace layers
    model.layers = new_layers
    
    # Remove positional embeddings (RoPE doesn't need them)
    del model.pos_embedding
    
    # Update buffers
    causal_mask_full = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device), diagonal=1)
    model.register_buffer('causal_mask_full', causal_mask_full, persistent=False)
    pos_ids_full = torch.arange(max_seq_len, dtype=torch.long, device=device)
    model.register_buffer('pos_ids_full', pos_ids_full, persistent=False)
    model.max_seq_len = max_seq_len
    
    # Update encode method to work without positional embeddings
    def new_encode(self, input_ids, pad_token_id=None):
        batch_size, seq_len = input_ids.shape
        
        # Only token embeddings (no positional)
        x = self.tok_embedding(input_ids)
        
        # Create causal mask for this sequence length
        causal_mask = self.causal_mask_full[:seq_len, :seq_len]
        
        # Create padding mask if pad_token_id is provided
        key_padding_mask = None
        if pad_token_id is not None:
            key_padding_mask = (input_ids == pad_token_id)
        
        # Apply transformer layers (now with RoPE)
        for layer in self.layers:
            x = layer(x, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        return x
    
    # Replace encode method
    import types
    model.encode = types.MethodType(new_encode, model)
    
    print(f"✓ Model converted to RoPE with max_seq_len: {max_seq_len}")

def extend_model_context(checkpoint_path, new_seq_len=512, method='rope', interpolation='linear', 
                        output_path=None, warmup_steps=2000):
    """
    Extend model context length from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        new_seq_len: New sequence length (default: 512)
        method: 'rope', 'interpolate', or 'rope_convert'
        interpolation: 'linear' or 'cosine' (for interpolate method)
        output_path: Output path for extended checkpoint
        warmup_steps: LR warmup steps for training continuation
    """
    print(f"🔧 Extending model context: {checkpoint_path}")
    print(f"   Method: {method}")
    print(f"   New seq_len: {new_seq_len}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model info
    model_config = checkpoint['model_config']
    old_seq_len = model_config['max_seq_len']
    
    if new_seq_len <= old_seq_len:
        print(f"Model already supports seq_len {old_seq_len} >= {new_seq_len}, no extension needed")
        return checkpoint_path
    
    print(f"   Current seq_len: {old_seq_len} → {new_seq_len}")
    
    # Create model
    from baseline_continue import SimpleLM  # Import model class
    
    model = SimpleLM(
        vocab_size=model_config['vocab_size'],
        hidden_size=model_config['hidden_size'],
        num_layers=model_config['num_layers'],
        max_seq_len=old_seq_len  # Start with old length
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Apply extension method
    if method == 'interpolate':
        extend_positional_embeddings_interpolation(model, new_seq_len, interpolation)
    elif method == 'rope':
        # Assume model already has RoPE - just extend
        model.extend_max_seq_len(new_seq_len)
    elif method == 'rope_convert':
        convert_to_rope(model, new_seq_len)
    else:
        raise ValueError(f"Unknown extension method: {method}")
    
    # Update checkpoint
    new_checkpoint = checkpoint.copy()
    new_checkpoint['model_state_dict'] = model.state_dict()
    new_checkpoint['model_config']['max_seq_len'] = new_seq_len
    new_checkpoint['extension_info'] = {
        'method': method,
        'old_seq_len': old_seq_len,
        'new_seq_len': new_seq_len,
        'interpolation': interpolation if method == 'interpolate' else None,
        'extended_at': datetime.now().isoformat(),
        'warmup_steps_recommended': warmup_steps
    }
    
    # Generate output path
    if output_path is None:
        base_name = os.path.splitext(checkpoint_path)[0]
        output_path = f"{base_name}_extended_{new_seq_len}.pt"
    
    # Save extended checkpoint
    torch.save(new_checkpoint, output_path)
    
    # Save summary
    summary_path = output_path.replace('.pt', '_extension_summary.json')
    summary = {
        'original_checkpoint': checkpoint_path,
        'extended_checkpoint': output_path,
        'extension_method': method,
        'old_seq_len': old_seq_len,
        'new_seq_len': new_seq_len,
        'interpolation_method': interpolation if method == 'interpolate' else None,
        'warmup_steps_recommended': warmup_steps,
        'extended_at': datetime.now().isoformat(),
        'usage_notes': {
            'training_continuation': f"Use --seq_length {new_seq_len} with LR warmup for {warmup_steps} steps",
            'load_checkpoint': f"Load {output_path} to continue training with extended context",
            'recommended_lr_schedule': "Reduce LR by 50% for first 2000 steps, then restore"
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ Context extension complete!")
    print(f"   Extended checkpoint: {output_path}")
    print(f"   Extension summary: {summary_path}")
    print(f"   🚀 Ready for training with seq_len {new_seq_len}")
    print(f"   💡 Recommended: Use LR warmup for {warmup_steps} steps")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Context Extension: Extend seq_len 256 → 512")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint to extend")
    parser.add_argument("--new_seq_len", type=int, default=512, help="New sequence length")
    parser.add_argument("--method", type=str, choices=['interpolate', 'rope', 'rope_convert'], 
                       default='rope_convert', help="Extension method")
    parser.add_argument("--interpolation", type=str, choices=['linear', 'cosine'], 
                       default='linear', help="Interpolation method (for interpolate)")
    parser.add_argument("--output", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Recommended warmup steps")
    
    args = parser.parse_args()
    
    # Extend context
    extended_checkpoint = extend_model_context(
        checkpoint_path=args.checkpoint,
        new_seq_len=args.new_seq_len,
        method=args.method,
        interpolation=args.interpolation,
        output_path=args.output,
        warmup_steps=args.warmup_steps
    )
    
    print(f"\n🎯 Next steps:")
    print(f"1. Continue training with: --load_checkpoint {extended_checkpoint} --seq_length {args.new_seq_len}")
    print(f"2. Use LR warmup for first {args.warmup_steps} steps")
    print(f"3. Monitor performance on longer sequences")

if __name__ == "__main__":
    main()