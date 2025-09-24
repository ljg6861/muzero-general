import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class FlashAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attn_mask=None, key_padding_mask=None):
        x = src
        x2, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False, is_causal=False)
        x = self.norm1(x + self.dropout1(x2))
        x2 = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = self.norm2(x + self.dropout2(x2))
        return x

class SimpleLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_layers=3, num_heads=4, max_seq_len=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.tok_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.layers = nn.ModuleList([
            FlashAttentionLayer(hidden_size, num_heads, dim_ff=hidden_size*4, dropout=0.0)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        causal_mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask_full', causal_mask, persistent=False)
        pos_ids_full = torch.arange(max_seq_len, dtype=torch.long)
        self.register_buffer('pos_ids_full', pos_ids_full, persistent=False)

    def encode(self, input_ids, pad_token_id=None):
        bsz, seqlen = input_ids.shape
        pos_ids = self.pos_ids_full[:seqlen].unsqueeze(0).expand(bsz, -1)
        x = self.tok_embedding(input_ids) + self.pos_embedding(pos_ids)
        attn_mask = self.causal_mask_full[:seqlen, :seqlen]
        key_padding_mask = (input_ids == pad_token_id) if pad_token_id is not None else None
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x

    def forward(self, input_ids, pad_token_id=None):
        h = self.encode(input_ids, pad_token_id)
        return self.lm_head(h)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=50, top_p=1.0, pad_token_id=None, eos_token_id=None):
        device = input_ids.device
        for _ in range(max_new_tokens):
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]
            logits = self.forward(input_ids, pad_token_id=pad_token_id)[:, -1, :]
            logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            if top_k > 0:
                v, ix = torch.topk(probs, min(top_k, probs.size(-1)))
                mask = torch.full_like(probs, float('-inf'))
                mask.scatter_(1, ix, torch.log(v))
                logits = mask
                probs = torch.softmax(logits, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cdf = torch.cumsum(sorted_probs, dim=-1)
                mask = cdf <= top_p
                mask[..., 1:] = mask[..., :-1].clone()
                mask[..., 0] = True
                probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = sorted_idx.gather(1, next_token)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return input_ids


def load_checkpoint_into_simple_lm(state_dict, model: SimpleLM):
    """Load state dict from training checkpoints, handling positional embedding resize and module prefixes."""
    sd = state_dict
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(sd.keys())
    if model_keys and ckpt_keys:
        if model_keys[0].startswith('module.') and not ckpt_keys[0].startswith('module.'):
            sd = {f'module.{k}': v for k, v in sd.items()}
        elif not model_keys[0].startswith('module.') and ckpt_keys[0].startswith('module.'):
            sd = {k.replace('module.', ''): v for k, v in sd.items()}
    # Handle positional embedding
    pos_key = 'module.pos_embedding.weight' if any(k.startswith('module.') for k in model.state_dict().keys()) else 'pos_embedding.weight'
    if pos_key in sd:
        cur_shape = model.state_dict()[pos_key].shape
        ckpt_pe = sd[pos_key]
        if ckpt_pe.shape != cur_shape:
            new_pe = model.state_dict()[pos_key].clone()
            n = min(cur_shape[0], ckpt_pe.shape[0])
            with torch.no_grad():
                new_pe[:n].copy_(ckpt_pe[:n])
            sd[pos_key] = new_pe
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected
