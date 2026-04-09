# bitnet/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .core import BitLinear
import math

class BitSubLN(nn.Module):
    """Layer normalization placed before quantization as Sub-LN."""
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.ln(x)

class BitSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, bits_act=8, groups=1, per_token_act=False, attn_dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        assert self.head_dim * n_head == d_model
        self.subln = BitSubLN(d_model)
        # Q,K,V projections - use BitLinear
        self.q_proj = BitLinear(d_model, d_model, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
        self.k_proj = BitLinear(d_model, d_model, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
        self.v_proj = BitLinear(d_model, d_model, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
        self.out_proj = BitLinear(d_model, d_model, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, kv_cache=None, attn_mask=None):
        """
        x: [B,T,D]
        kv_cache: optional dict with 'k' and 'v' tensors to append for generation
        Returns: out [B,T,D], optional updated kv_cache
        """
        B, T, D = x.shape
        x_ln = self.subln(x)  # Sub-LN as paper recommends (LayerNorm before quant)
        q = self.q_proj(x_ln).view(B, T, self.n_head, self.head_dim).transpose(1,2)  # B,H,T,hd
        k = self.k_proj(x_ln).view(B, T, self.n_head, self.head_dim).transpose(1,2)
        v = self.v_proj(x_ln).view(B, T, self.n_head, self.head_dim).transpose(1,2)

        # handle kv cache for efficient generation: concat along time dim
        if kv_cache is not None:
            # kv_cache['k']: [B, H, T_cache, head_dim]
            k = torch.cat([kv_cache['k'], k], dim=2)
            v = torch.cat([kv_cache['v'], v], dim=2)

        # update cache pointer
        new_cache = {'k': k, 'v': v}

        # scaled dot-product
        q = q / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # B,H,T_q, T_k
        if attn_mask is not None:
            attn_scores = attn_scores + attn_mask
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_probs, v)  # B,H,T,hd
        attn_out = attn_out.transpose(1,2).contiguous().view(B, T, D)
        out = self.out_proj(attn_out)
        return out, new_cache

class BitFFN(nn.Module):
    def __init__(self, d_model, d_ff, bits_act=8, groups=1, per_token_act=False):
        super().__init__()
        self.subln = BitSubLN(d_model)
        self.fc1 = BitLinear(d_model, d_ff, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
        self.fc2 = BitLinear(d_ff, d_model, bits_act=bits_act, groups=groups, per_token_act=per_token_act)

    def forward(self, x):
        x_ln = self.subln(x)
        y = self.fc2(F.gelu(self.fc1(x_ln)))
        return y

class BitTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff=None, bits_act=8, groups=1, per_token_act=False):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.attn = BitSelfAttention(d_model, n_head, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
        self.ffn = BitFFN(d_model, d_ff, bits_act=bits_act, groups=groups, per_token_act=per_token_act)

    def forward(self, x, kv_cache=None, attn_mask=None):
        attn_out, cache = self.attn(x, kv_cache=kv_cache, attn_mask=attn_mask)
        x = x + attn_out
        ffn_out = self.ffn(x)
        x = x + ffn_out
        return x, cache

class BitNetDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=160, n_layers=55, n_heads=8, d_ff=None,
                 bits_act=8, groups=1, max_seq_len=2048, per_token_act=False):
        super().__init__()
        NUM_FUNCTIONS = 8
        NUM_DOMAINS = 6
        NUM_LOCALIZATIONS = 6
        NUM_GO = 5

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            BitTransformerBlock(d_model, n_heads, d_ff=d_ff, bits_act=bits_act, groups=groups, per_token_act=per_token_act)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)  # keep final head high-precision       
        self.func_head = nn.Linear(d_model, NUM_FUNCTIONS)
        self.domain_head = nn.Linear(d_model, NUM_DOMAINS)
        self.loc_head = nn.Linear(d_model, NUM_LOCALIZATIONS)
        self.go_head = nn.Linear(d_model, NUM_GO)
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, attention_mask=None, kv_caches=None):
        """
        input_ids: [B, T]
        kv_caches: optional list of per-layer kv dicts for generation caching
        Return: logits [B,T,V]
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) + self.pos_embed(positions)
        new_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = None
            if kv_caches is not None:
                kv_cache = kv_caches[i]
            x, new_cache = layer(x, kv_cache=kv_cache)
            new_caches.append(new_cache)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits, x, new_caches

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, eos_token_id=None, temperature=1.0, top_k=50):
        """
        Simple autoregressive generation using per-token activation quantization (paper recommends per-token for inference).
        This uses the kv-cache mechanism for speed.
        """
        device = input_ids.device
        B, T = input_ids.shape
        kv_caches = [None] * len(self.layers)
        cur = input_ids
        for step in range(max_new_tokens):
            logits, _, kv_caches = self.forward(cur, kv_caches=kv_caches)
            # take last token logits
            last = logits[:, -1, :] / max(temperature, 1e-8)
            # top-k sampling
            # top-k sampling
            if top_k is not None:
                top_k_val = min(top_k, last.size(-1))  # ensures top_k <= vocab size
                v, ix = torch.topk(last, top_k_val, dim=-1)  # [B, top_k_val]

                # sample per batch
                probs = F.softmax(v, dim=-1)              # [B, top_k_val]
                next_token_idx = torch.multinomial(probs, num_samples=1)  # [B,1]
                next_token = ix.gather(dim=-1, index=next_token_idx)      # [B,1]
            else:
                probs = F.softmax(last, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)      # [B,1]

            # append new token
            cur = torch.cat([cur, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        return cur
    
    def predict_function(self, hidden_states, attention_mask=None):
        """
        hidden_states: [B, T, D]
        attention_mask: [B, T] or None
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        return self.func_head(pooled)

    def predict_domain(self, hidden_states, attention_mask=None):
        pooled = self._pool(hidden_states, attention_mask)
        return self.domain_head(pooled)

    def predict_localization(self, hidden_states, attention_mask=None):
        pooled = self._pool(hidden_states, attention_mask)
        return self.loc_head(pooled)

    def predict_go(self, hidden_states, attention_mask=None):
        pooled = self._pool(hidden_states, attention_mask)
        return self.go_head(pooled)

    def _pool(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        return pooled
