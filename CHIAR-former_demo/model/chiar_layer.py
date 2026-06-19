# model/chiar_layer.py — CHIAR Layer with RoPE attention
#
# Parameter design:
#   Each CHIARLayer has exactly ONE FFN regardless of operator selected.
#   This ensures parameter parity between baseline and CHIAR.
#
#   Structure per layer:
#     Operator sub-layer (DCT or Attention) + LayerNorm
#     Shared FFN sub-layer + LayerNorm
#
#   BaselineTransformer layers use same structure (Attn + FFN).
#   CHIAR routing layers select DCT or Attn, then pass through same FFN.

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dct_mix import DCTMix
from .rbf_mix import RBFMix
from .router  import SpectralRouter
from .rope    import RotaryEmbedding


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE.
    No FFN here — FFN lives at the CHIARLayer level (shared across operators).
    """
    def __init__(self, d_model, n_heads, dropout=0.1,
                 max_seq_len=512, rope_base=10000.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads
        self.qkv      = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.drop     = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(d_model)
        self.rope     = RotaryEmbedding(self.d_head, max_seq_len, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d  = x.shape
        residual = x
        q, k, v  = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k = self.rope(q, k)
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = attn.masked_fill(~mask, float("-inf"))
        attn = self.drop(F.softmax(attn, dim=-1))
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, d)
        return self.norm(residual + self.out_proj(out))


class DCTMixNoFFN(nn.Module):
    """
    DCT mixing sub-layer without FFN.
    FFN lives at CHIARLayer level.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        from .dct_mix import dct, idct
        self._dct  = dct
        self._idct = idct
        self.w    = nn.Parameter(torch.ones(d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self._idct(self._dct(x) * self.w))


class CHIARLayer(nn.Module):
    """
    Single CHIAR-Former layer.

    Structure:
      1. Operator sub-layer (DCT or Attn) — selected by SpectralRouter
      2. Shared FFN sub-layer + LayerNorm

    This ensures each layer has exactly the same number of parameters
    regardless of which operator is selected, enabling fair comparison
    with the baseline.
    """
    def __init__(self, d_model, n_heads, use_dct, use_rbf, use_attn,
                 n_random_features=64, rbf_gamma=1.0,
                 tau_low=0.855, tau_high=0.865,
                 routing_mode="threshold", dropout=0.1,
                 max_seq_len=512, rope_base=10000.0, **kwargs):
        super().__init__()
        self.use_dct  = use_dct
        self.use_rbf  = use_rbf
        self.use_attn = use_attn
        n_active      = sum([use_dct, use_rbf, use_attn])
        self.needs_routing = (n_active > 1)

        # Operator sub-layers (no FFN)
        if use_dct:
            self.dct_mix = DCTMixNoFFN(d_model, dropout)
        if use_rbf:
            # RBF keeps its own FFN (ablation only, not used in default)
            self.rbf_mix = RBFMix(d_model, n_random_features, rbf_gamma, dropout)
        if use_attn:
            self.attn = MultiHeadSelfAttention(
                d_model, n_heads, dropout, max_seq_len, rope_base)

        # Shared FFN for all operators in this layer
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout))
        self.ffn_norm = nn.LayerNorm(d_model)

        if self.needs_routing:
            self.router = SpectralRouter(
                d_model, tau_low, tau_high, routing_mode, n_ops=2)

    def forward(self, x: torch.Tensor):
        info = {"H": None, "gates": None, "op_idx": None}

        if not self.needs_routing:
            if self.use_dct:  x = self.dct_mix(x)
            elif self.use_rbf: x = self.rbf_mix(x)
            else:              x = self.attn(x)
        else:
            gates, H, op_idx = self.router(x)
            info.update({"H":      H.detach(),
                         "gates":  gates.detach(),
                         "op_idx": op_idx.detach()})
            outputs = []
            if self.use_dct:  outputs.append(self.dct_mix(x))
            if self.use_rbf:  outputs.append(self.rbf_mix(x))
            if self.use_attn: outputs.append(self.attn(x))
            stacked = torch.stack(outputs, dim=-1)
            x = (stacked * gates[..., :len(outputs)].unsqueeze(2)).sum(-1)

        # Shared FFN
        x = self.ffn_norm(x + self.ffn(x))
        return x, info
