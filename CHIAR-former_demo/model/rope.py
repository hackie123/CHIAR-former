# model/rope.py — Rotary Position Embedding (RoPE)
# Su et al. "RoFormer" arXiv:2104.09864
# Applied inside attention to Q and K. Zero learnable parameters.
# Used by: LLaMA, Qwen, Mistral, GPT-NeoX, Falcon

import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    def __init__(self, d_head: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t   = torch.arange(seq_len, device=self.inv_freq.device).float()
        emb = torch.cat([torch.outer(t, self.inv_freq)] * 2, dim=-1)
        self.register_buffer("cos_cache", emb.cos())
        self.register_buffer("sin_cache", emb.sin())

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]
        if T > self.cos_cache.shape[0]:
            self._build_cache(T * 2)
        cos = self.cos_cache[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cache[:T].unsqueeze(0).unsqueeze(0)
        return (q * cos + self._rotate_half(q) * sin,
                k * cos + self._rotate_half(k) * sin)
