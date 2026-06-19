# model/dct_mix.py — DCT Spectral Mixing Layer
# FFT-based Type-II DCT (torch.fft.dct does not exist in PyTorch).
# Mathematically identical to scipy DCT Type-II with norm="ortho".

import torch
import torch.nn as nn
import math


def dct(x: torch.Tensor) -> torch.Tensor:
    """Type-II DCT via FFT. Ortho-normalised. Input/output: (B, T, d)."""
    B, T, d = x.shape
    v  = torch.cat([x, x.flip(dims=[-1])], dim=-1)       # (B, T, 2d)
    Vc = torch.fft.rfft(v, dim=-1)                       # (B, T, d+1)
    k  = torch.arange(d, device=x.device, dtype=x.dtype)
    w  = 2 * torch.exp(-1j * math.pi * k / (2 * d))      # (d,)
    return (Vc[..., :d] * w).real / math.sqrt(2 * d)


def idct(x: torch.Tensor) -> torch.Tensor:
    """Type-III DCT (inverse of Type-II) via FFT. Input/output: (B, T, d)."""
    B, T, d = x.shape
    k  = torch.arange(d, device=x.device, dtype=x.dtype)
    w  = torch.exp(1j * math.pi * k / (2 * d)) * math.sqrt(2 * d)
    xc = torch.complex(x, torch.zeros_like(x)) * w       # (B, T, d)
    xc = torch.cat([xc, torch.zeros(B, T, 1, dtype=xc.dtype, device=x.device)], dim=-1)
    return torch.fft.irfft(xc, n=2*d, dim=-1)[..., :d]


class DCTMix(nn.Module):
    """
    Learnable spectral filter in DCT domain.
    DCTMix(X) = LN(X + FFN(iDCT(DCT(X) ⊙ w)))
    Complexity: O(d log d) per token.
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.w    = nn.Parameter(torch.ones(d_model))
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.ffn(idct(dct(x) * self.w)))