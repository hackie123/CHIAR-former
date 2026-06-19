# model/router.py — Spectral Router (per-token)
# Routes each token to DCT or Attention based on spectral entropy H(x).

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .dct_mix import dct as _dct


class SpectralRouter(nn.Module):
    """
    Per-token operator router.
    modes: threshold (default) | soft | hard

    threshold: H(x) > tau_mid → Attention, else → DCT
    hard:      learned argmax routing via STE
    soft:      learned weighted combination
    """
    def __init__(self, d_model, tau_low=0.855, tau_high=0.865,
                 mode="threshold", n_ops=2):
        super().__init__()
        self.tau_low  = tau_low
        self.tau_high = tau_high
        self.mode     = mode
        self.n_ops    = n_ops
        if mode in ("soft", "hard"):
            self.gate_proj = nn.Linear(d_model + 1, n_ops)

    def spectral_entropy(self, x: torch.Tensor) -> torch.Tensor:
        xf = _dct(x)
        p  = xf.pow(2)
        p  = p / (p.sum(-1, keepdim=True) + 1e-8)
        H  = -(p * (p + 1e-8).log()).sum(-1)
        return H / (math.log(x.shape[-1]) + 1e-8)   # normalise to [0,1]

    def routing_entropy(self, gates: torch.Tensor) -> torch.Tensor:
        """Operator utilisation entropy — used by collapse regulariser."""
        q = gates.mean(dim=[0, 1])
        return -(q * (q + 1e-8).log()).sum()

    def forward(self, x: torch.Tensor):
        H     = self.spectral_entropy(x)              # (B, T)
        tau_m = (self.tau_low + self.tau_high) / 2

        if self.mode == "threshold":
            op_idx = (H > tau_m).long()               # 0=DCT, 1=Attn
            gates  = F.one_hot(op_idx, self.n_ops).float()
        elif self.mode == "hard":
            feat   = torch.cat([x, H.unsqueeze(-1)], dim=-1)
            logit  = self.gate_proj(feat)
            op_idx = logit.argmax(-1)
            gates  = (F.one_hot(op_idx, self.n_ops).float()
                      - logit.softmax(-1)).detach() + logit.softmax(-1)
        else:  # soft
            feat   = torch.cat([x, H.unsqueeze(-1)], dim=-1)
            gates  = self.gate_proj(feat).softmax(-1)
            op_idx = gates.argmax(-1)

        return gates, H, op_idx