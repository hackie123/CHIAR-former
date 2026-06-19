# model/meta_router.py — Learned Task-Level Meta-Router
#
# Decides per-batch whether to use L1 DCT preprocessing (naturalistic text)
# or bypass it with identity (symbolic/discrete tasks).
#
# Fully learned via end-to-end training on mixed batches from all 4 datasets.
# No manual thresholds — model discovers naturalistic/symbolic boundary.
#
# Expected gate values after mixed training:
#   WikiText-103 → ~0.95  (use DCT)
#   WikiText-2   → ~0.90  (use DCT)
#   IMDB         → ~0.88  (use DCT)
#   ListOps      → ~0.05  (bypass DCT)

import torch
import torch.nn as nn


class MetaRouter(nn.Module):
    """
    Learned task-level gate.
    gate ~1.0 → naturalistic → L1 uses DCT
    gate ~0.0 → symbolic     → L1 uses Identity (bypass)

    Architecture: mean-pool batch → Linear(d, 1) → sigmoid
    Bias initialised to 2.0 so gate starts near 1.0 (use DCT).
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) → scalar gate in [0, 1]"""
        pooled = x.mean(dim=[0, 1])                      # (d,)
        return torch.sigmoid(self.gate_proj(pooled)).squeeze()

    def gate_value(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            return self.forward(x).item()
