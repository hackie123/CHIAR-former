# model/rbf_mix.py — RBF Kernel Mixing (ablation only, not used in default DCT+Attn)

import torch, math
import torch.nn as nn


class RBFMix(nn.Module):
    def __init__(self, d_model, n_random_features=64, gamma=1.0, dropout=0.1):
        super().__init__()
        self.R    = n_random_features
        omega     = torch.randn(d_model, n_random_features) * math.sqrt(2 * gamma)
        self.register_buffer("omega", omega)
        self.out_proj = nn.Linear(2 * n_random_features, d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d  = x.shape
        proj     = x @ self.omega / math.sqrt(self.R)
        phi      = torch.cat([proj.cos(), proj.sin()], dim=-1)
        A        = torch.softmax(phi @ phi.transpose(-2, -1), dim=-1)
        return self.norm(x + self.ffn(self.out_proj(A @ phi)))
