# model/chiar_classifier.py — CHIAR-Former Classification Head for LRA

import torch
import torch.nn as nn
from .chiar_former import CHIARFormer
from config import Config


class CHIARClassifier(nn.Module):
    """CHIARFormer backbone + mean-pool + linear classification head."""
    def __init__(self, cfg: Config, num_classes: int, vocab_size: int = None):
        super().__init__()
        if vocab_size is not None:
            cfg.vocab_size = vocab_size
        self.backbone   = CHIARFormer(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, num_classes))

    def forward(self, input_ids: torch.Tensor):
        x = self.backbone.emb_drop(self.backbone.token_emb(input_ids))
        meta_gate = self.backbone.meta_router(x) \
            if self.backbone.meta_router is not None \
            else torch.tensor(1.0, device=x.device)
        for i, layer in enumerate(self.backbone.layers):
            if i == 0:
                x_dct, _ = layer(x)
                x = meta_gate * x_dct + (1.0 - meta_gate) * x
            else:
                x, _ = layer(x)
        pooled = self.backbone.norm(x).mean(dim=1)
        return self.classifier(pooled), torch.tensor(0.0, device=x.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
