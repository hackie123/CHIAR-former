# model/chiar_former.py — CHIAR-Former v3
#
# Architecture:
#   Token Embedding (no positional embedding — RoPE handles it)
#   MetaRouter: learned task-level gate (naturalistic→DCT, symbolic→bypass)
#   L1:          DCT only             [fixed — spectral preprocessing]
#   L2 .. L(n-1): DCT | Attn routing  [SpectralRouter per token]
#   Ln:           Attn only           [fixed — accuracy anchor]
#   LayerNorm → LM Head (weight-tied)
#
# n_layers from config (default 8 for ~350M scale).

import torch
import torch.nn as nn
from .chiar_layer import CHIARLayer
from .meta_router import MetaRouter
from config import Config


class CHIARFormer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Token embedding only — no positional embedding
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # Learned task-level gate
        self.meta_router = MetaRouter(cfg.d_model) \
            if getattr(cfg, "use_meta_router", True) else None

        kw = dict(
            d_model           = cfg.d_model,
            n_heads           = cfg.n_heads,
            n_random_features = cfg.n_random_features,
            rbf_gamma         = cfg.rbf_gamma,
            tau_low           = cfg.tau_low,
            tau_high          = cfg.tau_high,
            routing_mode      = cfg.routing_mode,
            dropout           = cfg.dropout,
            max_seq_len       = cfg.max_seq_len,
            rope_base         = cfg.rope_base,
        )

        n      = cfg.n_layers
        layers = []
        for i in range(n):
            if i == 0:
                # L1: DCT only — spectral preprocessing
                layers.append(CHIARLayer(
                    use_dct=True, use_rbf=False, use_attn=False, **kw))
            elif i == n - 1:
                # Ln: Attention only — accuracy anchor
                layers.append(CHIARLayer(
                    use_dct=False, use_rbf=False, use_attn=True, **kw))
            else:
                # L2..L(n-1): DCT | Attn routing via spectral entropy
                layers.append(CHIARLayer(
                    use_dct=True, use_rbf=False, use_attn=True, **kw))

        self.layers  = nn.ModuleList(layers)
        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, input_ids: torch.Tensor,
                return_routing_info: bool = False):
        x = self.emb_drop(self.token_emb(input_ids))

        # MetaRouter: learned gate
        # gate ~1.0 → naturalistic → L1 uses DCT
        # gate ~0.0 → symbolic     → L1 uses Identity
        meta_gate = self.meta_router(x) \
            if self.meta_router is not None \
            else torch.tensor(1.0, device=x.device)

        routing_infos = []
        routing_loss  = torch.tensor(0.0, device=x.device)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x_dct, info = layer(x)
                # Soft blend: gate*DCT + (1-gate)*identity
                x = meta_gate * x_dct + (1.0 - meta_gate) * x
            else:
                x, info = layer(x)

            routing_infos.append(info)

            if self.cfg.use_collapse_reg and info["gates"] is not None:
                U = layer.router.routing_entropy(info["gates"])
                routing_loss = routing_loss - self.cfg.lambda_reg * U

        logits = self.lm_head(self.norm(x))

        if return_routing_info:
            return logits, routing_infos, routing_loss, meta_gate
        return logits, routing_loss

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
