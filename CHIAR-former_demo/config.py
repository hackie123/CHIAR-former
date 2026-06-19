# config.py — CHIAR-Former v3
# Default: 350M scale for A40 (48GB VRAM)

class Config:
    # ── Model ─────────────────────────────────────────────────────────────────
    vocab_size        = 50257
    d_model           = 1024
    n_heads           = 16
    n_layers          = 28
    max_seq_len       = 256
    dropout           = 0.1

    # ── RoPE (replaces learned absolute PE) ───────────────────────────────────
    rope_base         = 10000.0

    # ── Per-token spectral router ─────────────────────────────────────────────
    tau_low           = 0.8935    # updated by calibrate_tau.py after baseline run
    tau_high          = 0.8973
    routing_mode      = "threshold"
    layer_variant     = "dct_attn"

    # ── RBF (ablation only) ───────────────────────────────────────────────────
    rbf_gamma         = 1.0
    n_random_features = 64

    # ── MetaRouter ────────────────────────────────────────────────────────────
    use_meta_router   = True
    meta_mix_ratio    = 0.25     # fraction of non-WikiText103 in mixed batches

    # ── Collapse regulariser ──────────────────────────────────────────────────
    use_collapse_reg  = False
    lambda_reg        = 0.01

    # ── Training — tuned for A40 48GB ─────────────────────────────────────────
    batch_size        = 8
    grad_accum_steps  = 16       # effective batch = 128
    learning_rate     = 1e-4
    weight_decay      = 0.01
    max_epochs        = 5
    warmup_steps      = 1000
    clip_grad_norm    = 1.0

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every         = 100
    eval_every        = 1000
    checkpoint_dir    = "./checkpoints"
    device            = "cuda"


class SmallConfig(Config):
    """17M param config — used for existing ablation results."""
    d_model           = 256
    n_heads           = 4
    n_layers          = 4
    batch_size        = 32
    grad_accum_steps  = 4
    max_epochs        = 10
    warmup_steps      = 500
    eval_every        = 500
