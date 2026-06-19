# CHIAR-Former

**Chiaroscuro Attention: Spending Compute in the Dark**
*Operator Routing via Spectral Entropy Across Tasks and Scales*

Prateek Kumar Sikdar · AI Architect, Agentic AI Practice · Accenture, Bengaluru · 2026

[![arXiv](https://img.shields.io/badge/arXiv-2606.08327-b31b1b.svg)](https://arxiv.org/abs/2606.08327)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

---

## Overview

CHIAR-Former is an efficient transformer that routes each token to either **DCT spectral mixing** (O(d log d)) or **full self-attention** (O(n²d)) based on per-token spectral entropy H(x) — the information entropy of the token's DCT power spectrum.

The name comes from the Italian Renaissance painting technique *chiaroscuro* (light/dark), used by Caravaggio, Leonardo, and Rembrandt to spend illumination only where the eye needs detail. CHIAR-Former applies the same principle to computation: spend the attention budget where the signal is **dark** (high entropy, complex content words), and use cheap DCT mixing where the signal is **light** (low entropy, smooth function words).

**Key results at 400M parameters on WikiText-103:**

| Model | Params | Val PPL | Test PPL | FLOP reduction |
|---|---|---|---|---|
| Baseline (Full Attention + RoPE) | 404M | 23.73 | 23.58 | — |
| CHIAR-Former DCT+Attn | **400M** | 27.75 | 27.51 | **~37%** |
| CHIAR-Former Mixed Training | 400M | 28.81 | 28.56 | ~37% |

**Three core contributions:**

1. **Routing Collapse Discovery** — a three-operator system (DCT + RBF + Attention) collapses during training: RBF drops to 0% usage, revealing DCT+Attention as the sufficient operator pair.
2. **Theory-Grounded Spectral Router** — per-token routing via spectral entropy H(x), grounded in the Karhunen-Loève optimality of DCT for low-entropy (stationary) signals.
3. **Learned MetaRouter** — a task-level gate g = σ(Linear(mean(x))) ∈ [0,1] trained end-to-end on mixed batches, stabilising at g ≈ 0.22 — a compute–quality equilibrium between spectral structure and attention capacity.

---

## Repository Structure

```
CHIAR-former/                  ← Training codebase
│
├── config.py                  # Config (350M default) + SmallConfig (17M ablations)
├── train.py                   # WikiText-103 training: baseline + all CHIAR ablations
├── train_all.py               # Multi-dataset: WikiText-2, LRA (IMDB, ListOps), mixed
├── calibrate_tau.py           # Empirical tau calibration from trained baseline
├── analyze_all.py             # Generate all paper tables and figures
├── qualitative.py             # Text continuations + routing heatmap figure
├── test_forward.py            # 11 forward pass sanity tests (run before training)
├── requirements.txt
│
├── model/
│   ├── chiar_former.py        # Main model: embedding → MetaRouter → layers → LM head
│   ├── chiar_layer.py         # CHIARLayer: operator sub-layer + shared FFN
│   ├── router.py              # SpectralRouter: threshold / soft / hard modes
│   ├── meta_router.py         # MetaRouter: learned task-level gate
│   ├── dct_mix.py             # FFT-based Type-II DCT + learned spectral filter w
│   ├── rope.py                # Rotary Position Embedding (zero learnable params)
│   ├── rbf_mix.py             # RBF mixing (ablation only — collapses in training)
│   ├── chiar_classifier.py    # CHIAR classifier head for LRA tasks
│   └── __init__.py
│
├── data/
│   ├── wikitext.py            # WikiText-103 and WikiText-2 loaders (GPT-2 tokeniser)
│   ├── lra.py                 # LRA task loaders: IMDB (sentiment), ListOps (symbolic)
│   └── __init__.py
│
├── utils/
│   └── flop_counter.py        # FLOP estimation for baseline vs CHIAR comparison
│
└── checkpoints/               # Saved model checkpoints (created during training)

CHIAR-former_demo/             ← Interactive Streamlit heatmap app
│
├── app.py                     # Streamlit demo: live per-token routing visualisation
├── config.py                  # Mirrors training config (used to load checkpoint)
├── requirements.txt           # streamlit + torch + transformers
├── model/                     # Same model code as CHIAR-former/model/
├── data/                      # Same data loaders (unused in demo, kept for parity)
└── checkpoints/               # Place trained checkpoint here for live routing
```

---

## Architecture

CHIAR-Former v3 has the following layer structure for an N-layer model:

```
Token Embeddings  (no absolute PE — RoPE handles position inside attention)
        ↓
MetaRouter  g = σ( Linear( mean(x) ) )  ∈ [0, 1]
  gate ≈ 1 → naturalistic text → L1 uses DCT Mixing
  gate ≈ 0 → symbolic input   → L1 uses Identity bypass
        ↓
L1:          DCT Mixing only         [spectral preprocessing, soft-gated by MetaRouter]
L2 … L(N-1): SpectralRouter         [per-token: H(x) ≤ τ → DCT | H(x) > τ → Attention+RoPE]
LN:          Full Attention only     [accuracy anchor, always runs]
        ↓
LayerNorm → LM Head  (weight-tied with token embeddings)
```

Every layer uses a **shared FFN** (operator sub-layer + FFN + LayerNorm), ensuring exact parameter parity with the baseline regardless of which operator is selected.

### Key Components

**`model/dct_mix.py` — DCT Spectral Mixing**

Implements Type-II DCT via FFT (PyTorch has no `torch.fft.dct`):

```
DCTMix(X) = LN( X + FFN( iDCT( DCT(X) ⊙ w ) ) )
```

where `w ∈ ℝᵈ` is a learned per-frequency spectral filter. Complexity: O(d log d) per token.

**`model/router.py` — SpectralRouter**

Computes spectral entropy H(x) ∈ [0, 1] for each token embedding x:

```
p_i  = x̂_i² / Σ_j x̂_j²       (normalised DCT power spectrum)
H(x) = -1/log(d) · Σ p_i log p_i   (normalised to [0,1])
```

Routing modes:
- `threshold` (default) — H(x) > τ_mid → Attention, else → DCT. Theory-driven, no learnable parameters.
- `soft` — learned weighted combination via `gate_proj = Linear(d+1, 2)`.
- `hard` — learned argmax with Straight-Through Estimator (STE) for gradient flow.

**`model/meta_router.py` — MetaRouter**

Task-level gate trained end-to-end:

```python
pooled = x.mean(dim=[0, 1])           # (d,) — mean over batch B and sequence T
g = sigmoid( Linear(d, 1)(pooled) )   # scalar gate ∈ [0,1]
h1 = g · DCTMix(X) + (1-g) · X       # soft blend at L1
```

Bias initialised to 2.0 → g ≈ 0.88 at training start (favours DCT). Converges to g ≈ 0.22 — a stable equilibrium where spectral preprocessing and attention play complementary roles.

**`model/rope.py` — Rotary Position Embedding**

Encodes relative positions by rotating Q and K vectors inside every attention layer. Zero learnable parameters; generalises beyond training sequence length. Used by LLaMA, Qwen, Mistral, Falcon.

---

## Setup

**Requirements:** Python 3.10+, CUDA GPU (tested on NVIDIA RTX A5000 24GB and A40 48GB).

```bash
git clone https://github.com/<your-repo>/CHIAR-former.git
cd CHIAR-former

pip install -r requirements.txt
```

Verify the full installation before training:

```bash
python test_forward.py
```

All 11 tests must pass. These cover: RoPE rotation, DCT/iDCT round-trip, RBF mixing, SpectralRouter entropy bounds, MetaRouter gradient flow, full forward pass, routing info extraction, no absolute PE, backward pass, baseline parameter parity, and baseline forward.

---

## Training

Training runs in a fixed sequence. Each step depends on the previous checkpoint.

### Step 1 — Baseline (WikiText-103)

```bash
# 350M scale (default) — ~8 hrs on A40 / ~12 hrs on A5000
python train.py --baseline

# 17M scale (ablation)
python train.py --baseline --small
```

Saves to: `checkpoints/baseline_wikitext103_350M_best.pt`

### Step 2 — Calibrate τ

Run once after baseline training. Measures the 33rd/67th percentile of spectral entropy H(x) on the validation set and prints the values to copy into `config.py`.

```bash
python calibrate_tau.py
# → Update config.py: tau_low = 0.8935, tau_high = 0.8973  (example values)
```

### Step 3 — CHIAR-Former v3 (WikiText-103)

```bash
# Default: threshold routing, DCT+Attn variant (~10 hrs on A40)
python train.py --routing threshold --variant dct_attn

# Ablations: soft and hard routing
python train.py --routing soft  --variant dct_attn --small
python train.py --routing hard  --variant dct_attn --small

# With collapse regulariser
python train.py --routing threshold --variant dct_attn --small --reg

# Original 3-operator (DCT + RBF + Attn) — reproduces routing collapse
python train.py --routing threshold --variant rbf --small
```

### Step 4 — Multi-Dataset Training

```bash
# WikiText-2 (small corpus generalisation)
python train_all.py --dataset wikitext2 --baseline
python train_all.py --dataset wikitext2 --routing threshold --variant dct_attn

# LRA — IMDB sentiment (naturalistic) and ListOps (symbolic)
python train_all.py --dataset lra_imdb    --routing threshold --variant dct_attn
python train_all.py --dataset lra_listops --routing threshold --variant dct_attn
python train_all.py --dataset lra_all     --baseline
python train_all.py --dataset lra_all     --routing threshold --variant dct_attn
```

### Step 5 — Mixed Training (MetaRouter)

Trains on batches drawn from all four datasets simultaneously (WikiText-103, WikiText-2, IMDB, ListOps). The MetaRouter learns the naturalistic/symbolic boundary end-to-end. Logs the gate value g at each eval step.

```bash
python train_all.py --dataset mixed --routing threshold --variant dct_attn
# Saves to: checkpoints/chiar_v3_mixed_best.pt
# Watch: MetaGate(LM) descend from ~0.88 → ~0.22 over training
```

### Step 6 — Analysis and Figures

```bash
python analyze_all.py
# Outputs in checkpoints/analysis/:
#   table1_wikitext103.csv
#   table2_wikitext2.csv
#   table3_lra.csv
#   fig1_training_curves.png
#   fig2_metarouter_gate.png
```

### Step 7 — Qualitative Analysis and Routing Heatmap

```bash
# Full run: our models + HuggingFace comparison (FNet, BigBird, Longformer)
python qualitative.py \
    --ours_baseline checkpoints/baseline_wikitext103_350M_best.pt \
    --ours_chiar    checkpoints/chiar_threshold_dct_attn_wikitext103_350M_best.pt

# Heatmap only (faster)
python qualitative.py \
    --ours_chiar checkpoints/chiar_threshold_dct_attn_wikitext103_350M_best.pt \
    --heatmap_only

# Skip HuggingFace downloads (our models only)
python qualitative.py \
    --ours_baseline checkpoints/baseline_wikitext103_350M_best.pt \
    --ours_chiar    checkpoints/chiar_threshold_dct_attn_wikitext103_350M_best.pt \
    --skip_hf

# Outputs in checkpoints/analysis/:
#   fig_routing_heatmap.png         (the chiaroscuro heatmap figure)
#   routing_heatmap_data.json       (per-token routing decisions)
#   qualitative_continuations.csv
#   qualitative_table.tex           (ready to paste into paper)
```

---

## Configuration

`config.py` has two configs:

| Parameter | `Config` (350M) | `SmallConfig` (17M) | Description |
|---|---|---|---|
| `d_model` | 1024 | 256 | Embedding dimension |
| `n_heads` | 16 | 4 | Attention heads |
| `n_layers` | 28 | 4 | Total layers |
| `max_seq_len` | 256 | 256 | Max sequence length |
| `batch_size` | 8 | 32 | Per-GPU batch size |
| `grad_accum_steps` | 16 | 4 | Effective batch = 128 |
| `learning_rate` | 1e-4 | 1e-4 | Peak LR (cosine decay) |
| `tau_low` | 0.8935 | — | 33rd entropy percentile |
| `tau_high` | 0.8973 | — | 67th entropy percentile |
| `use_meta_router` | True | True | Enable MetaRouter gate |
| `meta_mix_ratio` | 0.25 | 0.25 | Fraction of non-WT103 batches in mixed training |
| `use_collapse_reg` | False | False | Operator utilisation entropy regulariser |
| `rope_base` | 10000.0 | 10000.0 | RoPE frequency base |

---

## Live Demo — Routing Heatmap App

The `CHIAR-former_demo/` folder contains a Streamlit app that visualises per-token routing decisions in real time. Type any text and see each token coloured by its operator assignment: **blue** for DCT Mixing (cheap, low-entropy), **red** for Full Attention (expensive, high-entropy).

### Setup

```bash
cd CHIAR-former_demo
pip install -r requirements.txt
```

### Place a checkpoint

```
CHIAR-former_demo/
└── checkpoints/
    └── chiar_threshold_dct_attn_wikitext103_350M_best.pt
```

Copy your trained checkpoint from `CHIAR-former/checkpoints/` to this path. If no checkpoint is found, the app runs in **demo mode** using a function-word heuristic (so you can explore the UI immediately without training).

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

### What the demo shows

- **Token heatmap** — every token colour-coded by routing decision.
- **Stats panel** — total tokens, DCT count (%), Attention count (%), and estimated FLOP reduction for the current input.
- **Routing insight** — plain-English explanation of what the model chose and why.
- **4 built-in examples** — climate policy (mixed routing), transformer architecture (technical), lighthouse narrative (naturalistic), and a symbolic MAX/MIN expression (symbolic → low gate).
- **Sidebar** — paper contributions summary with the six key findings.

The app gracefully handles long inputs (caps at 256 tokens) and falls back to heuristic routing when no checkpoint is available, making it useful for demonstration without GPU access.

---

## Reproducing Paper Results

The full experiment sequence to reproduce all tables and figures in the paper:

```bash
# 1. Baseline (350M, WikiText-103)
python train.py --baseline

# 2. Tau calibration
python calibrate_tau.py   # copy tau_low / tau_high into config.py

# 3. CHIAR v3 (350M, WikiText-103)
python train.py --routing threshold --variant dct_attn

# 4. CHIAR v3 Mixed Training
python train_all.py --dataset mixed --routing threshold --variant dct_attn

# 5. Ablations (17M, all routing modes)
python train.py --baseline --small
python train.py --routing soft      --variant dct_attn --small
python train.py --routing hard      --variant dct_attn --small
python train.py --routing threshold --variant dct_attn --small
python train.py --routing threshold --variant dct_attn --small --reg

# 6. Analysis (tables + MetaRouter gate figure)
python analyze_all.py

# 7. Routing heatmap figure
python qualitative.py \
    --ours_chiar checkpoints/chiar_threshold_dct_attn_wikitext103_350M_best.pt \
    --heatmap_only
```

Expected total GPU time on a single NVIDIA RTX A5000 (24GB): ~35–40 hours across all runs.

---

## Checkpoint Naming Convention

Checkpoints are saved to `checkpoints/` with the pattern:

```
{model}_{routing}_{variant}_{dataset}_{scale}_best.pt
```

Key checkpoints:

| File | Description |
|---|---|
| `baseline_wikitext103_350M_best.pt` | Full attention baseline, 404M params |
| `chiar_threshold_dct_attn_wikitext103_350M_best.pt` | CHIAR v3, 400M params, threshold routing |
| `chiar_v3_mixed_best.pt` | CHIAR v3 after mixed-dataset MetaRouter training |
| `baseline_wikitext103_17M_best.pt` | 17M baseline for ablation |

Each checkpoint contains: `{"step": int, "model": state_dict, "val_ppl": float, "cfg": dict}`.

---

## Citation

```bibtex
@article{sikdar2026chiar,
  title   = {Chiaroscuro Attention: Spending Compute in the Dark},
  author  = {Sikdar, Prateek Kumar},
  journal = {arXiv preprint arXiv:2606.08327},
  year    = {2026}
}
```

---

## Acknowledgements

The author thanks the open-source communities behind PyTorch, HuggingFace Transformers, and the WikiText dataset. Literature surveying and codebase development were assisted by Claude Sonnet 4.6 (Anthropic, 2026). All experiments were conducted on a single NVIDIA RTX A5000 (24GB VRAM) GPU instance via RunPod cloud infrastructure.
