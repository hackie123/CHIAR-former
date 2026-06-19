# CHIAR-Former

**Chiaroscuro Attention: Spending Compute in the Dark**
*Operator Routing via Spectral Entropy Across Tasks and Scales*

Prateek Kumar Sikdar · AI Architect, Agentic AI Practice · Accenture, Bengaluru · 2026

[![arXiv](https://img.shields.io/badge/arXiv-2606.08327-b31b1b.svg)](https://arxiv.org/abs/2606.08327)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-prateeksikdar%2FCHIAR--Former-orange.svg)](https://huggingface.co/prateeksikdar/CHIAR-Former)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)

---

## What is CHIAR-Former?

CHIAR-Former is an efficient language model that decides, for each token, whether it deserves expensive full self-attention or cheap spectral mixing — and routes accordingly. Function words like *the*, *of*, *and* are smooth and low-complexity; they go through **DCT spectral mixing** in O(d log d). Content words like *overwhelming*, *consensus*, *paradox* are rich and high-complexity; they go through **full self-attention** in O(n²d). The result is ~37% fewer FLOPs at 400M parameters with only a 3.93 perplexity cost on WikiText-103.

The name comes from *chiaroscuro* — the Renaissance painting technique of Caravaggio and Rembrandt that spends light only where the eye needs detail, leaving peripheral regions in inexpensive shadow. CHIAR-Former does the same with compute.

**Results at 400M parameters, WikiText-103:**

| Model | Params | Test PPL | FLOP Reduction |
|---|---|---|---|
| Full Attention Baseline | 404M | 23.58 | — |
| **CHIAR-Former** | **400M** | **27.51** | **~37%** |
| CHIAR-Former (mixed training) | 400M | 28.56 | ~37% |

---

## Quickstart

### 1. Install

```bash
git clone https://github.com/prateeksikdar/CHIAR-Former.git
cd CHIAR-Former
pip install torch>=2.1.0 transformers>=4.38.0 huggingface_hub
```

### 2. Download the pre-trained checkpoint

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='prateeksikdar/CHIAR-Former',
    filename='chiar_dct_attn_400M_best.pt',
    local_dir='./checkpoints'
)
```

### 3. Load the model and run inference

```python
import torch
from config import Config
from model import CHIARFormer
from transformers import AutoTokenizer

# Load tokeniser
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Load model
cfg = Config()
cfg.vocab_size = len(tokenizer)
model = CHIARFormer(cfg)

ckpt = torch.load('checkpoints/chiar_dct_attn_400M_best.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
model.eval()

# Run inference
text = "Despite the overwhelming scientific consensus on climate change,"
ids  = tokenizer.encode(text, return_tensors='pt')

with torch.no_grad():
    logits, _ = model(ids)

next_token = tokenizer.decode(logits[0, -1].argmax().item())
print(f"Next token: {next_token}")
```

### 4. See per-token routing decisions

```python
with torch.no_grad():
    logits, routing_infos, _, meta_gate = model(ids, return_routing_info=True)

# routing_infos[1] = first routing layer (L2)
op_idx  = routing_infos[1]['op_idx'][0]   # (T,) — 0=DCT, 1=Attention
entropy = routing_infos[1]['H'][0]         # (T,) — spectral entropy per token

tokens = [tokenizer.decode([t]) for t in ids[0]]
for tok, op, h in zip(tokens, op_idx, entropy):
    route = 'Attention' if op == 1 else 'DCT'
    print(f"{tok:15s}  H={h:.3f}  →  {route}")

print(f"\nMetaRouter gate: {meta_gate.item():.3f}  (1=DCT, 0=bypass)")
```

---

## Interactive Demo

The `CHIAR-former_demo/` folder is a Streamlit app where you type any text and see each token coloured by its routing decision in real time — **blue** for DCT Mixing (cheap), **red** for Full Attention (expensive).

### Setup

```bash
cd CHIAR-former_demo
pip install streamlit>=1.32.0 torch>=2.1.0 transformers>=4.38.0 huggingface_hub

# Download checkpoint into the demo's checkpoints folder
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='prateeksikdar/CHIAR-Former',
    filename='chiar_dct_attn_400M_best.pt',
    local_dir='./checkpoints'
)
"
```

### Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

**No GPU required** — the app runs on CPU. If no checkpoint is found it falls back to a function-word heuristic so you can still explore the interface.

### What you will see

- **Token heatmap** — every token colour-coded: blue = DCT (low entropy), red = Attention (high entropy).
- **Stats panel** — token count, DCT %, Attention %, estimated FLOP reduction for the input.
- **4 built-in examples** — climate policy text, transformer architecture description, a narrative passage, and a symbolic MAX/MIN expression.
- **Routing insight** — plain-English explanation of what the model chose and why.
- **Sidebar** — summary of the six key paper contributions.

---

## Model Details

CHIAR-Former routes each token through one of two operators based on **spectral entropy** H(x) ∈ [0,1] — the normalised information entropy of the token's DCT power spectrum. Low entropy means the token's frequency content is concentrated in a few components (smooth, cheap to process). High entropy means energy is spread across all frequencies (complex, needs full attention).

**Layer structure (400M, 28 layers):**

```
Token Embeddings   ← no absolute positional encoding; RoPE handles position inside attention
      ↓
MetaRouter         ← learned task-level gate g ∈ [0,1]; stabilises at g ≈ 0.22
      ↓
L1                 ← DCT Mixing only, soft-gated by MetaRouter
L2 … L27          ← per-token routing: H(x) ≤ τ → DCT  |  H(x) > τ → Full Attention + RoPE
L28                ← Full Attention only (accuracy anchor)
      ↓
LayerNorm → LM Head (weight-tied)
```

**Key design choices:**

- **RoPE** (Rotary Position Embedding) inside every attention layer — zero learnable positional parameters, generalises beyond training length.
- **Shared FFN per layer** — every layer has exactly one FFN regardless of which operator runs, ensuring fair parameter comparison with the baseline (400M vs 404M, a 1% difference).
- **Threshold routing** (default) — τ = midpoint of 33rd/67th entropy percentiles from the baseline embedding distribution. Theory-driven, no learnable routing parameters.
- **MetaRouter** — a scalar gate g = σ(Linear(mean(x))) trained end-to-end on mixed batches from four datasets. Stabilises at g ≈ 0.22 at scale, indicating a robust compute–quality equilibrium between spectral preprocessing and attention capacity.

---

## Available Checkpoints

All checkpoints are on Hugging Face at [`prateeksikdar/CHIAR-Former`](https://huggingface.co/prateeksikdar/CHIAR-Former).

| Checkpoint | Params | Test PPL | Description |
|---|---|---|---|
| `chiar_dct_attn_400M_best.pt` | 400M | 27.51 | Main model — WikiText-103, threshold routing |
| `chiar_v3_mixed_best.pt` | 400M | 28.56 | Mixed-dataset training (WikiText-103/2, IMDB, ListOps) |
| `baseline_wikitext103_350M_best.pt` | 404M | 23.58 | Full attention baseline for comparison |

**Download a specific checkpoint:**

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id='prateeksikdar/CHIAR-Former',
    filename='chiar_dct_attn_400M_best.pt',
    local_dir='./checkpoints'
)
```

**Download all checkpoints:**

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='prateeksikdar/CHIAR-Former',
    local_dir='./checkpoints'
)
```

Each checkpoint file contains: `{"step": int, "model": state_dict, "val_ppl": float, "cfg": dict}`.

---

## Repository Contents

```
CHIAR-Former/
├── model/
│   ├── chiar_former.py     # Main model class — load this for inference
│   ├── chiar_layer.py      # Single layer: operator sub-layer + shared FFN
│   ├── router.py           # SpectralRouter — computes H(x) and routes per token
│   ├── meta_router.py      # MetaRouter — learned task-level gate g
│   ├── dct_mix.py          # DCT spectral mixing (FFT-based, O(d log d))
│   ├── rope.py             # Rotary Position Embedding
│   └── rbf_mix.py          # RBF mixing (ablation only)
├── config.py               # Model hyperparameters (Config = 400M, SmallConfig = 17M)
├── test_forward.py         # 11 sanity tests — run to verify your install
├── qualitative.py          # Generate routing heatmap figures
├── requirements.txt

CHIAR-Former_demo/
├── app.py                  # Streamlit interactive demo
├── model/                  # Same model code as above
├── config.py               # Same config
└── requirements.txt
```

---

## Requirements

- Python 3.10+
- PyTorch 2.1+
- `transformers >= 4.38.0`
- `huggingface_hub`
- CUDA GPU recommended for inference at 400M scale; CPU works for the demo

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
