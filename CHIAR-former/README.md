# CHIAR-Former v3

**Chiaroscuro Attention: Spending Compute in the Dark**
Prateek Kumar Sikdar, Accenture, Bengaluru — June 2026

## Architecture

- **RoPE** inside attention — no absolute positional embedding
- **Per-token SpectralRouter** (L2..L7) — H(x) threshold → DCT or Attention
- **Learned MetaRouter** — task-level gate, bypasses L1 DCT for symbolic tasks
- **~150M params** — baseline and CHIAR within 3% of each other (fair comparison)
- **Hardware**: A40 48GB VRAM — no gradient checkpointing needed

## Setup

```bash
pip install --upgrade transformers==4.40.0 datasets==2.18.0
pip install -r requirements.txt
python test_forward.py   # all 11 tests must pass
```

## Run order

```bash
# Step 1: Baseline (~8 hrs on A40)
python train.py --baseline

# Step 2: Tau calibration
python calibrate_tau.py   # update tau_low/tau_high in config.py

# Step 3: CHIAR v3 (~10 hrs on A40)
python train.py --routing threshold --variant dct_attn

# Step 4: WikiText-2 (~1.5 hrs)
python train_all.py --dataset wikitext2 --baseline
python train_all.py --dataset wikitext2 --routing threshold --variant dct_attn

# Step 5: LRA — IMDB + ListOps (~3 hrs)
python train_all.py --dataset lra_all --baseline
python train_all.py --dataset lra_all --routing threshold --variant dct_attn

# Step 6: Mixed training — MetaRouter learns (~10 hrs)
python train_all.py --dataset mixed --routing threshold --variant dct_attn

# Step 7: Analysis
python analyze_all.py

# Step 8: Qualitative
python qualitative.py \
    --ours_baseline checkpoints/baseline_wikitext103_350M_best.pt \
    --ours_chiar    checkpoints/chiar_threshold_dct_attn_wikitext103_350M_best.pt

# Step 9: Download
python -c "import zipfile,os; z=zipfile.ZipFile('/workspace/results_final.zip','w',zipfile.ZIP_DEFLATED); [z.write(os.path.join(r,f)) for r,d,files in os.walk('checkpoints') for f in files]; z.close(); print('Done')"
```

## Folder structure

```
chiar_v3/
├── config.py              # 350M Config + 17M SmallConfig
├── train.py               # WikiText-103 (baseline + CHIAR)
├── train_all.py           # All datasets + mixed MetaRouter training
├── calibrate_tau.py       # Tau calibration after baseline
├── test_forward.py        # 11 forward pass tests
├── analyze_all.py         # Tables + figures
├── qualitative.py         # Text continuations + routing heatmap
├── requirements.txt
├── README.md
├── model/
│   ├── rope.py            # RoPE — zero learnable params
│   ├── meta_router.py     # Learned task-level gate
│   ├── chiar_former.py    # Main model
│   ├── chiar_layer.py     # Layer: shared FFN, DCT or Attn operator
│   ├── chiar_classifier.py
│   ├── router.py          # Per-token spectral entropy router
│   ├── dct_mix.py         # FFT-based DCT (no torch.fft.dct)
│   └── rbf_mix.py         # RBF ablation only
├── data/
│   ├── wikitext.py        # WikiText-103 + WikiText-2
│   └── lra.py             # IMDB + ListOps
└── utils/
    └── flop_counter.py
```
