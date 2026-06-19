# calibrate_tau.py — Empirical tau calibration from trained baseline embeddings
# Run once after baseline training. Prints tau_low and tau_high for config.py.

import torch
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
import sys, os, math
sys.path.insert(0, ".")
from config import Config
from model.dct_mix import dct as _dct


def spectral_entropy(x, d):
    xf    = _dct(x.unsqueeze(0)).squeeze(0)   # (T, d)
    power = xf.pow(2)
    p     = power / (power.sum(-1, keepdim=True) + 1e-8)
    H     = -(p * (p + 1e-8).log()).sum(-1)
    return H / (math.log(d) + 1e-8)


def main():
    cfg       = Config()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Find baseline checkpoint
    ckpt_path = os.path.join(cfg.checkpoint_dir,
                             "baseline_wikitext103_350M_best.pt")
    if not os.path.exists(ckpt_path):
        # Try SmallConfig checkpoint name
        ckpt_path = os.path.join(cfg.checkpoint_dir,
                                 "baseline_wikitext103_17M_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"No baseline checkpoint found in {cfg.checkpoint_dir}")
        print("Run: python train.py --baseline   first.")
        return

    print(f"Loading: {ckpt_path}")
    ckpt       = torch.load(ckpt_path, map_location=device)
    emb_weight = ckpt["model"]["token_emb.weight"].to(device)
    print(f"  Embedding shape: {emb_weight.shape}")

    ds    = load_dataset("wikitext", "wikitext-103-v1")
    texts = ds["validation"]["text"][:200]
    all_H = []

    for text in texts:
        if not text.strip(): continue
        ids = tokenizer.encode(text, add_special_tokens=False,
                               max_length=256, truncation=True)
        if not ids: continue
        ids_t = torch.tensor(ids, device=device)
        embs  = emb_weight[ids_t]              # (T, d)
        H     = spectral_entropy(embs, cfg.d_model)
        all_H.extend(H.cpu().numpy().tolist())

    all_H = np.array(all_H)
    p33, p67 = np.percentile(all_H, 33), np.percentile(all_H, 67)
    print(f"\nEntropy range: [{all_H.min():.4f}, {all_H.max():.4f}]")
    print(f"Mean: {all_H.mean():.4f}")
    print(f"\nUpdate config.py:")
    print(f"  tau_low  = {p33:.4f}")
    print(f"  tau_high = {p67:.4f}")


if __name__ == "__main__":
    main()