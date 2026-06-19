# analyze_all.py — Full paper analysis: tables, figures, MetaRouter gate distribution
#
# Usage: python analyze_all.py

import os, sys, json
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")
CKPT = "./checkpoints"
OUT  = os.path.join(CKPT, "analysis")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
    "figure.dpi":150,"savefig.dpi":200,"savefig.bbox":"tight"})

COLORS = {"baseline":"#2C3E50","soft":"#1A6FBF","hard":"#E74C3C",
          "threshold":"#8B4513","reg":"#27AE60","dct_attn":"#F39C12",
          "imdb":"#1A6FBF","listops":"#C0392B"}

def load_results(name):
    p = os.path.join(CKPT, f"{name}_results.json")
    return json.load(open(p)) if os.path.exists(p) else None

def load_log(name):
    p = os.path.join(CKPT, f"{name}_log.jsonl")
    if not os.path.exists(p): return None
    return pd.DataFrame([json.loads(l) for l in open(p)])

def save(name):
    plt.savefig(os.path.join(OUT, name)); plt.close()
    print(f"  Saved: {name}")

# ── Table 1: WikiText-103 ablation ────────────────────────────────────────────
print("── Table 1: WikiText-103 ──")
wt_runs = [
    ("baseline_wikitext103",             "Baseline (Full Attn)"),
    ("chiar_soft_wikitext103",           "CHIAR Soft"),
    ("chiar_hard_wikitext103",           "CHIAR Hard"),
    ("chiar_threshold_wikitext103",      "CHIAR Threshold"),
    ("chiar_threshold_reg_wikitext103",  "CHIAR Thresh+Reg"),
    ("chiar_threshold_dct_attn_wikitext103", "CHIAR DCT+Attn"),
    # backwards compat
    ("baseline_fullattention",           "Baseline (Full Attn)"),
    ("chiar_soft",                       "CHIAR Soft"),
    ("chiar_hard",                       "CHIAR Hard"),
    ("chiar_threshold",                  "CHIAR Threshold"),
    ("chiar_threshold_reg",              "CHIAR Thresh+Reg"),
    ("chiar_threshold_dct_attn",         "CHIAR DCT+Attn"),
]
seen, wt_rows = set(), []
for rn, label in wt_runs:
    if label in seen: continue
    r = load_results(rn)
    if r is None: continue
    seen.add(label)
    wt_rows.append({"Model":label,
                    "Val PPL":round(r.get("best_val_ppl",float("nan")),2),
                    "Test PPL":round(r.get("test_ppl",float("nan")),2)})
df1 = pd.DataFrame(wt_rows)
df1.to_csv(os.path.join(OUT,"table1_wikitext103.csv"), index=False)
print(df1.to_string(index=False))

# ── Table 2: WikiText-2 ───────────────────────────────────────────────────────
print("\n── Table 2: WikiText-2 ──")
wt2_rows = []
for rn, label in [("baseline_wikitext2","Transformer (ours)"),
                   ("baseline_ptb","Transformer (ours)"),
                   ("chiar_threshold_dct_attn_wikitext2","CHIAR DCT+Attn (ours)"),
                   ("chiar_threshold_dct_attn_ptb","CHIAR DCT+Attn (ours)")]:
    r = load_results(rn)
    if r and label not in [x["Model"] for x in wt2_rows]:
        wt2_rows.append({"Model":label,
                          "Test PPL":round(r.get("test_ppl",float("nan")),2),
                          "Source":"This work"})
df2 = pd.DataFrame(wt2_rows) if wt2_rows else pd.DataFrame(
    columns=["Model","Test PPL","Source"])
df2.to_csv(os.path.join(OUT,"table2_wikitext2.csv"), index=False)
print(df2.to_string(index=False))

# ── Table 3: LRA ─────────────────────────────────────────────────────────────
print("\n── Table 3: LRA ──")
lra_rows = []
published = {
    "Transformer [Tay et al.,2021]": {"imdb":64.27,"listops":36.37,"src":"Tay et al. 2021"},
    "FNet [Lee-Thorp et al.,2022]":  {"imdb":65.11,"listops":35.33,"src":"Tay et al. 2021"},
    "Performer [Choromanski et al.,2021]":{"imdb":65.40,"listops":18.01,"src":"Tay et al. 2021"},
    "Linformer [Wang et al.,2020]":  {"imdb":53.94,"listops":35.70,"src":"Tay et al. 2021"},
}
for task in ["imdb","listops"]:
    for prefix, label in [("baseline_lra_","Baseline (ours)"),
                           ("chiar_threshold_dct_attn_lra_","CHIAR DCT+Attn (ours)")]:
        r = load_results(f"{prefix}{task}")
        if r:
            lra_rows.append({"Model":label,"Task":task,
                "Acc":round(r.get("best_test_acc",float("nan")),2),"Source":"This work"})
for model_name, vals in published.items():
    for task in ["imdb","listops"]:
        lra_rows.append({"Model":model_name,"Task":task,
            "Acc":vals[task],"Source":vals["src"]})
df3 = pd.DataFrame(lra_rows)
df3.to_csv(os.path.join(OUT,"table3_lra.csv"), index=False)
print(df3.to_string(index=False))

# ── Figure 1: Training curves ─────────────────────────────────────────────────
print("\n── Figure 1: Training curves ──")
fig, axes = plt.subplots(1,2,figsize=(13,5))
for tag,col,lbl in [("baseline_wikitext103","baseline","Baseline"),
                     ("chiar_soft_wikitext103","soft","Soft"),
                     ("chiar_hard_wikitext103","hard","Hard"),
                     ("chiar_threshold_dct_attn_wikitext103","dct_attn","DCT+Attn"),
                     # backwards compat
                     ("baseline_fullattention","baseline","Baseline"),
                     ("chiar_soft","soft","Soft"),
                     ("chiar_hard","hard","Hard"),
                     ("chiar_threshold_dct_attn","dct_attn","DCT+Attn")]:
    df = load_log(tag)
    if df is None or "val_ppl" not in df.columns: continue
    if lbl in [l.get_label() for l in axes[0].lines]: continue
    axes[0].plot(df["step"], df["val_loss"], label=lbl,
                 color=COLORS.get(col,"gray"), lw=2)
    axes[1].plot(df["step"], df["val_ppl"],  label=lbl,
                 color=COLORS.get(col,"gray"), lw=2)
for ax, title in zip(axes,["Validation Loss","Validation PPL \u2193"]):
    ax.set_xlabel("Training Steps"); ax.set_title(title); ax.legend(fontsize=8)
plt.suptitle("CHIAR-Former Training Curves (WikiText-103)", fontweight="bold")
plt.tight_layout(); save("fig1_training_curves.png")

# ── Figure 2: MetaRouter gate distribution ────────────────────────────────────
print("\n── Figure 2: MetaRouter gate distribution ──")
mixed_log = load_log("chiar_v3_mixed")
if mixed_log is not None and "meta_gate_lm" in mixed_log.columns:
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(mixed_log["meta_gate_lm"].dropna(), bins=30, alpha=0.7,
            color=COLORS["dct_attn"], label="WikiText (naturalistic)")
    ax.axvline(0.5, color="red", lw=1.5, linestyle="--", label="Decision boundary")
    ax.set_xlabel("MetaRouter Gate Value (1=use DCT, 0=bypass)")
    ax.set_ylabel("Count")
    ax.set_title("MetaRouter Gate Distribution Across Datasets", fontweight="bold")
    ax.legend(); plt.tight_layout(); save("fig2_metarouter_gate.png")
else:
    print("  (Mixed training not run yet — skipping MetaRouter figure)")

print(f"\nAnalysis complete. Results in {OUT}/")
