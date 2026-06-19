# qualitative.py — Qualitative Analysis: Text Continuations + Routing Heatmap
#
# Option B: Pretrained HuggingFace models (FNet, BigBird, Longformer)
# Option C: Our trained CHIAR v3 and baseline (controlled comparison)
#
# Prompts:
#   1. long_range  — tests long-range dependency resolution
#   2. technical   — tests domain/technical language
#   3. narrative   — naturalistic text (CHIAR home turf)
#   4. symbolic    — symbolic reasoning (CHIAR weakness — honest reporting)
#   5. heatmap     — used for routing heatmap figure only
#
# Usage:
#   # Full run (Option B + C + heatmap)
#   python qualitative.py \
#       --ours_baseline checkpoints/baseline_wikitext103_350M_best.pt \
#       --ours_chiar    checkpoints/chiar_threshold_dct_attn_wikitext103_350M_best.pt
#
#   # Skip HuggingFace downloads (Option C only)
#   python qualitative.py --ours_baseline ... --ours_chiar ... --skip_hf
#
#   # Heatmap only
#   python qualitative.py --ours_chiar ... --heatmap_only

import os, sys, json, argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, ".")
from config import Config
from model  import CHIARFormer
from train  import BaselineTransformer

OUT = "./checkpoints/analysis"
os.makedirs(OUT, exist_ok=True)

# ── Prompts ───────────────────────────────────────────────────────────────────

PROMPTS = {
    "long_range": (
        "The scientist who won the Nobel Prize in 1998 for her groundbreaking "
        "research on cellular membrane transport mechanisms later became the "
        "first woman to lead the National Academy of Sciences. Her name was"
    ),
    "technical": (
        "The transformer architecture achieves parallelism by replacing "
        "recurrent connections with self-attention, allowing each token to"
    ),
    "narrative": (
        "The old lighthouse keeper had maintained the lamp for forty years, "
        "through storms that shook the foundations and winters that froze the"
    ),
    "symbolic": (
        "If MAX(3, MIN(7, 4), 2) equals X and MIN(X, 5) equals Y then Y is"
    ),
    "heatmap": (
        "Despite the overwhelming scientific consensus on climate change, "
        "many governments have struggled to implement effective carbon "
        "reduction policies because"
    ),
}

PROMPT_LABELS = {
    "long_range": "Long-range dependency",
    "technical":  "Technical domain",
    "narrative":  "Naturalistic narrative",
    "symbolic":   "Symbolic reasoning",
    "heatmap":    "Routing heatmap prompt",
}

MAX_NEW_TOKENS = 25

# Function words expected to route to DCT
FUNCTION_WORDS = {
    "the","a","an","of","in","to","and","or","but","is","was","are","were",
    "it","its","that","this","for","on","at","by","with","from","as","be",
    "been","have","has","had","do","did","not","no","so","if","then","than",
    "into","their","they","we","you","he","she","which","who","also","just",
}

# ── Our models (Option C) ─────────────────────────────────────────────────────

def load_our_model(ckpt_path, model_class, cfg, device):
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}"); return None
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = model_class(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  Loaded: {os.path.basename(ckpt_path)} "
          f"(Val PPL: {ckpt.get('val_ppl', 0.0):.2f})")
    return model


@torch.no_grad()
def generate_our(model, tokenizer, prompt, max_new=MAX_NEW_TOKENS,
                 device="cuda", return_routing=False):
    ids       = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated = ids.clone()
    routing_log = []

    for _ in range(max_new):
        inp = generated[:, -256:] if generated.shape[1] > 256 else generated
        if return_routing and hasattr(model, "layers"):
            logits, infos, _, _ = model(inp, return_routing_info=True)
            if infos[1]["op_idx"] is not None:
                last_op  = infos[1]["op_idx"][0, -1].item()
                last_tok = generated[0, -1].item()
                routing_log.append((last_tok, last_op))
        else:
            logits, _ = model(inp)
        next_tok  = logits[0, -1, :].argmax(-1, keepdim=True).unsqueeze(0)
        generated = torch.cat([generated, next_tok], dim=1)
        if next_tok.item() == tokenizer.eos_token_id:
            break

    text = tokenizer.decode(generated[0, ids.shape[1]:].tolist(),
                            skip_special_tokens=True)
    return (text, routing_log) if return_routing else text


# ── HuggingFace models (Option B) ────────────────────────────────────────────

HF_MODELS = {
    "FNet":       "google/fnet-base",
    "BigBird":    "google/bigbird-roberta-base",
    "Longformer": "allenai/longformer-base-4096",
}


def generate_hf(model_name, hf_id, prompt, max_new=MAX_NEW_TOKENS):
    try:
        from transformers import pipeline
        print(f"    Loading {model_name}...")
        gen = pipeline("text-generation", model=hf_id,
                       max_new_tokens=max_new, do_sample=False,
                       pad_token_id=50256)
        result = gen(prompt)[0]["generated_text"]
        return result[len(prompt):].strip()[:200]
    except Exception as e:
        print(f"    {model_name} failed: {e}")
        return "[unavailable]"


# ── Routing heatmap ───────────────────────────────────────────────────────────

def plot_routing_heatmap(prompt, routing_log, tokenizer, continuation):
    full_text = prompt + " " + continuation
    token_ids = tokenizer.encode(full_text)
    tokens    = [tokenizer.decode([t]) for t in token_ids]
    routing_map = {tok_id: op for tok_id, op in routing_log}

    # Routing for each token
    ops = []
    n_prompt = len(token_ids) - len(routing_log)
    for i, (tid, tok) in enumerate(zip(token_ids, tokens)):
        tok_clean = tok.strip().lower().rstrip(".,;:\"'")
        if i < n_prompt:
            ops.append(0 if tok_clean in FUNCTION_WORDS else 1)
        else:
            ops.append(routing_map.get(tid, 1))

    DCT_COL  = "#2471A3"
    ATTN_COL = "#C0392B"
    tpr      = 12   # tokens per row
    tok_w    = 1.0 / tpr
    tok_h    = 0.11
    n_rows   = (len(tokens) + tpr - 1) // tpr
    fig_h    = max(4, n_rows * 1.2 + 2)

    fig, ax = plt.subplots(figsize=(14, fig_h))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    y_top = 0.92
    for i, (tok, op) in enumerate(zip(tokens, ops)):
        row = i // tpr; col = i % tpr
        x   = col * tok_w + 0.005
        y   = y_top - row * (tok_h + 0.03)
        fc  = DCT_COL if op == 0 else ATTN_COL
        rect = mpatches.FancyBboxPatch(
            (x, y-tok_h), tok_w-0.008, tok_h,
            boxstyle="round,pad=0.005",
            facecolor=fc, alpha=0.88,
            edgecolor="white", linewidth=0.6)
        ax.add_patch(rect)
        display = tok.strip()[:9] or "·"
        ax.text(x+(tok_w-0.008)/2, y-tok_h/2, display,
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")

    # Stats
    n_dct  = ops.count(0); n_attn = ops.count(1)
    pct_dct = 100*n_dct/len(ops)
    stats = f"DCT: {n_dct} tokens ({pct_dct:.0f}%)  |  " \
            f"Attention: {n_attn} tokens ({100-pct_dct:.0f}%)"

    dct_p  = mpatches.Patch(color=DCT_COL,  label=f"DCT Mixing — O(d log d)  [{pct_dct:.0f}%]")
    attn_p = mpatches.Patch(color=ATTN_COL, label=f"Full Attention — O(n²d)  [{100-pct_dct:.0f}%]")
    ax.legend(handles=[dct_p, attn_p], loc="lower left",
              fontsize=9, framealpha=0.95, edgecolor=ATTN_COL)

    ax.set_title(
        "CHIAR-Former v3: Chiaroscuro Routing Heatmap\n"
        f"Prompt: \"{prompt[:70]}...\"\n"
        f"{stats}",
        fontsize=10, fontweight="bold", y=1.01)

    plt.tight_layout()
    path = os.path.join(OUT, "fig_routing_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight", facecolor="#F5F7FA")
    plt.close()
    print(f"  Saved: fig_routing_heatmap.png")

    with open(os.path.join(OUT, "routing_heatmap_data.json"), "w") as f:
        json.dump([{"token": t, "routing": "DCT" if o==0 else "Attention"}
                   for t, o in zip(tokens, ops)], f, indent=2)


# ── Table builder ─────────────────────────────────────────────────────────────

def build_table(results):
    import pandas as pd
    rows = []
    for pk in PROMPTS:
        if pk == "heatmap": continue
        row = {"Prompt": PROMPT_LABELS[pk]}
        row.update(results.get(pk, {}))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "qualitative_continuations.csv"), index=False)
    print("  Saved: qualitative_continuations.csv")

    # LaTeX
    models = list(list(results.values())[0].keys()) if results else []
    lines = [
        r"\begin{table*}[t]\centering\small",
        r"\caption{Qualitative text continuations (25 tokens). "
        r"CHIAR~v3 and Baseline: trained from scratch on WikiText-103 (350M). "
        r"FNet, BigBird, Longformer: pretrained HuggingFace models "
        r"(illustrative comparison only).}",
        r"\label{tab:qualitative}",
        r"\begin{tabular}{@{}p{2.2cm}" + "p{2.2cm}"*len(models) + r"@{}}",
        r"\toprule",
        "Prompt & " + " & ".join(f"\\textbf{{{m}}}" for m in models) + r"\\",
        r"\midrule",
    ]
    for pk in PROMPTS:
        if pk == "heatmap": continue
        conts = results.get(pk, {})
        vals  = " & ".join(
            r"\textit{" + conts.get(m,"—")[:55].replace("_"," ").replace("&","\\&") + r"...}"
            for m in models)
        lines.append(f"\\textbf{{{PROMPT_LABELS[pk]}}} & {vals} \\\\[4pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    with open(os.path.join(OUT, "qualitative_table.tex"), "w") as f:
        f.write("\n".join(lines))
    print("  Saved: qualitative_table.tex")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours_baseline", default=None)
    p.add_argument("--ours_chiar",    default=None)
    p.add_argument("--skip_hf",       action="store_true")
    p.add_argument("--heatmap_only",  action="store_true")
    args = p.parse_args()

    from transformers import AutoTokenizer
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg       = Config(); cfg.vocab_size = 50257
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    results = {}

    # ── Option C: Our models ─────────────────────────────────────────────────
    our_models = {}
    if args.ours_baseline:
        m = load_our_model(args.ours_baseline, BaselineTransformer, cfg, device)
        if m: our_models["Baseline (ours)"] = m
    if args.ours_chiar:
        m = load_our_model(args.ours_chiar, CHIARFormer, cfg, device)
        if m: our_models["CHIAR v3 (ours)"] = m
    if not our_models:
        print("No checkpoints — using random weights for demo.")
        our_models["Baseline (ours)"] = BaselineTransformer(cfg).to(device)
        our_models["CHIAR v3 (ours)"] = CHIARFormer(cfg).to(device)

    if not args.heatmap_only:
        print("\n── Option C: Our trained models ──")
        for pk, prompt in PROMPTS.items():
            if pk == "heatmap": continue
            results[pk] = {}
            print(f"\n  [{PROMPT_LABELS[pk]}]")
            print(f"  Prompt: {prompt[:70]}...")
            for mn, model in our_models.items():
                cont = generate_our(model, tokenizer, prompt, device=device)
                results[pk][mn] = cont
                print(f"    {mn}: {cont[:70]}...")

    # ── Option B: HuggingFace models ─────────────────────────────────────────
    if not args.skip_hf and not args.heatmap_only:
        print("\n── Option B: HuggingFace pretrained models ──")
        print("  Note: These use different training data — illustrative only.")
        for pk, prompt in PROMPTS.items():
            if pk == "heatmap": continue
            for mn, hf_id in HF_MODELS.items():
                cont = generate_hf(mn, hf_id, prompt)
                results[pk][mn] = cont
                print(f"    {mn}: {cont[:70]}...")

    # ── Routing heatmap ───────────────────────────────────────────────────────
    if "CHIAR v3 (ours)" in our_models:
        print("\n── Routing heatmap ──")
        chiar = our_models["CHIAR v3 (ours)"]
        cont, rlog = generate_our(chiar, tokenizer, PROMPTS["heatmap"],
                                  device=device, return_routing=True)
        print(f"  Continuation: {cont[:70]}...")
        print(f"  Tokens logged: {len(rlog)}")
        plot_routing_heatmap(PROMPTS["heatmap"], rlog, tokenizer, cont)

    # ── Save ─────────────────────────────────────────────────────────────────
    if not args.heatmap_only:
        print("\n── Building tables ──")
        build_table(results)
        with open(os.path.join(OUT, "qualitative_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nOutputs in {OUT}/:")
    print("  qualitative_continuations.csv")
    print("  qualitative_table.tex")
    print("  qualitative_results.json")
    print("  fig_routing_heatmap.png")
    print("  routing_heatmap_data.json")


if __name__ == "__main__":
    main()
