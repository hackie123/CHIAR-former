# train_all.py — Multi-dataset training including mixed MetaRouter training
#
# Usage:
#   # WikiText-103 ablations
#   python train_all.py --dataset wikitext103 --baseline
#   python train_all.py --dataset wikitext103 --routing threshold --variant dct_attn
#
#   # WikiText-2
#   python train_all.py --dataset wikitext2 --baseline
#   python train_all.py --dataset wikitext2 --routing threshold --variant dct_attn
#
#   # LRA tasks
#   python train_all.py --dataset lra_imdb    --routing threshold --variant dct_attn
#   python train_all.py --dataset lra_listops --routing threshold --variant dct_attn
#   python train_all.py --dataset lra_all     --baseline
#   python train_all.py --dataset lra_all     --routing threshold --variant dct_attn
#
#   # v3: Mixed training (MetaRouter learns naturalistic vs symbolic)
#   python train_all.py --dataset mixed --routing threshold --variant dct_attn

import os, sys, math, argparse, json, random
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from transformers import AutoTokenizer

sys.path.insert(0, ".")
from config import Config
from model  import CHIARFormer, CHIARClassifier
from data.wikitext import load_wikitext
from data.lra      import load_lra_task, LRA_TASKS
from train import BaselineTransformer, get_lr, evaluate as evaluate_lm


# ── Classification baseline ───────────────────────────────────────────────────
class BaselineClassifier(nn.Module):
    """Baseline classifier — same Attn+FFN structure as CHIARLayer."""
    def __init__(self, cfg, num_classes):
        super().__init__()
        from model.chiar_layer import MultiHeadSelfAttention
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)
        self.attns = nn.ModuleList([
            MultiHeadSelfAttention(cfg.d_model, cfg.n_heads,
                cfg.dropout, cfg.max_seq_len, cfg.rope_base)
            for _ in range(cfg.n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 4), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model * 4, cfg.d_model),
                nn.Dropout(cfg.dropout))
            for _ in range(cfg.n_layers)])
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)])
        self.norm       = nn.LayerNorm(cfg.d_model)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.d_model, num_classes))

    def forward(self, input_ids):
        x = self.emb_drop(self.token_emb(input_ids))
        for attn, ffn, ffn_norm in zip(self.attns, self.ffns, self.ffn_norms):
            x = attn(x)
            x = ffn_norm(x + ffn(x))
        return self.classifier(self.norm(x).mean(1)), torch.tensor(0.0, device=x.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Eval classification ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_cls(model, loader, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            logits, _ = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct    += (logits.argmax(-1) == y).sum().item()
        total      += len(y)
    model.train()
    return total_loss/total, 100.0*correct/total


def lr_cls(step, warmup, total, max_lr, min_lr=1e-6):
    if step < warmup: return max_lr * step / max(1, warmup)
    p = (step-warmup) / max(1, total-warmup)
    return min_lr + 0.5*(max_lr-min_lr)*(1+math.cos(math.pi*p))


# ── LM training ───────────────────────────────────────────────────────────────
def train_lm(cfg, model, tr, va, te, device, run_name, tag):
    from train import train
    return train(cfg, model, tr, va, te, device, run_name)


# ── Classification training ───────────────────────────────────────────────────
def train_cls(cfg, model, tr_ds, te_ds, num_classes, device, run_name, task):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, f"{run_name}_log.jsonl")
    bs = max(8, cfg.batch_size//4)
    tr_loader = DataLoader(tr_ds, batch_size=bs, shuffle=True,
                           num_workers=0, pin_memory=True)
    te_loader = DataLoader(te_ds, batch_size=bs*2, shuffle=False,
                           num_workers=0, pin_memory=True)
    # Lower LR for 350M+ classification fine-tuning — 5e-4 causes
    # output collapse (loss stuck at ln(num_classes)) at this scale.
    cls_lr  = min(cfg.learning_rate, 5e-5)
    opt     = AdamW(model.parameters(), lr=cls_lr,
                    weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler  = torch.cuda.amp.GradScaler()
    loss_fn = nn.CrossEntropyLoss()
    total_steps = len(tr_loader) * cfg.max_epochs
    warmup  = max(cfg.warmup_steps, total_steps // 20)
    print(f"\nRun: {run_name} | Task: {task} | Params: {model.count_parameters():,}")
    print(f"  LR: {cls_lr:.2e} | Warmup: {warmup} | Total steps: {total_steps}")

    step, best_acc = 0, 0.0
    opt.zero_grad()
    for epoch in range(cfg.max_epochs):
        for i, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            lr = lr_cls(step, warmup, total_steps, cls_lr)
            for pg in opt.param_groups: pg["lr"] = lr

            with torch.cuda.amp.autocast():
                logits, _ = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
            scaler.step(opt); scaler.update()
            opt.zero_grad()
            step += 1

            if step % 100 == 0:
                print(f"  E{epoch+1} S{step:5d} | Loss {loss.item():.4f} | LR {lr:.2e}")
            if step % 300 == 0:
                tl, acc = evaluate_cls(model, te_loader, device)
                print(f"\n  *** Test Acc: {acc:.2f}% (loss {tl:.4f}) ***\n")
                with open(log_path, "a") as f:
                    f.write(json.dumps({"step":step,"epoch":epoch+1,
                        "test_acc":acc,"task":task}) + "\n")
                if acc > best_acc:
                    best_acc = acc
                    torch.save({"step":step,"model":model.state_dict(),"test_acc":acc},
                               os.path.join(cfg.checkpoint_dir, f"{run_name}_best.pt"))
        print(f"Epoch {epoch+1} done")

    _, final_acc = evaluate_cls(model, te_loader, device)
    results = {"run_name":run_name,"task":task,"best_test_acc":best_acc,
               "final_test_acc":final_acc,"params":model.count_parameters()}
    with open(os.path.join(cfg.checkpoint_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Best: {best_acc:.2f}% | Final: {final_acc:.2f}%")
    return best_acc


# ── Mixed training (v3 MetaRouter) ────────────────────────────────────────────
def train_mixed(cfg, model, device):
    """
    Mixed training for MetaRouter learning.
    Batches sampled from all 4 datasets with configured mix ratio.
    MetaRouter learns to distinguish naturalistic vs symbolic from downstream loss.
    Eval/test done separately per dataset after training.
    """
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    run_name = "chiar_v3_mixed"
    log_path = os.path.join(cfg.checkpoint_dir, f"{run_name}_log.jsonl")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    print("Loading all datasets for mixed training...")

    # LM datasets (WikiText-103 and WikiText-2)
    tr103, va103, te103 = load_wikitext(cfg, tokenizer, "103")
    tr2,   va2,   te2   = load_wikitext(cfg, tokenizer, "2")

    # Classification datasets
    tr_imdb,    te_imdb,    _  = load_lra_task("imdb",    cfg, tokenizer)
    tr_listops, te_listops, _  = load_lra_task("listops", cfg, tokenizer)

    # Build loaders — WikiText-103 is dominant, so we oversample others
    bs     = cfg.batch_size
    lm_ldr = DataLoader(tr103,     batch_size=bs, shuffle=True, num_workers=0)
    wt2_ldr= DataLoader(tr2,       batch_size=bs, shuffle=True, num_workers=0)
    imdb_ldr = DataLoader(tr_imdb, batch_size=max(8,bs//4), shuffle=True, num_workers=0)
    lo_ldr   = DataLoader(tr_listops, batch_size=max(8,bs//4), shuffle=True, num_workers=0)

    val_ldr  = DataLoader(va103, batch_size=bs*2, shuffle=False, num_workers=0)

    # Pre-create classification heads BEFORE optimizer so their params
    # are included in AdamW (fixes "_cls_head_* never trained" bug).
    model._cls_head_imdb    = nn.Linear(cfg.d_model, 2).to(device)
    model._cls_head_listops = nn.Linear(cfg.d_model, 10).to(device)

    opt      = AdamW(model.parameters(), lr=cfg.learning_rate,
                     weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler   = torch.cuda.amp.GradScaler()
    lm_loss  = nn.CrossEntropyLoss()
    cls_loss = nn.CrossEntropyLoss()

    # total_steps = optimizer steps (accounts for grad accumulation)
    total_steps = (len(lm_ldr) * cfg.max_epochs) // cfg.grad_accum_steps
    print(f"Mixed training | Params: {model.count_parameters():,} | "
          f"Optimizer steps: {total_steps:,}")
    print(f"  WikiText-103: {len(tr103):,} | WikiText-2: {len(tr2):,} | "
          f"IMDB: {len(tr_imdb):,} | ListOps: {len(tr_listops):,}")

    # Cycle non-primary loaders
    wt2_iter  = iter(wt2_ldr)
    imdb_iter = iter(imdb_ldr)
    lo_iter   = iter(lo_ldr)

    def next_batch(it, loader):
        try: return next(it), it
        except StopIteration:
            it = iter(loader); return next(it), it

    step, best_val_ppl = 0, float("inf")
    g_lm = 0.0   # MetaRouter gate for logging — init before first use
    opt.zero_grad()
    mix = cfg.meta_mix_ratio  # e.g. 0.25 of steps use non-wikitext103 data

    for epoch in range(cfg.max_epochs):
        for i, (x_lm, y_lm) in enumerate(lm_ldr):
            lr = get_lr(step, cfg.warmup_steps, total_steps, cfg.learning_rate)
            for pg in opt.param_groups: pg["lr"] = lr

            with torch.cuda.amp.autocast():
                # Primary: WikiText-103 LM
                x_lm, y_lm = x_lm.to(device), y_lm.to(device)
                logits, rl  = model(x_lm)
                loss = lm_loss(logits.view(-1, cfg.vocab_size), y_lm.view(-1)) + rl

                # Mixed batch: sample one of the other datasets
                if random.random() < mix:
                    choice = random.choice(["wt2", "imdb", "listops"])
                    if choice == "wt2":
                        (x2, y2), wt2_iter = next_batch(wt2_iter, wt2_ldr)
                        x2, y2 = x2.to(device), y2.to(device)
                        logits2, _ = model(x2)
                        loss = loss + lm_loss(
                            logits2.view(-1, cfg.vocab_size), y2.view(-1))
                    elif choice == "imdb":
                        (xi, yi), imdb_iter = next_batch(imdb_iter, imdb_ldr)
                        xi, yi = xi.to(device), yi.to(device)
                        pooled = model.norm(
                            model.emb_drop(model.token_emb(xi))).mean(1)
                        loss = loss + cls_loss(model._cls_head_imdb(pooled), yi)
                    else:
                        (xlo, ylo), lo_iter = next_batch(lo_iter, lo_ldr)
                        xlo, ylo = xlo.to(device), ylo.to(device)
                        pooled = model.norm(
                            model.emb_drop(model.token_emb(xlo))).mean(1)
                        loss = loss + cls_loss(model._cls_head_listops(pooled), ylo)

            scaler.scale(loss / cfg.grad_accum_steps).backward()

            if (i+1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                scaler.step(opt); scaler.update()
                opt.zero_grad(); step += 1

                if step % cfg.log_every == 0:
                    # Log MetaRouter gate value for current LM batch
                    with torch.no_grad():
                        g_lm = model.meta_router.gate_value(
                            model.emb_drop(model.token_emb(x_lm)))
                    print(f"  E{epoch+1} S{step:6d} | Loss {loss.item():.4f} | "
                          f"LR {lr:.2e} | MetaGate(LM)={g_lm:.3f}")

                if step % cfg.eval_every == 0:
                    vl, vp = evaluate_lm(model, val_ldr, device, cfg)
                    print(f"\n  *** Val PPL: {vp:.2f} | MetaGate(LM)={g_lm:.3f} ***")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"step":step,"epoch":epoch+1,
                            "val_ppl":vp,"meta_gate_lm":g_lm}) + "\n")
                    if vp < best_val_ppl:
                        best_val_ppl = vp
                        torch.save({"step":step,"model":model.state_dict(),"val_ppl":vp},
                            os.path.join(cfg.checkpoint_dir, f"{run_name}_best.pt"))
                        print(f"  Saved best\n")
        print(f"Epoch {epoch+1} done")

    # Final eval on all datasets
    print("\n=== Final Evaluation (all datasets) ===")
    _, wt103_ppl = evaluate_lm(model, DataLoader(te103, batch_size=bs*2, num_workers=0), device, cfg)
    _, wt2_ppl   = evaluate_lm(model, DataLoader(te2,   batch_size=bs*2, num_workers=0), device, cfg)
    print(f"WikiText-103 Test PPL: {wt103_ppl:.2f}")
    print(f"WikiText-2   Test PPL: {wt2_ppl:.2f}")

    results = {"run_name":run_name,"wikitext103_ppl":wt103_ppl,
               "wikitext2_ppl":wt2_ppl,"best_val_ppl":best_val_ppl,
               "params":model.count_parameters()}
    with open(os.path.join(cfg.checkpoint_dir, f"{run_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
        choices=["wikitext103","wikitext2",
                 "lra_imdb","lra_listops","lra_all",
                 "mixed"])
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--routing",  default="threshold",
                   choices=["soft","hard","threshold"])
    p.add_argument("--variant",  default="dct_attn",
                   choices=["rbf","dct_attn"])
    p.add_argument("--reg",      action="store_true")
    p.add_argument("--small",    action="store_true", help="Use 17M SmallConfig")
    args = p.parse_args()

    from config import SmallConfig
    cfg    = SmallConfig() if args.small else Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {args.dataset}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    # ── Mixed training (v3 MetaRouter) ────────────────────────────────────────
    if args.dataset == "mixed":
        cfg.routing_mode  = args.routing
        cfg.layer_variant = args.variant
        model = CHIARFormer(cfg).to(device)
        train_mixed(cfg, model, device)
        return

    # ── LM datasets ───────────────────────────────────────────────────────────
    if args.dataset in ("wikitext103", "wikitext2"):
        ver = "103" if args.dataset == "wikitext103" else "2"
        tr, va, te = load_wikitext(cfg, tokenizer, ver)
        tag = f"wikitext{ver}"
        if args.baseline:
            model    = BaselineTransformer(cfg).to(device)
            run_name = f"baseline_{tag}"
        else:
            cfg.routing_mode     = args.routing
            cfg.use_collapse_reg = args.reg
            cfg.layer_variant    = args.variant
            rn = f"chiar_{args.routing}"
            if args.variant != "rbf": rn += f"_{args.variant}"
            if args.reg: rn += "_reg"
            run_name = f"{rn}_{tag}"
            model    = CHIARFormer(cfg).to(device)
        from train import train
        train(cfg, model, tr, va, te, device, run_name)

    # ── LRA classification ────────────────────────────────────────────────────
    elif args.dataset.startswith("lra"):
        cfg.max_epochs   = 4
        cfg.warmup_steps = 200
        cfg.learning_rate = 5e-5
        tasks = list(LRA_TASKS.keys()) \
            if args.dataset == "lra_all" \
            else [args.dataset.replace("lra_","")]

        all_results = {}
        for task in tasks:
            tr_ds, te_ds, num_classes = load_lra_task(task, cfg, tokenizer)
            if args.baseline:
                model    = BaselineClassifier(cfg, num_classes).to(device)
                run_name = f"baseline_lra_{task}"
            else:
                cfg.routing_mode  = args.routing
                cfg.layer_variant = args.variant
                rn = f"chiar_{args.routing}"
                if args.variant != "rbf": rn += f"_{args.variant}"
                run_name = f"{rn}_lra_{task}"
                model    = CHIARClassifier(cfg, num_classes).to(device)
            acc = train_cls(cfg, model, tr_ds, te_ds,
                            num_classes, device, run_name, task)
            all_results[task] = acc

        if len(all_results) > 1:
            avg = sum(all_results.values()) / len(all_results)
            print(f"\nLRA Summary:")
            for t, a in all_results.items():
                print(f"  {t:12s}: {a:.2f}%")
            print(f"  {'Average':12s}: {avg:.2f}%")
            tag = "baseline" if args.baseline else f"chiar_{args.routing}_{args.variant}"
            with open(os.path.join(cfg.checkpoint_dir, f"lra_summary_{tag}.json"),"w") as f:
                json.dump({"tasks":all_results,"average":avg}, f, indent=2)


if __name__ == "__main__":
    main()