# train.py — WikiText-103 training (baseline + CHIAR ablations)
# A40 48GB VRAM: no gradient checkpointing needed.
# Mixed precision: fp16 compute, fp32 weights.

import os, sys, math, argparse, json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer

torch.backends.cudnn.benchmark = True
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, ".")
from config import Config, SmallConfig
from model  import CHIARFormer
from data.wikitext import load_wikitext


class BaselineTransformer(nn.Module):
    """
    Standard Pre-LN Transformer baseline with RoPE.
    Each layer = MultiHeadSelfAttention (RoPE) + shared FFN + LayerNorm.
    Identical structure to CHIARLayer — full attention at every layer.
    Parameter count matches CHIARFormer for fair comparison.
    """
    def __init__(self, cfg):
        super().__init__()
        from model.chiar_layer import MultiHeadSelfAttention

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_drop  = nn.Dropout(cfg.dropout)

        # Attention sub-layers
        self.attns = nn.ModuleList([
            MultiHeadSelfAttention(
                cfg.d_model, cfg.n_heads, cfg.dropout,
                cfg.max_seq_len, cfg.rope_base)
            for _ in range(cfg.n_layers)])

        # Shared FFN sub-layers (one per layer, same as CHIARLayer)
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model * 4), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model * 4, cfg.d_model),
                nn.Dropout(cfg.dropout))
            for _ in range(cfg.n_layers)])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model) for _ in range(cfg.n_layers)])

        self.norm    = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
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

    def forward(self, input_ids):
        x = self.emb_drop(self.token_emb(input_ids))
        for attn, ffn, ffn_norm in zip(self.attns, self.ffns, self.ffn_norms):
            x = attn(x)
            x = ffn_norm(x + ffn(x))
        return self.lm_head(self.norm(x)), torch.tensor(0.0, device=x.device)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_lr(step, warmup, total, max_lr, min_lr=1e-6):
    if step < warmup: return max_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * p))


@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss, n = 0.0, 0
    loss_fn = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast():
            logits, _ = model(x)
        loss = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1))
        total_loss += loss.item() * y.numel()
        n          += y.numel()
    model.train()
    torch.cuda.empty_cache()
    avg = total_loss / n
    return avg, math.exp(min(avg, 20))


def train(cfg, model, train_ds, val_ds, test_ds, device, run_name):
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    log_path = os.path.join(cfg.checkpoint_dir, f"{run_name}_log.jsonl")

    tr_loader  = DataLoader(train_ds, batch_size=cfg.batch_size,
                            shuffle=True,  num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=cfg.batch_size * 2,
                            shuffle=False, num_workers=0, pin_memory=True)
    te_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size * 2,
                            shuffle=False, num_workers=0, pin_memory=True)

    opt         = AdamW(model.parameters(), lr=cfg.learning_rate,
                        weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    scaler      = torch.cuda.amp.GradScaler()
    loss_fn     = nn.CrossEntropyLoss()
    total_steps = len(tr_loader) * cfg.max_epochs // cfg.grad_accum_steps

    print(f"\nRun: {run_name}")
    print(f"  Params: {model.count_parameters():,} | Steps: {total_steps:,} | "
          f"LR: {cfg.learning_rate} | Epochs: {cfg.max_epochs}")

    step, best_val_ppl = 0, float("inf")
    opt.zero_grad()

    for epoch in range(cfg.max_epochs):
        for i, (x, y) in enumerate(tr_loader):
            x, y = x.to(device), y.to(device)
            lr   = get_lr(step, cfg.warmup_steps, total_steps, cfg.learning_rate)
            for pg in opt.param_groups: pg["lr"] = lr

            with torch.cuda.amp.autocast():
                logits, rl = model(x)
                loss = loss_fn(logits.view(-1, cfg.vocab_size), y.view(-1)) + rl

            scaler.scale(loss / cfg.grad_accum_steps).backward()

            if (i + 1) % cfg.grad_accum_steps == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad_norm)
                scaler.step(opt); scaler.update()
                opt.zero_grad(); step += 1

                if step % cfg.log_every == 0:
                    print(f"  E{epoch+1} S{step:6d} | "
                          f"Loss {loss.item():.4f} | "
                          f"PPL {math.exp(min(loss.item(),20)):8.2f} | "
                          f"LR {lr:.2e}")

                if step % cfg.eval_every == 0:
                    vl, vp = evaluate(model, val_loader, device, cfg)
                    print(f"\n  *** Val PPL: {vp:.2f} ***\n")
                    with open(log_path, "a") as f:
                        f.write(json.dumps({"step": step, "epoch": epoch+1,
                            "val_loss": vl, "val_ppl": vp, "lr": lr}) + "\n")
                    if vp < best_val_ppl:
                        best_val_ppl = vp
                        torch.save({"step": step, "model": model.state_dict(),
                                    "val_ppl": vp, "cfg": vars(cfg)},
                                   os.path.join(cfg.checkpoint_dir,
                                                f"{run_name}_best.pt"))
                        print(f"  Saved best\n")
        print(f"Epoch {epoch+1} done")

    _, test_ppl = evaluate(model, te_loader, device, cfg)
    results = {"run_name": run_name, "best_val_ppl": best_val_ppl,
               "test_ppl": test_ppl, "params": model.count_parameters(),
               "d_model": cfg.d_model, "n_layers": cfg.n_layers}
    with open(os.path.join(cfg.checkpoint_dir,
                           f"{run_name}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Done. Best Val PPL: {best_val_ppl:.2f} | Test PPL: {test_ppl:.2f}")
    return best_val_ppl, test_ppl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--routing",  default="threshold",
                   choices=["soft", "hard", "threshold"])
    p.add_argument("--variant",  default="dct_attn",
                   choices=["rbf", "dct_attn"])
    p.add_argument("--reg",      action="store_true")
    p.add_argument("--small",    action="store_true",
                   help="Use 17M SmallConfig")
    args = p.parse_args()

    cfg    = SmallConfig() if args.small else Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scale  = "17M" if args.small else "350M"
    print(f"Device: {device} | Scale: {scale} | "
          f"d_model={cfg.d_model} n_heads={cfg.n_heads} n_layers={cfg.n_layers}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    tr, va, te = load_wikitext(cfg, tokenizer, "103")
    tag = f"wikitext103_{scale}"

    if args.baseline:
        run_name = f"baseline_{tag}"
        model    = BaselineTransformer(cfg).to(device)
    else:
        cfg.routing_mode     = args.routing
        cfg.use_collapse_reg = args.reg
        cfg.layer_variant    = args.variant
        rn = f"chiar_{args.routing}"
        if args.variant != "rbf": rn += f"_{args.variant}"
        if args.reg: rn += "_reg"
        run_name = f"{rn}_{tag}"
        model    = CHIARFormer(cfg).to(device)

    train(cfg, model, tr, va, te, device, run_name)


if __name__ == "__main__":
    main()