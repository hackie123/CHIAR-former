# data/lra.py — IMDB and ListOps
# Published baselines (Tay et al. 2021, char-level, from scratch):
#   IMDB:    Transformer 64.27% | FNet 65.11% | Performer 65.40%
#   ListOps: Transformer 36.37% | FNet 35.33% | Performer 18.01%
import torch, numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset


class LRADataset(Dataset):
    def __init__(self, inputs, labels, seq_len):
        self.inputs = inputs; self.labels = labels; self.seq_len = seq_len

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        x = self.inputs[idx]
        if len(x) < self.seq_len:
            x = torch.cat([x, torch.zeros(self.seq_len-len(x), dtype=torch.long)])
        return x[:self.seq_len], self.labels[idx]


def load_imdb(cfg, tokenizer):
    print("  Loading IMDB..."); ds = load_dataset("imdb")
    seq_len = min(cfg.max_seq_len, 512)
    def process(split):
        inputs, labels = [], []
        for row in ds[split]:
            ids = tokenizer.encode(row["text"], add_special_tokens=True,
                                   max_length=seq_len, truncation=True)
            inputs.append(torch.tensor(ids, dtype=torch.long))
            labels.append(int(row["label"]))
        return inputs, torch.tensor(labels, dtype=torch.long)
    tr_in, tr_lb = process("train"); te_in, te_lb = process("test")
    print(f"    Train: {len(tr_lb):,} | Test: {len(te_lb):,}")
    return LRADataset(tr_in,tr_lb,seq_len), LRADataset(te_in,te_lb,seq_len), 2


def load_listops(cfg, tokenizer=None):
    print("  Generating ListOps locally...")
    seq_len = min(cfg.max_seq_len, 256)
    def generate(n, seed):
        rng = np.random.RandomState(seed); inputs, labels = [], []
        for _ in range(n):
            op = rng.randint(0,4); seq = [14, 10+op]
            vals = [rng.randint(0,10) for _ in range(rng.randint(4,9))]
            seq += vals + [15]
            if len(seq) < seq_len: seq += [16]*(seq_len-len(seq))
            if op==0: label=max(vals)
            elif op==1: label=min(vals)
            elif op==2: label=int(np.mean(vals))%10
            else: label=int(np.median(vals))%10
            inputs.append(torch.tensor(seq[:seq_len], dtype=torch.long))
            labels.append(label)
        return inputs, torch.tensor(labels, dtype=torch.long)
    tr_in,tr_lb = generate(96000,42); te_in,te_lb = generate(2000,99)
    print(f"    Train: {len(tr_lb):,} | Test: {len(te_lb):,}")
    return LRADataset(tr_in,tr_lb,seq_len), LRADataset(te_in,te_lb,seq_len), 10


LRA_TASKS = {"imdb": load_imdb, "listops": load_listops}

def load_lra_task(task_name, cfg, tokenizer):
    if task_name not in LRA_TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    return LRA_TASKS[task_name](cfg, tokenizer)
