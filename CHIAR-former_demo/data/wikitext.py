# data/wikitext.py — WikiText-103 and WikiText-2
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class WikiTextDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens; self.seq_len = seq_len
        self.n = (len(tokens)-1)//seq_len

    def __len__(self): return self.n

    def __getitem__(self, idx):
        s = idx*self.seq_len
        return self.tokens[s:s+self.seq_len], self.tokens[s+1:s+self.seq_len+1]


def load_wikitext(cfg, tokenizer, version="103"):
    name = f"wikitext-{version}-v1"
    print(f"Loading {name}...")
    ds = load_dataset("wikitext", name)

    def tokenize(split):
        all_tokens = []
        rows = ds[split]["text"]
        for i in range(0, len(rows), 1000):
            text = "\n".join(rows[i:i+1000])
            if text.strip():
                all_tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        return torch.tensor(all_tokens, dtype=torch.long)

    tr = tokenize("train"); va = tokenize("validation"); te = tokenize("test")
    print(f"  Train: {len(tr):,} | Val: {len(va):,} | Test: {len(te):,}")
    return (WikiTextDataset(tr, cfg.max_seq_len),
            WikiTextDataset(va, cfg.max_seq_len),
            WikiTextDataset(te, cfg.max_seq_len))
