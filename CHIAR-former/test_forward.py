# test_forward.py — Forward pass verification (all 9 tests)
import torch, sys
sys.path.insert(0, ".")
from config import Config
from model  import CHIARFormer, MetaRouter, RotaryEmbedding, SpectralRouter
from model.dct_mix import DCTMix
from model.rbf_mix import RBFMix
from train  import BaselineTransformer


def check(cond, msg):
    print(f"  {'[PASS]' if cond else '[FAIL]'}  {msg}")
    return cond


cfg = Config(); cfg.vocab_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}\n  CHIAR-Former v3 Forward Pass Tests\n{'='*60}")
print(f"Device: {device}")
B, T, d = 2, 16, cfg.d_model
x = torch.randn(B, T, d, device=device)

# Test 1: RoPE
print("\n── Test 1: RoPE ──")
rope = RotaryEmbedding(d // cfg.n_heads, max_seq_len=64)
q = torch.randn(B, cfg.n_heads, T, d // cfg.n_heads)
k = torch.randn(B, cfg.n_heads, T, d // cfg.n_heads)
qr, kr = rope(q, k)
check(qr.shape == q.shape, f"Q shape preserved: {qr.shape}")
check(not qr.isnan().any(), "No NaNs in rotated Q")
check(not torch.allclose(qr, q), "RoPE actually rotates (Q changed)")

# Test 2: DCT Mix
print("\n── Test 2: DCT Mix ──")
dct = DCTMix(d).to(device)
out = dct(x)
check(out.shape == x.shape, f"Shape preserved: {out.shape}")
check(not out.isnan().any(), "No NaNs")

# Test 3: RBF Mix
print("\n── Test 3: RBF Mix ──")
rbf = RBFMix(d).to(device)
out = rbf(x)
check(out.shape == x.shape, f"Shape preserved: {out.shape}")
check(not out.isnan().any(), "No NaNs")

# Test 4: Spectral Router
print("\n── Test 4: Spectral Router ──")
router = SpectralRouter(d, 0.855, 0.865, "threshold", n_ops=2).to(device)
gates, H, op_idx = router(x)
check(gates.shape == (B, T, 2), f"Gates shape: {gates.shape}")
check(H.shape == (B, T), f"Entropy shape: {H.shape}")
check((H >= 0).all() and (H <= 1).all(),
      f"H in [0,1]: min={H.min():.3f} max={H.max():.3f}")

# Test 5: MetaRouter
print("\n── Test 5: MetaRouter (learned) ──")
mr   = MetaRouter(d).to(device)
gate = mr(x)
check(gate.dim() == 0, f"Gate is scalar: {gate.item():.4f}")
check(0 <= gate.item() <= 1, f"Gate in [0,1]: {gate.item():.4f}")
gate.backward()
check(mr.gate_proj.weight.grad is not None, "Gradient flows through MetaRouter")

# Test 6: CHIARFormer full forward
print("\n── Test 6: CHIARFormer full forward ──")
model = CHIARFormer(cfg).to(device)
ids   = torch.randint(0, cfg.vocab_size, (B, T), device=device)
logits, rl = model(ids)
check(logits.shape == (B, T, cfg.vocab_size), f"Logits shape: {logits.shape}")
check(not logits.isnan().any(), "No NaNs in logits")
check(rl.item() == 0.0, "Routing loss is 0 (collapse reg off)")
print(f"  CHIAR Params: {model.count_parameters():,}")

# Test 7: MetaRouter gate
print("\n── Test 7: MetaRouter gate values ──")
logits, _, _, mg = model(ids, return_routing_info=True)
check(0 <= mg.item() <= 1, f"MetaRouter gate: {mg.item():.4f}")

# Test 8: No absolute positional embedding
print("\n── Test 8: No absolute positional embedding ──")
check(not hasattr(model, "pos_emb"), "pos_emb removed (RoPE used instead)")

# Test 9: Backward pass
print("\n── Test 9: Backward pass ──")
import torch.nn.functional as F
loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), ids.view(-1))
loss.backward()
check(model.token_emb.weight.grad is not None, "Gradients flow to embeddings")

# Test 10: Baseline parameter parity
print("\n── Test 10: Baseline parameter parity ──")
baseline = BaselineTransformer(cfg).to(device)
bp = baseline.count_parameters()
cp = model.count_parameters()
diff_pct = abs(bp - cp) / bp * 100
check(diff_pct < 5.0,
      f"Params within 5%: Baseline={bp:,} CHIAR={cp:,} diff={diff_pct:.2f}%")

# Test 11: Baseline forward
print("\n── Test 11: Baseline forward ──")
blog, _ = baseline(ids)
check(blog.shape == (B, T, cfg.vocab_size), f"Baseline logits shape: {blog.shape}")
check(not blog.isnan().any(), "No NaNs in baseline logits")

print(f"\n{'='*60}\nAll tests complete.")
