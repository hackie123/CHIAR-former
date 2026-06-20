# app.py — CHIAR-Former Live Routing Demo
# Streamlit app for HuggingFace Spaces
#
# Auto-downloads checkpoint from HuggingFace Hub on first run.
# If the model fails to load or inference fails, a hard error is shown.
# There is NO heuristic fallback — results are always from the real model.
#
# Usage (local):   streamlit run app.py
# Deployment:      push to HuggingFace Spaces (SDK: Streamlit)

import os, sys, time, threading, traceback
import streamlit as st
import torch

sys.path.insert(0, ".")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHIAR-Former: Live Routing Demo",
    page_icon="🎨",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
HF_REPO_ID  = "prateeksikdar/CHIAR-Former"
HF_FILENAME = "chiar_dct_attn_400M_best.pt"
CKPT_PATH   = os.path.join("checkpoints", HF_FILENAME)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0D1B2A; color: #F0F0F0; }
  .stApp { background-color: #0D1B2A; }
  .title-box {
    background: linear-gradient(135deg, #1A4E8C, #0D1B2A);
    border-left: 5px solid #F39C12;
    padding: 24px 28px; border-radius: 8px; margin-bottom: 24px;
  }
  .title-box h1 { color: #F39C12; font-size: 2.2rem; margin:0; }
  .title-box p  { color: #AABBCC; font-size: 1.0rem; margin-top:8px; }
  .chiaroscuro-box {
    background: #111E2E; border: 1px solid #2E4A6A;
    border-left: 4px solid #8E44AD; padding: 16px 20px;
    border-radius: 6px; margin-bottom: 20px;
    font-size: 0.92rem; color: #C8D8E8; line-height: 1.7;
  }
  .stat-card {
    background: #111E2E; border: 1px solid #2E4A6A;
    border-radius: 8px; padding: 16px; text-align: center; margin: 4px;
  }
  .stat-val { font-size: 1.8rem; font-weight: bold; color: #F39C12; }
  .stat-lbl { font-size: 0.8rem; color: #7A9AB5; margin-top: 4px; }
  .token-box {
    display: inline-block; padding: 5px 9px; margin: 3px 2px;
    border-radius: 5px; font-size: 0.95rem; font-weight: 600;
    font-family: monospace; color: white; vertical-align: middle;
  }
  .dct-token  { background-color: #1A5276; border: 1px solid #2E86C1; }
  .attn-token { background-color: #7B241C; border: 1px solid #C0392B; }
  .legend-dct  { background:#1A5276; color:white; padding:4px 12px;
                  border-radius:4px; font-size:0.85rem; }
  .legend-attn { background:#7B241C; color:white; padding:4px 12px;
                  border-radius:4px; font-size:0.85rem; }
  .loading-line { font-size:1.05rem; color:#85C1E9; padding:6px 0; line-height:1.8; }
  .result-header { font-size:1.2rem; font-weight:bold; color:#F39C12; margin-bottom:12px; }
  .contribution-box {
    background: #0A1520; border: 1px solid #1A3A55; border-radius: 6px;
    padding: 14px 18px; margin: 6px 0; font-size: 0.9rem; color: #B0C8E0;
  }
  .status-loaded {
    background:#1E4D2B; color:#58D68D; border:1px solid #27AE60;
    padding:6px 12px; border-radius:6px; font-size:0.85rem;
    display:block; margin-bottom:4px;
  }
  .status-failed {
    background:#2D0A0A; color:#F1948A; border:1px solid #7B1C1C;
    padding:6px 12px; border-radius:6px; font-size:0.85rem;
    display:block; margin-bottom:4px;
  }
  .status-loading {
    background:#1A2E4A; color:#85C1E9; border:1px solid #2E86C1;
    padding:6px 12px; border-radius:6px; font-size:0.85rem;
    display:block; margin-bottom:4px;
  }
  .error-box {
    background:#2D0A0A; border:1px solid #7B1C1C; border-radius:8px;
    padding:20px 24px; margin-top:16px;
  }
  .error-box h3 { color:#F1948A; margin:0 0 8px 0; font-size:1.1rem; }
  .error-step { color:#E59866; font-size:0.9rem; margin-bottom:6px; font-weight:600; }
  .error-trace {
    background:#1A0505; border:1px solid #5B1010; border-radius:4px;
    padding:12px; font-size:0.78rem; color:#F1948A;
    font-family:monospace; white-space:pre-wrap; word-break:break-all;
    max-height:300px; overflow-y:auto; margin-top:6px;
  }
  .link-row a { color:#85C1E9; text-decoration:none; margin-right:16px; font-size:0.88rem; }
  .link-row a:hover { text-decoration:underline; }
</style>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>🎨 CHIAR-Former: Live Routing Demo</h1>
  <p>Chiaroscuro Attention — Spending Compute in the Dark &nbsp;|&nbsp;
     Prateek Kumar Sikdar, Accenture &nbsp;|&nbsp; AAAI 2027</p>
  <div class="link-row" style="margin-top:12px;">
    <a href="https://arxiv.org/abs/2606.08327" target="_blank">📄 arXiv Paper</a>
    <a href="https://github.com/hackie123/CHIAR-former" target="_blank">💻 GitHub</a>
    <a href="https://huggingface.co/prateeksikdar/CHIAR-Former" target="_blank">🤗 Model Weights</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Chiaroscuro inspiration ───────────────────────────────────────────────────
st.markdown("""
<div class="chiaroscuro-box">
  <b style="color:#C39BD3;font-size:1.05rem;">🖼️ The Chiaroscuro Inspiration</b><br><br>
  In Renaissance painting, <i>chiaroscuro</i> (Italian: <i>chiaro</i> = light,
  <i>scuro</i> = dark) is the technique of using extreme contrasts of light and shadow
  to model form. Masters like <b>Caravaggio</b>, <b>Leonardo da Vinci</b>, and
  <b>Rembrandt</b> did not illuminate a canvas uniformly — they poured light exactly
  where the eye needed detail, leaving peripheral regions in inexpensive shadow.<br><br>
  <b>CHIAR-Former borrows this principle for computation.</b> Function words —
  <i>the</i>, <i>of</i>, <i>and</i> — are the shadowed periphery: smooth,
  low-frequency, cheap to process with DCT spectral mixing in
  <span style="color:#85C1E9;">O(d log d)</span>.
  Content words — <i>overwhelming</i>, <i>consensus</i>, <i>paradox</i> — are the
  illuminated focal points: rich, high-entropy, deserving full self-attention in
  <span style="color:#F1948A;">O(n²d)</span>.<br><br>
  The model measures this via <b>spectral entropy</b> H(x) — the information entropy
  of each token's DCT power spectrum. Low entropy = smooth = DCT (blue).
  High entropy = complex = Attention (red).
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# session_state["model_status"]:
#   "pending" — not attempted yet
#   "loaded"  — model ready
#   "failed"  — load failed; session_state["model_error"] has the details
# ══════════════════════════════════════════════════════════════════════════════

if "model_status" not in st.session_state:
    st.session_state["model_status"] = "pending"
    st.session_state["model_error"]  = {"step": "", "trace": ""}
    st.session_state["model"]        = None
    st.session_state["tokenizer"]    = None
    st.session_state["val_ppl"]      = None


def attempt_load():
    """
    Load CHIAR-Former into session_state.
    Six discrete steps — each fails loudly with its step name and full traceback.
    No fallback, no heuristic, no silent catch.
    Called once per session; subsequent calls are no-ops.
    """
    if st.session_state["model_status"] != "pending":
        return

    def fail(step, tb):
        st.session_state["model_status"] = "failed"
        st.session_state["model_error"]  = {"step": step, "trace": tb}

    # Step 1 — tokenizer
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        st.session_state["tokenizer"] = tok
    except Exception:
        fail("Step 1 — Load GPT-2 tokenizer", traceback.format_exc())
        return

    # Step 2 — download checkpoint
    if not os.path.exists(CKPT_PATH):
        try:
            from huggingface_hub import hf_hub_download
            os.makedirs("checkpoints", exist_ok=True)
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=HF_FILENAME,
                local_dir="./checkpoints",
            )
        except ImportError:
            fail(
                "Step 2 — Download checkpoint",
                "huggingface_hub is not installed.\n"
                "Fix: pip install huggingface_hub"
            )
            return
        except Exception:
            fail("Step 2 — Download checkpoint from HuggingFace Hub", traceback.format_exc())
            return

    # Step 3 — torch.load
    try:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
    except Exception:
        fail(f"Step 3 — torch.load('{CKPT_PATH}')", traceback.format_exc())
        return

    # Step 4 — build config
    try:
        from config import Config
        cfg = Config()
        cfg.vocab_size = len(tok)
        for k, v in ckpt.get("cfg", {}).items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    except Exception:
        fail("Step 4 — Build model Config", traceback.format_exc())
        return

    # Step 5 — instantiate model
    try:
        from model import CHIARFormer
        model = CHIARFormer(cfg)
    except Exception:
        fail("Step 5 — CHIARFormer.__init__(cfg)", traceback.format_exc())
        return

    # Step 6 — load weights
    try:
        model.load_state_dict(ckpt["model"])
        model.eval()
    except Exception:
        fail("Step 6 — model.load_state_dict(checkpoint)", traceback.format_exc())
        return

    # All steps passed
    st.session_state["model"]        = model
    st.session_state["val_ppl"]      = ckpt.get("val_ppl", None)
    st.session_state["model_status"] = "loaded"


def show_load_error():
    """Render a hard, detailed error block. No routing result is shown."""
    err  = st.session_state["model_error"]
    step = err.get("step", "unknown step")
    tb   = err.get("trace", "no traceback available")
    st.markdown(f"""
    <div class="error-box">
      <h3>🚫 Model failed to load — no routing results available</h3>
      <div class="error-step">Failed at: {step}</div>
      <div class="error-trace">{tb}</div>
      <br>
      <span style="color:#A04040;font-size:0.85rem;">
        Fix the error above and restart the app.
        Check that the checkpoint at <code>{CKPT_PATH}</code> matches the model
        architecture in <code>config.py</code> and that all dependencies
        (torch, transformers, huggingface_hub) are installed correctly.
      </span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Loading messages ──────────────────────────────────────────────────────────

LOADING_LINES = [
    ("⚡", "CHIAR-Former saves ~37% total FLOPs vs full attention on this input..."),
    ("🔵", "Function words → DCT Mixing — O(d log d) — cheap spectral processing..."),
    ("🔴", "Content words → Full Self-Attention — O(n²d) — rich dynamic routing..."),
    ("🧠", "MetaRouter learned this blend from 4 datasets — no manual thresholds..."),
    ("📐", "400M parameters — 4M fewer than the full-attention baseline..."),
    ("🎯", "SpectralRouter uses entropy H(x) to decide per token — theory-driven..."),
    ("🌊", "Like Caravaggio's brush — light only where the signal demands detail..."),
]


def typewriter_loading(placeholder):
    lines_shown = []
    for icon, text in LOADING_LINES:
        lines_shown.append(f'<div class="loading-line">{icon} &nbsp; {text}</div>')
        placeholder.markdown(
            '<div style="background:#0A1520;border:1px solid #1A3A55;'
            'border-radius:8px;padding:16px 20px;">'
            + "".join(lines_shown) + "</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1.2)


# ── Routing inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def get_routing(model, tokenizer, text):
    """Run forward pass and return per-token routing decisions from the model."""
    ids = tokenizer.encode(text, return_tensors="pt")
    ids = ids[:, :256]

    logits, infos, _, meta_gate = model(ids, return_routing_info=True)

    info   = infos[1] if len(infos) > 1 else infos[0]
    op_idx = info.get("op_idx")   # (1, T) — 0=DCT, 1=Attention
    H      = info.get("H")        # (1, T) — spectral entropy per token

    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]

    if op_idx is None:
        raise RuntimeError(
            "Model did not return op_idx in routing_info. "
            "Ensure return_routing_info=True is handled in CHIARFormer.forward()."
        )

    ops     = op_idx[0].tolist()
    entropy = H[0].tolist() if H is not None else [None] * len(tokens)
    gate    = meta_gate.item() if meta_gate is not None else None
    return tokens, ops, entropy, gate


# ── Render heatmap ────────────────────────────────────────────────────────────

def render_heatmap(tokens, ops, entropy, meta_gate):
    n_dct   = ops.count(0)
    n_attn  = ops.count(1)
    total   = len(ops)
    pct_dct = 100 * n_dct / total if total else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-val">{total}</div>
          <div class="stat-lbl">Tokens</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-val" style="color:#2E86C1;">{n_dct} ({pct_dct:.0f}%)</div>
          <div class="stat-lbl">→ DCT Mixing</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="stat-card">
          <div class="stat-val" style="color:#C0392B;">{n_attn} ({100-pct_dct:.0f}%)</div>
          <div class="stat-lbl">→ Full Attention</div></div>""", unsafe_allow_html=True)
    with col4:
        gate_str = f"{meta_gate:.3f}" if meta_gate is not None else "—"
        st.markdown(f"""<div class="stat-card">
          <div class="stat-val" style="color:#27AE60;">{gate_str}</div>
          <div class="stat-lbl">MetaRouter gate g</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <span class="legend-dct">🔵 DCT Mixing — O(d log d)</span> &nbsp;&nbsp;
    <span class="legend-attn">🔴 Full Attention — O(n²d)</span>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    html_tokens = []
    for tok, op, h in zip(tokens, ops, entropy):
        display = tok.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
        cls     = "dct-token" if op == 0 else "attn-token"
        title   = f"H={h:.3f}" if h is not None else ""
        html_tokens.append(
            f'<span class="token-box {cls}" title="{title}">{display}</span>'
        )

    st.markdown(
        '<div style="line-height:2.4;padding:16px;background:#0A1520;'
        'border:1px solid #1A3A55;border-radius:8px;">'
        + "".join(html_tokens) + "</div>",
        unsafe_allow_html=True,
    )

    if any(h is not None for h in entropy):
        st.markdown(
            '<span style="font-size:0.78rem;color:#4A6A8A;">'
            '💡 Hover over any token to see its spectral entropy H(x)</span>',
            unsafe_allow_html=True,
        )

    gate_line = ""
    if meta_gate is not None:
        regime = ("naturalistic text regime — DCT preprocessing active."
                  if meta_gate > 0.5
                  else "symbolic/structured input regime — DCT largely bypassed.")
        gate_line = f"MetaRouter gate g = <b>{meta_gate:.3f}</b> — {regime}<br>"

    st.markdown(f"""
    <div class="chiaroscuro-box" style="margin-top:12px;">
      <b style="color:#F39C12;">Routing insight for this input:</b><br>
      {n_dct} of {total} tokens ({100*n_dct//total if total else 0}%) →
      <span style="color:#85C1E9;"><b>DCT Mixing</b></span> O(d log d).<br>
      {n_attn} of {total} tokens ({100*n_attn//total if total else 0}%) →
      <span style="color:#F1948A;"><b>Full Attention</b></span> O(n²d).<br>
      {gate_line}<br>
      This is the chiaroscuro principle: illuminate only where the signal demands it.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚙️ Model Status")

    status = st.session_state["model_status"]

    if status == "pending":
        st.markdown(
            '<span class="status-loading">⏳ Not loaded — click Route Tokens</span>',
            unsafe_allow_html=True,
        )
    elif status == "loaded":
        ppl     = st.session_state["val_ppl"]
        ppl_str = f" · Val PPL {ppl:.2f}" if ppl else ""
        st.markdown(
            f'<span class="status-loaded">✅ Model loaded{ppl_str}</span>',
            unsafe_allow_html=True,
        )
    elif status == "failed":
        step = st.session_state["model_error"].get("step", "")
        st.markdown(
            f'<span class="status-failed">🚫 Load failed — {step}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 🔬 Key Contributions")

    contributions = [
        ("🔵 Spectral Entropy Routing",
         "Per-token routing via H(x) — theory-grounded in KL optimality of DCT "
         "for low-entropy signals. No black-box gate."),
        ("💥 Routing Collapse Discovery",
         "3-operator system (DCT + RBF + Attention) collapses to DCT+Attention "
         "during training — revealing the optimal operator subset."),
        ("🧠 Learned MetaRouter",
         "Task-level gate g = σ(Linear(mean(x))) trained end-to-end on mixed batches. "
         "Stabilises at g ≈ 0.22 — a compute–quality equilibrium at scale."),
        ("⚡ 37% FLOP Reduction",
         "Hard per-token routing saves 62.5% of attention FLOPs in routing "
         "layers → ~37% total reduction at 400M scale."),
        ("🌊 Chiaroscuro Principle",
         "Spend compute where the signal is dark (high entropy). "
         "Light tokens → DCT. Dark tokens → Attention."),
        ("📊 Scale Experiments",
         "16M ablations + 400M scaling study + mixed multi-task MetaRouter training."),
    ]

    for title, desc in contributions:
        st.markdown(f"""
        <div class="contribution-box">
          <b style="color:#F39C12;">{title}</b><br>
          <span style="font-size:0.82rem;">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.82rem;color:#5A7A9A;">
      <b style="color:#7A9AB5;">Links</b><br>
      <a href="https://arxiv.org/abs/2606.08327" target="_blank"
         style="color:#85C1E9;">📄 arXiv:2606.08327</a><br>
      <a href="https://github.com/hackie123/CHIAR-former" target="_blank"
         style="color:#85C1E9;">💻 GitHub repo</a><br>
      <a href="https://huggingface.co/prateeksikdar/CHIAR-Former" target="_blank"
         style="color:#85C1E9;">🤗 Model weights</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#4A6A8A;text-align:center;">
      Single NVIDIA RTX A5000 · 24GB VRAM<br>
      Prateek Kumar Sikdar · Accenture · Bengaluru · 2026
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN INPUT
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("### ✍️ Enter any text to see live routing")
st.markdown(
    '<span style="font-size:0.9rem;color:#7A9AB5;">'
    'Type a sentence — the model routes each token to DCT or Attention '
    'based on spectral entropy. '
    '<b style="color:#2E86C1;">Blue = DCT (cheap)</b>. '
    '<b style="color:#C0392B;">Red = Attention (expensive)</b>. '
    'Hover a token to see its entropy H(x).'
    '</span>',
    unsafe_allow_html=True,
)

default_texts = [
    "Despite the overwhelming scientific consensus on climate change, many governments have struggled to implement effective carbon reduction policies.",
    "The transformer architecture achieves parallelism by replacing recurrent connections with self-attention, allowing each token to attend to all others.",
    "The old lighthouse keeper had maintained the lamp for forty years through storms that shook the foundations and winters that froze the bay solid.",
    "If MAX(3, MIN(7, 4), 2) equals X and MIN(X, 5) equals Y then Y is the answer.",
]
EXAMPLE_LABELS = [
    "Example 1 — Climate policy (mixed routing)",
    "Example 2 — Transformer architecture (technical)",
    "Example 3 — Lighthouse narrative (naturalistic)",
    "Example 4 — Symbolic MAX/MIN expression",
]

col_input, col_example = st.columns([3, 1])
with col_input:
    user_text = st.text_area(
        label="Input text",
        value=default_texts[0],
        height=110,
        label_visibility="collapsed",
        placeholder="Type or paste any English text here...",
    )
with col_example:
    st.markdown("<br>", unsafe_allow_html=True)
    example = st.selectbox(
        "Try an example",
        ["— pick an example —"] + EXAMPLE_LABELS,
        label_visibility="collapsed",
    )
    if example != "— pick an example —":
        user_text = default_texts[EXAMPLE_LABELS.index(example)]

run_btn = st.button("🎨 Route Tokens", type="primary", use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# RUN
# ══════════════════════════════════════════════════════════════════════════════

if run_btn and user_text.strip():

    st.markdown("---")

    # ── Load model once per session ───────────────────────────────────────────
    if st.session_state["model_status"] == "pending":
        with st.spinner("Loading CHIAR-Former (downloading checkpoint on first run)..."):
            attempt_load()
        st.rerun()  # rerun so sidebar status updates before we proceed

    # ── Hard stop if load failed ──────────────────────────────────────────────
    if st.session_state["model_status"] == "failed":
        show_load_error()   # renders error block then calls st.stop()

    # ── At this point model is confirmed loaded ───────────────────────────────
    st.success("✅ Running real model routing — spectral entropy H(x) from trained CHIAR-Former.")

    st.markdown(
        '<div class="result-header">⏳ Routing in progress...</div>',
        unsafe_allow_html=True,
    )
    loading_ph  = st.empty()
    result_holder = {}

    def run_inference():
        try:
            result_holder["out"] = get_routing(
                st.session_state["model"],
                st.session_state["tokenizer"],
                user_text,
            )
        except Exception:
            result_holder["error"] = traceback.format_exc()

    t = threading.Thread(target=run_inference)
    t.start()
    typewriter_loading(loading_ph)
    t.join()
    loading_ph.empty()

    # ── Hard stop if inference failed ─────────────────────────────────────────
    if "error" in result_holder:
        st.markdown(f"""
        <div class="error-box">
          <h3>🚫 Inference failed — no routing results available</h3>
          <div class="error-step">Error during model forward pass:</div>
          <div class="error-trace">{result_holder["error"]}</div>
          <br>
          <span style="color:#A04040;font-size:0.85rem;">
            The model loaded successfully but threw an error during inference.
            Check that the input text is valid and that CHIARFormer.forward()
            supports the <code>return_routing_info=True</code> flag.
          </span>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Show results ──────────────────────────────────────────────────────────
    tokens, ops, entropy, gate = result_holder["out"]

    st.markdown(
        '<div class="result-header">🎨 Chiaroscuro Routing Heatmap</div>',
        unsafe_allow_html=True,
    )
    render_heatmap(tokens, ops, entropy, gate)

elif run_btn:
    st.warning("Please enter some text first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.82rem;color:#4A6A8A;padding:8px;">
  <b style="color:#5A7A9A;">CHIAR-Former</b> · Chiaroscuro Attention: Spending Compute in the Dark<br>
  Prateek Kumar Sikdar · AI Architect · Accenture · Bengaluru · 2026<br>
  <a href="https://arxiv.org/abs/2606.08327" target="_blank" style="color:#4A7A9A;">arXiv:2606.08327</a>
  &nbsp;|&nbsp;
  <a href="https://github.com/hackie123/CHIAR-former" target="_blank" style="color:#4A7A9A;">GitHub</a>
  &nbsp;|&nbsp;
  <a href="https://huggingface.co/prateeksikdar/CHIAR-Former" target="_blank" style="color:#4A7A9A;">HuggingFace</a>
</div>
""", unsafe_allow_html=True)
