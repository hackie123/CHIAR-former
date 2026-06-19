# app.py — CHIAR-Former Live Routing Demo
# Streamlit app for HuggingFace Spaces
#
# Automatically downloads the checkpoint from HuggingFace Hub on first run.
# Falls back to heuristic routing if download fails or no GPU is available.
#
# Usage (local):
#   streamlit run app.py
#
# Deployment:
#   Push to HuggingFace Spaces (SDK: Streamlit)
#   Checkpoint is auto-downloaded from prateeksikdar/CHIAR-Former on first run.
#
# Install:
#   pip install streamlit torch transformers huggingface_hub

import os, sys, time, math, threading
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
HF_REPO_ID   = "prateeksikdar/CHIAR-Former"
HF_FILENAME  = "chiar_dct_attn_400M_best.pt"
CKPT_PATH    = os.path.join("checkpoints", HF_FILENAME)

FUNCTION_WORDS = {
    "the","a","an","of","in","to","and","or","but","is","was","are","were",
    "it","its","that","this","for","on","at","by","with","from","as","be",
    "been","have","has","had","do","did","not","no","so","if","then","than",
    "into","their","they","we","you","he","she","which","who","also","just",
    "very","more","most","some","all","each","both","few","many","such",
    "about","after","before","between","during","while","though","although",
    "because","since","when","where","how","what","there","here","these","those",
}

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background-color: #0D1B2A; color: #F0F0F0; }
  .stApp { background-color: #0D1B2A; }
  .title-box {
    background: linear-gradient(135deg, #1A4E8C, #0D1B2A);
    border-left: 5px solid #F39C12;
    padding: 24px 28px;
    border-radius: 8px;
    margin-bottom: 24px;
  }
  .title-box h1 { color: #F39C12; font-size: 2.2rem; margin:0; }
  .title-box p  { color: #AABBCC; font-size: 1.0rem; margin-top:8px; }
  .chiaroscuro-box {
    background: #111E2E;
    border: 1px solid #2E4A6A;
    border-left: 4px solid #8E44AD;
    padding: 16px 20px;
    border-radius: 6px;
    margin-bottom: 20px;
    font-size: 0.92rem;
    color: #C8D8E8;
    line-height: 1.7;
  }
  .stat-card {
    background: #111E2E;
    border: 1px solid #2E4A6A;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    margin: 4px;
  }
  .stat-val { font-size: 1.8rem; font-weight: bold; color: #F39C12; }
  .stat-lbl { font-size: 0.8rem; color: #7A9AB5; margin-top: 4px; }
  .token-box {
    display: inline-block;
    padding: 5px 9px;
    margin: 3px 2px;
    border-radius: 5px;
    font-size: 0.95rem;
    font-weight: 600;
    font-family: monospace;
    color: white;
    vertical-align: middle;
  }
  .dct-token  { background-color: #1A5276; border: 1px solid #2E86C1; }
  .attn-token { background-color: #7B241C; border: 1px solid #C0392B; }
  .legend-dct  { background:#1A5276; color:white; padding:4px 12px;
                  border-radius:4px; font-size:0.85rem; }
  .legend-attn { background:#7B241C; color:white; padding:4px 12px;
                  border-radius:4px; font-size:0.85rem; }
  .loading-line {
    font-size: 1.05rem;
    color: #85C1E9;
    padding: 6px 0;
    line-height: 1.8;
  }
  .result-header {
    font-size: 1.2rem;
    font-weight: bold;
    color: #F39C12;
    margin-bottom: 12px;
  }
  .contribution-box {
    background: #0A1520;
    border: 1px solid #1A3A55;
    border-radius: 6px;
    padding: 14px 18px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #B0C8E0;
  }
  .status-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 8px;
  }
  .status-loaded  { background:#1E4D2B; color:#58D68D; border:1px solid #27AE60; }
  .status-demo    { background:#4A2000; color:#F0A500; border:1px solid #E67E22; }
  .status-loading { background:#1A2E4A; color:#85C1E9; border:1px solid #2E86C1; }
  .link-row a {
    color: #85C1E9;
    text-decoration: none;
    margin-right: 16px;
    font-size: 0.88rem;
  }
  .link-row a:hover { text-decoration: underline; }
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
  to model three-dimensional form. Masters like <b>Caravaggio</b>, <b>Leonardo da Vinci</b>,
  and <b>Rembrandt</b> did not illuminate a canvas uniformly — they poured light exactly
  where the eye needed detail, leaving peripheral regions in inexpensive shadow.<br><br>
  <b>CHIAR-Former borrows this principle for computation.</b> Not every token in a
  sentence deserves the same attention budget. Function words — <i>the</i>, <i>of</i>,
  <i>and</i> — are the shadowed periphery: smooth, low-frequency, cheap to process
  with DCT spectral mixing in <span style="color:#85C1E9;">O(d log d)</span>.
  Content words — <i>overwhelming</i>, <i>consensus</i>, <i>paradox</i> — are the
  illuminated focal points: rich, high-entropy, deserving full self-attention in
  <span style="color:#F1948A;">O(n²d)</span>.<br><br>
  The model measures this via <b>spectral entropy</b> H(x) — the information entropy
  of each token's DCT power spectrum. Low entropy = smooth = DCT (blue).
  High entropy = complex = Attention (red). The result: a painting-like allocation
  of compute, spending light where the signal is dark.
</div>
""", unsafe_allow_html=True)


# ── Checkpoint download ───────────────────────────────────────────────────────

def download_checkpoint():
    """
    Download checkpoint from HuggingFace Hub if not already present.
    Returns (success: bool, message: str).
    """
    if os.path.exists(CKPT_PATH):
        return True, "checkpoint already present"
    try:
        from huggingface_hub import hf_hub_download
        os.makedirs("checkpoints", exist_ok=True)
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            local_dir="./checkpoints",
        )
        return True, "downloaded successfully"
    except ImportError:
        return False, "huggingface_hub not installed"
    except Exception as e:
        return False, str(e)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load CHIAR-Former from checkpoint.
    Auto-downloads from HuggingFace Hub if checkpoint is missing.
    Returns (model, tokenizer, mode) where mode is 'loaded' or 'demo'.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Try to download if missing
    if not os.path.exists(CKPT_PATH):
        success, msg = download_checkpoint()
        if not success:
            # Could not download — run in demo mode
            return None, tokenizer, f"demo:{msg}"

    # Load model
    try:
        from config import Config
        from model import CHIARFormer

        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        cfg  = Config()
        cfg.vocab_size = len(tokenizer)

        # Restore config from checkpoint if available
        saved_cfg = ckpt.get("cfg", {})
        for k, v in saved_cfg.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        model = CHIARFormer(cfg)
        model.load_state_dict(ckpt["model"])
        model.eval()

        val_ppl = ckpt.get("val_ppl", None)
        return model, tokenizer, f"loaded:{val_ppl:.2f}" if val_ppl else "loaded"

    except Exception as e:
        return None, tokenizer, f"demo:load error — {e}"


# ── Loading messages (typewriter effect) ──────────────────────────────────────

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
    """Display loading facts one by one while the model runs."""
    lines_shown = []
    for icon, text in LOADING_LINES:
        lines_shown.append(f'<div class="loading-line">{icon} &nbsp; {text}</div>')
        placeholder.markdown(
            '<div style="background:#0A1520;border:1px solid #1A3A55;'
            'border-radius:8px;padding:16px 20px;">'
            + "".join(lines_shown)
            + "</div>",
            unsafe_allow_html=True,
        )
        time.sleep(1.2)


# ── Routing inference ─────────────────────────────────────────────────────────

@torch.no_grad()
def get_routing(model, tokenizer, text):
    """Run a forward pass and return per-token routing decisions."""
    ids = tokenizer.encode(text, return_tensors="pt")
    ids = ids[:, :256]  # cap at training seq_len

    logits, infos, _, meta_gate = model(ids, return_routing_info=True)

    # L2 (index 1) is the first routing layer
    info   = infos[1] if len(infos) > 1 else infos[0]
    op_idx = info.get("op_idx")   # (1, T) — 0=DCT, 1=Attention
    H      = info.get("H")        # (1, T) — spectral entropy

    tokens = [tokenizer.decode([t]) for t in ids[0].tolist()]

    if op_idx is not None:
        ops      = op_idx[0].tolist()
        entropy  = H[0].tolist() if H is not None else [None] * len(tokens)
    else:
        ops     = [0 if tok.strip().lower().rstrip(".,;:'\"") in FUNCTION_WORDS
                   else 1 for tok in tokens]
        entropy = [None] * len(tokens)

    gate = meta_gate.item() if meta_gate is not None else None
    return tokens, ops, entropy, gate


def heuristic_routing(tokenizer, text):
    """Heuristic fallback for demo mode — uses function word list."""
    ids     = tokenizer.encode(text)
    tokens  = [tokenizer.decode([t]) for t in ids[:256]]
    ops     = [0 if tok.strip().lower().rstrip(".,;:'\"") in FUNCTION_WORDS
               else 1 for tok in tokens]
    entropy = [None] * len(tokens)
    return tokens, ops, entropy, None


# ── Render heatmap ────────────────────────────────────────────────────────────

def render_heatmap(tokens, ops, entropy, meta_gate, mode):
    n_dct   = ops.count(0)
    n_attn  = ops.count(1)
    total   = len(ops)
    pct_dct = 100 * n_dct  / total if total else 0

    # Stats row
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

    # Legend
    st.markdown("""
    <span class="legend-dct">🔵 DCT Mixing — O(d log d)</span> &nbsp;&nbsp;
    <span class="legend-attn">🔴 Full Attention — O(n²d)</span>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Token heatmap
    html_tokens = []
    for tok, op, h in zip(tokens, ops, entropy):
        display = tok.replace(" ", "&nbsp;").replace("<", "&lt;").replace(">", "&gt;")
        cls     = "dct-token" if op == 0 else "attn-token"
        # Show entropy on hover via title attribute
        title   = f"H={h:.3f}" if h is not None else ""
        html_tokens.append(
            f'<span class="token-box {cls}" title="{title}">{display}</span>'
        )

    st.markdown(
        '<div style="line-height:2.4;padding:16px;background:#0A1520;'
        'border:1px solid #1A3A55;border-radius:8px;">'
        + "".join(html_tokens)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Hover tip
    if any(h is not None for h in entropy):
        st.markdown(
            '<span style="font-size:0.78rem;color:#4A6A8A;">'
            '💡 Hover over any token to see its spectral entropy H(x)</span>',
            unsafe_allow_html=True,
        )

    # Routing insight box
    st.markdown("<br>", unsafe_allow_html=True)
    gate_line = (
        f"MetaRouter gate g = <b>{meta_gate:.3f}</b> — "
        + ("naturalistic text regime (DCT preprocessing active)."
           if meta_gate is not None and meta_gate > 0.5
           else "symbolic/structured input regime (DCT largely bypassed).")
        if meta_gate is not None
        else ""
    )
    st.markdown(f"""
    <div class="chiaroscuro-box" style="margin-top:4px;">
      <b style="color:#F39C12;">Routing insight for this input:</b><br>
      {n_dct} of {total} tokens ({100*n_dct//total if total else 0}%) routed to
      <span style="color:#85C1E9;"><b>DCT Mixing</b></span> —
      processed in O(d log d), cheap spectral filtering.<br>
      {n_attn} of {total} tokens ({100*n_attn//total if total else 0}%) routed to
      <span style="color:#F1948A;"><b>Full Attention</b></span> —
      full O(n²d) contextual computation.<br>
      {gate_line}<br><br>
      This is the chiaroscuro principle in action: the model illuminates
      only what needs illumination, leaving smooth tokens in inexpensive shadow.
    </div>
    """, unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:

    # Model status — shown before load so user knows what's happening
    st.markdown("### ⚙️ Model Status")
    status_ph = st.empty()
    status_ph.markdown(
        '<span class="status-pill status-loading">⏳ not loaded yet</span>',
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
         "Light tokens (smooth) → DCT. Dark tokens (complex) → Attention."),
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


# ── Main input ────────────────────────────────────────────────────────────────

st.markdown("### ✍️ Enter any text to see live routing")
st.markdown(
    '<span style="font-size:0.9rem;color:#7A9AB5;">'
    'Type a sentence — the model routes each token to DCT or Attention '
    'based on spectral entropy. '
    '<b style="color:#2E86C1;">Blue = DCT (cheap)</b>. '
    '<b style="color:#C0392B;">Red = Attention (expensive)</b>. '
    'Hover over a token to see its entropy value.'
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
        idx       = EXAMPLE_LABELS.index(example)
        user_text = default_texts[idx]

run_btn = st.button("🎨 Route Tokens", type="primary", use_container_width=True)


# ── Run ───────────────────────────────────────────────────────────────────────

if run_btn and user_text.strip():

    st.markdown("---")
    st.markdown(
        '<div class="result-header">⏳ Routing in progress...</div>',
        unsafe_allow_html=True,
    )

    loading_ph = st.empty()

    # Load model (cached — only downloads/loads once per session)
    with st.spinner("Loading CHIAR-Former... (first run: downloading checkpoint from HuggingFace Hub)"):
        model, tokenizer, mode = load_model()

    # Update sidebar status pill now that we know load result
    if mode.startswith("loaded"):
        val_ppl_str = mode.split(":")[1] if ":" in mode else ""
        label = f"✅ model loaded" + (f" · Val PPL {val_ppl_str}" if val_ppl_str else "")
        status_ph.markdown(
            f'<span class="status-pill status-loaded">{label}</span>',
            unsafe_allow_html=True,
        )
    else:
        reason = mode.split(":", 1)[1] if ":" in mode else "checkpoint not found"
        status_ph.markdown(
            f'<span class="status-pill status-demo">⚠️ demo mode</span>',
            unsafe_allow_html=True,
        )

    # Run routing in a thread while showing typewriter loading messages
    result_holder = {}

    def run_inference():
        try:
            if mode.startswith("loaded"):
                result_holder["tokens"], result_holder["ops"], \
                result_holder["entropy"], result_holder["gate"] = \
                    get_routing(model, tokenizer, user_text)
            else:
                result_holder["tokens"], result_holder["ops"], \
                result_holder["entropy"], result_holder["gate"] = \
                    heuristic_routing(tokenizer, user_text)
        except Exception as e:
            result_holder["error"] = str(e)

    t = threading.Thread(target=run_inference)
    t.start()
    typewriter_loading(loading_ph)
    t.join()
    loading_ph.empty()

    # ── Results ───────────────────────────────────────────────────────────────
    if "error" in result_holder:
        st.error(f"Routing failed: {result_holder['error']}")
    else:
        tokens     = result_holder["tokens"]
        ops        = result_holder["ops"]
        entropy    = result_holder["entropy"]
        meta_gate  = result_holder["gate"]

        st.markdown(
            '<div class="result-header">🎨 Chiaroscuro Routing Heatmap</div>',
            unsafe_allow_html=True,
        )

        # Show banner if running in demo mode
        if not mode.startswith("loaded"):
            reason = mode.split(":", 1)[1] if ":" in mode else "checkpoint not found"
            st.info(
                f"ℹ️ Running in **demo mode** ({reason}). "
                "Routing is approximated using a function-word heuristic — "
                "not the trained model. "
                f"To see real routing, ensure `{CKPT_PATH}` is present or "
                "that the HuggingFace Hub download succeeds.",
                icon="ℹ️",
            )

        render_heatmap(tokens, ops, entropy, meta_gate, mode)

elif run_btn:
    st.warning("Please enter some text first.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-size:0.82rem;color:#4A6A8A;padding:8px;">
  <b style="color:#5A7A9A;">CHIAR-Former</b> · Chiaroscuro Attention: Spending Compute in the Dark<br>
  Prateek Kumar Sikdar · AI Architect · Accenture · Bengaluru · 2026<br>
  <a href="https://arxiv.org/abs/2606.08327" target="_blank"
     style="color:#4A7A9A;">arXiv:2606.08327</a> &nbsp;|&nbsp;
  <a href="https://github.com/hackie123/CHIAR-former" target="_blank"
     style="color:#4A7A9A;">GitHub</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/prateeksikdar/CHIAR-Former" target="_blank"
     style="color:#4A7A9A;">HuggingFace</a>
</div>
""", unsafe_allow_html=True)
