"""
Saffron Authenticity Detector — Streamlit frontend.

Run from the project root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

# ── project root on path ─────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT / "scripts"), str(_ROOT / "app")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Page config (MUST be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Saffron Authenticity Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — dark saffron theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #0d0d18;
    color: #e8e8f0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #13131f;
    border-right: 1px solid #2a2a3d;
}
[data-testid="stHeader"] { background: transparent; }

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f5a623 0%, #e85d04 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.15;
}
.hero p {
    color: #8888aa;
    font-size: 1.05rem;
    margin-top: .6rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploadDropzone"] {
    background: #16162a !important;
    border: 2px dashed #f5a62355 !important;
    border-radius: 16px !important;
    transition: border-color .25s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #f5a623bb !important;
}

/* ── Verdict cards ── */
.verdict-card {
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    animation: popIn .45s cubic-bezier(.34,1.56,.64,1) both;
    margin-top: 1rem;
}
.verdict-real {
    background: linear-gradient(135deg, #0d2d1a 0%, #0d3d22 100%);
    border: 2px solid #22c55e55;
    box-shadow: 0 0 40px #22c55e18;
}
.verdict-fake {
    background: linear-gradient(135deg, #2d0d0d 0%, #3d1010 100%);
    border: 2px solid #ef444455;
    box-shadow: 0 0 40px #ef444418;
}
.verdict-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem;
    font-weight: 700;
    letter-spacing: .05em;
    margin: 0;
}
.verdict-real  .verdict-label { color: #22c55e; }
.verdict-fake  .verdict-label { color: #ef4444; }
.verdict-emoji { font-size: 2.8rem; margin-bottom: .4rem; }
.verdict-conf  { color: #aaaacc; font-size: 1rem; margin-top: .5rem; }
.verdict-conf  strong { color: #e8e8f0; font-size: 1.15rem; }

/* ── Confidence bar ── */
.conf-bar-wrap {
    background: #1e1e2e;
    border-radius: 99px;
    height: 14px;
    overflow: hidden;
    margin: 1rem 0 .4rem;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s ease;
}
.bar-real { background: linear-gradient(90deg,#16a34a,#22c55e); }
.bar-fake { background: linear-gradient(90deg,#b91c1c,#ef4444); }

/* ── Info pill ── */
.method-pill {
    display: inline-block;
    background: #f5a62322;
    color: #f5a623;
    border: 1px solid #f5a62344;
    border-radius: 99px;
    padding: .25rem .85rem;
    font-size: .78rem;
    font-weight: 500;
    letter-spacing: .04em;
    margin-top: .6rem;
}

/* ── Section headers ── */
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #f5a623;
    text-transform: uppercase;
    letter-spacing: .08em;
    margin: 1.8rem 0 .6rem;
}

/* ── Colour chip row ── */
.chip-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: .5rem; }
.chip {
    border-radius: 10px;
    padding: .5rem .75rem;
    font-size: .82rem;
    font-family: 'Inter', sans-serif;
    color: #e8e8f0;
    border: 1px solid #ffffff18;
    min-width: 80px;
    text-align: center;
}

/* ── Sidebar ── */
.sidebar-label {
    font-size: .8rem;
    color: #6666aa;
    text-transform: uppercase;
    letter-spacing: .06em;
    margin-bottom: .3rem;
}
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.dot-green { background: #22c55e; box-shadow: 0 0 6px #22c55e88; }
.dot-orange { background: #f5a623; box-shadow: 0 0 6px #f5a62388; }

/* ── Animations ── */
@keyframes popIn {
    0%   { opacity:0; transform: scale(.88) translateY(10px); }
    100% { opacity:1; transform: scale(1)   translateY(0); }
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
.spinner {
    display: inline-block;
    width: 22px; height: 22px;
    border: 3px solid #f5a62333;
    border-top-color: #f5a623;
    border-radius: 50%;
    animation: spin .7s linear infinite;
    vertical-align: middle;
    margin-right: 8px;
}

/* ── Divider ── */
hr { border-color: #2a2a3d; }

/* ── Streamlit overrides ── */
.stButton>button {
    background: linear-gradient(135deg, #f5a623, #e85d04);
    color: #0d0d18;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    padding: .5rem 1.5rem;
    font-family: 'Space Grotesk', sans-serif;
}
.stButton>button:hover { opacity: .9; }
[data-testid="stExpander"] {
    background: #13131f;
    border: 1px solid #2a2a3d;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
RUN_DIR      = str(_ROOT / "runs" / "cv1")
MANIFEST     = str(_ROOT / "data" / "manifest.csv")


@st.cache_resource(show_spinner=False)
def _load_bundle():
    from predict import load_model
    return load_model(RUN_DIR, MANIFEST)


def _colour_chips_html(features: dict[str, float]) -> str:
    """Render the exact 12 colour statistics used by the baseline classifier."""
    chips = []
    colours = [
        "#c0392b", "#27ae60", "#2980b9", "#c0392b", "#27ae60", "#2980b9",
        "#8e44ad", "#f39c12", "#ecf0f1", "#8e44ad", "#f39c12", "#ecf0f1",
    ]
    for (label, value), colour in zip(features.items(), colours):
        chips.append(
            f'<div class="chip" style="background:{colour}22;border-color:{colour}44;">'
            f'<div style="font-size:.65rem;color:#aaaacc">{label}</div>'
            f'<div style="font-weight:600">{value:.3f}</div></div>'
        )
    return '<div class="chip-row">' + "".join(chips) + "</div>"


def _conf_bar(confidence: float, label: str) -> str:
    pct = int(confidence * 100)
    cls = "bar-real" if label == "REAL" else "bar-fake"
    return (
        f'<div class="conf-bar-wrap">'
        f'<div class="conf-bar-fill {cls}" style="width:{pct}%"></div>'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Saffron Detector")
    st.markdown("---")

    # Model status
    st.markdown('<div class="sidebar-label">Model Status</div>',
                unsafe_allow_html=True)
    with st.spinner("Loading model…"):
        bundle = _load_bundle()

    if bundle.method == "deep":
        n = len(bundle.folds)
        decoder_status = (
            "Wombat-OA tuned decoding"
            if bundle.decoder_tuned
            else "Default decoder · threshold 0.5"
        )
        st.markdown(
            f'<span class="status-dot dot-green"></span>'
            f'<b style="color:#22c55e">Deep Model Active</b><br>'
            f'<span style="color:#6666aa;font-size:.82rem">'
            f'ConvNeXt-Tiny + GNN + HAFM<br>'
            f'{decoder_status}<br>'
            f'{n} fold{"s" if n>1 else ""} ensembled</span>',
            unsafe_allow_html=True,
        )
    else:
        baseline_label = (
            "Random Forest classifier"
            if bundle.colour_clf is not None
            else "Colour heuristic · manifest unavailable"
        )
        st.markdown(
            '<span class="status-dot dot-orange"></span>'
            '<b style="color:#f5a623">Colour Baseline</b><br>'
            '<span style="color:#6666aa;font-size:.82rem">'
            f'RGB + HSV statistics<br>{baseline_label}<br>'
            'Train deep model for best results</span>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sidebar-label" style="margin-top:1rem">Model settings</div>',
                unsafe_allow_html=True)
    if bundle.method == "deep":
        st.caption(
            f"Input: {bundle.img_size} × {bundle.img_size} px · "
            f"MC-Dropout samples: {bundle.mc_samples}"
        )
    else:
        st.caption("Baseline features: 64 × 64 centre crop · MC-Dropout: not used")

    st.markdown("---")
    st.markdown('<div class="sidebar-label">About</div>', unsafe_allow_html=True)
    decoder_description = (
        "Wombat-OA tuned beam decoding"
        if bundle.decoder_tuned
        else "Wombat beam decoding with default parameters"
    )
    st.markdown(
        '<span style="color:#6666aa;font-size:.82rem">'
        'Detects adulterated saffron from a photo using a deep CNN+GNN+HAFM '
        f'model and {decoder_description}.<br><br>'
        'Works best on close-up photos of dry saffron threads under natural light.'
        '</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div style="font-size:3rem;margin-bottom:.3rem">🌿</div>
  <h1>Saffron Authenticity Detector</h1>
  <p>Upload a saffron photo — get an instant real or fake verdict</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your saffron image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.markdown("""
    <div style="text-align:center;padding:2rem 0;color:#44445a;font-size:.9rem">
        📷 &nbsp; Accepted formats: JPG · PNG · WebP<br>
        <span style="font-size:.78rem">Works best with close-up photos of dry saffron threads</span>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Image preview + analysis
# ─────────────────────────────────────────────────────────────────────────────
if bundle.method == "deep" and bundle.fallback_reason:
    st.info(f"Deep model weights loaded. {bundle.fallback_reason}")

try:
    pil_img = Image.open(uploaded).convert("RGB")
except (UnidentifiedImageError, OSError) as exc:
    st.error(f"Could not read the uploaded image: {exc}")
    st.stop()

if bundle.method == "colour":
    baseline_message = "Using the colour-statistics baseline."
    if bundle.fallback_reason:
        baseline_message += f" {bundle.fallback_reason}"
    st.warning(baseline_message)

col_img, col_res = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<div class="section-title">📸 Uploaded Image</div>',
                unsafe_allow_html=True)
    st.image(pil_img, use_container_width=True,
             caption=f"{uploaded.name}  ·  {pil_img.size[0]}×{pil_img.size[1]} px")

with col_res:
    st.markdown('<div class="section-title">🔬 Analysis</div>',
                unsafe_allow_html=True)

    # Run inference
    with st.spinner("Analysing…"):
        t0 = time.perf_counter()
        from predict import predict_image
        result = predict_image(pil_img, model_bundle=bundle)
        elapsed = time.perf_counter() - t0

    label      = result["label"]
    confidence = result["confidence"]
    p_fake     = result["p_fake"]
    method     = result["method"]

    # Verdict card
    card_cls = "verdict-real" if label == "REAL" else "verdict-fake"
    emoji     = "✅" if label == "REAL" else "🚫"
    colour    = "#22c55e" if label == "REAL" else "#ef4444"

    st.markdown(f"""
    <div class="verdict-card {card_cls}">
      <div class="verdict-emoji">{emoji}</div>
      <p class="verdict-label">{label}</p>
      <div class="verdict-conf">
        Confidence: <strong>{confidence*100:.1f}%</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bar
    st.markdown(_conf_bar(confidence, label), unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;justify-content:space-between;'
        f'color:#6666aa;font-size:.78rem">'
        f'<span>0%</span><span style="color:{colour};font-weight:600">'
        f'{confidence*100:.1f}%</span><span>100%</span></div>',
        unsafe_allow_html=True,
    )

    # Meta row
    method_lbl = "Deep Model (CNN+GNN+HAFM)" if method == "deep" else "Colour Baseline"
    folds_used = result.get("folds_used", 0)
    meta_txt = method_lbl
    if method == "deep" and folds_used > 1:
        meta_txt += f" · {folds_used}-fold ensemble"
    st.markdown(f'<div class="method-pill">⚙ {meta_txt}</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div style="color:#44445a;font-size:.75rem;margin-top:.4rem">'
        f'Processed in {elapsed*1000:.0f} ms</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Probability detail
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
with col_a:
    st.metric("P(Fake)", f"{p_fake*100:.1f}%",
              delta=f"{(p_fake-0.5)*100:+.1f}% from 50%",
              delta_color="inverse")
with col_b:
    thr = result.get("threshold", 0.5)
    st.metric("Decision Threshold", f"{thr*100:.1f}%")
with col_c:
    st.metric("Verdict", label,
              delta="↑ Confidence" if confidence > 0.7 else "⚠ Low confidence",
              delta_color="normal" if label == "REAL" else "inverse")

# ─────────────────────────────────────────────────────────────────────────────
# Colour feature breakdown
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">🎨 Colour Feature Breakdown</div>',
            unsafe_allow_html=True)
st.markdown(
    '<span style="color:#6666aa;font-size:.82rem">'
    'RGB + HSV means and standard deviations on a 64×64 centre crop of the uploaded image</span>',
    unsafe_allow_html=True,
)
st.markdown(
    _colour_chips_html(result["details"]["colour_features"]),
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Per-fold breakdown (deep model only)
# ─────────────────────────────────────────────────────────────────────────────
per_fold = result.get("per_fold_probs", [])
if per_fold and len(per_fold) > 1:
    with st.expander("📊 Per-fold probabilities (ensemble detail)"):
        import pandas as pd
        df = pd.DataFrame({
            "Fold": [f"Fold {i}" for i in range(len(per_fold))],
            "P(Fake)": [f"{p*100:.1f}%" for p in per_fold],
            "Verdict": ["FAKE" if p >= 0.5 else "REAL" for p in per_fold],
        })
        st.dataframe(df, hide_index=True, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# How it works
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("ℹ️ How it works"):
    if bundle.method == "deep" and bundle.decoder_tuned:
        decoder_step = (
            "**Wombat-OA tuned beam decoding** — diversity-penalised "
            "weighted beam search and log-probability fusion using parameters "
            "tuned on the held-out validation fold."
        )
    elif bundle.method == "deep":
        decoder_step = (
            "**Default beam decoding** — Wombat beam search and "
            "log-probability fusion with untuned defaults and a 0.5 decision "
            "threshold, because this checkpoint has no saved decoder metadata."
        )
    else:
        decoder_step = (
            "If deep weights are available, Wombat-OA tunes beam decoding "
            "parameters on a held-out validation fold."
        )

    st.markdown(f"""
**Model architecture** (when deep weights are available):
1. **ConvNeXt-Tiny backbone** extracts a spatial feature map from your photo
2. **Grid Graph-Attention Network (GNN)** applies multi-head attention over an 8-neighbourhood patch graph
3. **Hybrid Attention Fusion Module (HAFM)** — CNN global vector attends over GNN node embeddings
4. **Three heads** (CNN / GNN / HAFM) each predict a logit independently
5. **8 colour-preserving test-time views** (identity, flips, rotations, crops) are extracted
6. {decoder_step}

**Fallback** (when no weights exist):
RGB mean/std + HSV mean/std on a 64×64 crop → Random Forest trained on the full manifest. Achieves ~0.94 AUC on the saffron dataset.

**What "fake" means in this dataset:**
Adulterated saffron samples labelled with types `-`, `C`, `S`, `M` (adulterant types from filenames).
""")

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#33334a;font-size:.78rem;padding:.5rem 0">'
    'Saffron Authenticity Detector · ConvNeXt + GNN + HAFM + Wombat-OA · '
    'Sample-level group-CV · '
    '<span style="color:#f5a62366">🌿</span>'
    '</div>',
    unsafe_allow_html=True,
)
