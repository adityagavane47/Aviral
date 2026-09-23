"""
app.py — Project Aviral | INSAT Deep VFI Dashboard
====================================================
Streamlit dashboard featuring a "Dual-Player Time-Lapse" UI:

  LEFT  COLUMN:  Original sequence  — 2-frame jump  (t=0 → t=30)
  RIGHT COLUMN:  AI-enhanced sequence — smooth 3-frame (t=0 → t=15 → t=30)

Prominent SSIM, MSE, PSNR metrics displayed below.

HOW TO RUN:
-----------
    pip install streamlit torch xarray netCDF4 opencv-python scikit-image numpy
    streamlit run app.py

    The app auto-generates dummy satellite data if real .nc files are absent.
    To use real INSAT data, follow the instructions in data_processor.py.
"""

import time
import numpy as np
import streamlit as st
import torch
import cv2

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Aviral | INSAT VFI Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  CUSTOM CSS — Premium Dark UI
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #0f1f4b 0%, #1a1035 40%, #0d2d3a 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(56,189,248,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .hero-subtitle {
    color: #94a3b8;
    font-size: 1.0rem;
    margin-top: 6px;
    font-weight: 400;
  }
  .badge {
    display: inline-block;
    background: rgba(56,189,248,0.15);
    border: 1px solid rgba(56,189,248,0.4);
    color: #38bdf8;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-right: 8px;
    margin-top: 12px;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: linear-gradient(145deg, #131929, #0f1f3d);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s ease, border-color 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    border-color: rgba(56,189,248,0.5);
  }
  .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #64748b;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .metric-value {
    font-size: 2.0rem;
    font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(90deg, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .metric-desc {
    font-size: 0.72rem;
    color: #475569;
    margin-top: 6px;
  }

  /* ── Player labels ── */
  .player-label {
    text-align: center;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 8px 0 4px;
  }
  .player-original { color: #f59e0b; }
  .player-ai       { color: #34d399; }

  .frame-badge {
    display: inline-block;
    border-radius: 6px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 3px 2px;
  }
  .fb-t0  { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }
  .fb-t15 { background: rgba(52,211,153,0.15);  color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
  .fb-t30 { background: rgba(245,158,11,0.15);  color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }

  /* ── Section headers ── */
  .section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #cbd5e1;
    border-left: 4px solid #38bdf8;
    padding-left: 12px;
    margin: 20px 0 14px;
  }

  /* ── Timeline bar ── */
  .timeline {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    margin: 8px 0;
  }
  .tl-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    border: 2px solid;
    flex-shrink: 0;
  }
  .tl-line {
    height: 2px;
    flex: 1;
  }
  .tl-label {
    font-size: 0.65rem;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    color: #64748b;
    text-align: center;
    margin-top: 4px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #0d1424 !important;
    border-right: 1px solid rgba(99,179,237,0.15);
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1.0rem !important;
    padding: 14px 32px !important;
    transition: opacity 0.2s ease !important;
    letter-spacing: 0.04em !important;
    width: 100% !important;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
  }

  /* ── Divider ── */
  hr { border-color: rgba(99,179,237,0.12) !important; }

  /* ── Progress bar color ── */
  .stProgress > div > div { background-color: #38bdf8 !important; }

  /* ── Image captions ── */
  .frame-caption {
    text-align: center;
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  /* ── Status pill ── */
  .status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52,211,153,0.12);
    border: 1px solid rgba(52,211,153,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #34d399;
    font-weight: 600;
  }
  .dot-pulse {
    width: 7px; height: 7px;
    background: #34d399;
    border-radius: 50%;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.7); }
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HELPER UTILITIES
# ─────────────────────────────────────────────────────────────
def tensor_to_rgb(tensor: torch.Tensor, colormap: bool = True) -> np.ndarray:
    """
    Convert (1, 3, H, W) tensor [0,1] → uint8 RGB numpy array (H, W, 3).
    Optionally applies a false-color thermal colormap (COLORMAP_INFERNO).
    """
    arr = tensor.squeeze(0).detach().cpu().numpy()       # (3, H, W)
    arr = np.transpose(arr, (1, 2, 0))                   # (H, W, 3)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)

    if colormap:
        # Use only the first channel (all channels are identical from grayscale)
        gray = arr[..., 0]
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)  # BGR
        arr = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)            # RGB

    return arr


def animate_frames(
    container,
    frames: list,
    labels: list,
    delay: float = 0.9,
    loop: int = 2,
) -> None:
    """
    Simulates an animation by cycling through frames in a Streamlit container.

    Args:
        container: st.empty() placeholder
        frames:    list of np.ndarray RGB images
        labels:    list of caption strings
        delay:     seconds between frames
        loop:      number of loops
    """
    for _ in range(loop):
        for img, label in zip(frames, labels):
            container.image(img, caption=label, use_container_width=True)
            time.sleep(delay)


def display_metric_card(label: str, value: str, desc: str, col) -> None:
    """Renders a styled metric card into a Streamlit column."""
    col.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-desc">{desc}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  CACHED PIPELINE (runs once, re-runs only on sidebar changes)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_interpolator():
    """Load/cache the ML interpolator across Streamlit reruns."""
    from ml_engine import get_interpolator
    return get_interpolator()


@st.cache_data(show_spinner=False)
def run_pipeline(use_dummy: bool = True):
    """
    End-to-end pipeline: load data → interpolate → compute metrics.
    Cached so refreshing the page doesn't re-run everything.

    Returns:
        dict with: frame_t0, frame_t30, frame_t15, frame_naive, metrics
    """
    from data_processor import get_frame_tensors
    from ml_engine import run_interpolation, get_interpolator

    # 1. Load satellite data
    t0, t30, raw0, raw30 = get_frame_tensors(use_dummy=use_dummy)

    # 2. Interpolate t=15 min frame
    interp = get_interpolator()
    result = run_interpolation(t0, t30, interpolator=interp)

    return {
        "frame_t0":    t0,
        "frame_t30":   t30,
        "frame_t15":   result["frame_t15"],
        "frame_naive": result["frame_naive"],
        "metrics":     result["metrics"],
    }


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0 8px;'>
      <span style='font-size:2.4rem;'>🛰️</span>
      <div style='font-size:1.1rem; font-weight:800; color:#38bdf8; margin-top:4px;'>
        Project Aviral
      </div>
      <div style='font-size:0.75rem; color:#475569; margin-top:2px;'>
        INSAT-3D Deep VFI Pipeline
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("#### ⚙️ Pipeline Settings")

    use_dummy = st.toggle(
        "Use Synthetic Data",
        value=True,
        help="ON = auto-generate dummy INSAT data. OFF = load real .nc files from data/",
    )

    colormap_on = st.toggle(
        "False-Color Thermal Map",
        value=True,
        help="Apply INFERNO colormap (mimics standard meteorological IR visualization)",
    )

    anim_speed = st.select_slider(
        "Animation Speed",
        options=["Slow (1.2s)", "Normal (0.8s)", "Fast (0.4s)"],
        value="Normal (0.8s)",
    )
    speed_map = {"Slow (1.2s)": 1.2, "Normal (0.8s)": 0.8, "Fast (0.4s)": 0.4}
    frame_delay = speed_map[anim_speed]

    anim_loops = st.slider("Animation Loops", 1, 5, 3)

    st.divider()

    st.markdown("""
    #### 📂 Data File Locations
    ```
    data/
    ├── insat_t0.nc    ← t = 00 min
    └── insat_t30.nc   ← t = 30 min
    ```
    #### 🔧 Model Weights (RIFE)
    ```
    models/
    └── train_log/
        ├── flownet.pkl
        └── contextnet.pkl
    ```
    Set `USE_RIFE = True` in `ml_engine.py` after placing weights.
    """)

    st.divider()

    run_btn = st.button("🚀  Run VFI Pipeline", key="run_pipeline_btn")


# ─────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">🛰️ Project Aviral — Deep VFI Dashboard</p>
  <p class="hero-subtitle">
    INSAT-3D Thermal IR · 30-min → 15-min Temporal Super-Resolution
  </p>
  <span class="badge">INSAT-3D</span>
  <span class="badge">DEEP VFI</span>
  <span class="badge">OPTICAL FLOW</span>
  <span class="badge">REAL-TIME</span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────
# Auto-run on first load; also re-run when button clicked
if "pipeline_result" not in st.session_state or run_btn:

    with st.spinner("🔄  Running end-to-end VFI pipeline..."):
        progress = st.progress(0, text="Loading satellite data...")
        time.sleep(0.2)

        try:
            result = run_pipeline(use_dummy=use_dummy)
            st.session_state["pipeline_result"] = result

            progress.progress(30, text="Data loaded. Starting interpolation...")
            time.sleep(0.3)
            progress.progress(75, text="Interpolating t=15 min frame...")
            time.sleep(0.4)
            progress.progress(100, text="Computing metrics...")
            time.sleep(0.2)
            progress.empty()

        except Exception as e:
            st.error(f"❌ Pipeline error: {e}")
            st.info(
                "💡 Tip: Make sure you have installed all dependencies:\n"
                "```\npip install streamlit torch xarray netCDF4 "
                "opencv-python scikit-image numpy\n```"
            )
            st.stop()

result = st.session_state["pipeline_result"]
metrics = result["metrics"]

# ─────────────────────────────────────────────────────────────
#  STATUS BAR
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="status-pill"><span class="dot-pulse"></span>'
    ' Pipeline Active — Optical Flow Bidirectional Interpolation</div>',
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Quality Metrics — AI vs. Linear Baseline</div>',
            unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)

display_metric_card("SSIM",  f"{metrics['ssim']:.4f}",
                    "Structural Similarity ↑", m1)
display_metric_card("MSE",   f"{metrics['mse']:.2e}",
                    "Mean Squared Error ↓",    m2)
display_metric_card("PSNR",  f"{metrics['psnr']:.2f} dB",
                    "Peak Signal-to-Noise ↑",  m3)
display_metric_card("METHOD", "Optical Flow",
                    "Farneback Bidirectional",  m4)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DUAL-PLAYER TIME-LAPSE
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎬 Dual-Player Time-Lapse Viewer</div>',
            unsafe_allow_html=True)

# Convert tensors to display-ready RGB images
img_t0  = tensor_to_rgb(result["frame_t0"],  colormap=colormap_on)
img_t30 = tensor_to_rgb(result["frame_t30"], colormap=colormap_on)
img_t15 = tensor_to_rgb(result["frame_t15"], colormap=colormap_on)

col_orig, col_divider, col_ai = st.columns([5, 0.3, 5])

# ── LEFT: Original 2-frame jump ──
with col_orig:
    st.markdown(
        '<div class="player-label player-original">'
        '⚠️ Original — 2-Frame Jump (30-min gap)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align:center;">'
        '<span class="frame-badge fb-t0">t = 00 min</span>'
        '<span style="color:#475569; font-size:0.8rem;"> ➜ </span>'
        '<span class="frame-badge fb-t30">t = 30 min</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # Timeline visual
    st.markdown("""
    <div style="display:flex; align-items:center; margin: 8px 24px;">
      <div style="width:12px;height:12px;border-radius:50%;background:#f59e0b;flex-shrink:0;"></div>
      <div style="height:2px;flex:1;background:linear-gradient(90deg,#f59e0b,#ef4444);"></div>
      <div style="width:12px;height:12px;border-radius:50%;background:#f59e0b;flex-shrink:0;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0 12px;margin-bottom:8px;">
      <span style="font-size:0.65rem;color:#64748b;font-family:monospace;">t=0 min</span>
      <span style="font-size:0.65rem;color:#ef4444;font-family:monospace;">⚠ MISSING t=15</span>
      <span style="font-size:0.65rem;color:#64748b;font-family:monospace;">t=30 min</span>
    </div>
    """, unsafe_allow_html=True)

    orig_placeholder = st.empty()

    # Show static frames for original (no interpolated frame)
    orig_frames = [img_t0, img_t30]
    orig_labels = ["t = 00 min  |  Original", "t = 30 min  |  Original"]


# ── DIVIDER ──
with col_divider:
    st.markdown(
        '<div style="height:100%;display:flex;align-items:center;'
        'justify-content:center; padding-top:100px;">'
        '<div style="width:1px;height:300px;background:rgba(99,179,237,0.2);"></div>'
        '</div>',
        unsafe_allow_html=True
    )

# ── RIGHT: AI-enhanced 3-frame sequence ──
with col_ai:
    st.markdown(
        '<div class="player-label player-ai">'
        '✨ AI-Enhanced — 3-Frame Smooth (15-min resolution)</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align:center;">'
        '<span class="frame-badge fb-t0">t = 00 min</span>'
        '<span style="color:#475569; font-size:0.8rem;"> ➜ </span>'
        '<span class="frame-badge fb-t15">t = 15 min ✨</span>'
        '<span style="color:#475569; font-size:0.8rem;"> ➜ </span>'
        '<span class="frame-badge fb-t30">t = 30 min</span>'
        '</div>',
        unsafe_allow_html=True
    )

    # Timeline visual — 3 dots
    st.markdown("""
    <div style="display:flex; align-items:center; margin: 8px 24px;">
      <div style="width:12px;height:12px;border-radius:50%;background:#34d399;flex-shrink:0;"></div>
      <div style="height:2px;flex:1;background:linear-gradient(90deg,#34d399,#22d3ee);"></div>
      <div style="width:12px;height:12px;border-radius:50%;background:#22d3ee;border:2px solid #0ea5e9;flex-shrink:0;"></div>
      <div style="height:2px;flex:1;background:linear-gradient(90deg,#22d3ee,#34d399);"></div>
      <div style="width:12px;height:12px;border-radius:50%;background:#34d399;flex-shrink:0;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;padding:0 12px;margin-bottom:8px;">
      <span style="font-size:0.65rem;color:#64748b;font-family:monospace;">t=0 min</span>
      <span style="font-size:0.65rem;color:#22d3ee;font-weight:700;font-family:monospace;">✨ t=15 AI</span>
      <span style="font-size:0.65rem;color:#64748b;font-family:monospace;">t=30 min</span>
    </div>
    """, unsafe_allow_html=True)

    ai_placeholder = st.empty()

    # 3-frame sequence for AI column
    ai_frames = [img_t0, img_t15, img_t30]
    ai_labels = [
        "t = 00 min  |  AI-Enhanced",
        "t = 15 min  |  ✨ AI Interpolated",
        "t = 30 min  |  AI-Enhanced",
    ]


# ── Run Animation ──
st.markdown("<br>", unsafe_allow_html=True)
anim_col1, anim_col2, anim_col3 = st.columns([2, 1, 2])
with anim_col2:
    play_btn = st.button("▶  Play Animation", key="play_btn")

if play_btn:
    # Animate both columns simultaneously (loop several times)
    n_orig = len(orig_frames)
    n_ai   = len(ai_frames)
    max_frames = max(n_orig, n_ai)

    for loop_i in range(anim_loops):
        for fi in range(max_frames):
            orig_idx = min(fi, n_orig - 1)
            ai_idx   = fi % n_ai

            orig_placeholder.image(
                orig_frames[orig_idx],
                caption=orig_labels[orig_idx],
                use_container_width=True,
            )
            ai_placeholder.image(
                ai_frames[ai_idx],
                caption=ai_labels[ai_idx],
                use_container_width=True,
            )
            time.sleep(frame_delay)

    # After animation, show static final state
    orig_placeholder.image(img_t30, caption="t = 30 min | Original",
                            use_container_width=True)
    ai_placeholder.image(img_t15, caption="t = 15 min | ✨ AI Interpolated",
                          use_container_width=True)
else:
    # Default static display before animation plays
    orig_placeholder.image(img_t0, caption="t = 00 min | Original",
                            use_container_width=True)
    ai_placeholder.image(img_t15, caption="t = 15 min | ✨ AI Interpolated",
                          use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  SIDE-BY-SIDE STATIC COMPARISON
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-header">🔍 Frame-by-Frame Comparison</div>',
            unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.image(img_t0, caption="📡 t = 00 min  |  INSAT Observed",
             use_container_width=True)
with fc2:
    st.image(img_t15, caption="✨ t = 15 min  |  AI Interpolated (VFI)",
             use_container_width=True)
with fc3:
    st.image(img_t30, caption="📡 t = 30 min  |  INSAT Observed",
             use_container_width=True)


# ─────────────────────────────────────────────────────────────
#  PIPELINE ARCHITECTURE INFO
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🏗️ Pipeline Architecture & Technical Details", expanded=False):
    st.markdown("""
    ### End-to-End VFI Pipeline

    ```
    INSAT-3D NetCDF (.nc)
          │
          ▼
    [data_processor.py]
    • xarray → 2D float32 array
    • Percentile clip (p1–p99)
    • Min-max normalize → [0, 1]
    • Resize → 512×512 (bilinear)
    • Stack 3-channel tensor (1,3,H,W)
          │
          ▼
    [ml_engine.py]
    • Farneback Dense Optical Flow
    • Bidirectional warping (t0→t15 + t30→t15)
    • 50/50 blend → interpolated t=15 frame
    • SSIM / MSE / PSNR vs. linear baseline
          │
          ▼
    [app.py — This Dashboard]
    • Dual-Player Time-Lapse Viewer
    • False-color INFERNO thermal map
    • Live quality metrics display
    ```

    ### Upgrading to RIFE (Deep Learning)
    1. `git clone https://github.com/megvii-research/ECCV2022-RIFE`
    2. Download pretrained weights → `models/train_log/`
    3. Set `USE_RIFE = True` in `ml_engine.py`
    4. Restart the Streamlit app

    ### Key References
    - **RIFE**: Huang et al., ECCV 2022 — Real-Time Intermediate Flow Estimation
    - **INSAT-3D**: ISRO Geostationary Meteorological Satellite
    - **Farneback OF**: Gunnar Farneback, 2003 — Two-Frame Motion Estimation
    """)


# ─────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#334155; font-size:0.78rem; padding:16px 0 8px;'>
  🛰️ Project Aviral — INSAT Deep Video Frame Interpolation Pipeline<br>
  Built with PyTorch · OpenCV · xarray · Streamlit
</div>
""", unsafe_allow_html=True)
