"""
Streamlit dashboard for SLM inference optimization benchmarking.
DS-GA 1019 — NYU
"""

import sys
import os

import streamlit as st
import torch

sys.path.insert(0, os.path.dirname(__file__))

from src.config import MAX_NEW_TOKENS, TEMPERATURE, TOP_K, AVAILABLE_MODELS, format_prompt

st.set_page_config(
    page_title="Inference Optimization Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0f1117;
    --surface:  #161b27;
    --raised:   #1e2435;
    --border:   #252d3d;
    --accent:   #4a9eff;
    --accent-d: rgba(74,158,255,0.12);
    --green:    #34d399;
    --red:      #f87171;
    --hi:       #e4e8f0;
    --mid:      #6b7a99;
    --lo:       #363e52;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'IBM Plex Sans', sans-serif;
}

html, body, [class*="css"] {
    font-family: var(--sans) !important;
    background: var(--bg) !important;
    color: var(--hi) !important;
}
.stApp { background: var(--bg) !important; }
.block-container { padding: 2rem 2.5rem !important; max-width: 1300px !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem !important; }
[data-testid="stSidebarNav"] { display: none; }

/* Headings */
h1, h2, h3, h4 { font-family: var(--sans) !important; }
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: var(--hi) !important; letter-spacing: -0.01em !important; }
h2 { font-size: 1.05rem !important; font-weight: 600 !important; color: var(--hi) !important; }
h3 { font-size: 0.72rem !important; font-weight: 500 !important; color: var(--mid) !important;
     text-transform: uppercase !important; letter-spacing: 0.09em !important; margin-bottom: 0.5rem !important; }

/* Labels */
label, .stSelectbox label, .stSlider label, .stTextArea label, .stRadio label {
    font-family: var(--sans) !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    color: var(--mid) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

/* Inputs */
.stTextArea textarea, .stTextInput input {
    background: var(--raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--hi) !important;
    font-family: var(--mono) !important;
    font-size: 0.83rem !important;
    line-height: 1.6 !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-d) !important;
    outline: none !important;
}

/* Selectbox */
.stSelectbox > div > div {
    background: var(--raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    color: var(--hi) !important;
    font-size: 0.83rem !important;
    font-family: var(--sans) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
}
.stSlider [class*="StyledSliderBar"] > div:first-child { background: var(--accent) !important; }

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: var(--sans) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--lo) !important;
    font-family: var(--sans) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1.1rem !important;
    border-bottom: 2px solid transparent !important;
    transition: color 0.15s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--hi) !important;
    border-bottom-color: var(--accent) !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    padding: 0.85rem 1rem !important;
}
[data-testid="metric-container"] label { color: var(--mid) !important; font-size: 0.65rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--hi) !important;
    font-family: var(--mono) !important;
    font-size: 1.4rem !important;
    font-weight: 500 !important;
}

/* Alerts */
.stAlert {
    background: var(--raised) !important;
    border: 1px solid var(--border) !important;
    border-radius: 3px !important;
    font-size: 0.8rem !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 1rem 0 !important; }

/* Radio */
.stRadio [data-testid="stMarkdownContainer"] p {
    font-size: 0.83rem !important;
    color: var(--hi) !important;
    font-family: var(--sans) !important;
}

/* Success/info color override */
.stSuccess { border-left-color: var(--green) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Model loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_base_model(model_id: str):
    from src.model import load_model_and_tokenizer
    return load_model_and_tokenizer(model_id)


@st.cache_resource(show_spinner=False)
def load_quantized_model(model_id: str):
    from src.model import load_model_and_tokenizer
    from src.quantization import quantize_model
    model, tokenizer = load_model_and_tokenizer(model_id)
    return quantize_model(model), tokenizer


# ── Helpers ────────────────────────────────────────────────────────────────────

def speedup_color(x: float) -> str:
    if x >= 5:   return "#34d399"
    if x >= 2:   return "#4a9eff"
    if x >= 1:   return "#6b7a99"
    return "#f87171"


KNOWN_RESULTS = [
    {"label": "Baseline",           "tok_s": 17.92,  "speedup": 1.00, "mem_mb": 474.7},
    {"label": "KV-Cache",           "tok_s": 74.65,  "speedup": 4.17, "mem_mb": 474.7},
    {"label": "Quant + KV-Cache",   "tok_s": 22.44,  "speedup": 1.25, "mem_mb": 268.5},
    {"label": "Batch (bs=4) + KV",  "tok_s": 130.76, "speedup": 7.30, "mem_mb": 474.7},
    {"label": "All Combined",       "tok_s": 47.49,  "speedup": 2.65, "mem_mb": 268.5},
]

MODES = {
    "Baseline":              "baseline",
    "KV-Cache":              "kv_cache",
    "Quantized + KV-Cache":  "quantized",
    "Batched (bs=4) + KV":   "batched",
}


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Model")

    selected_model_label = st.selectbox(
        "model_select",
        list(AVAILABLE_MODELS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    selected_model_id = AVAILABLE_MODELS[selected_model_label]

    with st.spinner(f"Loading {selected_model_label}…"):
        try:
            model, tokenizer = load_base_model(selected_model_id)
        except Exception as e:
            st.error(f"Load failed: {e}")
            st.stop()

    from src.quantization import get_model_size_mb
    model_name = getattr(model.config, "_name_or_path", "unknown")
    mem_mb = get_model_size_mb(model)
    device = "CUDA" if torch.cuda.is_available() else "CPU"

    st.markdown(f"""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.72rem;line-height:2;color:#6b7a99;">
  <div><span style="color:#4a9eff;">{model_name}</span></div>
  <div>Device&nbsp;&nbsp;{device}</div>
  <div>Memory&nbsp;&nbsp;{mem_mb:.0f} MB</div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("### Generation parameters")
    max_new_tokens = st.slider("Max new tokens", 20, 400, MAX_NEW_TOKENS, step=10)
    temperature    = st.slider("Temperature",    0.1, 2.0, float(TEMPERATURE), step=0.05)
    top_k          = st.slider("Top-k",          1,   100, int(TOP_K), step=1)


# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="margin-bottom:1.6rem;">
  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:1.35rem;font-weight:600;
              color:#e4e8f0;letter-spacing:-0.01em;margin-bottom:0.2rem;">
    Inference Optimization Lab
  </div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#363e52;
              letter-spacing:0.04em;">
    DS-GA 1019 &nbsp;·&nbsp; {selected_model_label} &nbsp;·&nbsp; KV-Cache &nbsp;·&nbsp; INT8 Quantization &nbsp;·&nbsp; Async Batching &nbsp;·&nbsp; Numba JIT
  </div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_gen, tab_compare, tab_bench = st.tabs(["Generate", "Compare", "Benchmark Results"])


# ── TAB: Generate ──────────────────────────────────────────────────────────────

with tab_gen:
    col_l, col_r = st.columns([1.1, 1], gap="large")

    with col_l:
        st.markdown("### Prompt")
        prompt = st.text_area(
            "prompt_input",
            value="The meaning of life is",
            height=110,
            label_visibility="collapsed",
            placeholder="Enter a prompt…",
        )

        st.markdown("### Optimization")
        mode_label = st.radio("mode", list(MODES.keys()), label_visibility="collapsed")
        mode = MODES[mode_label]

        run_btn = st.button("Generate", use_container_width=True)

    with col_r:
        st.markdown("### Output")
        output_slot  = st.empty()
        metrics_slot = st.empty()

        if run_btn:
            if not prompt.strip():
                output_slot.warning("Enter a prompt first.")
            else:
                with st.spinner("Generating…"):
                    formatted_prompt = format_prompt(prompt, selected_model_id)
                    try:
                        if mode == "baseline":
                            from src.inference import generate_manual
                            result = generate_manual(model, tokenizer, formatted_prompt,
                                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

                        elif mode == "kv_cache":
                            from src.kv_cache import generate_with_kv_cache
                            result = generate_with_kv_cache(model, tokenizer, formatted_prompt,
                                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

                        elif mode == "quantized":
                            q_model, _ = load_quantized_model(selected_model_id)
                            from src.quantization import generate_quantized
                            result = generate_quantized(q_model, tokenizer, formatted_prompt,
                                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)

                        elif mode == "batched":
                            from src.async_batching import generate_batched
                            result = generate_batched(model, tokenizer, [formatted_prompt],
                                max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)[0]

                    except Exception as exc:
                        output_slot.error(f"Error: {exc}")
                        st.stop()

                text = result["text"]
                # Strip template prefix for chat models — show only the assistant reply
                if text.startswith(formatted_prompt):
                    continuation = text[len(formatted_prompt):]
                elif "<|assistant|>" in text:
                    continuation = text.split("<|assistant|>")[-1].strip()
                else:
                    continuation = text[len(prompt):]
                tok_s = result.get("tok_per_sec", 0)
                baseline_ref = 17.92
                speedup = tok_s / baseline_ref if baseline_ref else 0

                output_slot.markdown(f"""
<div style="background:#161b27;border:1px solid #252d3d;border-radius:3px;
            padding:1rem 1.1rem;min-height:7rem;font-family:'IBM Plex Mono',monospace;
            font-size:0.8rem;line-height:1.7;color:#6b7a99;word-break:break-word;">
  <span style="color:#c8d0e0;">{prompt} </span><span style="color:#8ba4cc;">{continuation.strip()}</span>
</div>
""", unsafe_allow_html=True)

                with metrics_slot.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tokens / sec",  f"{tok_s:.1f}")
                    m2.metric("Elapsed (s)",   f"{result['elapsed']:.2f}")
                    m3.metric("New tokens",    result["num_tokens"])

                    sc = speedup_color(speedup)
                    st.markdown(f"""
<div style="margin-top:0.5rem;padding:0.45rem 0.8rem;background:#161b27;
            border:1px solid #252d3d;border-radius:3px;
            font-family:'IBM Plex Mono',monospace;font-size:0.73rem;
            color:#6b7a99;display:flex;align-items:center;gap:0.6rem;">
  <span>Speedup vs baseline</span>
  <span style="color:{sc};font-weight:600;">{speedup:.2f}×</span>
  <span style="margin-left:auto;color:#363e52;font-size:0.67rem;">{mode_label}</span>
</div>
""", unsafe_allow_html=True)

        else:
            output_slot.markdown("""
<div style="background:#161b27;border:1px dashed #252d3d;border-radius:3px;
            padding:1rem 1.1rem;min-height:7rem;font-family:'IBM Plex Mono',monospace;
            font-size:0.75rem;color:#252d3d;display:flex;align-items:center;
            justify-content:center;">
  output will appear here
</div>
""", unsafe_allow_html=True)


# ── TAB: Compare ───────────────────────────────────────────────────────────────

with tab_compare:
    st.markdown("### Side-by-side comparison")

    cmp_prompt = st.text_area(
        "cmp_prompt",
        value="In a distant galaxy, a lone astronaut discovered",
        height=80,
        label_visibility="collapsed",
    )

    cc1, cc2 = st.columns(2)
    with cc1:
        mode_a_label = st.selectbox("Mode A", list(MODES.keys()), index=0, key="mode_a")
    with cc2:
        mode_b_label = st.selectbox("Mode B", list(MODES.keys()), index=1, key="mode_b")

    cmp_tokens = st.slider("Max tokens", 20, 200, 100, step=10)
    cmp_btn = st.button("Run Comparison", use_container_width=True)

    if cmp_btn and cmp_prompt.strip():
        def _run(label, prmpt, mx):
            key = MODES[label]
            fp = format_prompt(prmpt, selected_model_id)
            kw = dict(max_new_tokens=mx, temperature=temperature, top_k=top_k)
            if key == "baseline":
                from src.inference import generate_manual
                return generate_manual(model, tokenizer, fp, **kw)
            elif key == "kv_cache":
                from src.kv_cache import generate_with_kv_cache
                return generate_with_kv_cache(model, tokenizer, fp, **kw)
            elif key == "quantized":
                q_mdl, _ = load_quantized_model(selected_model_id)
                from src.quantization import generate_quantized
                return generate_quantized(q_mdl, tokenizer, fp, **kw)
            elif key == "batched":
                from src.async_batching import generate_batched
                return generate_batched(model, tokenizer, [fp], **kw)[0]

        def _extract_reply(res, prmpt):
            text = res["text"]
            if "<|assistant|>" in text:
                return text.split("<|assistant|>")[-1].strip()
            fp = format_prompt(prmpt, selected_model_id)
            if text.startswith(fp):
                return text[len(fp):].strip()
            return text[len(prmpt):].strip()

        with st.spinner("Running…"):
            res_a = _run(mode_a_label, cmp_prompt, cmp_tokens)
            res_b = _run(mode_b_label, cmp_prompt, cmp_tokens)

        ra, rb = st.columns(2)

        def _render_result(col, label, res, prmpt):
            tok_s = res.get("tok_per_sec", 0)
            reply = _extract_reply(res, prmpt)
            with col:
                st.markdown(f"""
<div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.78rem;font-weight:500;
            color:#6b7a99;margin-bottom:0.4rem;text-transform:uppercase;letter-spacing:0.06em;">
  {label}
</div>
<div style="background:#161b27;border:1px solid #252d3d;border-radius:3px;
            padding:0.9rem 1rem;font-family:'IBM Plex Mono',monospace;
            font-size:0.77rem;line-height:1.65;color:#8ba4cc;min-height:6rem;
            word-break:break-word;">
  {reply[:300]}{'…' if len(reply) > 300 else ''}
</div>
""", unsafe_allow_html=True)
                st.metric("Tokens / sec", f"{tok_s:.1f}")

        _render_result(ra, mode_a_label, res_a, cmp_prompt)
        _render_result(rb, mode_b_label, res_b, cmp_prompt)

        tok_a = res_a.get("tok_per_sec", 0)
        tok_b = res_b.get("tok_per_sec", 0)
        if min(tok_a, tok_b) > 0:
            ratio  = max(tok_a, tok_b) / min(tok_a, tok_b)
            winner = mode_a_label if tok_a >= tok_b else mode_b_label
            loser  = mode_b_label if tok_a >= tok_b else mode_a_label
            st.markdown(f"""
<div style="margin-top:0.8rem;padding:0.6rem 0.9rem;background:#161b27;
            border:1px solid #252d3d;border-radius:3px;
            font-family:'IBM Plex Mono',monospace;font-size:0.75rem;color:#6b7a99;">
  <span style="color:#e4e8f0;">{winner}</span> is
  <span style="color:#4a9eff;font-weight:500;">{ratio:.2f}×</span>
  faster than <span style="color:#e4e8f0;">{loser}</span> on this prompt.
</div>
""", unsafe_allow_html=True)


# ── TAB: Benchmark Results ──────────────────────────────────────────────────────

with tab_bench:
    st.markdown("### Benchmark results")
    st.markdown("""
<div style="font-family:'IBM Plex Mono',monospace;font-size:0.7rem;color:#363e52;
            margin-bottom:1.2rem;letter-spacing:0.02em;">
  GPT-2 124M · 10 prompts · 5 runs each · CPU · pre-computed
</div>
""", unsafe_allow_html=True)

    # Results table
    rows_html = ""
    for r in KNOWN_RESULTS:
        sp = r["speedup"]
        bar_pct = min(sp / 7.5 * 100, 100)
        sc = speedup_color(sp)
        rows_html += f"""
<tr>
  <td style="padding:0.55rem 0.8rem;border-bottom:1px solid #1a1f2c;
             font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#c8d0e0;">{r['label']}</td>
  <td style="padding:0.55rem 0.8rem;border-bottom:1px solid #1a1f2c;
             font-family:'IBM Plex Mono',monospace;font-size:0.78rem;color:#e4e8f0;
             text-align:right;font-weight:500;">{r['tok_s']:.2f}</td>
  <td style="padding:0.55rem 0.8rem;border-bottom:1px solid #1a1f2c;
             font-family:'IBM Plex Mono',monospace;font-size:0.78rem;text-align:right;">
    <span style="color:{sc};">{sp:.2f}×</span>
  </td>
  <td style="padding:0.55rem 1rem 0.55rem 0.8rem;border-bottom:1px solid #1a1f2c;min-width:140px;">
    <div style="height:4px;background:#1e2435;border-radius:2px;overflow:hidden;">
      <div style="width:{bar_pct:.1f}%;height:100%;background:{sc};border-radius:2px;"></div>
    </div>
  </td>
  <td style="padding:0.55rem 0.8rem;border-bottom:1px solid #1a1f2c;
             font-family:'IBM Plex Mono',monospace;font-size:0.78rem;
             color:#6b7a99;text-align:right;">{r['mem_mb']:.0f} MB</td>
</tr>"""

    st.markdown(f"""
<div style="border:1px solid #252d3d;border-radius:4px;overflow:hidden;margin-bottom:1.5rem;">
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="background:#161b27;">
        <th style="padding:0.45rem 0.8rem;text-align:left;font-family:'IBM Plex Sans',sans-serif;
                   font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                   color:#363e52;border-bottom:1px solid #252d3d;">Configuration</th>
        <th style="padding:0.45rem 0.8rem;text-align:right;font-family:'IBM Plex Sans',sans-serif;
                   font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                   color:#363e52;border-bottom:1px solid #252d3d;">Tok/s</th>
        <th style="padding:0.45rem 0.8rem;text-align:right;font-family:'IBM Plex Sans',sans-serif;
                   font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                   color:#363e52;border-bottom:1px solid #252d3d;">Speedup</th>
        <th style="padding:0.45rem 0.8rem;font-family:'IBM Plex Sans',sans-serif;
                   font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                   color:#363e52;border-bottom:1px solid #252d3d;"></th>
        <th style="padding:0.45rem 0.8rem;text-align:right;font-family:'IBM Plex Sans',sans-serif;
                   font-size:0.65rem;letter-spacing:0.08em;text-transform:uppercase;
                   color:#363e52;border-bottom:1px solid #252d3d;">Memory</th>
      </tr>
    </thead>
    <tbody style="background:#0f1117;">
      {rows_html}
    </tbody>
  </table>
</div>
""", unsafe_allow_html=True)

    # Findings
    st.markdown("### Findings")
    findings = [
        ("4.17×",  "KV-Cache alone is the dominant optimization — eliminates O(n) key/value recomputation per decode step."),
        ("7.30×",  "Batching (bs=4) combined with KV-Cache yields the best aggregate throughput."),
        ("−43%",   "INT8 quantization reduces model memory from 474 MB to 268 MB with negligible output quality loss."),
        ("2.65×",  "Full combination underperforms batch+KV alone: dequantization overhead on CPU dominates at larger batch sizes."),
    ]

    c1, c2 = st.columns(2)
    for i, (stat, desc) in enumerate(findings):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f"""
<div style="background:#161b27;border:1px solid #252d3d;border-left:2px solid #4a9eff;
            border-radius:3px;padding:0.75rem 0.9rem;margin-bottom:0.6rem;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:1.1rem;font-weight:500;
              color:#e4e8f0;margin-bottom:0.25rem;">{stat}</div>
  <div style="font-family:'IBM Plex Sans',sans-serif;font-size:0.78rem;
              color:#6b7a99;line-height:1.55;">{desc}</div>
</div>
""", unsafe_allow_html=True)
