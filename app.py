import streamlit as st
import time
import os
from utils.pdf_processor import extract_text_from_pdf, recursive_chunk_text
from utils.summarizer import load_model_and_tokenizer, summarize_chunks
from utils.qlora_trainer import run_qlora_finetuning

st.set_page_config(
    page_title="PrivSumAI",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource(show_spinner=False)
def get_model(use_4bit: bool, adapter_path):
    return load_model_and_tokenizer(use_4bit=use_4bit, adapter_path=adapter_path)

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:ital,wght@0,700;1,400&display=swap" rel="stylesheet">

<style>
:root {
  --bg:        #080D14;
  --bg2:       #0D1520;
  --bg3:       #111C2B;
  --bg4:       #162234;
  --em:        #10B981;
  --em2:       #34D399;
  --em3:       #064E3B;
  --gold:      #F59E0B;
  --gold2:     #FCD34D;
  --red:       #EF4444;
  --txt:       #E2EAF4;
  --txt2:      #8BA0BC;
  --txt3:      #4A6480;
  --border:    rgba(16,185,129,0.12);
  --border2:   rgba(16,185,129,0.25);
  --border3:   rgba(226,234,244,0.08);
  --r:         10px;
  --r2:        16px;
}

html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--txt) !important;
}

.block-container { padding: 0 2.5rem 4rem !important; max-width: 1100px !important; }

/* ── Sidebar ──────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg2) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--txt) !important; font-family: 'IBM Plex Sans', sans-serif !important; }
[data-testid="stSidebar"] label { color: var(--txt2) !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] p { color: var(--txt2) !important; font-size: 0.8rem !important; }
[data-testid="stSidebar"] [role="slider"] { background: var(--em) !important; border-color: var(--em) !important; box-shadow: 0 0 8px rgba(16,185,129,.4) !important; }
[data-testid="stSidebar"] [data-testid="stToggle"] input:checked + div { background: var(--em) !important; }
[data-testid="stSidebar"] [data-testid="stTextInput"] input {
  background: var(--bg3) !important; border-color: var(--border2) !important;
  color: var(--txt) !important; font-family: 'IBM Plex Mono', monospace !important;
  font-size: 0.8rem !important; border-radius: 6px !important;
}
[data-testid="stSidebar"] [data-testid="stSlider"] > div > div > div { background: var(--border) !important; }

/* ── Header ───────────────────────────────────────────────── */
.hdr { padding: 2.8rem 0 2rem; border-bottom: 1px solid var(--border); margin-bottom: 2rem; }
.hdr-eyebrow {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--em);
  display: flex; align-items: center; gap: 8px; margin-bottom: 0.8rem;
}
.hdr-eyebrow::before { content: ''; width: 20px; height: 1px; background: var(--em); }
.hdr h1 {
  font-family: 'Fraunces', serif !important; font-size: 3rem !important;
  font-weight: 700 !important; color: var(--txt) !important;
  line-height: 1.05 !important; margin: 0 0 0.5rem !important; letter-spacing: -0.03em !important;
}
.hdr h1 .accent { color: var(--em); font-style: italic; }
.hdr-sub { font-size: 0.92rem; color: var(--txt2); font-weight: 300; line-height: 1.6; margin-bottom: 1.4rem; max-width: 560px; }
.badges { display: flex; flex-wrap: wrap; gap: 6px; }
.badge {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem;
  letter-spacing: 0.06em; padding: 4px 12px; border-radius: 100px;
  border: 1px solid; display: inline-flex; align-items: center; gap: 4px;
}
.b-em  { color: var(--em);   border-color: rgba(16,185,129,.3); background: rgba(16,185,129,.06); }
.b-gd  { color: var(--gold); border-color: rgba(245,158,11,.3); background: rgba(245,158,11,.06); }
.b-wh  { color: var(--txt2); border-color: var(--border3);      background: rgba(226,234,244,.04); }

/* ── Tabs ─────────────────────────────────────────────────── */
[data-testid="stTabs"] [role="tablist"] {
  background: transparent !important; border-bottom: 1px solid var(--border3) !important;
  border-radius: 0 !important; padding: 0 !important; gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
  font-family: 'IBM Plex Sans', sans-serif !important; font-size: 0.84rem !important;
  font-weight: 500 !important; color: var(--txt3) !important;
  padding: 0.6rem 1.2rem !important; border-radius: 0 !important;
  border-bottom: 2px solid transparent !important; margin-bottom: -1px !important;
  transition: all .2s !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--em) !important; background: transparent !important;
  border-bottom: 2px solid var(--em) !important;
}
[data-testid="stTabs"] [role="tab"]:hover { color: var(--txt) !important; }

/* ── Radio ────────────────────────────────────────────────── */
[data-testid="stRadio"] label { color: var(--txt2) !important; font-size: 0.85rem !important; }
[data-testid="stRadio"] [data-testid="stMarkdownContainer"] p { color: var(--txt2) !important; }

/* ── File uploader ────────────────────────────────────────── */
[data-testid="stFileUploader"] {
  background: var(--bg2) !important; border: 1.5px dashed var(--border2) !important;
  border-radius: var(--r2) !important; transition: all .3s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--em) !important; box-shadow: 0 0 20px rgba(16,185,129,.1) !important; }
[data-testid="stFileUploader"] * { color: var(--txt2) !important; }
[data-testid="stFileUploadDropzone"] small { color: var(--txt3) !important; }

/* ── Buttons ──────────────────────────────────────────────── */
[data-testid="stButton"] > button {
  width: 100% !important; background: var(--em3) !important;
  border: 1px solid var(--em) !important; border-radius: var(--r) !important;
  color: var(--em2) !important; font-family: 'IBM Plex Sans', sans-serif !important;
  font-size: 0.88rem !important; font-weight: 600 !important;
  letter-spacing: 0.04em !important; padding: 0.72rem 2rem !important;
  transition: all .2s !important; box-shadow: 0 0 20px rgba(16,185,129,.08) !important;
  text-transform: uppercase !important;
}
[data-testid="stButton"] > button:hover {
  background: rgba(16,185,129,.2) !important;
  box-shadow: 0 0 30px rgba(16,185,129,.2) !important; transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
  background: transparent !important; border: 1px solid var(--border2) !important;
  color: var(--txt2) !important; text-transform: none !important;
  font-family: 'IBM Plex Mono', monospace !important; font-size: 0.78rem !important;
  letter-spacing: 0.03em !important;
}
[data-testid="stDownloadButton"] > button:hover {
  border-color: var(--em) !important; color: var(--em2) !important;
  background: rgba(16,185,129,.07) !important; transform: translateY(-1px) !important;
}

/* ── Progress bar ─────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--em) 0%, var(--em2) 100%) !important;
  border-radius: 100px !important;
}
[data-testid="stProgress"] > div { background: var(--bg3) !important; border-radius: 100px !important; }

/* ── Alerts ───────────────────────────────────────────────── */
[data-testid="stAlert"] {
  background: var(--bg3) !important; border-radius: var(--r) !important;
  border: 1px solid var(--border3) !important; color: var(--txt) !important;
}
[data-testid="stAlert"] p { color: var(--txt) !important; }
[data-testid="stAlert"][data-baseweb="notification"] { border-left: 3px solid var(--em) !important; }

/* ── Success/Info/Error ───────────────────────────────────── */
[data-testid="stNotification"] { background: var(--bg3) !important; color: var(--txt) !important; }
div[data-testid="stAlert"] > div { color: var(--txt) !important; }

/* ── Spinner ──────────────────────────────────────────────── */
[data-testid="stSpinner"] p { color: var(--txt2) !important; }

/* ── Toggle ───────────────────────────────────────────────── */
[data-testid="stToggle"] input:checked + div { background: var(--em) !important; }

/* ── Slider ───────────────────────────────────────────────── */
[data-testid="stSlider"] [role="slider"] { background: var(--em) !important; border-color: var(--em) !important; }
[data-testid="stSlider"] label { color: var(--txt2) !important; }
[data-testid="stSlider"] span { color: var(--txt3) !important; }

/* ── Privacy notice ───────────────────────────────────────── */
.pnotice {
  background: rgba(16,185,129,.04); border: 1px solid rgba(16,185,129,.15);
  border-left: 3px solid var(--em); border-radius: 0 var(--r) var(--r) 0;
  padding: 0.85rem 1.2rem; display: flex; gap: 10px; align-items: flex-start;
  font-size: 0.84rem; color: var(--txt2); font-weight: 300; margin-bottom: 1.4rem; line-height: 1.6;
}
.pnotice strong { color: var(--em2); }

/* ── Info cards ───────────────────────────────────────────── */
.info-strip { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin: 1.2rem 0; }
.icard {
  background: var(--bg3); border: 1px solid var(--border3); border-radius: var(--r);
  padding: 0.9rem 1.1rem; position: relative; overflow: hidden;
}
.icard::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--em), transparent);
}
.icard.gold::before { background: linear-gradient(90deg, var(--gold), transparent); }
.icard.wh::before   { background: linear-gradient(90deg, var(--txt3), transparent); }
.icard .lbl { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.16em; color: var(--txt3); margin-bottom: 0.4rem; }
.icard .val { font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 700; color: var(--txt); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.icard .val.em  { color: var(--em2); }
.icard .val.gd  { color: var(--gold2); }

/* ── Progress steps ───────────────────────────────────────── */
.steps {
  display: flex; align-items: center; gap: 0;
  background: var(--bg3); border: 1px solid var(--border3);
  border-radius: var(--r2); padding: 1rem 1.5rem; margin: 1rem 0;
}
.step { display: flex; align-items: center; flex: 1; gap: 8px; }
.step-dot {
  width: 28px; height: 28px; border-radius: 50%; flex-shrink: 0;
  display: flex; align-items: center; justify-content: center;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; font-weight: 500;
  border: 1.5px solid var(--border3); background: var(--bg3); color: var(--txt3);
  transition: all .4s;
}
.step-dot.active { border-color: var(--em); background: var(--em3); color: var(--em2); box-shadow: 0 0 12px rgba(16,185,129,.3); }
.step-dot.done   { border-color: var(--em); background: var(--em); color: var(--bg); }
.step-label { font-size: 0.76rem; color: var(--txt3); transition: color .4s; white-space: nowrap; }
.step-label.active { color: var(--em2); font-weight: 500; }
.step-label.done   { color: var(--em); }
.step-line { flex: 1; height: 1px; background: var(--border3); margin: 0 8px; transition: background .4s; }
.step-line.done { background: var(--em); }

/* ── Summary output ───────────────────────────────────────── */
.sum-header {
  display: flex; align-items: center; gap: 10px; margin: 1.8rem 0 0.9rem;
}
.sum-header .dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--em); box-shadow: 0 0 10px var(--em);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(.7)} }
.sum-header h3 {
  font-family: 'Fraunces', serif; font-size: 1.05rem; font-style: italic;
  font-weight: 400; color: var(--txt); margin: 0;
}
.sum-box {
  background: var(--bg2); border: 1px solid var(--border3);
  border-left: 3px solid var(--em); border-radius: 0 var(--r2) var(--r2) 0;
  padding: 1.6rem 1.8rem; font-size: 0.92rem; line-height: 1.9;
  color: var(--txt); font-weight: 300; position: relative;
}
.sum-box .stamp {
  position: absolute; top: 1rem; right: 1.2rem;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
  letter-spacing: 0.1em; color: var(--txt3); text-transform: uppercase;
}

/* ── Stat chips ───────────────────────────────────────────── */
.stats { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; margin-top: 1.2rem; }
.schip {
  background: var(--bg3); border: 1px solid var(--border3); border-radius: var(--r);
  padding: 0.85rem 1rem; display: flex; align-items: center; gap: 10px;
}
.schip .ico { font-size: 1.1rem; flex-shrink: 0; }
.schip .lbl { font-family: 'IBM Plex Mono', monospace; font-size: 0.58rem; text-transform: uppercase; letter-spacing: 0.12em; color: var(--txt3); }
.schip .val { font-family: 'Fraunces', serif; font-size: 1rem; font-weight: 700; color: var(--txt); }

/* ── Tcard (fine-tune / about) ────────────────────────────── */
.tcard {
  background: var(--bg3); border: 1px solid var(--border3);
  border-radius: var(--r); padding: 0.9rem 1.1rem;
  margin-bottom: 0.6rem; display: flex; gap: 12px; align-items: flex-start;
}
.tcard .num { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--em); font-weight: 500; padding-top: 3px; flex-shrink: 0; }
.tcard .ttl { font-size: 0.88rem; font-weight: 600; color: var(--txt); margin-bottom: 2px; }
.tcard .dsc { font-size: 0.81rem; color: var(--txt2); font-weight: 300; line-height: 1.55; }

/* ── Section label ────────────────────────────────────────── */
.sec-lbl {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.62rem;
  letter-spacing: 0.2em; text-transform: uppercase; color: var(--txt3);
  display: flex; align-items: center; gap: 8px; margin: 0 0 1rem;
}
.sec-lbl::after { content:''; flex:1; height:1px; background:var(--border3); }

/* ── Divider ──────────────────────────────────────────────── */
.divdr { height:1px; background:var(--border3); margin: 1.5rem 0; }

/* ── Sidebar labels ───────────────────────────────────────── */
.sb-lbl {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem;
  text-transform: uppercase; letter-spacing: 0.18em; color: var(--txt3);
  padding: 1rem 0 0.4rem; border-bottom: 1px solid var(--border3); margin-bottom: 0.6rem;
  display: block;
}

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border3); border-radius: 100px; }
::-webkit-scrollbar-thumb:hover { background: var(--em); }

/* ── Fix Streamlit markdown text color ───────────────────── */
.stMarkdown p, .stMarkdown li, .stMarkdown span { color: var(--txt2) !important; }
.stMarkdown strong { color: var(--txt) !important; }
.stMarkdown code { background: var(--bg4) !important; color: var(--em2) !important; border-radius: 4px !important; padding: 1px 5px !important; font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stMarkdownContainer"] p { color: var(--txt2) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--txt2) !important; }

/* ── Caption ──────────────────────────────────────────────── */
.stCaptionContainer p { color: var(--txt3) !important; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<span style="font-family:Fraunces,serif;font-size:1.05rem;font-weight:700;font-style:italic;color:#E2EAF4">PrivSumAI Settings</span>', unsafe_allow_html=True)

    st.markdown('<span class="sb-lbl">Model</span>', unsafe_allow_html=True)
    use_4bit = st.toggle("4-bit NF4 Quantization", value=True)
    max_new_tokens = st.slider("Max Output Tokens", 256, 768, 500, step=64)
    st.caption("Model stops naturally when done — this is the ceiling.")

    st.markdown('<span class="sb-lbl">Chunking</span>', unsafe_allow_html=True)
    chunk_size    = st.slider("Chunk Size (chars)", 500, 4000, 3000, step=500)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 50, 300, 50, step=50)

    st.markdown('<span class="sb-lbl">QLoRA Fine-Tuning</span>', unsafe_allow_html=True)
    run_finetune = st.toggle("Enable Fine-Tuning on Upload", value=False)
    if run_finetune:
        lora_r           = st.slider("LoRA Rank (r)", 4, 32, 8, step=4)
        lora_alpha       = st.slider("LoRA Alpha",    8, 64, 16, step=8)
        num_epochs       = st.slider("Epochs",        1,  5,  1)
        adapter_save_path= st.text_input("Adapter Save Path", value="./qlora_adapter")
    else:
        lora_r, lora_alpha, num_epochs, adapter_save_path = 8, 16, 1, "./qlora_adapter"

    st.markdown("""
    <div style="margin-top:1.8rem;padding:12px;background:rgba(16,185,129,.05);
         border:1px solid rgba(16,185,129,.15);border-radius:10px;
         font-size:.78rem;color:#8BA0BC;line-height:1.65">
      🔐 All processing is on-device.<br>Your documents never leave this machine.
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">Edge AI · TinyLlama-1.1B · QLoRA</div>
  <h1>Priv<span class="accent">Sum</span>AI</h1>
  <p class="hdr-sub">Privacy-preserving document summarization — runs entirely on your machine. No cloud. No API keys. No data leakage.</p>
  <div class="badges">
    <span class="badge b-em">🖥 Runs Locally</span>
    <span class="badge b-em">🔒 Zero Cloud Upload</span>
    <span class="badge b-gd">⚡ 4-bit Quantized</span>
    <span class="badge b-wh">🦙 TinyLlama-1.1B</span>
    <span class="badge b-wh">🎯 QLoRA Fine-Tune</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_summ, tab_tune, tab_about = st.tabs(["📄  Summarize", "🧠  Fine-Tune", "ℹ️  About"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ════════════════════════════════════════════════════════════════════════════
with tab_summ:
    st.markdown('<div class="sec-lbl">Upload document</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="pnotice">
      <span style="font-size:1rem;flex-shrink:0;margin-top:1px">🛡</span>
      <span>Your document is <strong>never uploaded</strong> to any server. Text extraction, chunking, and inference all run locally using your device's RAM and GPU.</span>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "Mode", ["Summarize one PDF", "Fine-tune on multiple PDFs"],
        horizontal=True, label_visibility="collapsed"
    )

    # ── Multi-doc fine-tune mode ──────────────────────────────────────────
    if mode == "Fine-tune on multiple PDFs" and run_finetune:
        uploaded_files = st.file_uploader(
            "Upload multiple PDFs",
            type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed"
        )
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader(
            "Drop a PDF here or click to browse",
            type=["pdf"], label_visibility="collapsed"
        )
        uploaded_files = []

    # Multi-doc training
    if uploaded_files and run_finetune:
        total_size = sum(f.size for f in uploaded_files)
        st.markdown(f"""
        <div class="info-strip">
          <div class="icard"><div class="lbl">📄 Documents</div><div class="val em">{len(uploaded_files)} PDFs</div></div>
          <div class="icard gold"><div class="lbl">💾 Total Size</div><div class="val gd">{round(total_size/1024,1)} KB</div></div>
          <div class="icard wh"><div class="lbl">🔒 Privacy</div><div class="val em">100% Local</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Files:** " + "  ·  ".join(f"`{f.name}`" for f in uploaded_files))

        if st.button("Train Adapter on All Documents", use_container_width=True):
            all_chunks = []
            for f in uploaded_files:
                with st.spinner(f"Processing {f.name}…"):
                    raw = extract_text_from_pdf(f.read())
                    if raw.strip():
                        ch = recursive_chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)
                        all_chunks.append(ch)
                        st.success(f"✓ {f.name} — {len(raw):,} chars, {len(ch)} chunks")
                    else:
                        st.warning(f"⚠ {f.name} — no text extracted, skipping")

            if not all_chunks:
                st.error("No text could be extracted from any uploaded PDF.")
                st.stop()

            import torch, gc
            st.cache_resource.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            total_ex = sum(len(c) for c in all_chunks)
            ft_bar = st.progress(0, text=f"Training on {len(all_chunks)} docs · {total_ex} examples…")
            with st.spinner("QLoRA training…"):
                adapter_path = run_qlora_finetuning(
                    all_chunks=all_chunks, lora_r=lora_r, lora_alpha=lora_alpha,
                    num_epochs=num_epochs, save_path=adapter_save_path,
                    progress_callback=lambda p: ft_bar.progress(p),
                )
            ft_bar.progress(1.0, text="Training complete ✓")
            st.success(f"✓ Adapter saved → `{adapter_path}`")
            st.info("Switch to **Summarize one PDF** mode to use this adapter.")

    # Single PDF summarize
    if uploaded_file:
        size_kb = round(uploaded_file.size / 1024, 1)
        st.markdown(f"""
        <div class="info-strip">
          <div class="icard"><div class="lbl">📄 Filename</div><div class="val">{uploaded_file.name}</div></div>
          <div class="icard gold"><div class="lbl">💾 Size</div><div class="val gd">{size_kb} KB</div></div>
          <div class="icard wh"><div class="lbl">🔒 Privacy</div><div class="val em">100% Local</div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        if st.button("Generate Summary", use_container_width=True):

            # ── Dynamic progress steps (5 when FT on, 4 when off) ──────────
            def mk_steps(states, labels, lines_done):
                dots = ""
                for i, (s, lbl) in enumerate(zip(states, labels)):
                    num = str(i+1).zfill(2)
                    dots += f'<div class="step"><div class="step-dot {s}">{num}</div><span class="step-label {s}">{lbl}</span></div>'
                    if i < len(states)-1:
                        ld = "done" if i < lines_done else ""
                        dots += f'<div class="step-line {ld}"></div>'
                return f'<div class="steps">{dots}</div>'

            if run_finetune:
                LBLS = ["Extract","Chunk","Fine-Tune","Load Model","Summarize"]
            else:
                LBLS = ["Extract","Chunk","Load Model","Summarize"]

            N = len(LBLS)

            def show_step(active_idx):
                states = []
                for i in range(N):
                    if i < active_idx:   states.append("done")
                    elif i == active_idx: states.append("active")
                    else:                states.append("")
                prog_box.markdown(mk_steps(states, LBLS, active_idx), unsafe_allow_html=True)

            prog_box = st.empty()
            show_step(0)

            # Step: Extract
            with st.spinner("Extracting text from PDF…"):
                pdf_bytes = uploaded_file.read()
                raw_text  = extract_text_from_pdf(pdf_bytes)

            if not raw_text.strip():
                st.error("⚠ Could not extract text — the PDF may be scanned or image-based.")
                st.stop()

            # Step: Chunk
            show_step(1)
            with st.spinner("Splitting into chunks…"):
                chunks = recursive_chunk_text(raw_text, chunk_size=chunk_size, overlap=chunk_overlap)
            st.success(f"✓ {len(raw_text):,} characters extracted → {len(chunks)} chunks")

            # Step: Fine-Tune (only when enabled)
            if run_finetune:
                show_step(2)
                import torch, gc
                st.cache_resource.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                ft_bar = st.progress(0, text="QLoRA fine-tuning…")
                with st.spinner("Training adapter on your document…"):
                    adapter_path = run_qlora_finetuning(
                        chunks=chunks, lora_r=lora_r, lora_alpha=lora_alpha,
                        num_epochs=num_epochs, save_path=adapter_save_path,
                        progress_callback=lambda p: ft_bar.progress(p),
                    )
                ft_bar.progress(1.0, text="Adapter saved ✓")
                st.success(f"✓ Adapter saved → `{adapter_save_path}`")
                adapter_to_load = adapter_path
                show_step(3)
            else:
                adapter_to_load = None
                show_step(2)

            # Step: Load Model
            with st.spinner("Loading TinyLlama… (cached after first load)"):
                model, tokenizer = get_model(use_4bit, adapter_to_load)

            # Step: Summarize
            show_step(N - 1)
            t0    = time.time()
            parts = []
            with st.spinner("Generating summary…"):
                for partial in summarize_chunks(model, tokenizer, chunks, max_new_tokens=max_new_tokens):
                    parts.append(partial)

            elapsed       = round(time.time() - t0, 1)
            final_summary = "\n\n".join(parts).strip()

            # All done
            show_step(N)

            if not final_summary:
                st.error("⚠ Model returned empty output. Try increasing Max Output Tokens or toggle off 4-bit.")
                st.stop()

            import html as _html
            safe = _html.escape(final_summary).replace("\n", "<br>")

            st.markdown(f"""
            <div class="sum-header"><div class="dot"></div><h3>Summary</h3></div>
            <div class="sum-box">
              <span class="stamp">TinyLlama · On-Device</span>
              {safe}
            </div>
            <div class="stats">
              <div class="schip"><span class="ico">⏱</span><div><div class="lbl">Time Taken</div><div class="val">{elapsed}s</div></div></div>
              <div class="schip"><span class="ico">📄</span><div><div class="lbl">Chunks</div><div class="val">{len(chunks)}</div></div></div>
              <div class="schip"><span class="ico">🔒</span><div><div class="lbl">Privacy</div><div class="val" style="font-size:.82rem;color:#34D399">On-Device</div></div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            st.download_button(
                "↓  Download Summary (.txt)",
                data=final_summary,
                file_name=f"{uploaded_file.name.replace('.pdf','')}_summary.txt",
                mime="text/plain",
            )

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — FINE-TUNE
# ════════════════════════════════════════════════════════════════════════════
with tab_tune:
    st.markdown('<div class="sec-lbl">Adapter manager</div>', unsafe_allow_html=True)
    adapter_dir = st.text_input("Adapter directory", value="./qlora_adapter", label_visibility="collapsed")

    if os.path.exists(adapter_dir):
        files = os.listdir(adapter_dir)
        if files:
            st.success(f"✓ {len(files)} file(s) found in `{adapter_dir}`")
            for f in files:
                fsize = os.path.getsize(os.path.join(adapter_dir, f)) / (1024*1024)
                st.markdown(f'<div class="tcard"><span class="num">FILE</span><div><div class="ttl">{f}</div><div class="dsc">{fsize:.2f} MB</div></div></div>', unsafe_allow_html=True)

            meta_path = os.path.join(adapter_dir, "training_meta.json")
            if os.path.exists(meta_path):
                import json
                with open(meta_path) as mf:
                    meta = json.load(mf)
                st.markdown(f"""
                <div class="tcard">
                  <span class="num">META</span>
                  <div>
                    <div class="ttl">Training details</div>
                    <div class="dsc">
                      Docs: {meta.get('num_documents','1')} &nbsp;·&nbsp;
                      Examples: {meta.get('num_examples','?')} &nbsp;·&nbsp;
                      Epochs: {meta.get('num_epochs','?')} &nbsp;·&nbsp;
                      LoRA r: {meta.get('lora_r','?')}
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("Adapter directory is empty — run a fine-tuning session first.")
    else:
        st.info("No adapter found at this path yet.")

    st.markdown('<div class="divdr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">How QLoRA works</div>', unsafe_allow_html=True)

    for num, ttl, dsc in [
        ("01","PDF → Examples","Document chunks become instruction-style training pairs."),
        ("02","Freeze base model","All 1.1B TinyLlama weights are frozen — zero full-model updates."),
        ("03","Inject adapters","Tiny rank-r matrices added to attention layers (q_proj, v_proj)."),
        ("04","4-bit training","Adapter trains in NF4 precision — fits in 4 GB VRAM with offload."),
        ("05","Save adapter","Only ~5 MB of adapter weights saved locally — fully portable."),
        ("06","Merge & infer","Adapter merges into base model at inference time for domain-aware output."),
    ]:
        st.markdown(f'<div class="tcard"><span class="num">{num}</span><div><div class="ttl">{ttl}</div><div class="dsc">{dsc}</div></div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="sec-lbl">Project overview</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
        <p style="font-size:.92rem;line-height:1.85;color:#8BA0BC;font-weight:300">
        <strong style="color:#E2EAF4;font-family:Fraunces,serif;font-style:italic">PrivSumAI</strong>
        is a fully offline, privacy-first document summarisation tool built for
        <strong style="color:#E2EAF4">M.Tech Data Science</strong> at RIT Kottayam.<br><br>
        Most AI summarization tools require uploading documents to cloud APIs.
        PrivSumAI solves this by running a quantized 1.1B LLM entirely on commodity
        hardware — no GPU required after setup, works fully offline.
        </p>
        """, unsafe_allow_html=True)
    with col_r:
        for lbl, val in [("👩‍💻 Author","Swathi P · 25MM2490"),("🎓 Guide","Dr. Jinesh N"),("🏛 Institution","RIT Kottayam · M.Tech DS")]:
            st.markdown(f'<div class="icard" style="margin-bottom:.6rem"><div class="lbl">{lbl}</div><div class="val" style="font-size:.88rem">{val}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divdr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">Tech stack</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for i, (ico, lbl, val) in enumerate([
        ("🦙","Base Model","TinyLlama-1.1B-Chat-v1.0"),
        ("⚡","Quantization","4-bit NF4 via bitsandbytes"),
        ("🎯","Fine-Tuning","QLoRA (PEFT) — adapter only"),
        ("📄","PDF Parsing","PyMuPDF + pdfplumber"),
        ("💻","UI","Streamlit"),
        ("🖥","Target HW","8–16 GB RAM · RTX 3050+"),
    ]):
        with (c1 if i % 2 == 0 else c2):
            st.markdown(f'<div class="tcard"><span style="font-size:1.1rem">{ico}</span><div><div class="ttl">{lbl}</div><div class="dsc">{val}</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divdr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-lbl">References</div>', unsafe_allow_html=True)
    for num, auth, ttl, venue in [
        ("[1]","Dettmers et al.","QLoRA: Efficient Finetuning of Quantized LLMs","NeurIPS 2023"),
        ("[2]","Zhang et al.","TinyLlama: An Open-Source Small Language Model","arXiv 2024"),
        ("[3]","Hu et al.","LoRA: Low-Rank Adaptation of Large Language Models","ICLR 2022"),
        ("[4]","Wolf et al.","Transformers: State-of-the-Art NLP","EMNLP 2020"),
        ("[5]","McMahan et al.","Communication-Efficient Learning from Decentralized Data","AISTATS 2017"),
    ]:
        st.markdown(f'<div class="tcard"><span class="num">{num}</span><div><div class="ttl">{ttl}</div><div class="dsc">{auth} · {venue}</div></div></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2.5rem;text-align:center;font-family:'IBM Plex Mono',monospace;
         font-size:.65rem;letter-spacing:.14em;text-transform:uppercase;color:#4A6480">
      Swathi P (25MM2490) · Guide: Dr. Jinesh N · RIT Kottayam
    </div>
    """, unsafe_allow_html=True)
