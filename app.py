import streamlit as st
import time
from utils.pdf_processor import extract_text_from_pdf
from utils.summarizer import (
    generate_summary, check_ollama_running,
    check_model_available, MODEL_NAME,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PrivSumAI",
    page_icon="🔒",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Fraunces:ital,wght@1,700&display=swap');

:root {
  --bg:      #070C12;
  --card:    #0D1520;
  --card2:   #111C2B;
  --border:  #1A2A3A;
  --em:      #10B981;
  --em2:     #34D399;
  --em-dim:  #065F46;
  --txt:     #E2EAF4;
  --txt2:    #7A94B0;
  --txt3:    #4A6480;
  --err:     #F87171;
  --warn:    #FBBF24;
  --radius:  12px;
}

html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--txt) !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Header ── */
.hdr {
  text-align: center;
  padding: 2.8rem 1rem 0.5rem;
  margin-bottom: 0.5rem;
}
.hdr-eyebrow {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--em);
  margin-bottom: 0.6rem;
}
.hdr h1 {
  font-family: 'Fraunces', serif;
  font-size: 3rem;
  font-weight: 700;
  font-style: italic;
  color: var(--txt);
  margin: 0;
  line-height: 1;
}
.hdr h1 .accent { color: var(--em); }
.hdr-sub {
  margin-top: 0.7rem;
  color: var(--txt2);
  font-size: 0.88rem;
  max-width: 480px;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.5;
}

/* ── Status pill ── */
.pill-wrap { text-align: center; margin: 1rem 0 1.5rem; }
.pill {
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
  padding: 0.32rem 0.9rem;
  border-radius: 999px;
  font-size: 0.73rem;
  font-weight: 600;
  letter-spacing: 0.04em;
}
.pill-ok  { background:#042B1A; color:var(--em2); border:1px solid var(--em-dim); }
.pill-err { background:#1C0A0A; color:var(--err); border:1px solid #7F1D1D; }
.pill-warn{ background:#1C1500; color:var(--warn);border:1px solid #92400E; }
.dot-live {
  width:7px; height:7px; border-radius:50%; background:var(--em2);
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* ── Control row ── */
.ctrl-row {
  display: flex;
  align-items: flex-end;
  gap: 1rem;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.2rem;
  margin-bottom: 1rem;
}
.ctrl-label {
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--txt3);
  margin-bottom: 0.3rem;
}
.token-badge {
  display: inline-block;
  background: var(--em-dim);
  color: var(--em2);
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0.15rem 0.55rem;
  border-radius: 6px;
  margin-left: 0.4rem;
}

/* ── Upload zone ── */
[data-testid="stFileUploaderDropzone"] {
  background: var(--card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.2s;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--em) !important;
}

/* ── File chip ── */
.file-chip {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.7rem 1rem;
  margin: 0.6rem 0;
  font-size: 0.88rem;
}
.file-chip .fname { color: var(--txt); font-weight: 500; }
.file-chip .fsize { color: var(--txt3); font-size: 0.78rem; margin-left: auto; }

/* ── Generate button ── */
.stButton > button {
  background: linear-gradient(135deg, #047857, #10B981) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  padding: 0.72rem 1.5rem !important;
  width: 100%;
  letter-spacing: 0.03em;
  transition: opacity .2s, transform .1s;
}
.stButton > button:hover  { opacity:.88; transform:translateY(-1px); }
.stButton > button:active { transform:translateY(0); }
.stButton > button:disabled { opacity:.4 !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
  background: var(--em) !important;
}

/* ── Step tracker ── */
.step-row {
  display: flex;
  gap: 0.5rem;
  margin: 1rem 0;
}
.step {
  flex: 1;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.5rem 0.7rem;
  font-size: 0.72rem;
  font-weight: 600;
  text-align: center;
  color: var(--txt3);
  letter-spacing: 0.05em;
}
.step.active {
  border-color: var(--em);
  color: var(--em2);
  background: #042B1A;
}
.step.done {
  border-color: var(--em-dim);
  color: var(--em);
}

/* ── Summary box ── */
.sum-box {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 0.4rem 0;
  margin-top: 0.5rem;
  overflow: hidden;
}
.sum-section {
  padding: 1rem 1.6rem;
  border-bottom: 1px solid var(--border);
}
.sum-section:last-child { border-bottom: none; }
.sum-label {
  display: block;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--em);
  margin-bottom: 0.55rem;
}
.sum-content {
  margin: 0;
  font-size: 0.95rem;
  line-height: 1.8;
  color: var(--txt);
  white-space: pre-wrap;
}

/* ── Meta bar ── */
.meta-bar {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-top: 0.7rem;
  padding: 0.5rem 0.2rem;
  font-size: 0.75rem;
  color: var(--txt3);
}
.meta-bar .m-item { display:flex; align-items:center; gap:0.3rem; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--txt2) !important;
  border-radius: 8px !important;
  font-size: 0.83rem !important;
  margin-top: 0.3rem;
  transition: border-color .2s, color .2s;
}
[data-testid="stDownloadButton"] button:hover {
  border-color: var(--em) !important;
  color: var(--em2) !important;
}

/* ── Alert / info ── */
[data-testid="stAlert"] {
  border-radius: 8px !important;
  font-size: 0.84rem !important;
}

/* ── About card ── */
.about-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.4rem 1.6rem;
  margin-bottom: 0.9rem;
}
.about-card h4 {
  margin: 0 0 0.6rem;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--em);
}
.about-card p, .about-card li {
  font-size: 0.9rem;
  color: var(--txt2);
  line-height: 1.65;
  margin: 0.25rem 0;
}
.about-card code {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.82rem;
  background: var(--card2);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 0.05rem 0.35rem;
  color: var(--em2);
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="hdr-eyebrow">🔒 Privacy-Preserving · On-Device AI · Gemma 3</div>
  <h1>Priv<span class="accent">Sum</span>AI</h1>
  <p class="hdr-sub">Structured document summarization — fully local, zero cloud, zero data leakage.</p>
</div>
""", unsafe_allow_html=True)

# ── Ollama status ──────────────────────────────────────────────────────────────
ollama_ok = check_ollama_running()
model_ok  = check_model_available(MODEL_NAME) if ollama_ok else False

if ollama_ok and model_ok:
    st.markdown(f"""
    <div class="pill-wrap">
      <span class="pill pill-ok">
        <span class="dot-live"></span> Ollama running &nbsp;·&nbsp; {MODEL_NAME} ready
      </span>
    </div>""", unsafe_allow_html=True)
elif ollama_ok and not model_ok:
    st.markdown(f"""
    <div class="pill-wrap">
      <span class="pill pill-warn">
        ⚠ Ollama running but {MODEL_NAME} not found — run: ollama pull {MODEL_NAME}
      </span>
    </div>""", unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="pill-wrap">
      <span class="pill pill-err">
        ✕ Ollama not running — open a terminal and run: ollama serve
      </span>
    </div>""", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_summ, tab_about = st.tabs(["📄 Summarize", "ℹ️ About"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_summ:

    # Output length control
    st.markdown('<div class="ctrl-label" style="margin-bottom:.3rem">Output length</div>', unsafe_allow_html=True)
    max_new_tokens = st.slider(
        "Max output tokens",
        min_value=300,
        max_value=2000,      # ← raised from 800
        value=900,           # ← sensible default for 4-section structured output
        step=100,
        label_visibility="collapsed",
        help="Controls how much the model writes. 900–1200 tokens gives a thorough structured summary.",
    )
    st.caption(f"🔢 **{max_new_tokens}** tokens max output · ~{max_new_tokens * 4} characters")

    # File uploader
    st.markdown('<div class="ctrl-label" style="margin-top:1rem;margin-bottom:.4rem">Upload document</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.markdown(f"""
        <div class="file-chip">
          <span>📄</span>
          <span class="fname">{uploaded_file.name}</span>
          <span class="fsize">{round(uploaded_file.size / 1024, 1)} KB</span>
        </div>""", unsafe_allow_html=True)

        ready = ollama_ok and model_ok
        if st.button("⚡  Generate Structured Summary", use_container_width=True, disabled=not ready):

            # ── Step tracker ──
            step_placeholder = st.empty()

            def show_steps(active: int):
                steps = ["① Extract", "② Analyse", "③ Structure"]
                html  = '<div class="step-row">'
                for i, s in enumerate(steps):
                    cls = "active" if i == active else ("done" if i < active else "step")
                    html += f'<div class="step {cls}">{s}</div>'
                html += "</div>"
                step_placeholder.markdown(html, unsafe_allow_html=True)

            # Step 1 — Extract
            show_steps(0)
            with st.spinner("Extracting text from PDF…"):
                pdf_bytes = uploaded_file.read()
                raw_text  = extract_text_from_pdf(pdf_bytes)

            if not raw_text.strip():
                st.error("Could not extract text. The PDF may be image-only (scanned).")
                st.stop()

            word_count_src = len(raw_text.split())
            show_steps(1)
            st.info(f"✅ Extracted **{len(raw_text):,}** characters · **{word_count_src:,}** words")

            # Step 2 — Generate
            show_steps(2)
            st.markdown('<div class="ctrl-label" style="margin-top:1rem">Summary</div>', unsafe_allow_html=True)
            stream_box = st.empty()

            t0 = time.time()
            try:
                final_summary = generate_summary(
                    raw_text,
                    max_tokens=max_new_tokens,
                    stream_placeholder=stream_box,
                )
                elapsed = round(time.time() - t0, 1)

                # Done — clear step tracker
                step_placeholder.empty()

                # Meta bar
                word_count_out = len(final_summary.split())
                st.markdown(f"""
                <div class="meta-bar">
                  <span class="m-item">⏱ {elapsed}s</span>
                  <span class="m-item">·</span>
                  <span class="m-item">📝 ~{word_count_out} words</span>
                  <span class="m-item">·</span>
                  <span class="m-item">🔒 {MODEL_NAME} · fully on-device</span>
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "⬇  Download summary as .txt",
                    data=final_summary,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_summary.txt",
                    mime="text/plain",
                )

            except ConnectionError as e:
                step_placeholder.empty()
                st.error(str(e))
            except Exception as e:
                step_placeholder.empty()
                st.error(f"Something went wrong: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class="about-card">
      <h4>What is PrivSumAI?</h4>
      <p>A privacy-preserving document summarization tool. Your PDFs are processed entirely
      on your local machine — nothing is sent to any server or cloud API.</p>
    </div>

    <div class="about-card">
      <h4>How it works</h4>
      <p>PrivSumAI uses <strong>Gemma 3 4B</strong> running locally via
      <a href="https://ollama.com" target="_blank" style="color:#34D399">Ollama</a>.
      The model is configured as a strict structured extraction tool — it outputs four
      fixed sections (Overview, Methodology, Outcomes, Keywords) with zero conversational filler.</p>
    </div>

    <div class="about-card">
      <h4>Getting started</h4>
      <p>1. Install Ollama from <code>ollama.com</code></p>
      <p>2. Open a terminal and run:</p>
      <p>&nbsp;&nbsp;&nbsp;<code>ollama pull gemma3</code></p>
      <p>&nbsp;&nbsp;&nbsp;<code>ollama serve</code></p>
      <p>3. Upload a PDF and click Generate.</p>
    </div>

    <div class="about-card">
      <h4>Tech stack</h4>
      <p>Streamlit · Gemma 3 4B · Ollama · PyMuPDF · QLoRA (fine-tuning phase)</p>
    </div>
    """, unsafe_allow_html=True)
