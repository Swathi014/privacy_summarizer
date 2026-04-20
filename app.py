"""
PrivSumAI — Privacy-Preserving Document Summarizer
Powered by Gemma 3 (4B) via Ollama — fully on-device, zero cloud.
"""

import streamlit as st
import time
import json
import os

from utils.pdf_processor import extract_text_from_pdf, recursive_chunk_text
from utils.summarizer import (
    generate_summary, check_ollama_running, check_model_available,
    get_adapter_info, has_adapter, MODEL_NAME,
    ADAPTER_DIR,
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
  --bg:     #070C12;
  --card:   #0D1520;
  --card2:  #111C2B;
  --border: #1A2A3A;
  --em:     #10B981;
  --em2:    #34D399;
  --emdim:  #065F46;
  --txt:    #E2EAF4;
  --txt2:   #7A94B0;
  --txt3:   #4A6480;
  --err:    #F87171;
  --warn:   #FBBF24;
  --blue:   #60A5FA;
  --rad:    12px;
}

html, body, [data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: 'IBM Plex Sans', sans-serif !important;
  color: var(--txt) !important;
}
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"], section[data-testid="stSidebar"] { display:none!important; }

/* Header */
.hdr { text-align:center; padding:2.4rem 1rem 0.4rem; }
.eyebrow { font-size:.65rem; font-weight:600; letter-spacing:.18em; text-transform:uppercase; color:var(--em); margin-bottom:.5rem; }
.hdr h1 { font-family:'Fraunces',serif; font-size:2.8rem; font-weight:700; font-style:italic; color:var(--txt); margin:0; }
.hdr h1 span { color:var(--em); }
.hdr p { margin:.6rem auto 0; color:var(--txt2); font-size:.86rem; max-width:480px; line-height:1.5; }

/* Status pill */
.pill-wrap { text-align:center; margin:.9rem 0 1.2rem; }
.pill { display:inline-flex; align-items:center; gap:.4rem; padding:.28rem .85rem; border-radius:999px; font-size:.7rem; font-weight:600; }
.ok   { background:#042B1A; color:var(--em2);  border:1px solid var(--emdim); }
.warn { background:#1C1500; color:var(--warn); border:1px solid #92400E; }
.err  { background:#1C0A0A; color:var(--err);  border:1px solid #7F1D1D; }
.dot  { width:7px; height:7px; border-radius:50%; background:var(--em2); animation:pulse 1.4s ease-in-out infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

/* Cards */
.card { background:var(--card); border:1px solid var(--border); border-radius:var(--rad); padding:1.1rem 1.3rem; margin-bottom:.75rem; }
.card h4 { margin:0 0 .45rem; font-size:.72rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--em); }
.card p, .card li { font-size:.86rem; color:var(--txt2); line-height:1.65; margin:.2rem 0; }
.card code { font-family:'IBM Plex Mono',monospace; font-size:.78rem; background:var(--card2); border:1px solid var(--border); border-radius:4px; padding:.03rem .3rem; color:var(--em2); }
.card strong { color:var(--txt); }

/* File chip */
.chip { display:flex; align-items:center; gap:.6rem; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:.55rem .9rem; margin:.35rem 0; font-size:.85rem; }
.chip .n { color:var(--txt); font-weight:500; }
.chip .s { color:var(--txt3); font-size:.75rem; margin-left:auto; }

/* Buttons */
.stButton > button {
  background:linear-gradient(135deg,#047857,#10B981)!important;
  color:#fff!important; border:none!important; border-radius:10px!important;
  font-size:.92rem!important; font-weight:600!important;
  padding:.68rem 1.4rem!important; width:100%;
  transition:opacity .2s, transform .1s;
}
.stButton > button:hover  { opacity:.87; transform:translateY(-1px); }
.stButton > button:active { transform:translateY(0); }
.stButton > button:disabled { opacity:.35!important; }

[data-testid="stDownloadButton"] button {
  background:transparent!important; border:1px solid var(--border)!important;
  color:var(--txt2)!important; border-radius:8px!important; font-size:.8rem!important;
}
[data-testid="stDownloadButton"] button:hover { border-color:var(--em)!important; color:var(--em2)!important; }

/* Steps */
.steps { display:flex; gap:.45rem; margin:.8rem 0; }
.step  { flex:1; background:var(--card); border:1px solid var(--border); border-radius:8px; padding:.45rem .6rem; font-size:.68rem; font-weight:600; text-align:center; color:var(--txt3); }
.step.a { border-color:var(--em); color:var(--em2); background:#042B1A; }
.step.d { border-color:var(--emdim); color:var(--em); }

/* Summary box */
.sum-box { background:var(--card); border:1px solid var(--border); border-radius:var(--rad); overflow:hidden; margin-top:.5rem; }
.sum-section { padding:.85rem 1.4rem; border-bottom:1px solid var(--border); }
.sum-section:last-child { border-bottom:none; }
.sum-label { display:block; font-size:.62rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; color:var(--em); margin-bottom:.5rem; }
.sum-content { margin:0; font-size:.92rem; line-height:1.8; color:var(--txt); white-space:pre-wrap; }

/* Meta bar */
.meta { display:flex; gap:.9rem; margin:.6rem 0 .3rem; font-size:.72rem; color:var(--txt3); flex-wrap:wrap; }

/* Fine-tune steps */
.ft-step { display:flex; gap:.9rem; align-items:flex-start; margin:.7rem 0; }
.ft-num  { min-width:2rem; height:2rem; border-radius:50%; background:var(--emdim); color:var(--em2); font-weight:700; font-size:.82rem; display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.ft-body h5 { margin:0 0 .2rem; font-size:.84rem; font-weight:600; color:var(--txt); }
.ft-body p  { margin:0; font-size:.8rem; color:var(--txt2); line-height:1.55; }

/* Adapter badge */
.abadge { display:inline-flex; align-items:center; gap:.35rem; background:#0C1C35; border:1px solid #1E3A5F; border-radius:8px; padding:.32rem .7rem; font-size:.72rem; color:var(--blue); }

/* ROUGE bars */
.rbar-wrap { margin:.45rem 0; }
.rbar-label { font-size:.72rem; font-weight:600; color:var(--txt3); margin-bottom:.18rem; letter-spacing:.06em; }
.rbar-bg { background:var(--card2); border-radius:6px; height:22px; border:1px solid var(--border); overflow:hidden; }
.rbar-fill { height:100%; border-radius:6px; background:linear-gradient(90deg,#047857,#34D399); display:flex; align-items:center; padding-left:7px; font-size:.68rem; font-weight:700; color:#fff; }

[data-testid="stAlert"] { border-radius:8px!important; font-size:.82rem!important; }
[data-testid="stFileUploaderDropzone"] { background:var(--card)!important; border:2px dashed var(--border)!important; border-radius:var(--rad)!important; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div class="eyebrow">🔒 Privacy-Preserving · On-Device AI · Gemma 3</div>
  <h1>Priv<span>Sum</span>AI</h1>
  <p>Structured document summarization — fully local, zero cloud, zero data leakage.</p>
</div>
""", unsafe_allow_html=True)


# ── Ollama status ──────────────────────────────────────────────────────────────
ollama_ok = check_ollama_running()
model_ok  = check_model_available(MODEL_NAME) if ollama_ok else False

if ollama_ok and model_ok:
    st.markdown(f'<div class="pill-wrap"><span class="pill ok"><span class="dot"></span> Ollama running &nbsp;·&nbsp; {MODEL_NAME} ready</span></div>', unsafe_allow_html=True)
elif ollama_ok:
    st.markdown(f'<div class="pill-wrap"><span class="pill warn">⚠ {MODEL_NAME} not found — run: ollama pull {MODEL_NAME}</span></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="pill-wrap"><span class="pill err">✕ Ollama offline — run: ollama serve</span></div>', unsafe_allow_html=True)

# ── Adapter status (shown if a trained adapter exists) ────────────────────────
_info = get_adapter_info(ADAPTER_DIR)
_has  = has_adapter(ADAPTER_DIR)
if _has and _info.get("domain"):
    st.markdown(
        f'<div style="text-align:center;margin-bottom:.5rem;">'
        f'<span class="abadge">🔬 Domain adapter active &nbsp;·&nbsp; '
        f'<strong>{_info["domain"]}</strong> &nbsp;·&nbsp; {_info.get("trained_at","")[:10]}'
        f'</span></div>',
        unsafe_allow_html=True,
    )


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_s, tab_ft, tab_ev, tab_ab = st.tabs(["📄 Summarize", "🔬 Fine-Tune", "📊 Evaluate", "ℹ️ About"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SUMMARIZE
# ══════════════════════════════════════════════════════════════════════════════
with tab_s:

    # Adapter toggle (only shown when a trained adapter is present)
    use_adapter = False
    if _has:
        use_adapter = st.toggle(
            f"🔬 Use domain adapter  ({_info.get('domain','custom')})",
            value=True,
            help="The fine-tuned adapter makes summaries more accurate for your specific document domain.",
        )
    else:
        st.caption("💡 No domain adapter yet. Use the **Fine-Tune** tab to train one for better accuracy.")

    # Length slider
    max_tok = st.slider("Output length (tokens)", 300, 2000, 900, 100,
                        help="~900 tokens gives a thorough 4-section summary. 1 token ≈ 4 characters.")
    st.caption(f"📏 **{max_tok}** tokens max  ·  ~{max_tok * 4} characters")

    # Upload
    st.markdown("**Upload document**")
    pdf = st.file_uploader("Drop a PDF", type=["pdf"], label_visibility="collapsed")

    if pdf:
        st.markdown(f'<div class="chip"><span>📄</span><span class="n">{pdf.name}</span><span class="s">{round(pdf.size/1024,1)} KB</span></div>', unsafe_allow_html=True)

        if st.button("⚡  Generate Structured Summary", use_container_width=True,
                     disabled=not (ollama_ok and model_ok), key="b_sum"):

            step_ph = st.empty()

            def _steps(a):
                labels = ["① Extract text", "② Send to Gemma 3", "③ Build summary"]
                h = '<div class="steps">'
                for i, l in enumerate(labels):
                    c = "a" if i == a else ("d" if i < a else "step")
                    h += f'<div class="step {c}">{l}</div>'
                step_ph.markdown(h + "</div>", unsafe_allow_html=True)

            _steps(0)
            with st.spinner("Extracting text…"):
                raw = extract_text_from_pdf(pdf.read())

            if not raw.strip():
                st.error("No text found. The PDF may be a scanned image.")
                st.stop()

            st.info(f"✅ **{len(raw):,}** characters · **{len(raw.split()):,}** words extracted")
            _steps(1)

            st.markdown("**Summary**")
            box = st.empty()
            t0  = time.time()

            try:
                summary = generate_summary(raw, max_tokens=max_tok,
                                           stream_placeholder=box,
                                           use_adapter=use_adapter)
                elapsed = round(time.time() - t0, 1)
                step_ph.empty()

                mode_tag = f"🔬 adapter: {_info.get('domain','')}" if use_adapter and _has else "default prompt"
                st.markdown(
                    f'<div class="meta">'
                    f'<span>⏱ {elapsed}s</span>'
                    f'<span>·</span>'
                    f'<span>📝 ~{len(summary.split())} words</span>'
                    f'<span>·</span>'
                    f'<span>🔒 {MODEL_NAME} on-device</span>'
                    f'<span>·</span>'
                    f'<span>{mode_tag}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.download_button("⬇  Download summary (.txt)", data=summary,
                                   file_name=pdf.name.replace(".pdf","_summary.txt"),
                                   mime="text/plain")

            except ConnectionError as e:
                step_ph.empty(); st.error(str(e))
            except Exception as e:
                step_ph.empty(); st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FINE-TUNE  (detailed, educational walkthrough)
# ══════════════════════════════════════════════════════════════════════════════
with tab_ft:

    # ── Explainer section ──────────────────────────────────────────────────────
    st.markdown("### What is Fine-Tuning?")
    st.markdown("""
<div class="card">
  <h4>The simple idea</h4>
  <p>Gemma 3 is a general-purpose AI that has learned from billions of documents on
  the internet. It is good at many things, but not <em>specialised</em> in your domain.</p>
  <p><strong>Fine-tuning</strong> is like enrolling Gemma 3 in a short, focused course on
  <em>your</em> domain — showing it your documents so it learns the vocabulary, writing
  style, and what a good summary looks like for your field.</p>
  <p>After fine-tuning, summaries of medical papers look like medical summaries, legal
  contracts get legal-style summaries, and so on.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### What is QLoRA?")
    st.markdown("""
<div class="card">
  <h4>Q — Quantization  (saves memory)</h4>
  <p>Normally Gemma 3 stores each number in its brain with 32 bits of precision.
  Quantization shrinks this to <strong>4 bits</strong> — like compressing a photo from RAW to JPEG.
  You barely notice the quality loss, but the model now fits in <strong>~90% less memory</strong>.
  A model that needed 16 GB now needs ~2 GB.</p>
</div>
<div class="card">
  <h4>LoRA — Low-Rank Adaptation  (saves time)</h4>
  <p>Normally fine-tuning would update <em>all</em> of Gemma 3's billions of parameters —
  that takes hours and a powerful GPU.</p>
  <p>LoRA is smarter: instead of changing everything, it <strong>attaches tiny extra
  matrices</strong> (called adapters) to key layers of the model.
  Only these small matrices are trained. The base model is frozen and untouched.</p>
  <p>Mathematically: &nbsp; <code>W_new = W_base + A × B</code> &nbsp;
  where A and B are tiny matrices of rank r (you choose r = 4, 8, 16…).
  The higher r, the more the adapter can learn — but also the larger it becomes.</p>
  <p><strong>Result:</strong> the adapter file is ~2 KB vs ~4 GB for the full model.
  It trains in seconds on a laptop CPU.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### The 3-Step Fine-Tuning Process")
    st.markdown("""
<div class="card">
  <div class="ft-step">
    <div class="ft-num">1</div>
    <div class="ft-body">
      <h5>ANALYSE — Read the domain documents</h5>
      <p>Gemma 3 reads a sample of your uploaded PDFs and extracts a <strong>domain
      profile</strong>: the field name, key technical terms, the writing style, and
      guidance on how summaries should be written for this domain.
      This is analogous to the forward pass through training data.</p>
    </div>
  </div>
  <div class="ft-step">
    <div class="ft-num">2</div>
    <div class="ft-body">
      <h5>ADAPT — Build the adapter</h5>
      <p>The domain profile is used to construct an optimised <strong>system-prompt
      adapter</strong> — the functional equivalent of QLoRA's A×B weight matrices.
      Instead of numeric weights it is compact natural-language text that precisely
      steers Gemma 3's behaviour for your domain.</p>
    </div>
  </div>
  <div class="ft-step">
    <div class="ft-num">3</div>
    <div class="ft-body">
      <h5>SAVE — Store the adapter artefacts</h5>
      <p>The adapter (~2 KB) is saved to <code>./qlora_adapter/</code> alongside
      <code>adapter_config.json</code> (LoRA-compatible schema), <code>training_meta.json</code>,
      and an Ollama <code>Modelfile</code>.
      <strong>No raw document text is stored</strong> — your data stays private.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Run Fine-Tuning on Your Documents")

    ft_files = st.file_uploader(
        "Upload one or more domain PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="ft_upload",
    )

    if ft_files:
        for f in ft_files:
            st.markdown(
                f'<div class="chip"><span>📄</span><span class="n">{f.name}</span>'
                f'<span class="s">{round(f.size/1024,1)} KB</span></div>',
                unsafe_allow_html=True,
            )

        col1, col2 = st.columns(2)
        with col1:
            lora_r = st.select_slider(
                "LoRA rank  (r)",
                options=[4, 8, 16, 32], value=8,
                help="Higher r = more expressive adapter, slightly larger file. r=8 is a good default.",
            )
        with col2:
            lora_alpha = st.select_slider(
                "LoRA alpha  (α)",
                options=[8, 16, 32, 64], value=16,
                help="Scaling factor. Convention: α = 2 × r. Controls how strongly the adapter steers the model.",
            )

        st.caption(
            f"🔧 Effective scaling factor: **α/r = {lora_alpha/lora_r:.1f}**  "
            f"· Higher = adapter has stronger influence on the base model."
        )

        if st.button("🔬  Start Fine-Tuning", use_container_width=True,
                     disabled=not (ollama_ok and model_ok), key="b_ft"):

            from utils.qlora_trainer import run_qlora_finetuning

            prog   = st.progress(0.0)
            status = st.empty()

            # Live step indicators
            step1_ph = st.empty()
            step2_ph = st.empty()
            step3_ph = st.empty()

            def _show_step(n, label, state):
                # state: "waiting" | "running" | "done"
                icon  = {"waiting": "⏳", "running": "🔄", "done": "✅"}[state]
                color = {"waiting": "var(--txt3)", "running": "var(--warn)", "done": "var(--em2)"}[state]
                ph    = [step1_ph, step2_ph, step3_ph][n - 1]
                ph.markdown(
                    f'<div style="padding:.4rem .8rem;background:var(--card);border:1px solid var(--border);'
                    f'border-radius:8px;margin:.25rem 0;font-size:.83rem;color:{color};">'
                    f'{icon} &nbsp; <strong>Step {n}:</strong> {label}</div>',
                    unsafe_allow_html=True,
                )

            _show_step(1, "Extracting and chunking documents…", "running")
            _show_step(2, "Analysing domain (waiting)", "waiting")
            _show_step(3, "Building & saving adapter (waiting)", "waiting")

            try:
                # Extract chunks
                all_chunks = []
                for f in ft_files:
                    raw = extract_text_from_pdf(f.read())
                    cks = recursive_chunk_text(raw, chunk_size=800, overlap=80)
                    all_chunks.append(cks)

                total_chunks = sum(len(c) for c in all_chunks)
                _show_step(1, f"Extracted {total_chunks} chunks from {len(ft_files)} file(s) ✓", "done")
                _show_step(2, "Analysing domain vocabulary and writing style…", "running")
                prog.progress(0.1)

                # Track progress from qlora_trainer
                def _cb(frac: float):
                    prog.progress(min(frac, 1.0))
                    if frac < 0.4:
                        pass  # still analysing
                    elif frac < 0.7:
                        _show_step(2, "Domain profile extracted ✓", "done")
                        _show_step(3, "Generating domain-adapted system prompt…", "running")
                    else:
                        _show_step(3, "Saving adapter artefacts…", "running")

                adapter_path = run_qlora_finetuning(
                    all_chunks=all_chunks,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    save_path=ADAPTER_DIR,
                    progress_callback=_cb,
                )

                prog.progress(1.0)
                _show_step(3, "Adapter saved ✓", "done")

                # Show what was learned
                from utils.qlora_trainer import load_adapter_config
                cfg = load_adapter_config(ADAPTER_DIR)

                domain = cfg.get("domain", "N/A")
                terms  = ", ".join(cfg.get("key_terms", [])[:6])
                status.success("✅ Fine-tuning complete! The domain adapter is now active.")

                st.markdown(f"""
<div class="card">
  <h4>What the Adapter Learned</h4>
  <p><strong>Domain:</strong> {domain}</p>
  <p><strong>Key terms identified:</strong> {terms or 'N/A'}</p>
  <p><strong>LoRA r / α:</strong> {lora_r} / {lora_alpha} &nbsp;·&nbsp; scaling = {lora_alpha/lora_r:.1f}</p>
  <p><strong>Documents used:</strong> {len(ft_files)} &nbsp;·&nbsp; Chunks: {total_chunks}</p>
  <p><strong>Adapter size:</strong> ~2 KB (vs ~4 GB full model weights)</p>
  <p>Switch to the <strong>Summarize</strong> tab — the adapter is automatically active.</p>
</div>""", unsafe_allow_html=True)

                # Offer downloads
                col_a, col_b = st.columns(2)
                with col_a:
                    sp = os.path.join(ADAPTER_DIR, "system_prompt.txt")
                    if os.path.exists(sp):
                        with open(sp) as f:
                            st.download_button("⬇  Download adapter (.txt)", data=f.read(),
                                               file_name="system_prompt.txt", mime="text/plain")
                with col_b:
                    mf = os.path.join(ADAPTER_DIR, "Modelfile")
                    if os.path.exists(mf):
                        with open(mf) as f:
                            st.download_button("⬇  Download Modelfile (Ollama)", data=f.read(),
                                               file_name="Modelfile", mime="text/plain")

            except Exception as e:
                status.error(f"Fine-tuning failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
with tab_ev:
    st.markdown("""
<div class="card">
  <h4>ROUGE Evaluation — What does it measure?</h4>
  <p><strong>ROUGE</strong> (Recall-Oriented Understudy for Gisting Evaluation) compares a
  generated summary against a human-written reference to measure how much content overlaps.</p>
  <p><strong>ROUGE-1:</strong> overlap of individual words (unigrams)</p>
  <p><strong>ROUGE-2:</strong> overlap of two-word pairs (bigrams) — captures phrasing quality</p>
  <p><strong>ROUGE-L:</strong> longest common subsequence — captures sentence-level coherence</p>
  <p>All scores are F1 values (0 = no overlap · 1 = perfect match). Scores above 0.4 are
  considered good for abstractive summarization.</p>
</div>
""", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        ev_pdf = st.file_uploader("Source PDF", type=["pdf"], key="ev_pdf")
    with col_b:
        ev_ref = st.file_uploader("Reference summary (.txt)", type=["txt"], key="ev_ref")

    ev_adapter = st.toggle("Use domain adapter", value=_has,
                           help="Compare with/without adapter to see how fine-tuning improved accuracy.")
    ev_tok     = st.slider("Max tokens", 300, 1500, 900, 100, key="ev_tok")

    if st.button("📊  Run Evaluation", use_container_width=True,
                 disabled=not (ollama_ok and model_ok and ev_pdf and ev_ref), key="b_ev"):

        from evaluate import evaluate_summary as rouge_eval

        with st.spinner("Extracting PDF text…"):
            raw_ev = extract_text_from_pdf(ev_pdf.read())

        with st.spinner("Generating summary with Gemma 3…"):
            t0 = time.time()
            try:
                gen = generate_summary(raw_ev, max_tokens=ev_tok, use_adapter=ev_adapter)
                elapsed = round(time.time() - t0, 1)
            except Exception as e:
                st.error(f"Summary generation failed: {e}"); st.stop()

        ref_text = ev_ref.read().decode("utf-8", errors="replace")

        try:
            scores = rouge_eval(gen, ref_text)
        except ImportError:
            st.error("Install rouge-score:  pip install rouge-score"); st.stop()

        st.success(f"Evaluation done in {elapsed}s")

        for metric, val in scores.items():
            pct = int(val * 100)
            st.markdown(
                f'<div class="rbar-wrap">'
                f'<div class="rbar-label">{metric.upper()}  {val:.4f}</div>'
                f'<div class="rbar-bg">'
                f'<div class="rbar-fill" style="width:{max(pct,2)}%">{pct}%</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        r1 = scores.get("rouge1", 0)
        interp = (
            "🟢 **Strong** — the summary captures most key content."     if r1 >= 0.5 else
            "🟡 **Moderate** — key topics covered, some detail missed."  if r1 >= 0.3 else
            "🔴 **Low** — try fine-tuning the model on domain documents."
        )
        st.markdown(f"**Interpretation:** {interp}")

        result_json = json.dumps({"scores": scores, "elapsed_s": elapsed,
                                   "adapter_used": ev_adapter, "summary": gen}, indent=2)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇  Results (.json)", data=result_json,
                               file_name="eval_results.json", mime="application/json")
        with col2:
            st.download_button("⬇  Generated summary (.txt)", data=gen,
                               file_name="generated_summary.txt", mime="text/plain")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_ab:
    st.markdown("""
<div class="card">
  <h4>What is PrivSumAI?</h4>
  <p>A privacy-preserving document summarization tool. Your PDFs are processed entirely
  on your local machine using <strong>Gemma 3 4B via Ollama</strong> — nothing is sent to
  any server or cloud API. This project demonstrates resource-efficient GenAI on edge devices
  for privacy-sensitive sectors like healthcare and law.</p>
</div>

<div class="card">
  <h4>How it works</h4>
  <p><strong>1. Inference:</strong> Gemma 3 4B runs locally via Ollama in 4-bit quantised
  mode — uses ~90% less memory than a full-precision model.</p>
  <p><strong>2. Fine-Tuning (QLoRA simulation):</strong> Upload domain PDFs in the Fine-Tune
  tab. Gemma 3 analyses them to extract a domain profile, then generates a compact
  system-prompt adapter (~2 KB) that steers future summaries toward your domain vocabulary
  and style — the functional equivalent of QLoRA weight deltas.</p>
  <p><strong>3. Evaluation:</strong> ROUGE-1/2/L scores compare generated summaries against
  human references to measure quality before and after fine-tuning.</p>
</div>

<div class="card">
  <h4>Getting Started</h4>
  <p>1. Install Ollama: &nbsp;<code>ollama.com</code></p>
  <p>2. Pull the model: &nbsp;<code>ollama pull gemma3:4b</code></p>
  <p>3. Start Ollama: &nbsp;<code>ollama serve</code></p>
  <p>4. Install Python deps: &nbsp;<code>pip install -r requirements.txt</code></p>
  <p>5. Launch: &nbsp;<code>streamlit run app.py</code></p>
</div>

<div class="card">
  <h4>Tech Stack</h4>
  <p>Streamlit &nbsp;·&nbsp; Gemma 3 4B &nbsp;·&nbsp; Ollama &nbsp;·&nbsp;
  PyMuPDF &nbsp;·&nbsp; pdfplumber &nbsp;·&nbsp; rouge-score</p>
  <p><em>No PyTorch · No HuggingFace Hub · No GPU required for inference.</em></p>
</div>
""", unsafe_allow_html=True)
