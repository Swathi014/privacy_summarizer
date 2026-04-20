"""
summarizer.py
─────────────
Gemma 3 document summarizer via Ollama.
Supports optional domain adapter for improved accuracy after fine-tuning.
"""

import requests
import json
import os

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "gemma3:4b"
ADAPTER_DIR = "./qlora_adapter"

MAX_SOURCE_CHARS = 12_000  # ~3 000 tokens; leaves room for structured output

DEFAULT_SYSTEM = (
    "You are a document analysis tool. Your only function is to extract and "
    "structure information from documents. You do not greet, explain, or converse. "
    "You output structured text only."
)

STOP_SEQS = [
    "I hope this", "Let me know", "Feel free",
    "Is there anything", "Please note", "I have summarized",
]


def check_ollama_running() -> bool:
    try:
        return requests.get("http://localhost:11434", timeout=3).status_code == 200
    except Exception:
        return False


def check_model_available(model: str = MODEL_NAME) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            return any(model in m["name"] for m in r.json().get("models", []))
    except Exception:
        pass
    return False


def get_adapter_system_prompt(adapter_path: str = ADAPTER_DIR) -> str:
    """Return the domain-adapted system prompt, or fall back to default."""
    p = os.path.join(adapter_path, "system_prompt.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return DEFAULT_SYSTEM


def get_adapter_info(adapter_path: str = ADAPTER_DIR) -> dict:
    """Return adapter metadata dict (config + meta merged), or {}."""
    info = {}
    for fname in ("adapter_config.json", "training_meta.json"):
        p = os.path.join(adapter_path, fname)
        if os.path.exists(p):
            with open(p) as f:
                info.update(json.load(f))
    return info


def has_adapter(adapter_path: str = ADAPTER_DIR) -> bool:
    return os.path.exists(os.path.join(adapter_path, "system_prompt.txt"))


def _build_user_prompt(doc_text: str) -> str:
    return (
        "Analyze the following document and produce a structured summary using "
        "EXACTLY these four sections. Do not add any text before or after them.\n\n"
        "**OVERVIEW**\n"
        "A concise paragraph: domain, core problem, and proposed solution.\n\n"
        "**METHODOLOGY**\n"
        "A paragraph: technical approach, tools, datasets, and architecture.\n\n"
        "**OUTCOMES**\n"
        "A paragraph: results, deliverables, or expected contributions.\n\n"
        "**KEYWORDS**\n"
        "Comma-separated list of 5-8 key technical terms.\n\n"
        "---\n"
        f"DOCUMENT:\n{doc_text}\n"
        "---\n\n"
        "**OVERVIEW**\n"
    )


def _render_html(text: str) -> str:
    """Convert **SECTION** markers to styled HTML for Streamlit rendering."""
    import re
    parts = re.compile(r'\*\*(OVERVIEW|METHODOLOGY|OUTCOMES|KEYWORDS)\*\*').split(text)
    if len(parts) <= 1:
        return (
            '<div class="sum-box"><div class="sum-section">'
            f'<p class="sum-content">{text}</p></div></div>'
        )
    html = '<div class="sum-box">'
    i = 1
    while i < len(parts):
        header  = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        html += (
            f'<div class="sum-section">'
            f'<span class="sum-label">{header}</span>'
            f'<p class="sum-content">{content}</p>'
            f'</div>'
        )
        i += 2
    return html + "</div>"


def generate_summary(
    text: str,
    max_tokens: int = 900,
    stream_placeholder=None,
    use_adapter: bool = True,
    adapter_path: str = ADAPTER_DIR,
) -> str:
    """
    Generate a structured 4-section summary using Gemma 3.

    If use_adapter=True and an adapter exists, the domain-adapted system
    prompt is used instead of the generic default.
    """
    safe_text  = text[:MAX_SOURCE_CHARS]
    sys_prompt = get_adapter_system_prompt(adapter_path) if use_adapter else DEFAULT_SYSTEM

    payload = {
        "model":  MODEL_NAME,
        "prompt": _build_user_prompt(safe_text),
        "system": sys_prompt,
        "stream": True,
        "options": {
            "num_predict":    max_tokens,
            "temperature":    0,
            "repeat_penalty": 1.15,
            "stop":           STOP_SEQS,
        },
    }

    accumulated = "**OVERVIEW**\n"

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                accumulated += chunk.get("response", "")

                if stream_placeholder:
                    stream_placeholder.markdown(
                        _render_html(accumulated + "▌"),
                        unsafe_allow_html=True,
                    )
                if chunk.get("done"):
                    break

        if stream_placeholder:
            stream_placeholder.markdown(_render_html(accumulated), unsafe_allow_html=True)

        return accumulated.strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot reach Ollama. Run:  ollama serve")
    except Exception as e:
        raise RuntimeError(f"Ollama generation error: {e}")
