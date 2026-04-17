"""
summarizer.py
─────────────
Ollama-based Gemma 3 summarizer — configured as a strict extraction tool,
NOT a conversational assistant.

Key design decisions:
  • temperature=0     → fully deterministic / greedy, kills chatbot-style rambling
  • Structured prompt → forces fixed-section output (Overview, Methodology, etc.)
  • Hard prefix seed  → model starts mid-sentence so it cannot open with filler phrases
  • stop sequences    → cuts off any trailing "I hope this helps!" tail
  • char limit raised → supports longer output without truncating source material
"""

import requests
import json

OLLAMA_URL  = "http://localhost:11434/api/generate"
MODEL_NAME  = "gemma3:4b"

# ~12 000 chars ≈ ~3 000 tokens; leaves room for a long structured output
MAX_SOURCE_CHARS = 12_000

# Stop if the model slips into chatbot mode
STOP_SEQUENCES = [
    "I hope this", "Let me know", "Feel free",
    "Is there anything", "Please note", "Note:",
    "I have summarized", "I summarized",
]


def _build_prompt(doc_text: str) -> str:
    return (
        "You are a document analysis tool. Your only function is to extract and "
        "structure information from documents. You do not greet, explain, or converse. "
        "You output structured text only.\n\n"
        "Analyze the following document and produce a structured summary using EXACTLY "
        "these four sections in this order. Do not add any section not listed. "
        "Do not write any text before or after the four sections.\n\n"
        "SECTION FORMAT:\n"
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


def check_ollama_running() -> bool:
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def check_model_available(model: str = MODEL_NAME) -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            return any(model in m for m in models)
    except Exception:
        pass
    return False


def _render_summary_html(text: str) -> str:
    import re
    section_pattern = re.compile(r'\*\*(OVERVIEW|METHODOLOGY|OUTCOMES|KEYWORDS)\*\*')
    parts = section_pattern.split(text)

    if len(parts) <= 1:
        return f'<div class="sum-box">{text}</div>'

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
    html += "</div>"
    return html


def generate_summary(
    text: str,
    max_tokens: int = 1000,
    stream_placeholder=None,
    model: str = MODEL_NAME,
) -> str:
    safe_text = text[:MAX_SOURCE_CHARS]
    prompt    = _build_prompt(safe_text)

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict":    max_tokens,
            "temperature":    0,
            "repeat_penalty": 1.15,
            "stop":           STOP_SEQUENCES,
        },
    }

    accumulated = "**OVERVIEW**\n"

    try:
        with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk  = json.loads(line)
                token  = chunk.get("response", "")
                accumulated += token

                if stream_placeholder is not None:
                    stream_placeholder.markdown(
                        _render_summary_html(accumulated + "▌"),
                        unsafe_allow_html=True,
                    )

                if chunk.get("done", False):
                    break

        if stream_placeholder is not None:
            stream_placeholder.markdown(
                _render_summary_html(accumulated),
                unsafe_allow_html=True,
            )

        return accumulated.strip()

    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            "Cannot reach Ollama. Make sure it is running: `ollama serve`"
        )
    except Exception as e:
        raise RuntimeError(f"Ollama generation error: {e}")
