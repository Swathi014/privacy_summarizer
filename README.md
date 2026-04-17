# PrivSumAI — Ollama Edition

Fast, private, on-device PDF summarizer using **Gemma 3 via Ollama**.  
No GPU required. No HuggingFace model loading. Responses stream in seconds.

---

## ⚡ Quick Setup

### 1. Install Ollama
Download from https://ollama.com and install it.

### 2. Pull Gemma 3 & start the server
Open a terminal:
```bash
ollama pull gemma3
ollama serve
```
Keep this terminal open while using the app.

### 3. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Why Ollama?

| | Old (HuggingFace transformers) | New (Ollama) |
|---|---|---|
| Startup time | ~1–2 minutes | ~1–2 seconds |
| RAM usage | Very high (loads full model in Python) | Managed natively by Ollama |
| GPU required | Yes (for reasonable speed) | No |
| First token delay | Long | Near-instant |

---

## File Structure

```
privacy_summarizer/
├── app.py                    # Streamlit UI (no sidebar)
├── requirements.txt
└── utils/
    ├── summarizer.py         # Ollama API client with streaming
    └── pdf_processor.py      # PDF text extraction + chunking
```
