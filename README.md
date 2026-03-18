# 🔒 PrivSumAI — Privacy-Preserving Document Summarizer

> **M.Tech Data Science Project** | RIT Kottayam  
> Presented by: Swathi P (25MM2490) | Guide: Dr. Jinesh N

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TinyLlama](https://img.shields.io/badge/Model-TinyLlama--1.1B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is PrivSumAI?

**PrivSumAI** is a fully offline, privacy-first document summarization tool. Unlike tools that upload your PDF to cloud APIs (ChatGPT, Gemini), PrivSumAI runs a quantized 1.1B language model **entirely on your local machine** — no internet needed after setup, no data ever leaves your device.

---

## Features

- 100% On-Device — No cloud, no API keys, no data leakage
- 4-bit NF4 Quantization — TinyLlama-1.1B fits in ~800 MB VRAM
- QLoRA Fine-Tuning — Train a custom adapter on your own documents
- Multi-Document Training — Fine-tune on multiple PDFs simultaneously
- Smart Extraction — Section-aware summarization (Abstract, Intro, Conclusion priority)
- GPU Accelerated — RTX 3050+ supported via CUDA; falls back to CPU
- Dark Theme UI — Clean Streamlit dashboard with live progress steps

---

## Installation

### Requirements
- Python 3.10.x (64-bit)
- Windows 10/11 or Linux
- 8 GB RAM minimum
- NVIDIA GPU with 4 GB+ VRAM (optional but recommended)

### Step 1 — Clone
```bash
git clone https://github.com/YOUR_USERNAME/privsumAI.git
cd privsumAI
```

### Step 2 — Install PyTorch with CUDA
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — bitsandbytes for Windows
```bash
pip install bitsandbytes --prefer-binary --extra-index-url https://jllllll.github.io/bitsandbytes-windows-webui
```

### Step 5 — HuggingFace login (one time)
```bash
python -c "from huggingface_hub import login; login(token='your_token_here')"
```

### Step 6 — Run
```bash
streamlit run app.py
```

> First run downloads TinyLlama (~2.2 GB). After that, works fully offline.

---

## Project Structure

```
privsumAI/
├── app.py                  # Streamlit web application
├── evaluate.py             # ROUGE evaluation script
├── requirements.txt
├── README.md
└── utils/
    ├── pdf_processor.py    # PDF extraction + chunking
    ├── summarizer.py       # Model loading + summarization
    └── qlora_trainer.py    # QLoRA fine-tuning
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Base Model | TinyLlama-1.1B-Chat-v1.0 |
| Quantization | bitsandbytes (4-bit NF4) |
| Fine-Tuning | PEFT / QLoRA |
| Framework | HuggingFace Transformers |
| PDF Parsing | PyMuPDF (fitz) |
| UI | Streamlit |
| Evaluation | rouge-score |

---

## References

1. Dettmers et al., *QLoRA*, NeurIPS 2023
2. Zhang et al., *TinyLlama*, arXiv 2024
3. Hu et al., *LoRA*, ICLR 2022
4. Wolf et al., *Transformers*, EMNLP 2020

---

*M.Tech Data Science, RIT Kottayam | 2025–2026*
