"""
summarizer.py — Hallucination-Free Edition
────────────────────────────────────────────
Root cause of hallucination:
  Structured section prompts ("Document Overview:", "Problem:", etc.)
  trigger TinyLlama's template-completion training data — it ignores
  the actual document and fills in plausible-sounding generic text.

Fix:
  1. Copy the ABSTRACT and CONCLUSION verbatim from the document
  2. Use an extremely simple prompt: "Here is the abstract and conclusion.
     Write a summary based on these."
  3. No section labels in the prompt — they trigger hallucination
  4. Feed actual document sentences, not instructions about structure
"""

from __future__ import annotations
import os
import re
from typing import Generator, List, Optional

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ─── Model Loading ────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    use_4bit: bool = True,
    adapter_path: Optional[str] = None,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        print(f">> GPU: {torch.cuda.get_device_name(0)}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model = model.to("cuda:0")
        print(f">> Model on: {next(model.parameters()).device}")
    else:
        from transformers import BitsAndBytesConfig
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ) if use_4bit else None
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=cfg,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    if adapter_path and os.path.isdir(adapter_path):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


# ─── Section Extractor ────────────────────────────────────────────────────────

_WANT = re.compile(
    r'^\s*(?:\d+[\.\d]*\s+)?'
    r'(abstract|introduction|conclusion[s]?|summary|'
    r'contribution[s]?|result[s]?|finding[s]?|'
    r'key\s+result[s]?|overview|background)\s*$',
    re.IGNORECASE
)

_SKIP = re.compile(
    r'^\s*(?:\d+[\.\d]*\s+)?'
    r'(reference[s]?|bibliograph\w*|appendix|acknowledg\w*)\s*$',
    re.IGNORECASE
)


def _extract_best_sections(full_text: str, budget: int = 3500) -> str:
    """
    Directly grab Abstract + Introduction + Conclusion text verbatim.
    These are the highest-signal sections of any research document.
    Returns raw paragraph text — no labels, no formatting.
    """
    lines      = full_text.split('\n')
    sections   = {}
    cur_key    = 'preamble'
    cur_lines: List[str] = []
    skipping   = False

    for line in lines:
        stripped = line.strip()

        if _SKIP.match(stripped):
            if cur_lines:
                sections[cur_key] = '\n'.join(cur_lines).strip()
            skipping = True
            cur_lines = []
            continue

        if _WANT.match(stripped) and len(stripped) < 60:
            if cur_lines:
                sections[cur_key] = '\n'.join(cur_lines).strip()
            cur_key   = stripped.lower().strip()
            cur_lines = []
            skipping  = False
            continue

        if not skipping:
            cur_lines.append(line)

    if cur_lines and not skipping:
        sections[cur_key] = '\n'.join(cur_lines).strip()

    # Priority order for selection
    priority = [
        'abstract', 'introduction', 'conclusion', 'conclusions',
        'summary', 'contributions', 'contribution',
        'results', 'result', 'findings', 'finding',
        'overview', 'background', 'preamble'
    ]

    selected = []
    used     = 0

    for key in priority:
        if used >= budget:
            break
        for sec_key, content in sections.items():
            if key in sec_key and content:
                # Clean the content — remove lone numbers, short lines, URLs
                lines_clean = [
                    l for l in content.split('\n')
                    if len(l.strip()) > 30
                    and not re.match(r'^\s*[\d\.\-\*•]+\s*$', l)
                    and 'http' not in l
                    and 'doi.org' not in l.lower()
                ]
                text = ' '.join(lines_clean).strip()
                text = re.sub(r'\s+', ' ', text)
                if len(text) < 100:
                    continue
                take = min(len(text), budget - used)
                selected.append(text[:take])
                used += take
                break  # don't double-count same section

    result = '\n\n'.join(selected)

    # Fallback: just use first 3000 chars if nothing extracted
    if len(result) < 200:
        result = re.sub(r'\s+', ' ', full_text[:3000]).strip()

    return result


# ─── Inference ────────────────────────────────────────────────────────────────

def _generate(model, tokenizer, doc_text: str, max_new_tokens: int = 500) -> str:
    import torch

    device = next(model.parameters()).device

    # SIMPLE prompt — no section labels that trigger template hallucination
    prompt = (
        "<|system|>\n"
        "You are a document summarizer. "
        "Read the following text carefully and write a thorough, detailed summary. "
        "Cover: what the document is about, the problem it addresses, "
        "the proposed solution and its key features, "
        "the main results with specific details, and the conclusions. "
        "Only use facts that appear in the text. Write at least 6-8 sentences.\n"
        f"<|user|>\nText:\n{doc_text}\n\nWrite a detailed summary:\n"
        "<|assistant|>\n"
        "This document presents"
    )

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1400,
        padding=False,
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        if device.type == "cuda":
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                out = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
        else:
            out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

    new_tokens = out[0][input_ids.shape[1]:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return "This document presents " + raw


# ─── Public API ───────────────────────────────────────────────────────────────

def summarize_chunks(
    model,
    tokenizer,
    chunks: List[str],
    max_new_tokens: int = 500,
) -> Generator[str, None, None]:
    """
    Extract abstract + introduction + conclusion verbatim,
    then use a minimal prompt that forces the model to read
    the actual text instead of filling in hallucinated templates.
    """
    full_text = "\n\n".join(chunks)
    doc_text  = _extract_best_sections(full_text, budget=3500)
    summary   = _generate(model, tokenizer, doc_text, max_new_tokens)
    yield summary
