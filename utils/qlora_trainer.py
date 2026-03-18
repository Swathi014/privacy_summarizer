"""
qlora_trainer.py
────────────────
Fine-tunes TinyLlama on one OR multiple documents using QLoRA.
Accepts either:
  - chunks: List[str]           — single document (original API)
  - all_chunks: List[List[str]] — multiple documents combined
Everything runs locally — no data is sent anywhere.
"""

from __future__ import annotations
import os
import json
import math
from typing import Callable, List, Optional

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ─── Dataset Preparation ──────────────────────────────────────────────────────

def _chunks_to_training_examples(chunks: List[str]) -> List[dict]:
    """Convert document chunks into instruction-style training examples."""
    examples = []
    for chunk in chunks:
        if len(chunk.strip()) < 80:
            continue
        prompt = (
            "<|system|>\nYou are a helpful summarizer.\n"
            f"<|user|>\nSummarize the following passage:\n{chunk.strip()}\n"
            "<|assistant|>\n"
        )
        examples.append({"text": prompt + chunk[:300].strip()})
    return examples


def _multi_doc_chunks_to_examples(all_chunks: List[List[str]]) -> List[dict]:
    """
    Combine chunks from multiple documents into one training dataset.
    all_chunks = [[doc1_chunk1, doc1_chunk2, ...], [doc2_chunk1, ...], ...]
    """
    examples = []
    for doc_chunks in all_chunks:
        examples.extend(_chunks_to_training_examples(doc_chunks))
    return examples


# ─── QLoRA Fine-Tuning ────────────────────────────────────────────────────────

def run_qlora_finetuning(
    chunks: Optional[List[str]] = None,
    all_chunks: Optional[List[List[str]]] = None,
    lora_r: int = 8,
    lora_alpha: int = 16,
    num_epochs: int = 1,
    save_path: str = "./qlora_adapter",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Fine-tune TinyLlama on one or multiple documents.

    Parameters
    ----------
    chunks      : chunks from a single document (original API, still supported)
    all_chunks  : list of chunk-lists from multiple documents
                  e.g. [[doc1_chunks], [doc2_chunks], [doc3_chunks]]
    lora_r      : LoRA rank
    lora_alpha  : LoRA alpha scaling
    num_epochs  : training epochs
    save_path   : where to save the adapter
    progress_callback : receives float in [0,1]

    Returns
    -------
    str — absolute path to saved adapter directory
    """
    import os, torch
    # Block all HuggingFace network calls — use local cache only
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM,
        BitsAndBytesConfig, TrainingArguments,
        Trainer, DataCollatorForLanguageModeling,
        TrainerCallback,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    # ── Resolve input ─────────────────────────────────────────────────────────
    if all_chunks is not None:
        # Multiple documents
        examples = _multi_doc_chunks_to_examples(all_chunks)
        num_docs = len(all_chunks)
    elif chunks is not None:
        # Single document (backwards compatible)
        examples = _chunks_to_training_examples(chunks)
        num_docs = 1
    else:
        raise ValueError("Provide either 'chunks' or 'all_chunks'.")

    if not examples:
        raise ValueError("No usable training examples extracted from the documents.")

    print(f">> QLoRA: {num_docs} document(s), {len(examples)} training examples")

    dataset = Dataset.from_list(examples)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=256,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # ── Model ─────────────────────────────────────────────────────────────────
    # Load on CPU to avoid VRAM conflict with summarizer cache.
    # Training still runs on GPU via accelerate — weights move to GPU
    # during the forward/backward pass automatically.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )

    # Build a device map: embed/norm on CPU, transformer layers on GPU
    # This lets us train even when full model doesn't fit in VRAM at once
    import accelerate
    max_gpu_mem = {0: "3GiB"}   # reserve 1 GB headroom on RTX 3050
    max_cpu_mem = "12GiB"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        max_memory={0: max_gpu_mem[0], "cpu": max_cpu_mem},
        trust_remote_code=True,
        local_files_only=True,
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # ── LoRA adapters ─────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=save_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # accumulate more, lower peak VRAM
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        optim="paged_adamw_8bit",        # 8-bit optimizer saves ~500 MB VRAM
        max_grad_norm=0.3,
    )

    total_steps = math.ceil(
        len(tokenized) / training_args.per_device_train_batch_size
    ) * num_epochs

    class _ProgCB(TrainerCallback):
        def __init__(self):
            self.step = 0
        def on_log(self, args, state, control, **kwargs):
            self.step += 1
            if progress_callback:
                progress_callback(min(self.step / max(total_steps, 1), 0.99))

    # ── Train ─────────────────────────────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
        callbacks=[_ProgCB()],
    )
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    meta = {
        "base_model": MODEL_ID,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "num_epochs": num_epochs,
        "num_documents": num_docs,
        "num_examples": len(examples),
    }
    with open(os.path.join(save_path, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return os.path.abspath(save_path)
