"""
evaluate.py
───────────
Evaluate the summarizer against reference summaries using ROUGE metrics.
Run from the project root:

    python evaluate.py --pdf path/to/doc.pdf --reference path/to/ref.txt

Or import `evaluate_summary()` programmatically.
"""

from __future__ import annotations
import argparse
import json
from typing import Dict


def evaluate_summary(generated: str, reference: str) -> Dict[str, float]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Parameters
    ----------
    generated  : model-generated summary string
    reference  : ground-truth reference summary string

    Returns
    -------
    dict with keys rouge1, rouge2, rougeL (F1 scores, 0-1 range)
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }
    except ImportError:
        raise ImportError("Install rouge-score: pip install rouge-score")


def _run_cli():
    parser = argparse.ArgumentParser(description="Evaluate PrivSumAI output with ROUGE")
    parser.add_argument("--pdf", required=True, help="Path to the PDF to summarise")
    parser.add_argument("--reference", required=True, help="Path to reference summary .txt")
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--adapter", default=None, help="Path to QLoRA adapter (optional)")
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default=None, help="Save JSON results to this path")
    args = parser.parse_args()

    from utils.pdf_processor import extract_text_from_pdf, recursive_chunk_text
    from utils.summarizer import load_model_and_tokenizer, summarize_chunks

    print("📄 Extracting text from PDF…")
    with open(args.pdf, "rb") as f:
        raw = extract_text_from_pdf(f.read())
    chunks = recursive_chunk_text(raw, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"   → {len(chunks)} chunks")

    print("🦙 Loading model…")
    model, tokenizer = load_model_and_tokenizer(
        use_4bit=args.use_4bit,
        adapter_path=args.adapter,
    )

    print("✍️  Generating summary…")
    parts = list(summarize_chunks(model, tokenizer, chunks, max_new_tokens=args.max_tokens))
    generated = "\n\n".join(parts)

    with open(args.reference, "r", encoding="utf-8") as f:
        reference = f.read()

    print("📊 Computing ROUGE scores…")
    scores = evaluate_summary(generated, reference)

    print("\n" + "=" * 40)
    print("  ROUGE Evaluation Results")
    print("=" * 40)
    for key, val in scores.items():
        print(f"  {key.upper():10s}: {val:.4f}")
    print("=" * 40)

    if args.output:
        result = {"scores": scores, "generated_summary": generated}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    _run_cli()
