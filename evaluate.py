"""
evaluate.py — ROUGE evaluation for PrivSumAI
=============================================
Usage:
    python evaluate.py --pdf myfile.pdf --reference myref.txt
"""
from __future__ import annotations
import argparse, json
from typing import Dict


def evaluate_summary(generated: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L F1 scores (0 to 1)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        s = scorer.score(reference, generated)
        return {
            "rouge1": round(s["rouge1"].fmeasure, 4),
            "rouge2": round(s["rouge2"].fmeasure, 4),
            "rougeL": round(s["rougeL"].fmeasure, 4),
        }
    except ImportError:
        raise ImportError("Run: pip install rouge-score")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf",        required=True)
    parser.add_argument("--reference",  required=True)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--adapter",    default="./qlora_adapter")
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--output",     default=None)
    args = parser.parse_args()

    from utils.pdf_processor import extract_text_from_pdf
    from utils.summarizer import generate_summary, get_adapter_info

    if not args.no_adapter:
        info = get_adapter_info(args.adapter)
        if info:
            print(f"Adapter: domain='{info.get('domain','?')}' trained={info.get('trained_at','?')}")

    print("Extracting PDF…")
    with open(args.pdf, "rb") as f:
        raw = extract_text_from_pdf(f.read())

    print("Generating summary…")
    gen = generate_summary(raw, max_tokens=args.max_tokens,
                           use_adapter=not args.no_adapter, adapter_path=args.adapter)

    with open(args.reference, encoding="utf-8") as f:
        ref = f.read()

    scores = evaluate_summary(gen, ref)
    print("\n=== ROUGE Results ===")
    for k, v in scores.items():
        print(f"  {k.upper()}: {v:.4f}  {'█'*int(v*20)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"scores": scores, "summary": gen}, f, indent=2)
        print(f"Saved to {args.output}")
