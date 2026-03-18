"""
pdf_processor.py
────────────────
Handles PDF text extraction and recursive chunking.
All processing is local — no network calls.
"""

from __future__ import annotations
import io
import re
from typing import List


# ─── PDF Text Extraction ──────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract plain text from a PDF given its raw bytes.
    Uses PyMuPDF (fitz) for high-quality text extraction.
    Falls back to pdfplumber if fitz is unavailable.
    """
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text: List[str] = []
        for page in doc:
            pages_text.append(page.get_text("text"))
        doc.close()
        return "\n".join(pages_text)

    except ImportError:
        pass  # try fallback

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
        return "\n".join(pages_text)

    except ImportError:
        raise ImportError(
            "No PDF library found. Install one of:\n"
            "  pip install pymupdf\n"
            "  pip install pdfplumber"
        )


# ─── Text Cleaning ────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Light normalisation: collapse multiple blank lines, fix spacing.
    """
    # Replace multiple newlines with double newline
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


# ─── Recursive Chunking ───────────────────────────────────────────────────────

def recursive_chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 100,
    separators: List[str] | None = None,
) -> List[str]:
    """
    Split text into overlapping chunks using a hierarchy of separators,
    similar to LangChain's RecursiveCharacterTextSplitter.

    Parameters
    ----------
    text        : full document text
    chunk_size  : target maximum characters per chunk
    overlap     : number of characters to repeat across adjacent chunks
    separators  : priority-ordered list of split strings

    Returns
    -------
    List of text chunks
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    text = clean_text(text)
    return _split_recursive(text, chunk_size, overlap, separators)


def _split_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    separators: List[str],
) -> List[str]:
    """Internal recursive splitter."""

    # Base case: text already fits in one chunk
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Try separators from highest to lowest priority
    for sep in separators:
        if sep and sep in text:
            parts = text.split(sep)
            break
    else:
        # No separator found — hard split
        parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        return [p for p in parts if p.strip()]

    chunks: List[str] = []
    current = ""

    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
                # Add overlap: carry last `overlap` chars into next chunk
                current = current[-overlap:].strip() + (sep if sep.strip() else " ") + part.strip()
            else:
                # Single part is larger than chunk_size — recurse with next separator
                next_seps = separators[separators.index(sep) + 1 :] if sep in separators else []
                if next_seps:
                    sub_chunks = _split_recursive(part, chunk_size, overlap, next_seps)
                    chunks.extend(sub_chunks)
                    current = sub_chunks[-1][-overlap:] if sub_chunks else ""
                else:
                    chunks.extend(
                        [part[i : i + chunk_size] for i in range(0, len(part), chunk_size - overlap)]
                    )
                    current = ""

    if current.strip():
        chunks.append(current.strip())

    return [c for c in chunks if c.strip()]
