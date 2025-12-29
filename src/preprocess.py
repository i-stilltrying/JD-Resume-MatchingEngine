# src/preprocess.py
# Text cleanup + helper utilities:
# --normalize text (lowercase, fix spaces)
# --tokenize text (for BM25)
# --split text into chunks (for chunk-based semantic scoring)
# --split resume into sentences (for top_sentences explanation)




from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Iterable

_WHITESPACE_RE = re.compile(r"\s+")
_NON_TEXT_RE = re.compile(r"[^\w\s\-\+\#\.\/]")

_SENT_SPLIT_RE = re.compile(r"(?<=[\.!\?])\s+|\n+")

def normalize_text(text: str) -> str:
    """Light normalization for resume/JD text."""
    text = text.replace("\u00a0", " ")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text.lower()

def simple_tokenize(text: str) -> List[str]:
    """Tokenizer for BM25 (keeps characters seen in skills like C++, C#, node.js)."""
    text = normalize_text(text)
    text = _NON_TEXT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return [t for t in text.split(" ") if t]

@dataclass(frozen=True)
class PreprocessConfig:
    max_chars: int = 20000  # truncate extremely long resumes for speed/stability

def preprocess_for_model(text: str, cfg: PreprocessConfig = PreprocessConfig()) -> str:
    text = normalize_text(text)
    if len(text) > cfg.max_chars:
        text = text[: cfg.max_chars]
    return text

def chunk_tokens(tokens: List[str], chunk_size: int = 300, overlap: int = 60) -> List[List[str]]:
    """Chunk a token list into overlapping windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    if not tokens:
        return [[]]

    chunks: List[List[str]] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        end = min(start + chunk_size, len(tokens))
        chunks.append(tokens[start:end])
        if end >= len(tokens):
            break
    return chunks

def join_tokens(tokens: List[str]) -> str:
    return " ".join(tokens).strip()

def split_sentences(text: str, min_chars: int = 25, max_sentences: int = 200) -> List[str]:
    """Simple sentence split for explanations (works decently for bullet resumes)."""
    text = text.replace("\u2022", " ")  # bullet
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p and p.strip()]
    # Keep informative sentences only
    out: List[str] = []
    for p in parts:
        if len(p) >= min_chars:
            out.append(p)
        if len(out) >= max_sentences:
            break
    return out
