# src/matching.py
# Main scoring brain.
# For each resume, it computes:
# --BM25 score (keyword overlap)
# --Semantic score using embeddings on resume chunks (meaning match)
# --Hybrid score = semantic + bm25
# --Must-have penalty if buckets missing
# --Seniority penalty if JD implies seniority / years requirement
# Explanations
# --top_terms (important overlapping JD keywords)
# --top_sentences (top 3 resume sentences most similar to JD)
# Produces final score and returns ranked list.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from .preprocess import (
    preprocess_for_model,
    simple_tokenize,
    chunk_tokens,
    join_tokens,
    split_sentences,
)
from .skills import bucket_coverage, apply_must_have_penalty, missing_bucket_count
from .seniority import compute_seniority_signals, apply_seniority_penalty


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(np.min(x))
    mx = float(np.max(x))
    if abs(mx - mn) < 1e-12:
        return np.ones_like(x, dtype=float) * 0.5
    return (x - mn) / (mx - mn)


def _top_terms_from_overlap(
    jd_tokens: List[str],
    resume_tokens: List[str],
    bm25: Optional[BM25Okapi],
    top_n: int = 10,
) -> List[str]:
    """Explain lexical match via overlapping JD tokens, ranked by BM25 IDF when available."""
    jd_set = set(jd_tokens)
    res_set = set(resume_tokens)
    overlap = list(jd_set.intersection(res_set))
    if not overlap:
        return []

    idf = getattr(bm25, "idf", {}) if bm25 is not None else {}
    overlap.sort(key=lambda t: float(idf.get(t, 0.0)), reverse=True)
    return overlap[:top_n]


def _top_sentences_by_semantic(
    embedder: SentenceTransformer,
    jd_emb: np.ndarray,
    resume_text: str,
    top_k: int = 3,
) -> List[str]:
    sents = split_sentences(resume_text)
    if not sents:
        return []
    sent_embs = embedder.encode(sents, normalize_embeddings=True)
    # embeddings are normalized -> dot product equals cosine
    scores = np.dot(sent_embs, jd_emb)
    top_idx = np.argsort(-scores)[:top_k]
    return [sents[int(i)] for i in top_idx]


def _semantic_score_with_chunking(
    embedder: SentenceTransformer,
    jd_emb: np.ndarray,
    resume_text: str,
    chunk_size: int,
    overlap: int,
    top_k_avg: int = 1,
) -> Tuple[float, float]:
    """Return (semantic_score_0to1, best_chunk_score_0to1)."""
    tokens = simple_tokenize(resume_text)
    chunks_tok = chunk_tokens(tokens, chunk_size=chunk_size, overlap=overlap)
    chunks = [join_tokens(toks) for toks in chunks_tok]
    # Handle empty text robustly
    chunks = [c for c in chunks if c.strip()] or [""]
    chunk_embs = embedder.encode(chunks, normalize_embeddings=True)
    scores = np.dot(chunk_embs, jd_emb)  # [-1,1]
    # Convert to [0,1]
    scores_01 = (scores + 1.0) / 2.0
    # Aggregate: top-k average (k=1 => max)
    k = max(1, min(top_k_avg, len(scores_01)))
    topk = np.sort(scores_01)[-k:]
    agg = float(np.mean(topk))
    best = float(np.max(scores_01))
    return agg, best


@dataclass
class MatchingConfig:
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Chunking for semantic score
    chunk_size_tokens: int = 320
    chunk_overlap_tokens: int = 80
    semantic_top_k_avg: int = 1  # 1=max, >1=top-k average

    # Optional reranker
    use_cross_encoder: bool = False
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    rerank_top_n: int = 5

    # Hybrid weights
    w_sem: float = 0.7
    w_bm25: float = 0.3

    # Must-have penalty
    must_have_penalty_factor: float = 0.85  # applied if missing 2+ buckets


class ResumeMatcher:
    def __init__(self, cfg: MatchingConfig = MatchingConfig()):
        self.cfg = cfg
        self.embedder = SentenceTransformer(cfg.embed_model_name)
        self.cross_encoder = CrossEncoder(cfg.cross_encoder_name) if cfg.use_cross_encoder else None

    def score(self, job_description: str, resumes: Dict[str, str]) -> List[Dict[str, object]]:
        """Score resumes vs JD; returns ranked rows with explainability columns."""
        jd_text = preprocess_for_model(job_description)
        jd_emb = self.embedder.encode([jd_text], normalize_embeddings=True)[0]

        resume_items = list(resumes.items())
        resume_ids = [rid for rid, _ in resume_items]
        resume_texts = [preprocess_for_model(t) for _, t in resume_items]

        # ---------- BM25 setup ----------
        tokenized_resumes = []
        for t in resume_texts:
            toks = simple_tokenize(t)
            tokenized_resumes.append(toks if toks else ["__empty__"])

        bm25 = BM25Okapi(tokenized_resumes)
        jd_tokens = simple_tokenize(jd_text)
        jd_tokens = jd_tokens if jd_tokens else ["__empty__"]
        bm25_raw = np.array(bm25.get_scores(jd_tokens), dtype=float)
        bm25_scaled = _minmax(bm25_raw)

        # ---------- Semantic (chunked) ----------
        sem_raw = np.zeros(len(resume_ids), dtype=float)
        sem_best_chunk = np.zeros(len(resume_ids), dtype=float)
        top_sents: List[List[str]] = []

        for i, t in enumerate(resume_texts):
            agg, best = _semantic_score_with_chunking(
                self.embedder,
                jd_emb,
                t,
                chunk_size=self.cfg.chunk_size_tokens,
                overlap=self.cfg.chunk_overlap_tokens,
                top_k_avg=self.cfg.semantic_top_k_avg,
            )
            sem_raw[i] = agg
            sem_best_chunk[i] = best
            top_sents.append(_top_sentences_by_semantic(self.embedder, jd_emb, t, top_k=3))

        sem_scaled = _minmax(sem_raw)
        sem_best_scaled = _minmax(sem_best_chunk)

        # ---------- Hybrid score ----------
        hybrid = self.cfg.w_sem * sem_scaled + self.cfg.w_bm25 * bm25_scaled

        # ---------- Must-have buckets + penalty ----------
        bucket_hits_list: List[Dict[str, int]] = []
        bucket_matched_list: List[Dict[str, List[str]]] = []
        gated = np.zeros_like(hybrid)

        for i, t in enumerate(resume_texts):
            hits, matched = bucket_coverage(t)
            bucket_hits_list.append(hits)
            bucket_matched_list.append(matched)
            gated[i] = apply_must_have_penalty(
                float(hybrid[i]),
                hits,
                penalty_factor=self.cfg.must_have_penalty_factor,
            )

        # ---------- Seniority penalty (NEW) ----------
        # Applies after bucket gating. Does NOT hard-filter; just adjusts ranking.
        gated_seniority = np.zeros_like(gated)

        jd_required_years_list: List[Optional[float]] = []
        resume_years_est_list: List[Optional[float]] = []
        seniority_penalty_factor_list: List[float] = []

        for i, t in enumerate(resume_texts):
            sig = compute_seniority_signals(jd_text, t)
            new_score, factor = apply_seniority_penalty(float(gated[i]), sig)
            gated_seniority[i] = new_score

            jd_required_years_list.append(sig.jd_required_years)
            resume_years_est_list.append(sig.resume_years_est)
            seniority_penalty_factor_list.append(float(factor))

        # ---------- Explanations (top terms from overlap) ----------
        top_terms_list: List[List[str]] = []
        for i, _t in enumerate(resume_texts):
            res_toks = tokenized_resumes[i] if tokenized_resumes[i] != ["__empty__"] else []
            top_terms_list.append(_top_terms_from_overlap(jd_tokens, res_toks, bm25, top_n=10))

        # ---------- Optional cross-encoder rerank ----------
        if self.cross_encoder is not None:
            # pick top-N based on seniority-adjusted gated score
            top_idx = np.argsort(-gated_seniority)[: self.cfg.rerank_top_n]
            pairs = [(jd_text, resume_texts[i]) for i in top_idx]
            ce_raw = np.array(self.cross_encoder.predict(pairs), dtype=float)
            ce_sigmoid = 1.0 / (1.0 + np.exp(-ce_raw))
            ce_scaled_top = _minmax(ce_sigmoid)

            ce_full = np.full(shape=(len(resume_ids),), fill_value=np.nan, dtype=float)
            for j, i in enumerate(top_idx):
                ce_full[i] = float(ce_scaled_top[j])

            final_raw = gated_seniority.copy()
            for i in top_idx:
                # blend reranker (already bucket+seniority adjusted)
                final_raw[i] = 0.7 * gated_seniority[i] + 0.3 * ce_full[i]

            final_scaled = _minmax(final_raw)
        else:
            ce_full = np.full(shape=(len(resume_ids),), fill_value=np.nan, dtype=float)
            final_scaled = _minmax(gated_seniority)

        # ---------- Build rows ----------
        rows: List[Dict[str, object]] = []
        for i, rid in enumerate(resume_ids):
            hits = bucket_hits_list[i]
            miss = missing_bucket_count(hits)

            rows.append(
                {
                    "resume_id": rid,
                    "semantic": float(sem_scaled[i]),
                    "semantic_best_chunk": float(sem_best_scaled[i]),
                    "bm25": float(bm25_scaled[i]),
                    "hybrid": float(hybrid[i]),
                    "hybrid_gated": float(gated[i]),
                    "hybrid_gated_seniority": float(gated_seniority[i]),
                    "final": float(final_scaled[i]),
                    "cross_encoder": (float(ce_full[i]) if not np.isnan(ce_full[i]) else None),

                    # Explanations
                    "top_terms": ", ".join(top_terms_list[i]),
                    "top_sentences": " || ".join(top_sents[i]),

                    # Must-have buckets (Teradata JD)
                    "bucket_agentic_llm_systems": int(hits["agentic_llm_systems"]),
                    "bucket_vector_memory_planning": int(hits["vector_memory_planning"]),
                    "bucket_observability_eval_reliability": int(hits["observability_eval_reliability"]),
                    "bucket_ui_ux_frontend": int(hits["ui_ux_frontend"]),
                    "bucket_backend_api_devops": int(hits["backend_api_devops"]),
                    "missing_bucket_count": int(miss),

                    # Seniority diagnostics (NEW)
                    "jd_required_years": jd_required_years_list[i],
                    "resume_years_est": resume_years_est_list[i],
                    "seniority_penalty_factor": float(seniority_penalty_factor_list[i]),
                }
            )

        rows.sort(key=lambda x: float(x["final"]), reverse=True)
        return rows
