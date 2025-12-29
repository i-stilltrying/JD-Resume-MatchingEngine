# src/evaluate.py
# Runs the same scoring pipeline as cli.py
# Then compares results against labels.csv
# Prints metrics:
# --Spearman correlation
# --DCG@5, nDCG@10
# In short: “Check if ranking matches manual labels.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .extract import extract_text
from .matching import ResumeMatcher, MatchingConfig
from .redact import redact_pii


def load_resumes(
    folder: Path,
    do_redact_pii: bool = False,
    save_debug_text: bool = True,
    outputs_dir: Path | None = None,
) -> dict[str, str]:
    resumes: dict[str, str] = {}
    files = sorted(list(folder.glob("*.txt")) + list(folder.glob("*.pdf")))

    extracted_dir = None
    redacted_dir = None
    if save_debug_text:
        if outputs_dir is None:
            outputs_dir = Path("outputs")
        extracted_dir = outputs_dir / "extracted"
        redacted_dir = outputs_dir / "redacted"
        extracted_dir.mkdir(parents=True, exist_ok=True)
        redacted_dir.mkdir(parents=True, exist_ok=True)

    for p in files:
        extracted = extract_text(p)
        if not extracted.strip():
            print(f"[WARN] Empty text extracted from: {p.name} (PDF might be scanned/image-based)")

        if save_debug_text and extracted_dir is not None:
            (extracted_dir / f"{p.name}.txt").write_text(extracted, encoding="utf-8", errors="ignore")

        txt = extracted
        if do_redact_pii:
            txt = redact_pii(txt)
            if save_debug_text and redacted_dir is not None:
                (redacted_dir / f"{p.name}.txt").write_text(txt, encoding="utf-8", errors="ignore")

        resumes[p.name] = txt

    if not resumes:
        raise ValueError(f"No .txt or .pdf resumes found in: {folder}")
    return resumes


def ndcg_at_k(relevances: list[float], k: int) -> float:
    rel = np.asarray(relevances[:k], dtype=float)
    if rel.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rel.size + 2))
    dcg = float(np.sum((2**rel - 1) * discounts))
    ideal = np.sort(rel)[::-1]
    idcg = float(np.sum((2**ideal - 1) * discounts))
    return 0.0 if idcg == 0.0 else dcg / idcg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True, help="Path to job description .txt")
    ap.add_argument("--resumes", required=True, help="Folder with resume .txt/.pdf files")
    ap.add_argument("--labels", required=True, help="CSV with columns: resume_id,label (0/0.5/1)")
    ap.add_argument("--redact_pii", action="store_true", help="Redact basic PII (email/phone/url) before scoring")

    ap.add_argument("--use_cross_encoder", action="store_true")
    ap.add_argument("--rerank_top_n", type=int, default=10)

    ap.add_argument("--chunk_size_tokens", type=int, default=320)
    ap.add_argument("--chunk_overlap_tokens", type=int, default=80)
    ap.add_argument("--semantic_top_k_avg", type=int, default=1)

    args = ap.parse_args()

    jd = Path(args.jd).read_text(encoding="utf-8", errors="ignore")

    # Save debug next to labels file by default (outputs/...)
    out_root = Path("outputs")
    resumes = load_resumes(Path(args.resumes), do_redact_pii=args.redact_pii, save_debug_text=True, outputs_dir=out_root)
    labels = pd.read_csv(Path(args.labels))

    label_map = dict(zip(labels["resume_id"], labels["label"]))
    missing = set(resumes.keys()) - set(label_map.keys())
    if missing:
        raise ValueError(f"Missing labels for: {sorted(missing)}")

    cfg = MatchingConfig(
        use_cross_encoder=args.use_cross_encoder,
        rerank_top_n=args.rerank_top_n,
        chunk_size_tokens=args.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        semantic_top_k_avg=args.semantic_top_k_avg,
    )
    matcher = ResumeMatcher(cfg)
    ranked = matcher.score(jd, resumes)

    df = pd.DataFrame(ranked)
    df["label"] = df["resume_id"].map(label_map).astype(float)

    spearman = spearmanr(df["final"].values, df["label"].values).correlation
    ndcg5 = ndcg_at_k(df["label"].tolist(), k=min(5, len(df)))
    ndcg10 = ndcg_at_k(df["label"].tolist(), k=min(10, len(df)))

    cols = [
        "resume_id",
        "label",
        "final",
        "semantic",
        "bm25",
        "missing_bucket_count",
        "top_terms",
    ]
    print(df[cols].to_string(index=False))

    print("\nMetrics on synthetic set:")
    print(f"Spearman correlation (score vs label): {spearman:.3f}")
    print(f"nDCG@5:  {ndcg5:.3f}")
    print(f"nDCG@10: {ndcg10:.3f}")
    print(f"\nDebug text saved under: {out_root / 'extracted'} and {out_root / 'redacted'}")


if __name__ == "__main__":
    main()
