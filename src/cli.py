  # Main entrypoint: load → extract → redact → score → save CSV
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .extract import extract_text
from .matching import ResumeMatcher, MatchingConfig
from .redact import redact_pii


def load_resumes(
    folder: Path,
    do_redact_pii: bool = False,
    save_debug_text: bool = True,
    outputs_dir: Path | None = None,
) -> dict[str, str]:
    """
    Load resumes (.txt/.pdf). Optionally save extracted/redacted text for audit/debug.

    - Extracts text via src/extract.py
    - Optionally redacts PII via src/redact.py
    - Saves debug text:
        outputs/extracted/<filename>.txt
        outputs/redacted/<filename>.txt
    """
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jd", required=True, help="Path to job description .txt")
    ap.add_argument("--resumes", required=True, help="Folder with resume .txt/.pdf files")
    ap.add_argument("--output", required=True, help="Output CSV path")
    ap.add_argument("--redact_pii", action="store_true", help="Redact basic PII (email/phone/url) before scoring")

    # Cross-encoder reranker
    ap.add_argument("--use_cross_encoder", action="store_true", help="Enable cross-encoder reranking for top-N")
    ap.add_argument("--rerank_top_n", type=int, default=10, help="How many top resumes to rerank with cross-encoder")
    ap.add_argument(
        "--cross_encoder_name",
        type=str,
        default="cross-encoder/ms-marco-MiniLM-L6-v2",
        help="HF model name for cross-encoder reranker",
    )

    # Chunking knobs
    ap.add_argument("--chunk_size_tokens", type=int, default=320)
    ap.add_argument("--chunk_overlap_tokens", type=int, default=80)
    ap.add_argument("--semantic_top_k_avg", type=int, default=1)

    args = ap.parse_args()

    jd_path = Path(args.jd)
    resumes_dir = Path(args.resumes)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Debug text is saved next to the output CSV (outputs/extracted, outputs/redacted)
    out_root = out_path.parent

    jd = jd_path.read_text(encoding="utf-8", errors="ignore")
    resumes = load_resumes(
        resumes_dir,
        do_redact_pii=args.redact_pii,
        save_debug_text=True,
        outputs_dir=out_root,
    )

    cfg = MatchingConfig(
        use_cross_encoder=args.use_cross_encoder,
        rerank_top_n=args.rerank_top_n,
        cross_encoder_name=args.cross_encoder_name,
        chunk_size_tokens=args.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        semantic_top_k_avg=args.semantic_top_k_avg,
    )

    matcher = ResumeMatcher(cfg)
    results = matcher.score(jd, resumes)

    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)

    # Print key columns for terminal readability
    show_cols = [
        "resume_id",
        "final",
        "semantic",
        "bm25",
        "cross_encoder",
        "missing_bucket_count",
        "jd_required_years",
        "resume_years_est",
        "seniority_penalty_factor",
        "top_terms",
    ]

    # only print columns that exist (avoids crashes if schema changes)
    show_cols = [c for c in show_cols if c in df.columns]
    print(df[show_cols].to_string(index=False))

    print(f"\nSaved: {out_path}")
    print(f"Debug text saved under: {out_root / 'extracted'} and {out_root / 'redacted'}")


if __name__ == "__main__":
    main()
