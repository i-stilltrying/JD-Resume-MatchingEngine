# src/extract.py
# Converts resume files into plain text.
# --.txt → reads normally
# --.pdf → extracts text using PyMuPDF


from __future__ import annotations

from pathlib import Path

def extract_text(path: Path) -> str:
    """Extract resume text from .txt or .pdf.

    - .txt: read UTF-8 (errors ignored)
    - .pdf: uses PyMuPDF (fitz) to extract page text
    """
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            import fitz  # PyMuPDF
        except Exception as e:
            raise RuntimeError(
                "PDF support requires PyMuPDF. Install with: pip install pymupdf"
            ) from e

        doc = fitz.open(path)
        parts: list[str] = []
        for page in doc:
            parts.append(page.get_text("text"))
        doc.close()
        return "\n".join(parts)

    raise ValueError(f"Unsupported file type: {path.name} ({suffix})")
