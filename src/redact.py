# src/redact.py
# If enabled, removes basic PII from text:
# --emails → [EMAIL]
# --phone numbers → [PHONE]
# --URLs → [URL]

from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"(?i)\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+\b")

# Phone numbers: match phone-like sequences, then validate by digit count.
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s\-()]{8,}\d)(?!\w)")

def _redact_phone_candidates(text: str) -> str:
    def repl(m: re.Match) -> str:
        s = m.group(1)
        digits = re.sub(r"\D", "", s)
        if 10 <= len(digits) <= 13:
            return "[PHONE]"
        return s
    return _PHONE_CANDIDATE_RE.sub(repl, text)

def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _URL_RE.sub("[URL]", text)
    text = _redact_phone_candidates(text)
    return text
