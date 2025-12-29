from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple

from .preprocess import normalize_text

# --- Seniority keyword sets (simple + explainable) ---
_SENIOR_TERMS = {
    "senior", "sr", "staff", "principal", "lead", "manager", "architect", "director"
}
_JUNIOR_TERMS = {
    "intern", "internship", "trainee", "junior", "jr", "student", "fresher", "entry-level"
}

# --- Patterns ---
_YEARS_RE = re.compile(r"\b(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|year|yrs|yr)\b", re.I)

_YEAR_RANGE_RE = re.compile(
    r"\b(19\d{2}|20\d{2})\s*(?:-|–|to)\s*(present|current|now|19\d{2}|20\d{2})\b",
    re.I,
)

_MONTH_YEAR_RANGE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s+(19\d{2}|20\d{2})\s*(?:-|–|to)\s*(?:present|current|now|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\w*\s+(19\d{2}|20\d{2}))\b",
    re.I,
)


@dataclass(frozen=True)
class SenioritySignals:
    jd_required_years: Optional[float]
    jd_senior_title: bool
    resume_years_est: Optional[float]
    resume_senior_terms: bool
    resume_junior_terms: bool


def _extract_required_years_from_jd(jd_text: str) -> Optional[float]:
    """Extract required years from JD, e.g., '5+ years'. Returns max found."""
    t = normalize_text(jd_text)
    vals = []
    for m in _YEARS_RE.finditer(t):
        try:
            vals.append(float(m.group(1)))
        except Exception:
            continue
    return max(vals) if vals else None


def _jd_has_senior_title(jd_text: str) -> bool:
    """
    Detect if JD head suggests a senior role.
    We look at the first 250 chars to avoid catching random 'lead' in the body.
    """
    t = normalize_text(jd_text)
    head = t[:250]
    return any(term in head.split() for term in ("senior", "staff", "principal")) or "senior" in head or "principal" in head or "staff" in head


def _extract_resume_years(text: str) -> Optional[float]:
    """Estimate years of experience from resume text (best-effort)."""
    t = normalize_text(text)

    # 1) explicit years mentions
    yrs = []
    for m in _YEARS_RE.finditer(t):
        try:
            v = float(m.group(1))
            if 0.0 < v <= 50.0:
                yrs.append(v)
        except Exception:
            continue
    if yrs:
        return max(yrs)

    # 2) year ranges
    spans = []
    today_year = date.today().year
    for m in _YEAR_RANGE_RE.finditer(t):
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = today_year if end_raw in {"present", "current", "now"} else int(end_raw)
        if 1950 <= start <= today_year and start <= end <= today_year + 1:
            span = end - start
            if 0 <= span <= 50:
                spans.append(float(span))
    if spans:
        return max(spans)

    # 3) month-year ranges (approx by years)
    spans2 = []
    for m in _MONTH_YEAR_RANGE_RE.finditer(t):
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = today_year if end_raw in {"present", "current", "now"} else int(end_raw)
        if 1950 <= start <= today_year and start <= end <= today_year + 1:
            span = end - start
            if 0 <= span <= 50:
                spans2.append(float(span))
    if spans2:
        return max(spans2)

    return None


def _resume_term_flags(text: str) -> Tuple[bool, bool]:
    tokens = set(normalize_text(text).split())
    senior = any(t in tokens for t in _SENIOR_TERMS)
    junior = any(t in tokens for t in _JUNIOR_TERMS)
    return senior, junior


def compute_seniority_signals(jd_text: str, resume_text: str) -> SenioritySignals:
    jd_req = _extract_required_years_from_jd(jd_text)
    jd_sen = _jd_has_senior_title(jd_text)

    # NEW: if JD implies senior but no explicit years, assume an implied minimum.
    # Tune this number based on your org: 4/5/6 are common.
    if jd_req is None and jd_sen:
        jd_req = 5.0

    res_years = _extract_resume_years(resume_text)
    res_sen, res_jun = _resume_term_flags(resume_text)

    return SenioritySignals(
        jd_required_years=jd_req,
        jd_senior_title=jd_sen,
        resume_years_est=res_years,
        resume_senior_terms=res_sen,
        resume_junior_terms=res_jun,
    )


def seniority_penalty_factor(sig: SenioritySignals) -> float:
    """
    Return multiplicative penalty factor in [0.55, 1.00].

    Changes vs previous:
    - Stronger penalties for being under the required years
    - Penalize unknown resume years if JD requires years (small penalty)
    """
    factor = 1.0

    # NEW: if JD requires years but we can't estimate resume years, apply a mild penalty
    if sig.jd_required_years is not None and sig.resume_years_est is None:
        factor *= 0.92

    # years mismatch penalty (only if both known)
    if sig.jd_required_years is not None and sig.resume_years_est is not None:
        gap = sig.jd_required_years - sig.resume_years_est

        # NEW: stronger curve (tuneable)
        if gap >= 4.0:
            factor *= 0.55
        elif gap >= 3.0:
            factor *= 0.60
        elif gap >= 2.0:
            factor *= 0.65
        elif gap >= 1.0:
            factor *= 0.85

    # senior title but resume looks junior (intern/student etc.)
    if sig.jd_senior_title and sig.resume_junior_terms and not sig.resume_senior_terms:
        factor *= 0.80

    # clamp
    if factor < 0.55:
        factor = 0.55
    if factor > 1.0:
        factor = 1.0
    return factor


def apply_seniority_penalty(score: float, sig: SenioritySignals) -> Tuple[float, float]:
    """Apply seniority penalty; returns (new_score, factor_applied)."""
    f = seniority_penalty_factor(sig)
    return score * f, f
