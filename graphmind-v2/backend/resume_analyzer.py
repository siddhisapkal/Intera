"""Resume PDF analyzer – extracts structured info and returns graph signals."""
from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None  # type: ignore

# ─────────────────────────── text extraction ────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("pypdf is not installed. Add 'pypdf' to requirements.txt.")
    reader = PdfReader(io.BytesIO(file_bytes))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        parts.append(text)
    return "\n".join(parts)


# ─────────────────────────── heuristic parsers ──────────────────────────────

_SKILL_KEYWORDS = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "react", "vue", "angular", "node", "django", "flask", "fastapi", "spring",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "docker", "kubernetes", "aws", "gcp", "azure", "terraform", "linux",
    "machine learning", "deep learning", "nlp", "pytorch", "tensorflow",
    "pandas", "numpy", "scikit-learn", "git", "graphql", "rest", "grpc",
}

_SECTION_HEADERS = re.compile(
    r"^\s*(experience|work experience|employment|education|skills|"
    r"projects|certifications|summary|objective|profile|"
    r"technical skills|achievements)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _extract_email(text: str) -> str | None:
    m = re.search(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text, re.IGNORECASE)
    return m.group(0) if m else None


def _extract_name(text: str) -> str | None:
    """Very heuristic: first non-blank line that looks like a name."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        words = line.split()
        if 2 <= len(words) <= 4 and all(re.match(r"[A-Za-z][a-z'-]*$", w) for w in words):
            return line
        break
    return None


def _extract_skills(text: str) -> list[str]:
    lowered = text.lower()
    found = [skill for skill in _SKILL_KEYWORDS if skill in lowered]
    return sorted(set(found))


def _extract_companies(text: str) -> list[str]:
    """Simple heuristic: lines near dates that look like company names."""
    date_pattern = re.compile(r"\b(20\d{2}|19\d{2}|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", re.IGNORECASE)
    companies: list[str] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if date_pattern.search(line_stripped) and 3 < len(line_stripped) < 80:
            # line before or current might be company name
            for candidate in (line_stripped, lines[i-1].strip() if i > 0 else ""):
                if candidate and not date_pattern.match(candidate) and re.search(r"[A-Z]", candidate):
                    companies.append(candidate[:60])
                    break
    return list(dict.fromkeys(companies))[:6]  # dedupe, limit


def _extract_education(text: str) -> list[str]:
    degree_pattern = re.compile(
        r"\b(b\.?tech|m\.?tech|b\.?e|m\.?e|bsc|msc|b\.?sc|m\.?sc|"
        r"bachelor|master|phd|mba|b\.?com|mca|bca)\b",
        re.IGNORECASE,
    )
    results: list[str] = []
    for line in text.splitlines():
        if degree_pattern.search(line):
            results.append(line.strip()[:120])
    return results[:4]


# ─────────────────────────── graph signals ──────────────────────────────────

def build_graph_signals(
    *,
    user_id: str,
    name: str | None,
    email: str | None,
    skills: list[str],
    companies: list[str],
    education: list[str],
) -> list[dict]:
    signals: list[dict] = []

    if name:
        signals.append({
            "entity": name,
            "entity_type": "Person",
            "relation": "IS_NAMED",
            "confidence": 0.95,
            "linked_to_action": True,
            "raw_text": f"Resume name: {name}",
        })

    if email:
        signals.append({
            "entity": email,
            "entity_type": "Email",
            "relation": "HAS_EMAIL",
            "confidence": 0.95,
            "linked_to_action": True,
            "raw_text": f"Resume email: {email}",
        })

    for skill in skills:
        signals.append({
            "entity": skill,
            "entity_type": "Skill",
            "relation": "HAS_SKILL",
            "confidence": 0.88,
            "linked_to_action": True,
            "raw_text": f"Skill from resume: {skill}",
        })

    for company in companies:
        signals.append({
            "entity": company,
            "entity_type": "Company",
            "relation": "WORKED_AT",
            "confidence": 0.80,
            "linked_to_action": True,
            "raw_text": f"Company from resume: {company}",
        })

    for edu in education:
        signals.append({
            "entity": edu[:80],
            "entity_type": "Education",
            "relation": "STUDIED_AT",
            "confidence": 0.85,
            "linked_to_action": True,
            "raw_text": f"Education from resume: {edu}",
        })

    return signals


# ─────────────────────────── main entry ─────────────────────────────────────

def analyze_resume(*, user_id: str, file_bytes: bytes) -> dict:
    """Parse a PDF resume and return structured info + graph signals."""
    text = extract_text_from_pdf(file_bytes)
    name = _extract_name(text)
    email = _extract_email(text)
    skills = _extract_skills(text)
    companies = _extract_companies(text)
    education = _extract_education(text)

    signals = build_graph_signals(
        user_id=user_id,
        name=name,
        email=email,
        skills=skills,
        companies=companies,
        education=education,
    )

    return {
        "name": name,
        "email": email,
        "skills": skills,
        "companies": companies,
        "education": education,
        "signals": signals,
        "text_preview": text[:600],
    }
