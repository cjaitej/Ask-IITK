"""Course-code extraction, for the exact-match half of retrieval.

Dense embeddings are bad at identifiers: with 182 near-identical course pages
indexed, CS346 scores about as well as CS345. Codes get a metadata filter
instead of a similarity search.
"""

from __future__ import annotations

import re
from typing import List

# CS345, CS 345, CS698B, ESO207, ESC111.
_CODE_RE = re.compile(r"\b(CS|ESO|ESC)\s*-?\s*(\d{3}[A-Z]?)\b", re.IGNORECASE)

MAX_CODES = 3  # distinct codes looked up per question


def extract_course_codes(text: str) -> List[str]:
    """Normalised, de-duplicated codes in order of first appearance."""
    seen: List[str] = []
    for prefix, number in _CODE_RE.findall(text or ""):
        code = (prefix + number).upper()
        if code not in seen:
            seen.append(code)
    return seen[:MAX_CODES]


def all_course_codes(text: str, limit: int = 60) -> List[str]:
    """Every code a chunk mentions, for the `course_codes` payload field.

    A timetable row names courses without being any course's own page.
    """
    seen: List[str] = []
    for prefix, number in _CODE_RE.findall(text or ""):
        code = (prefix + number).upper()
        if code not in seen:
            seen.append(code)
        if len(seen) >= limit:
            break
    return seen


def is_course_code(label: str) -> bool:
    """True when the label is itself a code, i.e. a course detail page."""
    return bool(label) and bool(re.fullmatch(r"(CS|ESO|ESC)\d{3}[A-Z]?", label.upper()))
