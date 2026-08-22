"""Date grounding for time-relative questions.

"What is the present semester?" carries no year, so it matches the 2025
calendar as readily as the 2026 one. augment_query() adds the year before
embedding; format_today() gives the prompt today's date.

Working out which semester it actually is stays out of here on purpose — that
has to come from the calendar, or it goes stale when the calendar changes.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

# Words that make a question depend on when it is asked.
RELATIVE_TERMS = frozenset(
    {
        "present", "current", "currently", "now", "today", "ongoing",
        "running", "this", "upcoming", "next", "latest", "recent",
    }
)

# Campus shorthand BGE has no training signal for.
ABBREVIATIONS = {
    "sem": "semester",
    "sems": "semesters",
    "dept": "department",
    "prof": "professor",
    "profs": "professors",
    "ug": "undergraduate",
    "pg": "postgraduate",
    "hol": "holiday",
}

_WORD = re.compile(r"[a-z]+")


def today() -> date:
    """Wrapped so tests can pin it."""
    return date.today()


def format_today(when: Optional[date] = None) -> str:
    """Human date for the prompt, e.g. 'Saturday, 22 August 2026'."""
    when = when or today()
    return when.strftime("%A, %d %B %Y").replace(" 0", " ")


def expand_abbreviations(question: str) -> str:
    """Expand shorthand, keeping the original word alongside it.

    The abbreviation may appear in the source text too, so dropping it would
    trade one miss for another.
    """
    out = []
    for token in question.split():
        bare = _WORD.findall(token.lower())
        expansion = ABBREVIATIONS.get(bare[0]) if bare else None
        out.append(f"{token} {expansion}" if expansion else token)
    return " ".join(out)


def is_time_relative(question: str) -> bool:
    return any(w in RELATIVE_TERMS for w in _WORD.findall(question.lower()))


def augment_query(question: str, when: Optional[date] = None) -> str:
    """The string actually sent to the embedding model.

    The year goes on only for time-relative questions; adding it to "when was
    the CSE department founded?" would drag it toward the calendar.
    """
    query = expand_abbreviations(question)
    if is_time_relative(question):
        when = when or today()
        query = f"{query} (as of {when.strftime('%B %Y')}, academic year {when.year})"
    return query
