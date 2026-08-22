"""Follow-up handling for the chat UI.

"Who teaches it?" embeds to nothing useful, so the embedding query needs the
missing noun put back. Done here by heuristic rather than a condense-question
LLM call, which would double the request count on a 15/minute free tier. The
model gets the raw turns through the prompt and resolves the reference itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Words that point at something said earlier rather than naming it.
_DEICTIC = frozenset(
    {
        "it", "its", "it's", "that", "this", "these", "those", "they", "them",
        "their", "he", "him", "his", "she", "her", "hers", "there", "same",
        "one", "ones", "both", "another",
    }
)

# "and the classroom?" is a follow-up even with no pronoun in it.
_FRAGMENT_STARTS = ("and ", "what about", "how about", "also ", "or ", "then ")

MAX_HISTORY_TURNS = 6      # what the prompt sees (3 exchanges)
MAX_RETRIEVAL_LOOKBACK = 2  # how many earlier user turns feed the embedding

_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class Turn:
    role: str  # "user" | "assistant"
    content: str


def needs_context(question: str) -> bool:
    """True when the question cannot stand on its own."""
    lowered = question.strip().lower()
    if not lowered:
        return False
    if lowered.startswith(_FRAGMENT_STARTS):
        return True
    words = _WORD.findall(lowered)
    if any(w in _DEICTIC for w in words):
        return True
    # "who teaches?" — too short to name a subject.
    return len(words) <= 4


def recent(history: Optional[Sequence[Turn]], limit: int = MAX_HISTORY_TURNS) -> List[Turn]:
    return list(history or [])[-limit:]


def retrieval_query(question: str, history: Optional[Sequence[Turn]] = None) -> str:
    """The question with earlier subjects restored, for embedding.

    User turns only — answers quote the documents at length and would swamp
    the question's own signal.
    """
    if not history or not needs_context(question):
        return question

    earlier = [t.content.strip() for t in history if t.role == "user" and t.content.strip()]
    if not earlier:
        return question

    context = " ".join(earlier[-MAX_RETRIEVAL_LOOKBACK:])
    return f"{context} {question}"


def render_history(history: Optional[Sequence[Turn]]) -> str:
    """The conversation block for the prompt, or "" if there is none."""
    turns = recent(history)
    if not turns:
        return ""
    lines = [
        ("User: " if t.role == "user" else "Assistant: ") + t.content.strip()
        for t in turns
        if t.content.strip()
    ]
    if not lines:
        return ""
    return (
        "--- CONVERSATION SO FAR ---\n" + "\n".join(lines) + "\n--- END CONVERSATION ---\n\n"
    )
