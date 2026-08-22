"""Tests for temporal grounding.

The bug these cover: "what is the present sem name?" returned "I could not
find this" even though the 2026 calendar was indexed. Retrieval scored 0.54
and ranked the 2025 calendar above the 2026 one, and the prompt never told the
model what day it was.
"""

from __future__ import annotations

from datetime import date

from rag.temporal import (
    augment_query,
    expand_abbreviations,
    format_today,
    is_time_relative,
)

AUG_2026 = date(2026, 8, 22)


def test_abbreviation_is_expanded_but_the_original_is_kept() -> None:
    out = expand_abbreviations("present sem dates")
    assert "sem" in out and "semester" in out


def test_expansion_survives_punctuation() -> None:
    assert "semester" in expand_abbreviations("what is the present sem?")


def test_expansion_leaves_unknown_words_alone() -> None:
    assert expand_abbreviations("when is convocation") == "when is convocation"


def test_time_relative_detection() -> None:
    assert is_time_relative("what is the present sem name?")
    assert is_time_relative("which semester is running now")
    assert not is_time_relative("When was the CSE department founded?")
    assert not is_time_relative("What is the course CS335?")


def test_relative_question_gets_anchored_to_the_year() -> None:
    query = augment_query("what is the present sem name?", AUG_2026)
    assert "2026" in query
    assert "semester" in query


def test_absolute_question_is_not_anchored() -> None:
    """Appending a year here would drag a founding-date question toward the
    academic calendar."""
    query = augment_query("When was the CSE department founded?", AUG_2026)
    assert "2026" not in query


def test_format_today_is_readable_and_unpadded() -> None:
    assert format_today(AUG_2026) == "Saturday, 22 August 2026"
    assert format_today(date(2026, 8, 5)) == "Wednesday, 5 August 2026"
