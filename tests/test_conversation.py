"""Tests for multi-turn follow-up handling.

The failure these guard against: in a chat box, "who teaches it?" embeds to
nothing useful on its own, so retrieval wanders to unrelated passages and the
model answers about the wrong subject.
"""

from __future__ import annotations

from rag.conversation import (
    MAX_HISTORY_TURNS,
    Turn,
    needs_context,
    render_history,
    retrieval_query,
)

HISTORY = [
    Turn("user", "What is the course CS771?"),
    Turn("assistant", "CS771 is Introduction to Machine Learning [1]."),
]


def test_pronoun_question_needs_context() -> None:
    assert needs_context("who teaches it?")
    assert needs_context("what about that course")
    assert needs_context("and the classroom?")


def test_short_question_needs_context() -> None:
    assert needs_context("who teaches?")


def test_self_contained_question_does_not() -> None:
    assert not needs_context("What are the eligibility requirements for M.Tech?")
    assert not needs_context("Who teaches CS771 Introduction to Machine Learning?")


def test_followup_query_carries_the_earlier_subject() -> None:
    query = retrieval_query("who teaches it?", HISTORY)
    assert "CS771" in query
    assert "who teaches it?" in query


def test_self_contained_query_is_left_alone() -> None:
    q = "What are the eligibility requirements for M.Tech?"
    assert retrieval_query(q, HISTORY) == q


def test_query_ignores_assistant_turns() -> None:
    """Answers are long and quote documents; folding them in would drown out
    the question's own signal."""
    query = retrieval_query("who teaches it?", HISTORY)
    assert "Introduction to Machine Learning" not in query


def test_no_history_is_a_no_op() -> None:
    assert retrieval_query("who teaches it?", []) == "who teaches it?"
    assert retrieval_query("who teaches it?", None) == "who teaches it?"


def test_render_history_labels_and_caps_turns() -> None:
    assert render_history([]) == ""
    long = [Turn("user", "q%d" % i) for i in range(20)]
    rendered = render_history(long)
    assert rendered.count("User:") == MAX_HISTORY_TURNS
    assert "q19" in rendered and "q0" not in rendered


def test_render_history_marks_both_roles() -> None:
    rendered = render_history(HISTORY)
    assert "User: What is the course CS771?" in rendered
    assert "Assistant: CS771 is Introduction" in rendered
