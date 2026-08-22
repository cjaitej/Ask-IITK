"""The two retrieval-shaping rules: exact code lookup and source capping.

Both guard regressions that actually happened once course pages became 83% of
the index — naming a course stopped retrieving it, and every result came
from one source.
"""

from __future__ import annotations

from types import SimpleNamespace

from rag.course_codes import all_course_codes, extract_course_codes, is_course_code
from rag.pipeline import RagPipeline


def hit(source_id: str, node_id: str):
    """Minimal stand-in for a search Hit."""
    return SimpleNamespace(
        node=SimpleNamespace(node_id=node_id, metadata={"source_id": source_id})
    )


# ---------------------------------------------------------------- code parsing

def test_codes_are_extracted_and_normalised() -> None:
    assert extract_course_codes("pre-requisites for CS345?") == ["CS345"]
    assert extract_course_codes("Who teaches CS 771?") == ["CS771"]
    assert extract_course_codes("ESO207 and ESC111") == ["ESO207", "ESC111"]


def test_codes_are_deduplicated_and_capped() -> None:
    assert extract_course_codes("CS345 CS345 CS345") == ["CS345"]
    assert len(extract_course_codes("CS1 CS201 CS202 CS203 CS204 CS205")) <= 3


def test_no_codes_in_ordinary_questions() -> None:
    assert extract_course_codes("What are the eligibility requirements?") == []


def test_all_course_codes_collects_every_mention() -> None:
    """A timetable row names a course without being that course's page."""
    row = "CS330 | Operating Systems | CS345 | Algorithms II"
    assert all_course_codes(row) == ["CS330", "CS345"]


def test_is_course_code_distinguishes_pages() -> None:
    assert is_course_code("CS698B") and is_course_code("ESO207")
    assert not is_course_code("Faculty") and not is_course_code("")


# ------------------------------------------------------------- diversification

def test_diversify_caps_a_dominant_source() -> None:
    """With enough variety available, the cap holds and no backfill is needed."""
    hits = [hit("cse_courses", str(i)) for i in range(10)]
    hits.insert(6, hit("cse_faculty", "fac"))
    hits.insert(8, hit("cse_timetable", "tt"))
    hits.insert(9, hit("admissions", "adm"))

    kept = RagPipeline._diversify(hits, limit=6, max_per_source=3)
    sources = [h.node.metadata["source_id"] for h in kept]

    assert sources.count("cse_courses") == 3, "dominant source must be capped"
    assert {"cse_faculty", "cse_timetable", "admissions"} <= set(sources), (
        "lower-ranked sources must get through — this is the whole point"
    )


def test_diversify_backfill_only_tops_up_what_the_cap_left_short() -> None:
    """11 candidates, limit 6, cap 3: the faculty chunk still gets in, and the
    remaining slots go back to the dominant source rather than sitting empty."""
    hits = [hit("cse_courses", str(i)) for i in range(10)]
    hits.insert(6, hit("cse_faculty", "fac"))

    kept = RagPipeline._diversify(hits, limit=6, max_per_source=3)
    sources = [h.node.metadata["source_id"] for h in kept]

    assert len(kept) == 6
    assert "cse_faculty" in sources


def test_diversify_preserves_order_within_the_cap() -> None:
    hits = [hit("a", "1"), hit("b", "2"), hit("a", "3")]
    kept = RagPipeline._diversify(hits, limit=3, max_per_source=2)
    assert [h.node.node_id for h in kept] == ["1", "2", "3"]


def test_diversify_backfills_rather_than_returning_short() -> None:
    """A genuinely single-source question must still fill its context."""
    hits = [hit("only", str(i)) for i in range(8)]
    kept = RagPipeline._diversify(hits, limit=6, max_per_source=3)
    assert len(kept) == 6


def test_diversify_handles_fewer_hits_than_the_limit() -> None:
    kept = RagPipeline._diversify([hit("a", "1")], limit=6, max_per_source=3)
    assert len(kept) == 1
