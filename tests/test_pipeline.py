"""Context assembly and the citation contract — no index or API key needed.

Citation mapping is what makes an answer checkable, so it is tested against a
model that cites correctly, one that cites nothing, and one that invents a
passage number.
"""

from __future__ import annotations

from rag.pipeline import _cited_sources
from rag.prompts import build_context

PASSAGES = [
    {
        "rank": 1,
        "score": 0.81,
        "text": "Classes commence July 30, 2026 (Thu).",
        "url": "https://www.iitk.ac.in/doaa/data/Calendar-2026.pdf",
        "source_id": "academic_calendar",
        "source_name": "IITK Academic Calendar (DOAA)",
        "department": "Academic Affairs (DOAA)",
        "doc_type": "pdf",
        "heading": "2026-27-I",
        "page": 1,
        "chunk_id": "academic_calendar::abc::0000",
    },
    {
        "rank": 2,
        "score": 0.66,
        "text": "Admission to M.Tech requires a valid GATE score.",
        "url": "https://www.iitk.ac.in/admissions",
        "source_id": "admissions",
        "source_name": "IITK Admissions",
        "department": "Admissions",
        "doc_type": "html",
        "heading": "Admissions",
        "page": None,
        "chunk_id": "admissions::page::0002",
    },
]


def test_build_context_numbers_passages_and_shows_the_page() -> None:
    context = build_context(PASSAGES, max_chars=10_000)

    assert context.startswith("[1] IITK Academic Calendar (DOAA)")
    assert "(page 1)" in context          # PDFs carry a page locator
    assert "[2] IITK Admissions" in context
    assert "(page None)" not in context   # HTML must not fake one


def test_build_context_respects_the_char_budget() -> None:
    context = build_context(PASSAGES, max_chars=120)
    assert "[1]" in context
    assert "[2]" not in context


def test_cited_sources_keeps_only_what_the_model_cited() -> None:
    sources = _cited_sources("Classes commence July 30, 2026 [1].", PASSAGES)

    assert len(sources) == 1
    assert sources[0]["url"] == PASSAGES[0]["url"]
    assert sources[0]["page"] == 1
    # The label must match the marker in the answer text, or the UI mislabels it.
    assert sources[0]["citation"] == 1


def test_cited_source_keeps_its_original_passage_number() -> None:
    sources = _cited_sources("Only the second one matters [2].", PASSAGES)

    assert len(sources) == 1
    assert sources[0]["citation"] == 2
    assert sources[0]["url"] == PASSAGES[1]["url"]


def test_cited_sources_falls_back_when_nothing_is_cited() -> None:
    sources = _cited_sources("Classes commence in July.", PASSAGES)
    assert len(sources) == 2  # show everything retrieved rather than nothing


def test_cited_sources_drops_an_invented_citation_number() -> None:
    sources = _cited_sources("See [1] and also [7].", PASSAGES)

    assert len(sources) == 1
    assert sources[0]["url"] == PASSAGES[0]["url"]


def test_cited_sources_deduplicates_the_same_document() -> None:
    duplicate = dict(PASSAGES[0], rank=3)
    sources = _cited_sources("[1][2][3]", PASSAGES + [duplicate])

    urls = [s["url"] for s in sources]
    assert len(urls) == len(set(urls)) == 2
