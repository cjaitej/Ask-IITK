"""Retrieval and generation: embed -> top-k from Qdrant -> Gemini -> answer.

retrieve() works on its own without a Gemini key, which is what the eval
script scores and what /retrieve exposes.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rag.config import get_settings
from rag.prompts import ANSWER_TEMPLATE, SYSTEM_PROMPT, build_context
from rag.conversation import Turn, render_history, retrieval_query
from rag.course_codes import extract_course_codes
from rag.temporal import augment_query, format_today

settings = get_settings()

_CITATION_RE = re.compile(r"\[(\d+)\]")
_RETRY_AFTER_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


class RateLimited(RuntimeError):
    """Gemini refused on quota.

    Its own type because the free tier allows 15 requests/minute, so this is
    routine during eval runs and traffic bursts. As a 500 it looked like a bug.
    """

    def __init__(self, message: str, retry_after: int = 60) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _as_rate_limit(exc: Exception) -> Optional[RateLimited]:
    """Spot a quota rejection without importing the provider SDK here."""
    text = str(exc)
    if "429" not in text and "RESOURCE_EXHAUSTED" not in text:
        return None
    match = _RETRY_AFTER_RE.search(text)
    wait = int(float(match.group(1))) + 1 if match else 60
    return RateLimited(
        "Gemini rate limit reached ({} requests/minute on the free tier). "
        "Retry in about {}s.".format(15, wait),
        retry_after=wait,
    )


@dataclass
class Passage:
    rank: int
    score: float
    text: str
    url: str
    source_id: str
    source_name: str
    department: str
    doc_type: str
    heading: str = ""
    page: Optional[int] = None
    chunk_id: str = ""

    def as_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "score": round(self.score, 4),
            "text": self.text,
            "url": self.url,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "department": self.department,
            "doc_type": self.doc_type,
            "heading": self.heading,
            "page": self.page,
            "chunk_id": self.chunk_id,
        }


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[Dict] = field(default_factory=list)
    retrieved: List[Dict] = field(default_factory=list)
    latency_ms: int = 0
    model: str = ""


class RagPipeline:
    """Embedding model, Qdrant client and LLM, held for the process."""

    def __init__(self) -> None:
        from rag.embeddings import get_embed_model
        from rag.vector_store import get_qdrant_client

        self._client = get_qdrant_client()
        self._embed_model = get_embed_model()
        self._llm = None  # lazily built; retrieval must work without a key

    # -- retrieval ---------------------------------------------------------

    def _search(
        self, vector: List[float], k: int, key: str = "", value: str = ""
    ) -> List:
        """Top-k for an already-embedded query, optionally filtered."""
        from rag.vector_store import match_filter, search

        return search(
            self._client,
            vector,
            limit=k,
            where=match_filter(key, value) if key else None,
        )

    @staticmethod
    def _diversify(hits: List, limit: int, max_per_source: int) -> List:
        """Take `limit` hits, at most `max_per_source` from any one source.

        Course pages are 83% of the index, so a plain top-k returns nothing
        else: the faculty chunk for "Sunil Simon's designation" sat at rank 13.
        Capped hits are kept as backfill so a single-source question still
        fills its context.
        """
        kept, overflow, per_source = [], [], {}
        for hit in hits:
            source = (hit.node.metadata or {}).get("source_id", "?")
            if per_source.get(source, 0) < max_per_source:
                per_source[source] = per_source.get(source, 0) + 1
                kept.append(hit)
            else:
                overflow.append(hit)
            if len(kept) >= limit:
                return kept
        return (kept + overflow)[:limit]

    def _exact_code_hits(self, codes: List[str], vector: List[float]) -> List:
        """Chunks for explicitly named courses, own-page first.

        Two passes per code: `course_code` is the course's own page (contents,
        pre-requisites), `course_codes` also catches the timetable row and
        catalogue line that mention it (instructor, room, credits).
        """
        hits, seen = [], set()
        for code in codes:
            # The course's own page — the exact match.
            passes = [self._search(vector, settings.code_top_k, "course_code", code)]
            # Rows that merely name it, spread across sources so the timetable
            # is not buried under sibling course pages.
            mentions = self._search(
                vector, settings.code_mentions_top_k * 3, "course_codes", code
            )
            passes.append(
                self._diversify(
                    mentions, settings.code_mentions_top_k, settings.max_per_source
                )
            )
            for group in passes:
                for hit in group:
                    if hit.node.node_id not in seen:
                        seen.add(hit.node.node_id)
                        hits.append(hit)
        return hits


    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[Sequence[Turn]] = None,
    ) -> List[Passage]:
        k = top_k or settings.top_k
        # Restore what a follow-up left out, expand shorthand, add the year.
        resolved = retrieval_query(question, history)
        query = augment_query(resolved)

        # Embedded once and reused below — a question naming two course codes
        # issues five searches against the same query. get_query_embedding is
        # what applies BGE's query-side prefix.
        vector = self._embed_model.get_query_embedding(query)

        # Exact-key pass first: similarity search cannot tell CS345 from CS346.
        hits = self._exact_code_hits(extract_course_codes(resolved), vector)
        seen_ids = {h.node.node_id for h in hits}

        deep = self._search(vector, k * settings.fetch_multiplier)
        fresh = [h for h in deep if h.node.node_id not in seen_ids]
        for hit in self._diversify(fresh, k, settings.max_per_source):
            seen_ids.add(hit.node.node_id)
            hits.append(hit)

        passages: List[Passage] = []
        for rank, hit in enumerate(hits, start=1):
            meta = hit.node.metadata or {}
            passages.append(
                Passage(
                    rank=rank,
                    score=float(hit.score or 0.0),
                    text=hit.node.get_content(),
                    url=meta.get("url", ""),
                    source_id=meta.get("source_id", ""),
                    source_name=meta.get("source_name", ""),
                    department=meta.get("department", ""),
                    doc_type=meta.get("doc_type", ""),
                    heading=meta.get("heading", ""),
                    page=meta.get("page"),
                    chunk_id=meta.get("chunk_id", hit.node.node_id),
                )
            )
        return passages

    # -- generation --------------------------------------------------------

    def _get_llm(self):
        if self._llm is not None:
            return self._llm
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env to enable generation; "
                "retrieval works without it."
            )
        from llama_index.llms.google_genai import GoogleGenAI

        self._llm = GoogleGenAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            temperature=0.1,
        )
        return self._llm

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        history: Optional[Sequence[Turn]] = None,
    ) -> Answer:
        started = time.perf_counter()
        passages = self.retrieve(question, top_k=top_k, history=history)

        if not passages:
            return Answer(
                question=question,
                answer="I could not find this in the indexed IIT Kanpur sources.",
                latency_ms=int((time.perf_counter() - started) * 1000),
                model=settings.gemini_model,
            )

        passage_dicts = [p.as_dict() for p in passages]
        context = build_context(passage_dicts, settings.max_context_chars)
        prompt = ANSWER_TEMPLATE.format(
            system=SYSTEM_PROMPT,
            today=format_today(),
            history=render_history(history),
            context=context,
            question=question,
        )

        try:
            response = self._get_llm().complete(prompt)
        except Exception as exc:
            limited = _as_rate_limit(exc)
            if limited is not None:
                raise limited from exc
            raise
        text = str(response).strip()

        return Answer(
            question=question,
            answer=text,
            sources=_cited_sources(text, passage_dicts),
            retrieved=[
                {k: v for k, v in p.items() if k != "text"} for p in passage_dicts
            ],
            latency_ms=int((time.perf_counter() - started) * 1000),
            model=settings.gemini_model,
        )

    def stats(self) -> Dict:
        """Collection stats through the client already open here."""
        from rag.vector_store import collection_stats

        return collection_stats(self._client)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # pragma: no cover - shutdown best effort
            pass


def _cited_sources(answer_text: str, passages: List[Dict]) -> List[Dict]:
    """Map the [n] markers in the answer back to real documents.

    Falls back to every retrieved passage when the model cited nothing.
    """
    numbers = sorted({int(n) for n in _CITATION_RE.findall(answer_text)})
    chosen = [passages[n - 1] for n in numbers if 1 <= n <= len(passages)]
    if not chosen:
        chosen = passages

    sources: List[Dict] = []
    seen = set()
    for passage in chosen:
        key = (passage["url"], passage["page"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                # The marker this passage carried in the prompt, so the UI can
                # match an inline [n] to its document.
                "citation": passage["rank"],
                "source_name": passage["source_name"],
                "department": passage["department"],
                "url": passage["url"],
                "page": passage["page"],
                "heading": passage["heading"],
                "score": passage["score"],
            }
        )
    return sources


_PIPELINE: Optional[RagPipeline] = None


def get_pipeline() -> RagPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = RagPipeline()
    return _PIPELINE


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or "What is the academic calendar for 2026?"
    pipeline = get_pipeline()
    try:
        for passage in pipeline.retrieve(question):
            print(
                "{:>2}. {:.3f}  {}  {}".format(
                    passage.rank,
                    passage.score,
                    passage.source_id,
                    (passage.heading or passage.url)[:70],
                )
            )
    finally:
        # Local mode locks data/qdrant_local; release it before exit.
        pipeline.close()
