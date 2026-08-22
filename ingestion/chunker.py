"""Parse every document in the manifest into token-bounded chunks.

HTML becomes heading-scoped chunks of ~300-500 tokens with overlap. PDFs do
the same where the font-size heuristics found headings, and fall back to
fixed 400-token windows where they did not.

Output: data/chunks/chunks.jsonl.

Run:  python -m ingestion.chunker
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

from ingestion.parser import Block, parse_html, parse_pdf
from rag.config import get_settings
from rag.course_codes import all_course_codes, is_course_code

settings = get_settings()

_WORD_RE = re.compile(r"\S+")


def count_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~1.3 tokens per whitespace word)."""
    return int(len(_WORD_RE.findall(text)) * 1.3) + 1


def _split_sentences(text: str) -> List[str]:
    """Sentence-ish split that also respects existing line breaks."""
    parts: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        parts.extend(p.strip() for p in re.split(r"(?<=[.!?;:])\s+", line) if p.strip())
    return parts


def _window(units: List[str], target: int, overlap: int, hard_max: int) -> List[str]:
    """Pack units into ~target-token windows, carrying `overlap` tokens forward."""
    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = count_tokens(unit)

        # A single oversized unit gets hard-split.
        if unit_tokens > hard_max:
            if current:
                chunks.append(" ".join(current))
                current, current_tokens = [], 0
            words = unit.split()
            step = max(1, int(hard_max / 1.3))
            for i in range(0, len(words), step):
                chunks.append(" ".join(words[i : i + step]))
            continue

        if current_tokens + unit_tokens > target and current:
            chunks.append(" ".join(current))
            # Overlap: carry the tail of the finished chunk forward.
            carry: List[str] = []
            carry_tokens = 0
            for prev in reversed(current):
                prev_tokens = count_tokens(prev)
                if carry_tokens + prev_tokens > overlap:
                    break
                carry.insert(0, prev)
                carry_tokens += prev_tokens
            current, current_tokens = list(carry), carry_tokens

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        chunks.append(" ".join(current))

    return [c for c in chunks if c.strip()]


def _blocks_to_chunks(blocks: Iterable[Block], fixed_window: bool) -> List[Dict]:
    """Group blocks by heading, then window each group into chunks."""
    grouped: List[Dict] = []
    for block in blocks:
        key = (block.heading, block.page)
        if grouped and grouped[-1]["key"] == key:
            grouped[-1]["texts"].append(block.text)
        else:
            grouped.append(
                {
                    "key": key,
                    "heading": block.heading,
                    "page": block.page,
                    "texts": [block.text],
                }
            )

    target = (
        settings.pdf_fallback_tokens if fixed_window else settings.chunk_target_tokens
    )
    out: List[Dict] = []

    for group in grouped:
        body = "\n".join(group["texts"])
        units = _split_sentences(body)
        for text in _window(
            units,
            target=target,
            overlap=settings.chunk_overlap_tokens,
            hard_max=settings.chunk_max_tokens,
        ):
            if count_tokens(text) < settings.chunk_min_tokens:
                continue
            out.append(
                {"text": text, "heading": group["heading"], "page": group["page"]}
            )

    return out


def _merge_short_tail(chunks: List[Dict]) -> List[Dict]:
    """Fold a too-short trailing chunk into its predecessor within a heading."""
    if len(chunks) < 2:
        return chunks
    merged = [chunks[0]]
    for chunk in chunks[1:]:
        prev = merged[-1]
        if (
            count_tokens(chunk["text"]) < settings.chunk_min_tokens
            and chunk["heading"] == prev["heading"]
            and count_tokens(prev["text"]) + count_tokens(chunk["text"])
            <= settings.chunk_max_tokens
        ):
            prev["text"] = prev["text"] + " " + chunk["text"]
        else:
            merged.append(chunk)
    return merged


def chunk_document(record: Dict) -> List[Dict]:
    """Parse + chunk one manifest entry into fully-tagged chunk dicts."""
    path = Path(settings.data_dir.parent / record["path"])
    if not path.exists():
        print("[chunk] MISSING " + str(path))
        return []

    if record["doc_type"] == "html":
        _title, blocks = parse_html(path)
        raw_chunks = _blocks_to_chunks(blocks, fixed_window=False)
        strategy = "html_heading"
        # Every CSE page is titled "CSE - IIT Kanpur", so the curated source
        # name wins. Detail pages carry their own label (the course code).
        doc_title = record.get("doc_label") or record["source_name"]
    else:
        blocks, used_structure = parse_pdf(path)
        if not blocks:
            # No text layer, so a scan. Say so rather than index nothing.
            print("[chunk] NO TEXT LAYER (scanned?) " + record["url"])
            return []
        raw_chunks = _blocks_to_chunks(blocks, fixed_window=not used_structure)
        strategy = "pdf_heading" if used_structure else "pdf_fixed_window"
        doc_title = Path(record["url"]).name

    raw_chunks = _merge_short_tail(raw_chunks)

    chunks: List[Dict] = []
    for i, chunk in enumerate(raw_chunks):
        heading = chunk["heading"]
        # Title + heading trail up front: good retrieval signal, and it keeps
        # the context visible to the LLM. IITK repeats the page name as both
        # <title> and <h1>, so drop the duplicate segments.
        segments = [
            s.strip()
            for s in heading.split(" > ")
            if s.strip() and s.strip().lower() not in doc_title.lower()
        ]
        prefix = " — ".join([doc_title] + segments) + "\n\n"
        chunk_id = record["doc_id"] + "::" + format(i, "04d")
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": prefix + chunk["text"],
                "metadata": {
                    "source_id": record["source_id"],
                    "source_name": record["source_name"],
                    "department": record["department"],
                    "url": record["url"],
                    "doc_type": record["doc_type"],
                    "doc_title": doc_title,
                    "heading": heading,
                    "page": chunk["page"],  # None for HTML
                    "crawled_at": record["crawled_at"],
                    "sha256": record.get("sha256"),
                    "chunk_strategy": strategy,
                    # course_code is the page's own course; course_codes is
                    # every code mentioned, which is what makes a timetable
                    # row findable by code.
                    "course_code": (
                        doc_title.upper() if is_course_code(doc_title) else None
                    ),
                    "course_codes": all_course_codes(prefix + chunk["text"]) or None,
                    "n_tokens": count_tokens(chunk["text"]),
                },
            }
        )
    return chunks


def build_chunks() -> List[Dict]:
    settings.ensure_dirs()
    if not settings.manifest_path.exists():
        raise SystemExit(
            "No manifest at "
            + str(settings.manifest_path)
            + " — run `python -m ingestion.crawler` first."
        )

    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    all_chunks: List[Dict] = []

    for record in manifest:
        chunks = chunk_document(record)
        strategy = chunks[0]["metadata"]["chunk_strategy"] if chunks else "-"
        print(
            "[chunk] {:4s} {:4d} chunks ({})  {}".format(
                record["doc_type"], len(chunks), strategy, record["url"][:80]
            )
        )
        all_chunks.extend(chunks)

    with open(settings.chunks_path, "w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    tokens = [c["metadata"]["n_tokens"] for c in all_chunks] or [0]
    print(
        "\n[done] {} chunks -> {}\n       tokens: min={} avg={} max={}".format(
            len(all_chunks),
            settings.chunks_path,
            min(tokens),
            sum(tokens) // len(tokens),
            max(tokens),
        )
    )
    return all_chunks


def load_chunks() -> List[Dict]:
    with open(settings.chunks_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


if __name__ == "__main__":
    build_chunks()
