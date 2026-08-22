"""Turn raw HTML and PDFs into ordered Blocks carrying their heading.

Structure only — chunking is ingestion/chunker.py. HTML opens a new section
at every h1-h3; PDFs find headings by font size and tell the caller when
they could not, so it can fall back to fixed windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pdfplumber
from bs4 import BeautifulSoup, Tag

BOILERPLATE_TAGS = ("script", "style", "noscript", "nav", "footer", "header", "form")
HEADING_TAGS = ("h1", "h2", "h3")
TEXT_TAGS = ("p", "li", "td", "th", "dd", "dt", "pre", "blockquote", "h4", "h5", "h6")
CONTENT_SELECTORS = ("main", "article", "#content", ".content", "#main", ".main")
DEDUPE_MIN_CHARS = 80  # below this, repeated text is a label, not a duplicate


@dataclass
class Block:
    """A contiguous piece of text plus the heading trail it belongs to."""

    text: str
    heading_path: List[str] = field(default_factory=list)
    page: Optional[int] = None  # PDFs only

    @property
    def heading(self) -> str:
        return " > ".join(h for h in self.heading_path if h)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------


def _main_content(soup: BeautifulSoup) -> Tag:
    """Pick the content container, without losing the page.

    Some IITK templates put a small `#main` beside the real content, so a
    candidate only wins if it holds most of the body text.
    """
    body = soup.body or soup
    body_len = len(body.get_text(strip=True))

    best: Optional[Tag] = None
    best_len = 0
    for selector in CONTENT_SELECTORS:
        node = soup.select_one(selector)
        if not node:
            continue
        length = len(node.get_text(strip=True))
        if length > best_len:
            best, best_len = node, length

    if best is not None and best_len >= 0.6 * body_len and best_len > 400:
        return best
    return body


def _is_navigation(node: Tag, text: str) -> bool:
    """True for a list item that is really just a menu link.

    DOAA sidebar menus are <li><a>...</a></li> and retrieve badly — all link
    text, no statements. Prose with an inline link is kept.

    <li> only: table cells are data. The CSE courses table links every code,
    and treating those as menu items threw away 90% of the page.
    """
    if node.name != "li":
        return False
    anchors = node.find_all("a")
    if not anchors:
        return False
    anchor_len = sum(len(_clean(a.get_text(" ", strip=True))) for a in anchors)
    return anchor_len >= 0.9 * len(text)


def _is_link_row(row: Tag, cells: List[Tag], text: str) -> bool:
    """True for a menu laid out as a table row.

    The DOAA calendar packs 29 links into 6 cells. The CSE courses table links
    its codes too, so the link ratio cannot separate them — but a menu puts
    several links in one cell where a data row has at most one.
    """
    anchors = row.find_all("a")
    if not anchors or len(anchors) <= len(cells):
        return False
    anchor_len = sum(len(_clean(a.get_text(" ", strip=True))) for a in anchors)
    return anchor_len >= 0.9 * len(text.replace(" | ", ""))


def _row_text(row: Tag) -> str:
    """Render a row as one unit: 'CS330 | Operating Systems'.

    Cell by cell would scatter a record across chunk boundaries and cut the
    code from its name.
    """
    cells = [
        _clean(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])
    ]
    return " | ".join(c for c in cells if c)


def parse_html(path: Path) -> Tuple[str, List[Block]]:
    """(page_title, blocks) for one saved HTML file.

    heading_path is the h1-h3 trail only; the title comes back separately so
    the chunker decides how to render the two together.
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")

    for tag in soup(list(BOILERPLATE_TAGS)):
        tag.decompose()

    title = _clean(soup.title.get_text()) if soup.title else ""
    root = _main_content(soup)

    blocks: List[Block] = []
    heading_path: List[str] = ["", "", ""]  # h1, h2, h3
    buffer: List[str] = []
    seen_text: set[str] = set()
    consumed: set[int] = set()  # cells already emitted as part of a table row

    def flush() -> None:
        if not buffer:
            return
        text = _clean("\n".join(buffer))
        buffer.clear()
        if len(text) >= 40:  # drop nav crumbs and one-word table cells
            blocks.append(Block(text=text, heading_path=[h for h in heading_path if h]))

    for node in root.descendants:
        if not isinstance(node, Tag):
            continue

        if node.name in HEADING_TAGS:
            flush()
            text = _clean(node.get_text(" ", strip=True))
            if not text:
                continue
            level = int(node.name[1]) - 1  # h1 -> 0
            heading_path[level] = text
            for deeper in range(level + 1, 3):
                heading_path[deeper] = ""
            continue

        if node.name == "tr" and not node.find("table"):
            cells = node.find_all(["td", "th"])
            text = _row_text(node)
            if text and not _is_link_row(node, cells, text):
                buffer.append(text)
            consumed.update(id(cell) for cell in cells)
            continue

        if node.name in TEXT_TAGS:
            if id(node) in consumed:  # already emitted as part of its table row
                continue
            # Skip containers whose text the nested tags will pick up anyway.
            if node.find(list(TEXT_TAGS)):
                continue
            text = _clean(node.get_text(" ", strip=True))
            if not text:
                continue
            if _is_navigation(node, text):
                continue
            # Long text only. Short strings repeat legitimately: "Assistant
            # Professor" appears once per person, and deduping those deleted
            # most of the faculty roster.
            if len(text) >= DEDUPE_MIN_CHARS:
                if text in seen_text:
                    continue
                seen_text.add(text)
            buffer.append(text)

    flush()
    return title, blocks


# --------------------------------------------------------------------------
# PDF
# --------------------------------------------------------------------------


def _line_records(page) -> List[Dict]:
    """Group words into lines, keeping each line's largest font size.

    Words, not chars: pdfplumber works out the spaces from x-gaps, and joining
    raw chars loses every one of them.
    """
    try:
        words = page.extract_words(
            extra_attrs=["size", "fontname"], keep_blank_chars=False
        )
    except Exception:
        return []

    if not words:
        return []

    lines: Dict[int, List[dict]] = {}
    for word in words:
        key = int(round(word["top"] / 3.0))  # ~3pt tolerance binds a line together
        lines.setdefault(key, []).append(word)

    records = []
    for key in sorted(lines):
        row = sorted(lines[key], key=lambda w: w["x0"])
        text = _clean(" ".join(w["text"] for w in row))
        if not text:
            continue
        sizes = [round(float(w.get("size", 0.0)), 1) for w in row]
        fonts = [str(w.get("fontname", "")) for w in row]
        records.append(
            {
                "text": text,
                "max_size": max(sizes),
                "bold": any("bold" in f.lower() or "black" in f.lower() for f in fonts),
                "top": row[0]["top"],
            }
        )
    return records


def parse_pdf(path: Path) -> Tuple[List[Block], bool]:
    """(blocks, used_heading_structure).

    A heading is a line meaningfully larger than the modal body font, or bold
    and short. Under 2 headings in the whole document means the caller should
    use fixed windows instead.

    A scan with no text layer returns ([], False) so the chunker can report it
    rather than silently index nothing.
    """
    all_lines: List[Dict] = []
    with pdfplumber.open(str(path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for record in _line_records(page):
                record["page"] = page_no
                all_lines.append(record)

    if not all_lines:
        return [], False

    # Modal size, weighted by characters, is the body text size.
    size_counts: Dict[float, int] = {}
    for line in all_lines:
        size_counts[line["max_size"]] = size_counts.get(line["max_size"], 0) + len(
            line["text"]
        )
    body_size = max(size_counts, key=size_counts.get)

    def is_heading(line: Dict) -> bool:
        if len(line["text"]) > 120 or len(line["text"]) < 3:
            return False
        if line["max_size"] >= body_size * 1.15:
            return True
        return line["bold"] and line["max_size"] > body_size and len(line["text"]) < 80

    heading_count = sum(1 for line in all_lines if is_heading(line))
    used_structure = heading_count >= 2

    blocks: List[Block] = []
    current_heading = ""
    buffer: List[str] = []
    buffer_page: Optional[int] = None

    def flush() -> None:
        nonlocal buffer, buffer_page
        if buffer:
            text = _clean("\n".join(buffer))
            if text:
                blocks.append(
                    Block(
                        text=text,
                        heading_path=[current_heading] if current_heading else [],
                        page=buffer_page,
                    )
                )
        buffer = []
        buffer_page = None

    for line in all_lines:
        if used_structure and is_heading(line):
            flush()
            current_heading = line["text"]
            continue
        if buffer_page is None:
            buffer_page = line["page"]
        buffer.append(line["text"])

    flush()
    return blocks, used_structure


def pdf_has_text_layer(path: Path) -> bool:
    """True if the PDF carries extractable text (i.e. is not a pure scan)."""
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages[:3]:
            if (page.extract_text() or "").strip():
                return True
    return False
