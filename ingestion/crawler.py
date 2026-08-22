"""Fetch the URLs in sources.yaml, plus the PDFs and detail pages they link.

Everything lands in data/raw/ with a manifest recording the crawl time,
source URL and SHA256. No open-ended crawling.

Run:  python -m ingestion.crawler
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup

from rag.config import Source, get_settings

settings = get_settings()

PDF_HREF_RE = re.compile(r"\.pdf(\?.*)?$", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": settings.user_agent})
    return s


def _safe_name(url: str, suffix: str) -> str:
    """Stable, filesystem-safe filename for a URL."""
    parsed = urlparse(url)
    stem = f"{parsed.netloc}{parsed.path}".strip("/")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", unquote(stem)) or "index"
    stem = stem[:110].strip("_")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return f"{stem}__{digest}{suffix}"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fix_encoding(resp: requests.Response) -> None:
    """Sniff the encoding when the header does not declare one.

    IITK serves `Content-Type: text/html` with no charset, which per RFC means
    Latin-1 — so requests decodes UTF-8 pages as Latin-1 and mangles every
    non-ASCII character. The course timetable is full of them.
    """
    declared = resp.headers.get("Content-Type", "").lower()
    if "charset=" not in declared:
        resp.encoding = resp.apparent_encoding or "utf-8"


def fetch_html(session: requests.Session, source: Source) -> Optional[Dict]:
    """Download one source page and persist it under data/raw/html."""
    print(f"[html] GET {source.url}")
    try:
        resp = session.get(source.url, timeout=settings.request_timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[html] FAILED {source.url}: {exc}")
        return None

    _fix_encoding(resp)
    crawled_at = _now()
    filename = _safe_name(source.url, ".html")
    path = settings.raw_html_dir / filename
    path.write_text(resp.text, encoding="utf-8", errors="ignore")

    return {
        "doc_id": f"{source.id}::page",
        "doc_type": "html",
        "source_id": source.id,
        "source_name": source.name,
        "department": source.department,
        "url": resp.url,
        "requested_url": source.url,
        "path": str(path.relative_to(settings.data_dir.parent)),
        "crawled_at": crawled_at,
        "sha256": _sha256_bytes(resp.content),
        "status_code": resp.status_code,
        "bytes": len(resp.content),
    }


def discover_pdf_links(html: str, base_url: str, source: Source) -> List[str]:
    """Collect PDF links on the page, filtered by the source's pdf_include list."""
    soup = BeautifulSoup(html, "lxml")
    found: List[str] = []
    seen = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not PDF_HREF_RE.search(href):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue

        if source.pdf_include:
            name = unquote(urlparse(absolute).path.rsplit("/", 1)[-1])
            if not any(pat.lower() in name.lower() for pat in source.pdf_include):
                continue

        seen.add(absolute)
        found.append(absolute)
        if len(found) >= source.max_pdfs:
            break

    return found


def discover_links(html: str, base_url: str, source: Source) -> List[str]:
    """Same-host links matching the source's `link_pattern`, capped.

    One hop and pattern-gated. The CSE catalogue links a page per course, and
    those carry the units and pre-requisites the catalogue itself does not.
    """
    if not source.follow_links or not source.link_pattern:
        return []

    pattern = re.compile(source.link_pattern, re.IGNORECASE)
    host = urlparse(base_url).netloc
    soup = BeautifulSoup(html, "lxml")

    found: List[str] = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        absolute = urljoin(base_url, anchor["href"].strip()).split("#")[0]
        if absolute in seen or absolute == base_url:
            continue
        if urlparse(absolute).netloc != host:
            continue
        if not pattern.search(absolute):
            continue
        seen.add(absolute)
        found.append(absolute)
        if len(found) >= source.max_links:
            break
    return found


def fetch_linked_page(
    session: requests.Session, url: str, source: Source
) -> Optional[Dict]:
    """Fetch one detail page discovered from a hub page."""
    try:
        resp = session.get(url, timeout=settings.request_timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[link] FAILED {url}: {exc}")
        return None

    _fix_encoding(resp)
    stem = unquote(urlparse(resp.url).path.rsplit("/", 1)[-1]).rsplit(".", 1)[0]
    filename = _safe_name(url, ".html")
    path = settings.raw_html_dir / filename
    path.write_text(resp.text, encoding="utf-8", errors="ignore")

    return {
        "doc_id": f"{source.id}::{stem}",
        "doc_type": "html",
        # Every CSE page is titled "CSE - IIT Kanpur", so the label comes from
        # the URL instead — otherwise all 186 course pages cite identically.
        "doc_label": stem,
        "source_id": source.id,
        "source_name": source.name,
        "department": source.department,
        "url": resp.url,
        "parent_url": source.url,
        "path": str(path.relative_to(settings.data_dir.parent)),
        "crawled_at": _now(),
        "sha256": _sha256_bytes(resp.content),
        "status_code": resp.status_code,
        "bytes": len(resp.content),
    }


def fetch_pdf(session: requests.Session, url: str, source: Source) -> Optional[Dict]:
    """Download one PDF, store it by name and record its SHA256 + provenance."""
    print(f"[pdf ] GET {url}")
    try:
        resp = session.get(url, timeout=settings.request_timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[pdf ] FAILED {url}: {exc}")
        return None

    if not resp.content.startswith(b"%PDF"):
        print(f"[pdf ] SKIP (not a PDF body) {url}")
        return None

    digest = _sha256_bytes(resp.content)
    filename = _safe_name(url, ".pdf")
    path = settings.raw_pdf_dir / filename
    path.write_bytes(resp.content)

    return {
        "doc_id": f"{source.id}::{digest[:12]}",
        "doc_type": "pdf",
        "source_id": source.id,
        "source_name": source.name,
        "department": source.department,
        "url": resp.url,
        "parent_url": source.url,
        "path": str(path.relative_to(settings.data_dir.parent)),
        "crawled_at": _now(),
        "sha256": digest,
        "status_code": resp.status_code,
        "bytes": len(resp.content),
    }


def crawl() -> List[Dict]:
    """Crawl every source and write data/raw/manifest.json."""
    settings.ensure_dirs()
    session = _session()
    manifest: List[Dict] = []

    for source in settings.load_sources():
        page = fetch_html(session, source)
        if page is None:
            continue
        manifest.append(page)

        html = Path(settings.data_dir.parent / page["path"]).read_text(
            encoding="utf-8", errors="ignore"
        )

        detail_urls = discover_links(html, page["url"], source)
        if detail_urls:
            print(f"[link] {len(detail_urls)} detail page(s) for source '{source.id}'")
            for i, link in enumerate(detail_urls, start=1):
                time.sleep(settings.link_crawl_delay_seconds)
                record = fetch_linked_page(session, link, source)
                if record:
                    manifest.append(record)
                if i % 25 == 0:
                    print(f"[link]   {i}/{len(detail_urls)}")

        if not source.follow_pdfs:
            continue

        links = discover_pdf_links(html, page["url"], source)
        print(f"[pdf ] {len(links)} candidate PDF(s) for source '{source.id}'")

        for link in links:
            time.sleep(settings.crawl_delay_seconds)
            record = fetch_pdf(session, link, source)
            if record:
                manifest.append(record)

        time.sleep(settings.crawl_delay_seconds)

    settings.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    settings.manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    html_n = sum(1 for m in manifest if m["doc_type"] == "html")
    pdf_n = sum(1 for m in manifest if m["doc_type"] == "pdf")
    print(f"\n[done] {html_n} HTML page(s), {pdf_n} PDF(s) -> {settings.manifest_path}")
    return manifest


if __name__ == "__main__":
    crawl()
