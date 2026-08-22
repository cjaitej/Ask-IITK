"""Unit tests for the parts of ingestion that are easy to get subtly wrong:
heading scoping, token windowing and overlap.

Run:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

from ingestion.chunker import _window, count_tokens
from ingestion.parser import parse_html
from rag.config import get_settings

settings = get_settings()

SAMPLE_HTML = """
<html><head><title>Test Page</title></head>
<body>
  <script>var x = 1;</script>
  <main>
    <h1>Admissions</h1>
    <p>The institute admits students through JEE Advanced every year in July.</p>
    <h2>Fee Structure</h2>
    <p>The semester fee for undergraduate students is listed in the fee table.</p>
    <ul><li>Tuition fee is charged per semester and is revised by the Senate.</li></ul>
    <h2>Hostel</h2>
    <p>All admitted students are allotted a hall of residence on joining campus.</p>
  </main>
</body></html>
"""


def test_parse_html_scopes_text_to_headings(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(SAMPLE_HTML, encoding="utf-8")

    title, blocks = parse_html(path)

    assert title == "Test Page"
    assert blocks, "expected at least one block"

    headings = [b.heading for b in blocks]
    assert any("Fee Structure" in h for h in headings)
    assert any("Hostel" in h for h in headings)

    # Text lands under the heading it followed, not the previous one.
    hostel = next(b for b in blocks if "Hostel" in b.heading)
    assert "hall of residence" in hostel.text


def test_parse_html_drops_scripts(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text(SAMPLE_HTML, encoding="utf-8")
    _, blocks = parse_html(path)
    assert all("var x" not in b.text for b in blocks)


def test_window_respects_target_and_overlaps() -> None:
    units = ["word " * 30 for _ in range(20)]  # ~39 tokens each
    chunks = _window(units, target=200, overlap=60, hard_max=500)

    assert len(chunks) > 1
    assert all(count_tokens(c) <= 500 for c in chunks)

    # Consecutive chunks share text — that is the overlap doing its job.
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]


def test_window_hard_splits_an_oversized_unit() -> None:
    monster = "token " * 2000
    chunks = _window([monster], target=400, overlap=60, hard_max=500)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 520 for c in chunks)


def test_sources_yaml_is_the_locked_set() -> None:
    """Sources are curated, not discovered — an unexpected id means someone
    widened the crawl, which is the thing v1 exists to prevent."""
    sources = settings.load_sources()
    assert {s.id for s in sources} == {
        "admissions",
        "academic_calendar",
        "cse_department",
        "cse_faculty",
        "cse_courses",
        "cse_timetable",
        "cse_btech",
        "cse_mtech",
        "cse_ms",
        "cse_phd",
        "cse_minors",
    }
    assert len({s.id for s in sources}) == len(sources), "duplicate source id"
    assert all(s.url.startswith("https://") for s in sources)
    assert all(s.department for s in sources), "department mapping is manual"


TABLE_HTML = """
<html><head><title>CSE - IIT Kanpur</title></head><body><main>
  <h2>Courses Offered</h2>
  <table>
    <tr><th>SNo</th><th>Code</th><th>Course Name</th></tr>
    <tr><td>1</td><td><a href="/c/CS335">CS335</a></td><td><a href="/c/CS335">Compiler Design</a></td></tr>
    <tr><td>2</td><td><a href="/c/CS340">CS340</a></td><td><a href="/c/CS340">Theory of Computation</a></td></tr>
  </table>
  <table>
    <tr>
      <td><a href="/a.pdf">Calendar 2026</a> <a href="/b.pdf">Calendar 2025</a> <a href="/c.pdf">Holidays 2026</a></td>
      <td><a href="/d.pdf">Calendar 2024</a> <a href="/e.pdf">Holidays 2024</a></td>
    </tr>
  </table>
</main></body></html>
"""

REPEATED_HTML = """
<html><head><title>CSE - IIT Kanpur</title></head><body><main>
  <h2>Faculty</h2>
  <p>Ada Lovelace</p><p>Assistant Professor</p><p>Analytical Engines</p>
  <p>Alan Turing</p><p>Assistant Professor</p><p>Computability and Cryptanalysis</p>
</main></body></html>
"""


def test_table_row_is_kept_as_one_unit(tmp_path: Path) -> None:
    """A course code separated from its name is useless — the row must survive
    as a single line."""
    path = tmp_path / "courses.html"
    path.write_text(TABLE_HTML, encoding="utf-8")
    _, blocks = parse_html(path)

    text = "\n".join(b.text for b in blocks)
    assert "CS335 | Compiler Design" in text
    assert "CS340 | Theory of Computation" in text


def test_link_only_table_row_is_dropped(tmp_path: Path) -> None:
    """A row packing many links into few cells is a menu, not data."""
    path = tmp_path / "courses.html"
    path.write_text(TABLE_HTML, encoding="utf-8")
    _, blocks = parse_html(path)

    text = "\n".join(b.text for b in blocks)
    assert "Calendar 2024" not in text  # the link-menu row
    assert "CS335" in text              # the data table alongside it survives


def test_repeated_short_labels_are_not_deduplicated(tmp_path: Path) -> None:
    """Deduping every repeat deleted most of the faculty roster: each person
    shares the designation of the person above them."""
    path = tmp_path / "faculty.html"
    path.write_text(REPEATED_HTML, encoding="utf-8")
    _, blocks = parse_html(path)

    text = "\n".join(b.text for b in blocks)
    assert text.count("Assistant Professor") == 2
    assert "Ada Lovelace" in text and "Alan Turing" in text


HUB_HTML = """
<html><body><main>
  <table>
    <tr><td>1</td><td><a href="CS345.html">CS345</a></td><td><a href="CS345.html">Algorithms II</a></td></tr>
    <tr><td>2</td><td><a href="ESO207.html">ESO207</a></td><td><a href="ESO207.html">Data Structures</a></td></tr>
  </table>
  <a href="Faculty.html">Faculty</a>
  <a href="https://example.com/pages/CS999.html">offsite</a>
  <a href="CS345.html#top">same page again</a>
</main></body></html>
"""


def _courses_source():
    return next(s for s in settings.load_sources() if s.id == "cse_courses")


def test_discover_links_finds_course_pages(tmp_path: Path) -> None:
    from ingestion.crawler import discover_links

    links = discover_links(
        HUB_HTML, "https://www.cse.iitk.ac.in/pages/Courses.html", _courses_source()
    )
    assert "https://www.cse.iitk.ac.in/pages/CS345.html" in links
    assert "https://www.cse.iitk.ac.in/pages/ESO207.html" in links


def test_discover_links_excludes_non_matching_and_offsite(tmp_path: Path) -> None:
    """The pattern is the whole safety net: without it this becomes a crawl."""
    from ingestion.crawler import discover_links

    links = discover_links(
        HUB_HTML, "https://www.cse.iitk.ac.in/pages/Courses.html", _courses_source()
    )
    assert not any("Faculty" in u for u in links)
    assert not any("example.com" in u for u in links)
    assert len(links) == len(set(links)), "fragments must not create duplicates"


def test_discover_links_respects_max_links() -> None:
    from ingestion.crawler import discover_links

    source = _courses_source().model_copy(update={"max_links": 1})
    links = discover_links(
        HUB_HTML, "https://www.cse.iitk.ac.in/pages/Courses.html", source
    )
    assert len(links) == 1


def test_discover_links_off_by_default() -> None:
    from ingestion.crawler import discover_links

    source = _courses_source().model_copy(update={"follow_links": False})
    assert discover_links(HUB_HTML, "https://www.cse.iitk.ac.in/pages/Courses.html", source) == []
