# AskIITK

RAG over a hand-picked set of official IIT Kanpur sources. Ask a question, get an
answer with citations back to the exact page or PDF it came from, over a local
FastAPI endpoint.

A plain retrieve→generate loop: one `/chat` endpoint plus a one-file question
page. No agent framework, no router, no hybrid search — none of it earns its
place at this size.

The corpus grew from 3 sources to 11, driven by measured failures rather than
guesswork. That is a wider corpus, not a wider architecture: nothing was
opened up to crawling, and every page is pinned in `sources.yaml`.

## Stack

| Piece | Choice |
|---|---|
| Embeddings | BGE Small (`BAAI/bge-small-en-v1.5`, 384-dim) |
| Vector store | Qdrant — collection `iitk_documents_v1`, float16 vectors, indexed payload |
| LLM | Gemini 2.5 Flash |
| Framework | LlamaIndex — embedding + LLM wrappers only; Qdrant is talked to directly |
| API | FastAPI |
| Container | one self-contained image; `docker compose up` |

## Sources (locked)

Defined in [sources.yaml](sources.yaml). No open-ended crawling — only these URLs,
plus PDFs they link that match each source's `pdf_include` filter.

| Source | URL |
|---|---|
| Admissions | https://www.iitk.ac.in/new/admissions |
| Academic Calendar (DOAA) | https://www.iitk.ac.in/doaa/academic-calendar |
| CSE Department | https://www.cse.iitk.ac.in/ |
| CSE Faculty | https://www.cse.iitk.ac.in/pages/Faculty.html |
| CSE Courses Offered | https://www.cse.iitk.ac.in/pages/Courses.html (+182 linked course pages) |
| CSE Course Timetable | https://www.cse.iitk.ac.in/pages/CourseTimetable.html |
| CSE BTech / MTech / MS / PhD | `pages/Program*.html` |
| CSE Minor Programmes | https://www.cse.iitk.ac.in/pages/MinorPrograms.html |

The calendar source also pulls 4 linked PDFs (the 2025 and 2026 calendars, the
2026 holiday list, and the eMasters calendar).

Department tagging is a manual `source → department` mapping in `sources.yaml`.
At this size, auto-extraction would be more machinery than the problem needs.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt

cp .env.example .env            # then add your GEMINI_API_KEY
```

Get a Gemini key at https://aistudio.google.com/apikey. Crawl, chunk, embed and
retrieve all work without one — only answer generation needs it.

## Build the index

```bash
python -m scripts.build_all
```

Each stage also runs on its own:

```bash
python -m ingestion.crawler     # fetch HTML + linked PDFs -> data/raw/
python -m ingestion.chunker     # parse + chunk            -> data/chunks/chunks.jsonl
python -m ingestion.indexer     # embed + push to Qdrant
```

Embeddings are cached in `data/chunks/embeddings.npy` and keyed to a fingerprint of
the chunk set, so re-indexing unchanged chunks costs nothing. `--recreate` drops the
collection first; `--no-cache` forces a re-embed.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then open **http://localhost:8000/ui** — a chat box. Messages thread the way
you expect, `[n]` markers in an answer are clickable and scroll to the source
that backs them, and the thread survives a page reload via `localStorage`
("New chat" clears it).

**Follow-ups work.** Ask "Who teaches CS771?", then "and the classroom?" — the
browser sends the prior turns, and the server resolves the reference on both
sides: the earlier subject is folded into the embedding query, and the turns go
into the prompt. See `rag/conversation.py`.

Answers render as Markdown — bold labels, bullets, tables. It is one static
file (`api/static/index.html`) served by the same app: no extra process, no
build step, no dependency.

```bash
curl -s -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "When does the odd semester begin in the 2026 academic calendar?"}'
```

```json
{
  "answer": "... [1]",
  "sources": [
    {
      "source_name": "IITK Academic Calendar (DOAA)",
      "department": "Academic Affairs (DOAA)",
      "url": "https://www.iitk.ac.in/doaa/data/Calendar-2026.pdf",
      "page": 1,
      "heading": "Academic Calendar 2026-27",
      "score": 0.71
    }
  ],
  "latency_ms": 2140,
  "model": "gemini-2.5-flash"
}
```

Other endpoints: `GET /ui` (the question page, and `/` redirects to it),
`GET /health` (index size, whether generation is enabled), `POST /retrieve`
(retrieval only, no LLM call), and Swagger UI at http://localhost:8000/docs.

Each source carries a `citation` number matching the `[n]` marker in the answer
text, so a reader can trace any claim to the document it came from.

## Evaluation

39 hand-verified questions in [tests/eval_questions.yaml](tests/eval_questions.yaml),
covering all eleven sources — course contents and pre-requisites, faculty
research areas, instructors and timings, programme structure, and two
time-relative questions.

```bash
python -m tests.run_eval              # retrieval only — no API key needed
python -m tests.run_eval --generate   # full answers through Gemini
```

It scores the two mechanical things — was the right source retrieved, and was
it cited — and writes `data/processed/eval_run.json` with `answer_correct: null`
per row for you to fill in. Answer quality is graded by hand on purpose.

Last run: **retrieval@6 39/39, top-1 source 32/39**. Retrieval@6 only says the
right source appeared somewhere in the top 6 — it is a floor, not a score.

Unit tests for chunking and the citation contract:

```bash
pytest -q
```

## How the index is stored

`ensure_collection` in [rag/vector_store.py](rag/vector_store.py) creates the
collection explicitly rather than letting it appear on first write, which is
what makes these possible:

- **float16 vectors.** BGE normalises into roughly ±0.2, where float16 still
  carries more precision than cosine ranking can use. Half the bytes, and the
  eval set does not shift.
- **A payload of only what a query reads** — prompt text, the fields an answer
  cites, and the two filter keys. Build-time diagnostics (`sha256`,
  `chunk_strategy`, `n_tokens`) stay in `chunks.jsonl`, which is the record of
  what was indexed.
- **Payload indexes** on `course_code`, `course_codes`, `source_id` and
  `doc_type`. The course-code filters are why naming a course retrieves that
  course, so they should not be a full scan as the corpus grows. Server mode
  only — the embedded client scans regardless.

Points go through `qdrant-client` directly. LlamaIndex's vector store wrote a
`_node_content` blob holding a second copy of the text and every metadata
field — 41% of the payload — and created the collection implicitly, leaving
nowhere to declare an index.

`embeddings.npy` is float32, not the float64 numpy picks for lists of Python
floats. An older float64 cache is rewritten in place rather than re-embedded.

### local vs server

`QDRANT_MODE` in `.env` decides:

- **`local`** (default) — embedded on-disk Qdrant under `data/qdrant_local/`. No
  Docker needed. One process at a time may hold the directory, so stop the API
  before re-indexing. Payload indexes are a no-op here and are skipped.
- **`server`** — a Qdrant container. Nothing else in the code changes:

  ```bash
  docker compose -f docker-compose.yml -f docker-compose.server.yml up -d qdrant
  ```

  Then set `QDRANT_MODE=server` and re-run `python -m ingestion.indexer`.

## Deployment

One self-contained image, deployed to **Azure Container Apps**. Both slow things
are baked in at build time, so the running container needs no sidecar, no volume
and no network at startup:

- **BGE Small** → `/models` (`HF_HUB_OFFLINE=1` at runtime, so it can never
  reach out for a model file)
- **the Qdrant index** → `/app/data/qdrant_local`, embedded mode

The index is rebuilt inside the builder stage from `data/chunks/`. `embeddings.npy`
is a cache keyed by a chunk fingerprint, so that step is a copy, not a re-embed.
**The image is the index**: when sources change, re-run the ingestion locally and
rebuild.

```bash
python -m scripts.build_all --recreate   # crawl -> chunk -> embed -> index
docker compose build                     # bakes the refreshed chunks into the image
```

torch is installed from PyTorch's CPU index. Resolved from PyPI it pulls the
CUDA build and its `nvidia-*` wheels — several GB of GPU runtime for an image
that only ever runs BGE Small on a Container Apps vCPU.

`data/chunks/` is committed for the same build: it is the one part of `data/`
the build context needs, and without it a fresh clone has nothing to index.

### Local run

```bash
docker compose up --build           # -> http://localhost:8000/ui
```

`GEMINI_API_KEY` comes from `.env` at runtime and is never baked into the image.
Without a key the app still starts and `/health` and `/retrieve` work; only
`/chat` needs Gemini, and it returns 503 without one.

Or without compose:

```bash
docker run --rm -p 8000:8000 -e GEMINI_API_KEY=... askiitk:v1
```

No `QDRANT_MODE` or `QDRANT_URL` needed — the image defaults to the baked-in
embedded index.

### Against a Qdrant server

The overlay swaps the baked-in index for a real Qdrant with a persistent volume,
which is what you want while iterating on the corpus: re-load the collection
without rebuilding the image.

```bash
docker compose -f docker-compose.yml -f docker-compose.server.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.server.yml run --rm indexer
```

`indexer` is a one-shot service built from the Dockerfile's `builder` stage —
the only stage carrying `ingestion/` and `data/chunks/`. `down -v` drops the
volume and the collection with it.

### Azure

```powershell
az login
.\deploy\azure.ps1
```

The script is idempotent — re-run it to ship a change. It creates the resource
group, an ACR, a Container Apps environment and the app, then prints the URL.
`az acr build` builds the image in Azure, so nothing large is uploaded and the
architecture always matches (linux/amd64).

| Setting | Value | Why |
|---|---|---|
| CPU / memory | 1.0 / 2.0Gi | torch + BGE Small resident |
| replicas | 0–2 | scale to zero; idle costs nothing |
| ingress | external, port 8000 | HTTPS and a public FQDN come free |
| `GEMINI_API_KEY` | Container Apps secret | never in the image — `.dockerignore` excludes `.env` |

`-MinReplicas 1` keeps one replica warm. At zero, the first request after an idle
period pays a cold start (~30s) while torch and the model load.

```powershell
.\deploy\azure.ps1 -Tag v2 -MinReplicas 1     # ship an update, stay warm
az containerapp logs show -n iitk-rag -g iitk-rag-rg --follow
az group delete -n iitk-rag-rg --yes --no-wait  # tear it all down
```

### Qdrant as a separate service

Still supported and unchanged — set `QDRANT_MODE=server` and `QDRANT_URL`. That
is the right shape once the index is large enough to want writes at runtime, or
shared across replicas. For 352 chunks it would be two services doing one
service's work.

## What is in the index

197 documents, 352 chunks: 193 HTML pages and 4 PDFs.

| Source | Chunks | Answers questions about |
|---|---|---|
| `cse_courses` | 293 | course contents, pre-requisites, units — one page per course |
| `academic_calendar` | 12 | semester dates, holidays, exams (all from PDFs) |
| `cse_department` | 7 | department history, news, awards |
| `admissions` | 4 | entrance exams, eligibility, age limits |
| `cse_faculty` | 4 | who works on what, phone, office, email format |
| `cse_timetable` | 3 | instructors, timings, classrooms, credits |
| `cse_btech` / `mtech` / `ms` / `phd` / `minors` | 13 | programme structure, requirements |

The 182 per-course pages are followed one hop from the catalogue, gated by
`link_pattern` in `sources.yaml`. Still not a crawl: same host, one hop, an
explicit regex and a hard cap.

## Known gaps

Measured, not guessed — a batch of 18 realistic questions leaves 8 unanswered:

- **Institute-wide topics are not indexed**: fees, hostels, placements
  (`spo.iitk.ac.in`), campus location, convocation. These live on separate
  subdomains; pulling them in is the design's "expand to full site crawl", a v2
  item, not a config tweak.
- **Course prerequisites** are not published on any page currently indexed.
- **Counting questions fail** ("how many faculty are there?"). The roster spans
  four chunks and top-k retrieval only ever sees some of them, so the model
  cannot count. This is structural to RAG, not a corpus gap — it needs an
  aggregate query path, not more documents.

## Layout

```
sources.yaml            the source list + department mapping
ingestion/
  crawler.py            fetch HTML, discover + download linked PDFs
  parser.py             HTML heading walk; PDF font-size heading detection
  chunker.py            token windowing with overlap -> chunks.jsonl
  indexer.py            BGE Small embeddings -> Qdrant
rag/
  config.py             settings, paths, source loading
  embeddings.py         BGE Small, with the query-side instruction prefix
  vector_store.py       Qdrant client, collection schema, search
  prompts.py            the answer prompt and citation contract
  temporal.py           today's date + query expansion for "current"/"present"
  conversation.py       follow-up resolution for the chat UI
  course_codes.py       course-code extraction for exact lookup
  pipeline.py           retrieve -> generate
api/main.py             POST /chat, /retrieve, /health, GET /ui
api/static/index.html   chat UI served at /ui
tests/
  eval_questions.yaml   39 hand-verified questions
  run_eval.py           scoring harness
  test_ingestion.py     parsing + chunking
  test_pipeline.py      context assembly + citation mapping
  test_temporal.py      date grounding
  test_conversation.py  follow-up resolution
scripts/build_all.py    crawl -> chunk -> embed -> index in one command
Dockerfile              two-stage: bake the model + index, then a slim runtime
docker-compose.yml      the self-contained image, one command
docker-compose.server.yml  overlay: app + a real Qdrant + a one-shot indexer
deploy/azure.ps1        build in ACR + deploy to Azure Container Apps
data/                   raw/, processed/, qdrant_local/ gitignored; chunks/ committed
```

## Design notes worth knowing

**Heading text is prepended to every chunk.** A chunk that reads "Semester fee is
₹X" is useless in isolation; "Academic Calendar 2026 — Fee Structure: Semester fee
is ₹X" retrieves better *and* gives the LLM the context to cite correctly.

**BGE needs an asymmetric prefix.** Queries get
`"Represent this sentence for searching relevant passages: "`, passages get nothing.
This is set once in `rag/embeddings.py` — applying it to both sides, or neither,
quietly degrades retrieval.

**Course codes are looked up exactly, not semantically.** Dense embeddings are
poor at identifiers: once 182 near-identical course pages were indexed, "the
pre-requisites for CS345" matched the *shape* of a course page and CS346 scored
as well as CS345. `rag/course_codes.py` pulls any code named in the question and
`rag/pipeline.py` runs two metadata-filtered passes for it — `course_code` for
the course's own page, `course_codes` for the timetable row and catalogue line
that merely mention it — before similarity search fills the rest.

**No single source may flood the results.** Course pages are 83% of the index,
so plain top-k returned nothing else: the faculty chunk answering "Sunil Simon's
designation" sat at rank 13 and the timetable naming an instructor at rank 25.
`_diversify` fetches deep and allows at most `max_per_source` chunks from one
source, backfilling so a genuinely single-source question still fills its
context. This is a cheap stand-in for the v2 reranker, not a replacement.

**Follow-ups are resolved without a second LLM call.** The usual fix is a
"condense question" step that rewrites "who teaches it?" into a standalone
question via the model. On a 15-requests-per-minute free tier that doubles the
spend on every turn, so `rag/conversation.py` uses a heuristic instead: if the
question carries a pronoun, opens with a fragment like "and…", or is under five
words, the previous user turns are prepended to the *embedding* query. The full
turns still go to the model, which is what resolves the reference in prose.

**The pipeline knows what day it is.** "What is the present semester?" failed
twice over: retrieval had no year to anchor on, and the prompt never said what
today's date was. `rag/temporal.py` appends the current year to time-relative
queries before embedding, and the prompt carries today's date so the model can
locate itself in the calendar. It deliberately does *not* compute the current
semester in Python — that answer has to come from the calendar document, or it
goes stale the moment the calendar changes.

**Table rows stay whole.** A `<tr>` is emitted as one unit —
`CS335 | Compiler Design` — rather than cell by cell. Splitting a row scatters a
record across chunk boundaries and severs the code from its name, which makes a
190-row course table unretrievable.

**PDF chunking degrades gracefully.** `parse_pdf` finds headings by comparing each
line's font size against the document's modal body size. If fewer than two headings
turn up, it reports that and the chunker falls back to fixed 400-token windows. The
strategy actually used is recorded per chunk as `chunk_strategy`.

**Citations are verified, not trusted.** The prompt numbers each passage; the
pipeline parses `[n]` markers out of the answer and maps them back to real URLs.
A number the model invents outside the range is dropped rather than rendered as a
source.

## v2 — not built

LangGraph planner/router/validate nodes, Faculty and Compare tools, Streamlit UI,
full-site crawl, auto-metadata extraction, a 100-question benchmark, hybrid
search, reranking, GraphRAG, live sync. See the design doc's backlog. (Docker
Compose has since been pulled forward — see Deployment.)
