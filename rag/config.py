"""Settings shared by ingestion, retrieval and the API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Source(BaseModel):
    """One entry from sources.yaml."""

    id: str
    name: str
    url: str
    department: str
    follow_pdfs: bool = True
    max_pdfs: int = 6
    pdf_include: List[str] = Field(default_factory=list)

    # One hop only, same page, matching link_pattern, capped at max_links.
    follow_links: bool = False
    link_pattern: str = ""
    max_links: int = 0


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths ---
    data_dir: Path = PROJECT_ROOT / "data"
    sources_file: Path = PROJECT_ROOT / "sources.yaml"

    # --- chunking ---
    chunk_target_tokens: int = 400      # 300-500 band from the design
    chunk_min_tokens: int = 60          # below this a chunk is merged/dropped
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 60
    pdf_fallback_tokens: int = 400      # fixed window when no heading structure

    # --- embeddings ---
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    embed_batch_size: int = 32

    # --- vector store ---
    # "local"  -> embedded on-disk Qdrant, no Docker needed
    # "server" -> Qdrant running in Docker at qdrant_url
    qdrant_mode: Literal["local", "server"] = "local"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "iitk_documents_v1"

    # --- retrieval + generation ---
    top_k: int = 6
    # Extra chunks per named course code, on top of top_k.
    code_top_k: int = 3           # the course's own page
    code_mentions_top_k: int = 5  # timetable and catalogue rows naming it
    # Course pages are 83% of the index, so a plain top-k returns nothing else.
    # Fetch deep, then cap per source.
    fetch_multiplier: int = 6
    max_per_source: int = 3
    gemini_model: str = "gemini-2.5-flash"
    gemini_api_key: str = ""
    max_context_chars: int = 24_000

    # --- crawler ---
    user_agent: str = (
        "AskIITK/1.0 (academic project; contact: cjaitej@gmail.com)"
    )
    request_timeout: int = 30
    crawl_delay_seconds: float = 1.0
    # Detail pages are small and same-host, so a shorter gap is still polite.
    link_crawl_delay_seconds: float = 0.5

    # --- derived paths ---
    @property
    def raw_html_dir(self) -> Path:
        return self.data_dir / "raw" / "html"

    @property
    def raw_pdf_dir(self) -> Path:
        return self.data_dir / "raw" / "pdf"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def chunks_dir(self) -> Path:
        return self.data_dir / "chunks"

    @property
    def qdrant_local_path(self) -> Path:
        return self.data_dir / "qdrant_local"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "raw" / "manifest.json"

    @property
    def chunks_path(self) -> Path:
        return self.chunks_dir / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.chunks_dir / "embeddings.npy"

    def ensure_dirs(self) -> None:
        for d in (
            self.raw_html_dir,
            self.raw_pdf_dir,
            self.processed_dir,
            self.chunks_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def load_sources(self) -> List[Source]:
        with open(self.sources_file, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return [Source(**s) for s in raw["sources"]]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
