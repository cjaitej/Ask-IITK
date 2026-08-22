"""Embed chunks.jsonl and push the vectors into Qdrant.

Vectors are cached in data/chunks/embeddings.npy, keyed to a fingerprint of
the chunk set, so re-indexing unchanged chunks costs nothing.

Run:  python -m ingestion.indexer            # embed (cached) + push
      python -m ingestion.indexer --recreate # drop the collection first
"""

from __future__ import annotations

import argparse
import json
import uuid
from typing import Dict, List

import numpy as np
from qdrant_client import models

from ingestion.chunker import load_chunks
from rag.config import get_settings
from rag.embeddings import get_embed_model
from rag.vector_store import (
    PAYLOAD_FIELDS,
    TEXT_KEY,
    ensure_collection,
    get_qdrant_client,
)

UPSERT_BATCH = 128

settings = get_settings()


def _fingerprint(chunks: List[Dict]) -> str:
    """Identifies the chunk set a cached embedding file belongs to."""
    import hashlib

    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk["chunk_id"].encode("utf-8"))
        h.update(chunk["text"].encode("utf-8"))
    h.update(settings.embed_model.encode("utf-8"))
    return h.hexdigest()


def embed_chunks(chunks: List[Dict], use_cache: bool = True) -> np.ndarray:
    """Embed every chunk, reusing data/chunks/embeddings.npy when still valid."""
    meta_path = settings.embeddings_path.with_suffix(".meta.json")
    fingerprint = _fingerprint(chunks)

    if use_cache and settings.embeddings_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") == fingerprint:
            vectors = np.load(settings.embeddings_path)
            if vectors.dtype != np.float32:
                # Older caches are float64: same numbers, twice the bytes.
                vectors = vectors.astype(np.float32)
                np.save(settings.embeddings_path, vectors)
                print("[embed] cache rewritten as float32")
            print("[embed] cache hit: {} vectors reused".format(vectors.shape[0]))
            return vectors

    model = get_embed_model()
    texts = [c["text"] for c in chunks]
    print(
        "[embed] embedding {} chunks with {} ...".format(len(texts), settings.embed_model)
    )
    # float32, not the float64 numpy picks for lists of Python floats. The model
    # emits float32 and Qdrant stores narrower still.
    vectors = np.asarray(
        model.get_text_embedding_batch(texts, show_progress=True), dtype=np.float32
    )

    settings.chunks_dir.mkdir(parents=True, exist_ok=True)
    np.save(settings.embeddings_path, vectors)
    meta_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "model": settings.embed_model,
                "n": int(vectors.shape[0]),
                "dim": int(vectors.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        "[embed] saved {} x {} -> {}".format(
            vectors.shape[0], vectors.shape[1], settings.embeddings_path
        )
    )
    return vectors


CHUNK_ID_NAMESPACE = uuid.UUID("6f9c1a52-0f2e-4c1b-9c0e-1f2a3b4c5d6e")


def point_id(chunk_id: str) -> str:
    """Chunk ids are neither a UUID nor an int, which is all Qdrant accepts.

    UUID5 is deterministic, so re-indexing updates a point rather than adding
    a duplicate. The readable id stays in the payload.
    """
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk_id))


def build_points(chunks: List[Dict], vectors: np.ndarray) -> List[models.PointStruct]:
    """One point per chunk, carrying only the PAYLOAD_FIELDS a query reads."""
    points: List[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        metadata = chunk["metadata"]
        payload = {
            key: metadata[key]
            for key in PAYLOAD_FIELDS
            if metadata.get(key) is not None
        }
        payload["chunk_id"] = chunk["chunk_id"]
        payload[TEXT_KEY] = chunk["text"]
        points.append(
            models.PointStruct(
                id=point_id(chunk["chunk_id"]),
                vector=[float(x) for x in vector],
                payload=payload,
            )
        )
    return points


def index(recreate: bool = False, use_cache: bool = True) -> int:
    chunks = load_chunks()
    if not chunks:
        raise SystemExit("No chunks found — run `python -m ingestion.chunker` first.")

    vectors = embed_chunks(chunks, use_cache=use_cache)
    if vectors.shape[1] != settings.embed_dim:
        raise SystemExit(
            "Embedding dim {} != configured embed_dim {}".format(
                vectors.shape[1], settings.embed_dim
            )
        )

    client = get_qdrant_client()
    try:
        ensure_collection(client, recreate=recreate)

        points = build_points(chunks, vectors)
        for start in range(0, len(points), UPSERT_BATCH):
            client.upsert(
                collection_name=settings.qdrant_collection,
                points=points[start : start + UPSERT_BATCH],
                wait=True,
            )

        count = client.count(settings.qdrant_collection, exact=True).count
        print(
            "[done] collection '{}' now holds {} points ({} mode)".format(
                settings.qdrant_collection, count, settings.qdrant_mode
            )
        )
        return count
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and push to Qdrant.")
    parser.add_argument(
        "--recreate", action="store_true", help="drop the collection before indexing"
    )
    parser.add_argument(
        "--no-cache", action="store_true", help="ignore the saved embeddings cache"
    )
    args = parser.parse_args()
    index(recreate=args.recreate, use_cache=not args.no_cache)
