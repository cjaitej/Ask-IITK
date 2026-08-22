"""Qdrant access for collection `iitk_documents_v1`.

QDRANT_MODE picks the backend: "local" is an embedded on-disk store under
data/qdrant_local (one process at a time), "server" is a Qdrant container at
QDRANT_URL. Nothing else in the code cares which.

Points go through qdrant-client directly rather than LlamaIndex's vector
store, which wrote a `_node_content` copy of every chunk (41% of the payload)
and created the collection implicitly, leaving nowhere to declare a payload
index. Hit/Node keep the shape the ranking helpers expect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from qdrant_client import QdrantClient, models

from rag.config import get_settings

# What a query actually reads: prompt text, the fields an answer cites, and the
# two filter keys. Build-time diagnostics (sha256, chunk_strategy, n_tokens)
# stay in chunks.jsonl rather than being copied into every point.
PAYLOAD_FIELDS = (
    "chunk_id",
    "url",
    "source_id",
    "source_name",
    "department",
    "doc_type",
    "doc_title",
    "heading",
    "page",
    "crawled_at",
    "course_code",
    "course_codes",
)

TEXT_KEY = "text"

# course_code is the course's own page; course_codes lists every code a chunk
# mentions. Qdrant indexes each element of a list, so both are keywords.
INDEXED_FIELDS = {
    "course_code": models.PayloadSchemaType.KEYWORD,
    "course_codes": models.PayloadSchemaType.KEYWORD,
    "source_id": models.PayloadSchemaType.KEYWORD,
    "doc_type": models.PayloadSchemaType.KEYWORD,
}


@dataclass
class Node:
    """The retrieval-facing half of a point."""

    node_id: str
    text: str = ""
    metadata: Dict = field(default_factory=dict)

    def get_content(self) -> str:
        return self.text


@dataclass
class Hit:
    node: Node
    score: float


def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    if settings.qdrant_mode == "server":
        return QdrantClient(url=settings.qdrant_url, timeout=60)

    settings.qdrant_local_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_local_path))


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the collection and its payload indexes.

    Vectors are float16: BGE normalises into roughly ±0.2, where float16 still
    carries more precision than cosine ranking can use. Half the bytes, and the
    eval set does not shift.
    """
    settings = get_settings()
    name = settings.qdrant_collection

    if recreate and client.collection_exists(name):
        client.delete_collection(name)
        print("[index] dropped collection " + name)

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=settings.embed_dim,
                distance=models.Distance.COSINE,
                datatype=models.Datatype.FLOAT16,
            ),
            # Only read for the few points a search returns, so keeping them
            # off the heap costs nothing.
            on_disk_payload=True,
        )
        print("[index] created collection {} (dim {})".format(name, settings.embed_dim))

    if settings.qdrant_mode != "server":
        # The embedded client scans instead, and warns if asked for an index.
        return

    for field_name, schema in INDEXED_FIELDS.items():
        client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
            wait=True,
        )
    print("[index] payload indexes: " + ", ".join(INDEXED_FIELDS))


def _to_hit(scored) -> Hit:
    payload = dict(scored.payload or {})
    text = payload.pop(TEXT_KEY, "")
    return Hit(
        node=Node(node_id=str(scored.id), text=text, metadata=payload),
        score=float(scored.score or 0.0),
    )


def search(
    client: QdrantClient,
    vector: Sequence[float],
    limit: int,
    where: Optional[models.Filter] = None,
) -> List[Hit]:
    settings = get_settings()
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=list(vector),
        limit=limit,
        query_filter=where,
        with_payload=True,
        with_vectors=False,
    )
    return [_to_hit(point) for point in result.points]


def match_filter(key: str, value: str) -> models.Filter:
    """key == value, or `value in key` when the key holds a list."""
    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )


def collection_stats(client: Optional[QdrantClient] = None) -> dict:
    """Point count and existence.

    Pass the open client if there is one — local mode locks the storage folder,
    so a second client in the same process just fails.
    """
    settings = get_settings()
    owned = client is None
    client = client or get_qdrant_client()
    try:
        exists = client.collection_exists(settings.qdrant_collection)
        count = (
            client.count(settings.qdrant_collection, exact=True).count if exists else 0
        )
        return {
            "collection": settings.qdrant_collection,
            "exists": exists,
            "points": count,
            "mode": settings.qdrant_mode,
        }
    finally:
        if owned:
            client.close()
