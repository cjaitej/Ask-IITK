"""BGE Small embedding model, loaded once per process."""

from __future__ import annotations

from functools import lru_cache

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from rag.config import get_settings

# BGE wants this prefix on queries only; passages are embedded bare. It lives
# here so no call site can get it wrong.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    settings = get_settings()
    return HuggingFaceEmbedding(
        model_name=settings.embed_model,
        query_instruction=BGE_QUERY_INSTRUCTION,
        embed_batch_size=settings.embed_batch_size,
        normalize=True,
    )
