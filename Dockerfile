# syntax=docker/dockerfile:1

# Self-contained image: the model and the index are baked in, so a cold start
# is just "load from disk" with no network and no sidecar.
#
#   BGE Small   -> /models
#   Qdrant index -> /app/data/qdrant_local (embedded)
#
# The index is rebuilt from data/chunks/ here, but embeddings.npy is a cache
# keyed by chunk fingerprint, so that step copies rather than re-embeds. When
# sources change, re-run the ingestion locally and rebuild.
#
# data/chunks/ is committed for this reason: it is the one part of data/ the
# build context needs.

# ---------------------------------------------------------------- builder ---
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    LLAMA_INDEX_CACHE_DIR=/models \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv

WORKDIR /app

# torch from PyTorch's CPU index. From PyPI it resolves to the CUDA build and
# its nvidia-* wheels: several GB of GPU runtime for a model that runs on one
# vCPU. This line is worth 4 GB.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what get_embed_model() needs, so editing a prompt does not invalidate
# the two-minute model fetch below. If embeddings.py grows an import beyond
# these, the next line fails loudly rather than baking a stale model.
COPY rag/__init__.py rag/config.py rag/embeddings.py ./rag/
COPY sources.yaml .

# Warm the model cache through get_embed_model(), the same call the app makes.
# A bare SentenceTransformer() fills HF_HOME, but HuggingFaceEmbedding passes
# llama-index's get_cache_dir() as cache_folder, which overrides HF_HOME — so
# warming the wrong one leaves a model the app cannot find, and
# HF_HUB_OFFLINE=1 turns that into a startup crash. Hence also
# LLAMA_INDEX_CACHE_DIR=/models in both stages.
RUN python -c "from rag.embeddings import get_embed_model; get_embed_model()"

# Build the embedded index. The .lock left behind belongs to the build
# container, and the runtime cannot open the folder until it is gone.
COPY rag/ ./rag/
COPY ingestion/ ./ingestion/
COPY data/chunks/ ./data/chunks/

ENV QDRANT_MODE=local
RUN python -m ingestion.indexer --recreate \
    && rm -f data/qdrant_local/.lock

# ---------------------------------------------------------------- runtime ---
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    LLAMA_INDEX_CACHE_DIR=/models \
    HF_HUB_OFFLINE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000 \
    QDRANT_MODE=local \
    QDRANT_COLLECTION=iitk_documents_v1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /models /models
# Owned by app: embedded Qdrant writes a .lock here at startup.
COPY --from=builder --chown=app:app /app/data/qdrant_local ./data/qdrant_local

COPY rag/ ./rag/
COPY api/ ./api/
COPY sources.yaml .

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

# Shell form so $PORT can be overridden at runtime.
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
