# syntax=docker/dockerfile:1

# The llama.cpp project ships an official runtime image containing the
# `llama-server` binary and all of its shared libraries. Building on it avoids
# compiling llama.cpp (several minutes per build) and guarantees ABI match.
FROM ghcr.io/ggml-org/llama.cpp:server AS llama

FROM llama AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PATH="/opt/venv/bin:/app:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama" \
    HF_HOME=/models/.hf

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies in an isolated venv (Ubuntu 24.04 is PEP 668 managed).
COPY requirements.txt .
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.txt

# Bake the embedding model into the image so inference needs no network.
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as F; F()(['warmup'])"

COPY . .
RUN chmod +x entrypoint.sh

ENV CORPUS_DIR=/app/corpus \
    CHROMA_DIR=/data/chroma \
    LLM_BASE_URL=http://127.0.0.1:8081/v1 \
    LLM_MODEL_PATH=/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    LLM_CTX=4096

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=5 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# The base image sets ENTRYPOINT ["/app/llama-server"]; we start both processes
# from our own script instead.
ENTRYPOINT []
CMD ["./entrypoint.sh"]
