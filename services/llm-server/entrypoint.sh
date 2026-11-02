#!/usr/bin/env bash
# Start the local GGUF model (llama.cpp) and the RAG API in one container.
set -euo pipefail

MODEL="${LLM_MODEL_PATH:-/models/qwen2.5-0.5b-instruct-q4_k_m.gguf}"
LLAMA_BIN="${LLAMA_BIN:-/app/llama-server}"
LLM_HOST="127.0.0.1"
LLM_PORT="${LLM_PORT:-8081}"
CTX="${LLM_CTX:-4096}"
THREADS="${LLM_THREADS:-$(nproc)}"

if [[ ! -f "$MODEL" ]]; then
  echo "[entrypoint] FATAL: GGUF not found at $MODEL" >&2
  echo "[entrypoint] run scripts/fetch_model.sh or mount one into /models" >&2
  exit 1
fi

echo "[entrypoint] starting llama-server ($MODEL, ctx=$CTX, threads=$THREADS)"
"$LLAMA_BIN" \
  --model "$MODEL" \
  --host "$LLM_HOST" \
  --port "$LLM_PORT" \
  --ctx-size "$CTX" \
  --threads "$THREADS" \
  --parallel 1 \
  --no-webui &
LLAMA_PID=$!

# If the model process dies, exit so `restart: unless-stopped` brings us back.
( while kill -0 "$LLAMA_PID" 2>/dev/null; do sleep 5; done; echo "[entrypoint] llama-server exited" >&2; kill -TERM $$ ) &

echo "[entrypoint] starting RAG API on :8000"
exec gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 4 --timeout 300 main:app
