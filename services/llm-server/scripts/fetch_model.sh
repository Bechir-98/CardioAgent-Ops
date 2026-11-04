#!/usr/bin/env bash
# Download the GGUF used by llm-server. The file is ~1 GB and git-ignored.
set -euo pipefail

MODEL="${1:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
REPO="${GGUF_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

mkdir -p "$DEST_DIR"

if [[ -f "$DEST_DIR/$MODEL" ]]; then
  echo "already present: $DEST_DIR/$MODEL"
  exit 0
fi

echo "downloading $REPO/$MODEL -> $DEST_DIR"
curl -fL --progress-bar -o "$DEST_DIR/$MODEL" \
  "https://huggingface.co/${REPO}/resolve/main/${MODEL}"

ls -lh "$DEST_DIR/$MODEL"
