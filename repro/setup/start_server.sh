#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_BIN="$SCRIPT_DIR/llama.cpp.build/bin/llama-server"
MODEL="$SCRIPT_DIR/../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [[ ! -x "$LLAMA_BIN" ]]; then
    echo "llama-server not found at $LLAMA_BIN — run install_llamacpp.sh first" >&2
    exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "Model not found at $MODEL — run download_model.sh first" >&2
    exit 1
fi

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 127.0.0.1 --port 8000 \
    --metrics --slots \
    -c 32768 \
    --n-gpu-layers 999 \
    --parallel 1 \
    --seed 42 \
    --timeout 600
