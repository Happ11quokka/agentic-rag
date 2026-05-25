#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
FILE="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

if [[ -f "$MODELS_DIR/$FILE" ]]; then
    echo "Already downloaded: $MODELS_DIR/$FILE"
    exit 0
fi

if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "Installing huggingface-hub CLI..."
    pip install --user 'huggingface_hub[cli]'
fi

huggingface-cli download \
    bartowski/Meta-Llama-3.1-8B-Instruct-GGUF \
    "$FILE" \
    --local-dir "$MODELS_DIR"

ls -lh "$MODELS_DIR/$FILE"
