#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="$(cat "$SCRIPT_DIR/LLAMACPP_COMMIT")"
SRC_DIR="$SCRIPT_DIR/llama.cpp.src"
INSTALL_DIR="$SCRIPT_DIR/llama.cpp.build"

if [[ ! -d "$SRC_DIR" ]]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$SRC_DIR"
fi
cd "$SRC_DIR"
git fetch --tags origin
git checkout "$TAG"

cmake -B "$INSTALL_DIR" -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$INSTALL_DIR" --config Release -j

echo "Built. Binary at: $INSTALL_DIR/bin/llama-server"
"$INSTALL_DIR/bin/llama-server" --version
