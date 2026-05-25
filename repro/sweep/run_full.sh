#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
LOG_DIR="$REPRO/results/sweep_logs"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"

cd "$REPRO"
source .venv/bin/activate

# Required for langchain-openai to hit our local llama-server (not api.openai.com).
# Without these the entire sweep crashes within seconds of the first ChatOpenAI() call.
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-local}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"

CONFIG="$1"
CACHE_MODE="${2:-default}"   # "default" (cache OFF) or "cache_on"
RUN_ID="$(/usr/bin/python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "sweep/configs/$CONFIG")"
OUT="results/raw/${RUN_ID}.jsonl"

# Start llama-server in background if not running. Pick variant by CACHE_MODE.
if ! curl -s -o /dev/null http://localhost:8000/health; then
    if [[ "$CACHE_MODE" == "cache_on" ]]; then
        echo "Starting llama-server with --cache-reuse ENABLED (Fig 9 variant)..."
        ./setup/start_server_cache_on.sh > "$LOG_DIR/llamaserver_cacheon_$ts.log" 2>&1 &
    else
        echo "Starting llama-server (default, cache-reuse OFF)..."
        ./setup/start_server.sh > "$LOG_DIR/llamaserver_$ts.log" 2>&1 &
    fi
    sleep 10
fi

# Warm up slot so /slots reports full schema for the first poll of the first query.
# Without this, kv_cache_max_tokens may underreport for the first 1-2 queries.
curl -s -m 30 -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"local","messages":[{"role":"user","content":"hi"}],"max_tokens":4}' \
    > /dev/null 2>&1 || echo "(warm-up POST failed; continuing anyway)"

# Run sweep with --resume (default)
echo "Running $CONFIG → $OUT"
python -m sweep.sweep_runner --config "sweep/configs/$CONFIG" --out "$OUT" \
    2>&1 | tee "$LOG_DIR/sweep_${RUN_ID}_$ts.log"

echo "Done. JSONL at: $OUT"
