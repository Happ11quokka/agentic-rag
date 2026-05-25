#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
LOG_DIR="$REPRO/results/sweep_logs"
mkdir -p "$LOG_DIR"
ts="$(date +%Y%m%d_%H%M%S)"

cd "$REPRO"
source .venv/bin/activate

CONFIG="$1"
RUN_ID="$(/usr/bin/python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['run_id'])" "sweep/configs/$CONFIG")"
OUT="results/raw/${RUN_ID}.jsonl"

# Start llama-server in background if not running
if ! curl -s -o /dev/null http://localhost:8000/health; then
    echo "Starting llama-server..."
    ./setup/start_server.sh > "$LOG_DIR/llamaserver_$ts.log" 2>&1 &
    sleep 10
fi

# Run sweep with --resume (default)
echo "Running $CONFIG → $OUT"
python -m sweep.sweep_runner --config "sweep/configs/$CONFIG" --out "$OUT" \
    2>&1 | tee "$LOG_DIR/sweep_${RUN_ID}_$ts.log"

echo "Done. JSONL at: $OUT"
