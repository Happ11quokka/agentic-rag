#!/usr/bin/env bash
# Wait for the collection and the models, re-measure search latency, then run
# prefetched-toolcall.
#
# The re-measure is not optional. Each question is capped at
# QUESTION_TIMEOUT_SECONDS=600 and can issue up to MAX_SEARCHES=8 searches. The
# only latency figure we have for this collection is 295 s per search, taken at
# searchCacheBudgetGBRatio=0.01; at that speed two searches exhaust the budget
# and every question times out, which produces failure counts rather than a
# measurement. The ratio is now 0.10, so the number has to be taken again before
# the run means anything.
set -uo pipefail

ROOT=/Users/imdonghyeon/agentic_rag
OUT=$ROOT/prefetched_toolcall_run.log
COLLECTION=wikipedia_2024_06_bge_m3_en_v1
export LLAMA_SERVER=$ROOT/repro/setup/llama.cpp.build/bin/llama-server

cd "$ROOT" || exit 1
log() { echo "$(date '+%H:%M:%S') $*" >> "$OUT"; }
: > "$OUT"

log "waiting for models and collection"
while true; do
    have_models=$(find experiment/models -name "*.gguf" -size +100M 2>/dev/null | wc -l | tr -d ' ')
    loaded=$(uv run python -c "
from pymilvus import MilvusClient
import warnings; warnings.filterwarnings('ignore')
try:
    s = MilvusClient(uri='http://localhost:19530', timeout=90).get_load_state('$COLLECTION')
    print(100 if str(s.get('state')).endswith('Loaded') else s.get('progress', -1))
except Exception:
    print(-1)
" 2>/dev/null | tail -1)
    if [ "$have_models" = "2" ] && [ "$loaded" = "100" ]; then break; fi
    log "models=${have_models}/2 load=${loaded}%"
    sleep 300
done
log "ready"

# 1. What does one search actually cost now? This decides whether the run is viable.
{
    echo
    echo "=== search latency at searchCacheBudgetGBRatio=0.10 ==="
} >> "$OUT"
caffeinate -ims uv run wikipedia-inspect --backend milvus \
    --bundle-dir /Users/imdonghyeon/.cache/wikipedia_diskann \
    --runs 3 --search-list 100 --load-timeout 3600 >> "$OUT" 2>&1

MEDIAN=$(grep -oE "median=[0-9.]+ ms" "$OUT" | tail -1 | grep -oE "[0-9.]+")
if [ -n "$MEDIAN" ]; then
    BUDGET=$(awk "BEGIN{printf \"%.1f\", 600000/$MEDIAN}")
    log "median ${MEDIAN} ms -> about ${BUDGET} searches fit in the 600 s question budget (MAX_SEARCHES=8)"
else
    log "could not read a median; running the smoke anyway to see the failure"
fi

# 2. One task, one repetition. Bounded by the same 600 s cap, so this costs at
#    most a few minutes and answers whether the whole path runs at all.
{
    echo
    echo "=== prefetched-toolcall smoke (1 task x 1 repetition) ==="
} >> "$OUT"
caffeinate -ims uv run python -c "
from experiment import prefetched_toolcall
answers = iter(['1', '1'])
prefetched_toolcall.prompt_and_run(input_fn=lambda prompt: next(answers))
" >> "$OUT" 2>&1

log "smoke exit=$?"
