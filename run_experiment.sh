#!/usr/bin/env bash
# Wait for the collection to load, measure search latency, then run the paired
# prefetch experiment. Restartable: it polls the server, so it can be killed and
# relaunched without disturbing an in-flight load.
#
# Docker Desktop's file-sharing service has died twice under this workload
# ("service fs failed: injecting event blocked for 60s"), each time costing a
# full reload. So every step here is retried rather than abandoned, and the log
# records which attempt produced the numbers.
set -uo pipefail

ROOT=/Users/imdonghyeon/agentic_rag
OUT=$ROOT/wikipedia_diskann_experiment.log
JSON=$ROOT/wikipedia_diskann_prefetch_10m.json
BUNDLE=/Users/imdonghyeon/.cache/wikipedia_diskann
COLLECTION=wikipedia_2024_06_bge_m3_en_v1

log() { echo "$(date '+%H:%M:%S') $*" >> "$OUT"; }

progress() {
    cd "$ROOT" || return 1
    uv run python -c "
from pymilvus import MilvusClient
import warnings; warnings.filterwarnings('ignore')
try:
    s = MilvusClient(uri='http://localhost:19530', timeout=90).get_load_state('$COLLECTION')
    print(100 if str(s.get('state')).endswith('Loaded') else s.get('progress', -1))
except Exception:
    print(-1)
" 2>/dev/null | tail -1
}

wait_for_docker() {
    for _ in $(seq 1 120); do
        curl -sf http://localhost:9091/healthz >/dev/null 2>&1 && return 0
        sleep 15
    done
    return 1
}

: > "$OUT"
log "waiting for load"

stalled=0
last=-2
while true; do
    pct=$(progress)
    [ "$pct" = "100" ] && { log "loaded"; break; }
    if [ "$pct" = "$last" ]; then stalled=$((stalled + 1)); else stalled=0; fi
    if [ "$stalled" -ge 48 ]; then log "ABORT: stuck at ${pct}% for 4h"; exit 1; fi
    last=$pct
    sleep 300
done

for attempt in 1 2 3; do
    log "search benchmark, attempt ${attempt}"
    caffeinate -ims uv run wikipedia-inspect --backend milvus \
        --bundle-dir "$BUNDLE" --runs 6 --search-list 100 --load-timeout 7200 >> "$OUT" 2>&1
    grep -q "summary:" "$OUT" && break
    log "benchmark died; waiting for docker, then retrying"
    wait_for_docker || log "docker did not come back"
    while [ "$(progress)" != "100" ]; do sleep 300; done
done

MEDIAN=$(grep -oE "median=[0-9.]+ ms" "$OUT" | tail -1 | grep -oE "[0-9.]+")
if [ -n "$MEDIAN" ]; then
    DECODE=$(awk "BEGIN{printf \"%.2f\", $MEDIAN/1000}")
else
    DECODE=2.00
fi
log "paired prefetch experiment; decode gap ${DECODE}s matched to median ${MEDIAN:-unknown} ms"

for attempt in 1 2 3; do
    caffeinate -ims uv run wikipedia-prefetch \
        --bundle-dir "$BUNDLE" \
        --episodes 3 --limit 5 --search-list 100 \
        --decode-seconds "$DECODE" --think-seconds 0.4 --wrong-hops 2 \
        --json "$JSON" >> "$OUT" 2>&1
    grep -q "hit rate:" "$OUT" && break
    log "experiment died on attempt ${attempt}; waiting for docker"
    wait_for_docker || log "docker did not come back"
    while [ "$(progress)" != "100" ]; do sleep 300; done
done

log "done"
