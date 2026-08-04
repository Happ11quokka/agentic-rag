#!/usr/bin/env bash
# Wait for the 10M collection to finish loading, then measure it.
#
# Polls the server rather than watching a client process. load_collection is a
# server-side job: the client only polls it, so a client that times out or is
# killed does not stop the load. Tying this script to a client would abort the
# run over a client-side deadline while the load was still healthy.
set -uo pipefail

ROOT=/Users/imdonghyeon/agentic_rag
OUT=$ROOT/wikipedia_diskann_prefetch_10m.log
JSON=$ROOT/wikipedia_diskann_prefetch_10m.json
BUNDLE=/Users/imdonghyeon/.cache/wikipedia_diskann
COLLECTION=wikipedia_2024_06_bge_m3_en_v1

progress() {
    cd "$ROOT" || return 1
    uv run python -c "
from pymilvus import utility, connections
import warnings; warnings.filterwarnings('ignore')
try:
    connections.connect(uri='http://localhost:19530', timeout=60)
    print(utility.loading_progress('$COLLECTION')['loading_progress'].rstrip('%'))
except Exception:
    print('-1')
" 2>/dev/null | tail -1
}

: > "$OUT"
echo "$(date '+%H:%M:%S') waiting for the collection to load..." >> "$OUT"

stalled=0
last=-1
while true; do
    pct=$(progress)
    if [ "$pct" = "100" ]; then
        echo "$(date '+%H:%M:%S') loaded" >> "$OUT"
        break
    fi
    if [ "$pct" = "$last" ]; then
        stalled=$((stalled + 1))
    else
        stalled=0
    fi
    # 40 polls x 5 min = 3.3 h with no movement. The load is slow by nature
    # (~134 GB copied MinIO -> local on one spindle), so only a long flat
    # stretch means something is actually wrong.
    if [ "$stalled" -ge 40 ]; then
        echo "$(date '+%H:%M:%S') ABORT: load stuck at ${pct}% for over 3 hours" >> "$OUT"
        exit 1
    fi
    if [ "$pct" = "-1" ]; then
        echo "$(date '+%H:%M:%S') milvus unreachable" >> "$OUT"
    fi
    last=$pct
    sleep 300
done

cd "$ROOT" || exit 1

{
    echo
    echo "=== search latency on the loaded collection (baseline retrieval cost) ==="
} >> "$OUT"
caffeinate -ims uv run wikipedia-inspect --backend milvus \
    --bundle-dir "$BUNDLE" --runs 10 --search-list 100 --load-timeout 3600 >> "$OUT" 2>&1

{
    echo
    echo "=== paired prefetch experiment ==="
} >> "$OUT"
# --decode-seconds stands in for the target model decoding the next thought.
# There is no target model on this machine, so it is an input to the result and
# the report prints it as such.
caffeinate -ims uv run wikipedia-prefetch \
    --bundle-dir "$BUNDLE" \
    --episodes 4 \
    --limit 5 \
    --search-list 100 \
    --decode-seconds 2.0 \
    --think-seconds 0.4 \
    --wrong-hops 2 \
    --json "$JSON" >> "$OUT" 2>&1

echo "exit=$? at $(date '+%H:%M:%S')" >> "$OUT"
