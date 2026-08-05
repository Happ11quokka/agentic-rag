#!/usr/bin/env bash
# Wait for the running benchmark to finish, then run the paired prefetch
# experiment sized to the retrieval cost it measures.
#
# Cold search on the loaded 10M collection measured ~5.6 min. At 5 episodes
# (30 target searches plus decode gaps matched to that retrieval) the paired
# run is 4-5 hours; 3 episodes keeps a full baseline-vs-prefetch comparison at
# roughly half that. Episode count is a sample-size choice, and is reported.
set -uo pipefail

ROOT=/Users/imdonghyeon/agentic_rag
OUT=$ROOT/wikipedia_diskann_prefetch_10m.log
JSON=$ROOT/wikipedia_diskann_prefetch_10m.json
BUNDLE=/Users/imdonghyeon/.cache/wikipedia_diskann

while pgrep -f "[w]ikipedia-inspect" > /dev/null 2>&1; do
    sleep 30
done

if ! grep -q "summary:" "$OUT"; then
    echo "$(date '+%H:%M:%S') ABORT: benchmark produced no summary" >> "$OUT"
    tail -3 "$OUT"
    exit 1
fi

# Match the decode gap to the retrieval it has to hide. prefetch can hide at
# most min(retrieval, decode) per hop, so a gap picked without knowing the
# retrieval decides the answer before measuring it.
MEDIAN=$(grep -oE "median=[0-9.]+ ms" "$OUT" | tail -1 | grep -oE "[0-9.]+")
if [ -n "$MEDIAN" ]; then
    DECODE=$(awk "BEGIN{printf \"%.2f\", $MEDIAN/1000}")
else
    DECODE=2.00
fi

{
    echo
    echo "=== paired prefetch experiment ==="
    echo "decode gap ${DECODE}s, matched to the measured median retrieval (${MEDIAN:-unknown} ms)"
} >> "$OUT"

cd "$ROOT" || exit 1
caffeinate -ims uv run wikipedia-prefetch \
    --bundle-dir "$BUNDLE" \
    --episodes 3 \
    --limit 5 \
    --search-list 100 \
    --decode-seconds "$DECODE" \
    --think-seconds 0.4 \
    --wrong-hops 2 \
    --json "$JSON" >> "$OUT" 2>&1

echo "exit=$? at $(date '+%H:%M:%S')" >> "$OUT"
