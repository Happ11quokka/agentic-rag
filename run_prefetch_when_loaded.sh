#!/usr/bin/env bash
# Wait for the 10M load+benchmark to finish, then run the paired prefetch
# experiment against the loaded collection.
#
# The collection stays loaded after `wikipedia-inspect` exits (load state
# survives the client and even a restart), so the experiment does not pay the
# load cost again.
set -uo pipefail

ROOT=/Users/imdonghyeon/agentic_rag
INSPECT_LOG=$ROOT/wikipedia_diskann_inspect_10m_v5.log
OUT=$ROOT/wikipedia_diskann_prefetch_10m.log
JSON=$ROOT/wikipedia_diskann_prefetch_10m.json

echo "waiting for wikipedia-inspect to finish..." > "$OUT"
while pgrep -f "wikipedia-inspect" > /dev/null 2>&1; do
    sleep 60
done

if grep -qE "MilvusException|wikipedia-inspect: error:" "$INSPECT_LOG" 2>/dev/null; then
    {
        echo "ABORT: the load/benchmark failed, so there is nothing loaded to measure against."
        grep -E "MilvusException|error:" "$INSPECT_LOG" | tail -2
    } >> "$OUT"
    exit 1
fi

if ! grep -q "summary:" "$INSPECT_LOG" 2>/dev/null; then
    echo "ABORT: inspect exited without producing a summary; not assuming the collection is loaded." >> "$OUT"
    tail -5 "$INSPECT_LOG" >> "$OUT"
    exit 1
fi

{
    echo "=== load + search benchmark (baseline retrieval cost) ==="
    grep -E "collection loaded in|latency_ms|summary:" "$INSPECT_LOG"
    echo
    echo "=== paired prefetch experiment ==="
} >> "$OUT"

cd "$ROOT" || exit 1
# --decode-seconds 2 stands in for the target model's decode. There is no
# target model on this machine, so this is an input to the result, not a
# measurement of one, and the report prints it as such.
caffeinate -ims uv run wikipedia-prefetch \
    --bundle-dir /Users/imdonghyeon/.cache/wikipedia_diskann \
    --episodes 4 \
    --limit 5 \
    --search-list 100 \
    --decode-seconds 2.0 \
    --think-seconds 0.4 \
    --wrong-hops 2 \
    --json "$JSON" >> "$OUT" 2>&1

echo "exit=$?" >> "$OUT"
