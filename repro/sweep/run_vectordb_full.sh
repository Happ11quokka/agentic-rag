#!/usr/bin/env bash
# Full-dataset (7,405) ReAct × Cohere-vectorDB + rerank sweep, crash-resilient.
#
# Design goals (per request):
#   - run the rerank baseline over ALL 7,405 HotpotQA questions
#   - survive network drops: embed calls retry with backoff (vector_search.py)
#   - survive the libomp/MPS startup SEGFAULT: auto-restart and resume
#   - run until the Cohere API quota is hit, then STOP CLEANLY (circuit breaker)
#   - stop anytime (Ctrl-C) and resume later with progress preserved (--resume)
#
# Resume is automatic: completed queries are appended to the SAME JSONL with
# fsync, and sweep_runner --resume skips every sample_idx already present. This
# driver wraps that in a restart loop because the faiss+torch rerank path can
# segfault on the FIRST query of a process (OMP libomp clash); past startup it is
# stable. Each restart cleans failed rows and resumes, so no work is lost.
#
# Just re-run this script to continue after any stop. It is idempotent.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
cd "$REPRO"

CONFIG="react_vectordb_rerank_full.yaml"
OUT="results/raw/react_vectordb_rerank.jsonl"
TOTAL=7405
MAX_NOPROGRESS=12     # consecutive restarts with ZERO new rows → give up (real breakage)
LOG_DIR="results/sweep_logs"
mkdir -p "$LOG_DIR"

# Keep the machine awake for as long as this driver runs.
caffeinate -dimsu -w $$ &
echo "[driver] caffeinate holding system awake (pid $!)"

count_rows() { [[ -f "$OUT" ]] && wc -l < "$OUT" | tr -d ' ' || echo 0; }

clean_failed() {   # drop timeout/error rows from THIS file only so --resume retries them
    [[ -f "$OUT" ]] || return 0
    source .venv/bin/activate
    python - "$OUT" <<'PY'
import json, sys
path = sys.argv[1]
rows = [json.loads(l) for l in open(path) if l.strip()]
kept, dropped = [], 0
for r in rows:
    meta = r.get("meta", {}) or {}
    fa = r.get("final_answer", "") or ""
    if meta.get("timeout") or meta.get("error") or fa.startswith("<TIMEOUT>") or fa.startswith("<ERROR>"):
        dropped += 1
    else:
        kept.append(r)
if dropped:
    with open(path, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[driver] cleaned {dropped} failed row(s); {len(kept)} good rows remain")
PY
}

# One-time backup before the first append/clean of this session.
if [[ -f "$OUT" ]]; then
    bak_dir="results/raw/pre_full_backup"; mkdir -p "$bak_dir"
    bak="$bak_dir/react_vectordb_rerank.$(date +%Y%m%d_%H%M%S).jsonl"
    cp "$OUT" "$bak"
    echo "[driver] backed up $(count_rows) existing rows → $bak"
fi

attempt=0
noprogress=0
while true; do
    done_before=$(count_rows)
    if [[ "$done_before" -ge "$TOTAL" ]]; then
        echo "[driver] ✅ full dataset complete ($done_before / $TOTAL)"; break
    fi

    attempt=$((attempt + 1))
    clean_failed
    echo "[driver] === attempt $attempt — $(count_rows) / $TOTAL done — (re)starting sweep ==="
    echo "[driver]     (Ctrl-C to stop; re-run this script to resume)"

    attempt_log="$LOG_DIR/vectordb_full_attempt_$(date +%Y%m%d_%H%M%S).log"
    # run_full.sh owns venv/env/llama-server/warmup/--resume. It may segfault; that
    # is fine — set -uo (no -e) lets us inspect the exit and loop.
    sweep/run_full.sh "$CONFIG" >"$attempt_log" 2>&1
    rc=$?
    tail -2 "$attempt_log" | sed 's/^/[sweep] /'

    # Circuit breaker fired (in-process consecutive-failure limit) → API quota /
    # server down. Stop cleanly so the user can resume later (e.g. quota reset).
    if grep -q "circuit-breaker" "$attempt_log"; then
        echo "[driver] ⏸  circuit breaker tripped (likely Cohere quota / server down)."
        echo "[driver]     $(count_rows) / $TOTAL saved. Re-run this script to resume later."
        break
    fi

    done_after=$(count_rows)
    if [[ "$done_after" -ge "$TOTAL" ]]; then
        echo "[driver] ✅ full dataset complete ($done_after / $TOTAL)"; break
    fi

    if [[ "$done_after" -gt "$done_before" ]]; then
        echo "[driver] progress: +$((done_after - done_before)) rows (now $done_after / $TOTAL), exit=$rc — restarting"
        noprogress=0
    else
        noprogress=$((noprogress + 1))
        echo "[driver] no progress this attempt (exit=$rc, likely startup segfault) — retry $noprogress/$MAX_NOPROGRESS"
        if [[ "$noprogress" -ge "$MAX_NOPROGRESS" ]]; then
            echo "[driver] ⛔ $MAX_NOPROGRESS restarts with zero progress — stopping."
            echo "[driver]     Check $attempt_log . $(count_rows) / $TOTAL saved; re-run to resume."
            break
        fi
        sleep 5
    fi
done

echo "[driver] stopped at $(count_rows) / $TOTAL rows in $OUT"
