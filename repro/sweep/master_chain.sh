#!/usr/bin/env bash
# Master chain: runs all 7 sweeps + analysis sequentially.
# - Default-cache sweeps share one llama-server (no restart between them)
# - Server restarts only at cache mode transitions (default <-> cache_on)
# - --resume means a failure on one sweep doesn't lose work
# - Wrapped in nohup + disown for full detachment from the calling shell

set -uo pipefail   # NOT -e: continue past per-sweep failures
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
LOG_DIR="$REPRO/results/sweep_logs"
MASTER_LOG="$LOG_DIR/master_chain.log"
mkdir -p "$LOG_DIR"

cd "$REPRO"

log() {
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] $*" | tee -a "$MASTER_LOG"
}

start_time=$(date +%s)
log "=================================="
log "  MASTER CHAIN START — PID $$"
log "=================================="
log "Plan: 7 sweeps + 9 plot scripts, expected ~60h wall-clock"
log ""

# ====== Phase A: default-cache sweeps (share single llama-server) ======
log ">>> Phase A: default-cache sweeps (fig13, fig14, fig15)"
for cfg in fig13_pareto.yaml fig14_iteration.yaml fig15_fewshot.yaml; do
    log "----- Starting: $cfg"
    a_start=$(date +%s)
    sweep/run_full.sh "$cfg" </dev/null >> "$MASTER_LOG" 2>&1 \
        && log "----- Finished: $cfg in $(( $(date +%s) - a_start ))s" \
        || log "----- WARN: $cfg exited non-zero in $(( $(date +%s) - a_start ))s — continuing"
done

# ====== Transition: default -> cache_on ======
log ">>> Killing default llama-server, switching to cache_on"
pkill -f llama-server >/dev/null 2>&1 || true
sleep 5

# ====== Phase B: cache-on sweep (different server flags) ======
log ">>> Phase B: cache-on sweep (fig13_pareto_cache_on)"
b_start=$(date +%s)
sweep/run_full.sh fig13_pareto_cache_on.yaml cache_on </dev/null >> "$MASTER_LOG" 2>&1 \
    && log "----- Finished: fig13_pareto_cache_on.yaml in $(( $(date +%s) - b_start ))s" \
    || log "----- WARN: fig13_pareto_cache_on.yaml failed"

# ====== Transition: cache_on -> default ======
log ">>> Killing cache_on llama-server, back to default for Fig 16"
pkill -f llama-server >/dev/null 2>&1 || true
sleep 5

# ====== Phase C: Fig 16 sweeps (default cache) ======
log ">>> Phase C: Fig 16 sweeps (3 panels)"
for cfg in fig16a_reflexion_sequential.yaml fig16b_lats_sequential.yaml fig16c_lats_parallel.yaml; do
    log "----- Starting: $cfg"
    c_start=$(date +%s)
    sweep/run_full.sh "$cfg" </dev/null >> "$MASTER_LOG" 2>&1 \
        && log "----- Finished: $cfg in $(( $(date +%s) - c_start ))s" \
        || log "----- WARN: $cfg exited non-zero"
done

# ====== Phase D: Stop server, run all plots ======
log ">>> All sweeps complete. Stopping llama-server."
pkill -f llama-server >/dev/null 2>&1 || true
sleep 2

log ">>> Phase D: Running 9 plot scripts"
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-local}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
for fig in 4 5 7 8 9 13 14 15 16; do
    log "----- Plotting fig$fig"
    python -m analysis.plot_fig${fig} >> "$MASTER_LOG" 2>&1 \
        && log "----- fig$fig PNG generated" \
        || log "----- WARN: plot_fig$fig failed (data may be missing)"
done

total=$(( $(date +%s) - start_time ))
hours=$(( total / 3600 ))
log ""
log "=================================="
log "  MASTER CHAIN END — total ${hours}h"
log "=================================="
log "Raw data:  $REPRO/results/raw/*.jsonl"
log "Figures:   $REPRO/results/figures/*.png"
log "Logs:      $MASTER_LOG"
