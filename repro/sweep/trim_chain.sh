#!/usr/bin/env bash
# OPTION B trim chain: finish fig13_pareto (LATS 5 more + LLMCompiler 50) + 5 plots.
# Expected total: ~2.5 hours.
# All earlier ReAct/Reflexion/LATS rows are preserved via --resume.

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
LOG_DIR="$REPRO/results/sweep_logs"
MASTER_LOG="$LOG_DIR/trim_chain.log"
mkdir -p "$LOG_DIR"
cd "$REPRO"

log() {
    echo "[$(date +%Y-%m-%d_%H:%M:%S)] $*" | tee -a "$MASTER_LOG"
}

start_time=$(date +%s)
log "=================================="
log "  OPTION B TRIM CHAIN — PID $$"
log "=================================="
log "Plan: finish fig13_pareto + 5 plots (~2.5h)"
log ""

# ====== Run fig13_pareto (resume mode, only LATS 5 + LLMCompiler 50 remaining) ======
log ">>> Finishing fig13_pareto (LATS 8-target + LLMCompiler 50)"
sweep/run_full.sh fig13_pareto.yaml </dev/null >> "$MASTER_LOG" 2>&1 \
    && log "----- Finished: fig13_pareto.yaml" \
    || log "----- WARN: fig13_pareto.yaml exited non-zero"

# ====== Kill llama-server, run 5 plots ======
log ">>> Stopping llama-server"
pkill -f llama-server >/dev/null 2>&1 || true
sleep 2

log ">>> Running 5 plot scripts (Fig 4, 5, 7, 8, 13)"
source .venv/bin/activate
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-dummy-local}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
for fig in 4 5 7 8 13; do
    log "----- Plotting fig$fig"
    python -m analysis.plot_fig${fig} >> "$MASTER_LOG" 2>&1 \
        && log "----- fig$fig PNG generated" \
        || log "----- WARN: plot_fig$fig failed (may need data)"
done

total=$(( $(date +%s) - start_time ))
minutes=$(( total / 60 ))
log ""
log "=================================="
log "  TRIM CHAIN END — total ${minutes}min"
log "=================================="
log "Figures in: $REPRO/results/figures/"
