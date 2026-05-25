#!/usr/bin/env bash
# Remove timeout/error rows from JSONL files so --resume retries them.
# Use when you know the network was unstable during part of the sweep.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="$SCRIPT_DIR/.."
cd "$REPRO"

source .venv/bin/activate

python <<'EOF'
import json
from pathlib import Path

RAW = Path("results/raw")
backup_dir = RAW / "pre_cleanup_backup"
backup_dir.mkdir(exist_ok=True)

total_kept = 0
total_dropped = 0

for jsonl in sorted(RAW.glob("*.jsonl")):
    rows = [json.loads(l) for l in jsonl.read_text().splitlines() if l.strip()]
    kept = []
    dropped = []
    for r in rows:
        meta = r.get("meta", {})
        is_failure = (
            meta.get("timeout") or
            meta.get("error") or
            r.get("final_answer", "").startswith("<TIMEOUT>") or
            r.get("final_answer", "").startswith("<ERROR>")
        )
        if is_failure:
            dropped.append(r)
        else:
            kept.append(r)

    if not dropped:
        print(f"  {jsonl.name}: nothing to drop ({len(kept)} kept)")
        continue

    # Backup
    backup = backup_dir / jsonl.name
    backup.write_text(jsonl.read_text())

    # Rewrite without failures
    with open(jsonl, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    print(f"  {jsonl.name}: dropped {len(dropped)} failed, kept {len(kept)} (backup: {backup})")
    total_kept += len(kept)
    total_dropped += len(dropped)

print()
print(f"TOTAL: kept {total_kept}, dropped {total_dropped}")
print("Now re-run master_chain.sh — --resume will retry the dropped cells.")
EOF
