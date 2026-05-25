"""Top-level sweep runner: parses config, owns MetricsCollector lifecycle,
iterates cells, supports --resume via append-only JSONL with fsync."""
import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Iterable

from measurement.metrics_collector import MetricsCollector
from measurement.trace_schema import QueryTrace
from sweep.cells import Cell, enumerate_cells, ResumeKey
from sweep.agent_runner import run_one


def _start_collector(base_url: str = "http://localhost:8000") -> MetricsCollector:
    return MetricsCollector(base_url).start()


def run_one_for_cell(cell: Cell, collector: MetricsCollector) -> QueryTrace:
    """Thin wrapper around run_one that passes the cell's extra_kwargs through."""
    return run_one(
        agent_type=cell.agent_type,
        fewshot=cell.fewshot,
        iteration_limit=cell.iteration_limit,
        sample_idx=cell.sample_idx,
        collector=collector,
        extra_kwargs=cell.extra_kwargs,
    )


def _resume_key_from_row(row: dict) -> ResumeKey:
    """Reconstruct the canonical ResumeKey from a JSONL row.

    Must match the shape returned by Cell.resume_key() in sweep/cells.py:
    (agent_type, (fewshot, iteration_limit), sample_idx)
    """
    from sweep.cells import _signature
    return (
        row["agent_type"],
        _signature(row.get("fewshot", 0), row.get("iteration_limit", 0)),
        row["sample_idx"],
    )


def read_done_tuples(path: str) -> set[ResumeKey]:
    """Read existing JSONL and return the set of completed resume keys.
    Tolerates a malformed trailing line from a crash by skipping it.
    """
    if not os.path.exists(path):
        return set()
    done: set[ResumeKey] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # malformed line — likely truncated last row from a crash; ignore
                continue
            done.add(_resume_key_from_row(row))
    return done


def append_jsonl_fsync(path: str, trace: QueryTrace) -> None:
    line = trace.model_dump_json() + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def run_sweep(cfg: dict, *, out_path: str, resume: bool = True,
              base_url: str = "http://localhost:8000") -> None:
    """Execute one sweep config end-to-end."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cells = list(enumerate_cells(cfg))

    if resume:
        done = read_done_tuples(out_path)
        before = len(cells)
        cells = [c for c in cells if c.resume_key() not in done]
        skipped = before - len(cells)
        print(f"[resume] skipping {skipped}/{before} cells already done", file=sys.stderr)

    collector = _start_collector(base_url=base_url)
    try:
        for i, cell in enumerate(cells):
            print(f"[{i+1}/{len(cells)}] {cell.agent_type} "
                  f"{cell.sweep_var_name}={cell.sweep_var_val} sample={cell.sample_idx}",
                  file=sys.stderr, flush=True)
            trace = run_one_for_cell(cell, collector)
            trace.run_id = cfg.get("run_id", "")
            append_jsonl_fsync(out_path, trace)
    finally:
        collector.stop()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to sweep YAML")
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--base-url", default="http://localhost:8000")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_sweep(cfg, out_path=args.out, resume=not args.no_resume, base_url=args.base_url)


if __name__ == "__main__":
    main()
