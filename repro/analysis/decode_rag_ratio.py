#!/usr/bin/env python
"""Summarize the decode-RAG baseline from a sweep JSONL of QueryTrace rows.

The headline metric is the *decode-RAG ratio* = retrieval time / end-to-end
time = ``tool_total_ms / (e2e_latency_s * 1000)``. This is the fraction of
wall-clock a decode-time RAG prefetch scheme could hope to hide. We report it
alongside the latency breakdown (tool vs LLM vs overhead), tool-call counts,
and EM accuracy.

Usage:
    python analysis/decode_rag_ratio.py results/raw/react_vectordb_smoke.jsonl
"""
import argparse
import json
import statistics as st


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(xs):
    return f"mean={st.mean(xs):.1%} median={st.median(xs):.1%} min={min(xs):.1%} max={max(xs):.1%}"


def ms(xs):
    return f"mean={st.mean(xs):.0f} median={st.median(xs):.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    args = ap.parse_args()

    rows = load(args.jsonl)
    n = len(rows)
    n_to = sum(1 for r in rows if r.get("meta", {}).get("timeout"))
    n_err = sum(1 for r in rows if r.get("meta", {}).get("error"))
    # "clean" = finished without wall-clock timeout / dispatch error
    ok = [r for r in rows
          if not r.get("meta", {}).get("timeout") and not r.get("meta", {}).get("error")
          and r.get("e2e_latency_s", 0) > 0]

    print(f"file: {args.jsonl}")
    print(f"rows={n}  clean={len(ok)}  timeouts={n_to}  errors={n_err}")
    if not ok:
        print("no clean rows to summarize")
        return

    ratios = [r["tool_total_ms"] / (r["e2e_latency_s"] * 1000.0) for r in ok]
    print()
    print(f"decode-RAG ratio (tool/e2e): {pct(ratios)}")
    print(f"n_tool_calls:  mean={st.mean([r['n_tool_calls'] for r in ok]):.2f} "
          f"median={st.median([r['n_tool_calls'] for r in ok])} "
          f"max={max(r['n_tool_calls'] for r in ok)}")
    print(f"n_llm_calls:   mean={st.mean([r['n_llm_calls'] for r in ok]):.2f}")
    print()
    print("latency breakdown (ms):")
    print(f"  tool_total_ms: {ms([r['tool_total_ms'] for r in ok])}")
    print(f"  llm_total_ms:  {ms([r['llm_total_ms'] for r in ok])}")
    print(f"  e2e_ms:        {ms([r['e2e_latency_s'] * 1000.0 for r in ok])}")
    print(f"  per-search ms: mean="
          f"{st.mean([r['tool_total_ms'] / r['n_tool_calls'] for r in ok if r['n_tool_calls']]):.0f}")
    print()
    acc = sum(1 for r in rows if r.get("correct")) / n
    print(f"EM accuracy (all rows): {acc:.1%}  ({sum(1 for r in rows if r.get('correct'))}/{n})")


if __name__ == "__main__":
    main()
