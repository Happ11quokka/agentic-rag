"""Recover per-agent tool-call estimates from existing JSONL traces.

ReAct tool counts are measured directly via LangChain callbacks.
Reflexion/LATS/LLMCompiler tool wrappers bypass the callback chain
(see design spec §11 L4) so n_tool_calls is recorded as 0.
This script reconstructs the missing counts from the structural fields
that ARE recorded: n_llm_calls, n_reflections, n_tree_expansions.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

JSONL = Path(__file__).resolve().parents[1] / "results" / "raw" / "fig13_pareto.jsonl"
OUT = Path(__file__).resolve().parents[1] / "results" / "aggregated" / "tool_calls_recovered.json"

# Inferred from src/agents/LATS/hotpotqa/search.py
LATS_N_GENERATE = 5
# Average length of each LATS child trajectory before terminal — estimated from
# react.n_tool_calls / react.n_llm_calls ≈ 0.966 + tree structure overhead.
LATS_ACTIONS_PER_CHILD = 3.0


def recover():
    by_agent = defaultdict(lambda: {
        "rows": 0,
        "n_llm_calls": [],
        "n_tool_calls_measured": [],
        "n_reflections": [],
        "n_tree_expansions": [],
        "correct": [],
        "n_tool_calls_estimated": [],
        "estimation_method": "",
    })

    with JSONL.open() as f:
        for line in f:
            r = json.loads(line)
            a = r["agent_type"]
            s = by_agent[a]
            s["rows"] += 1
            n_llm = r.get("n_llm_calls", 0)
            n_refl = r.get("n_reflections", 0)
            n_tree = r.get("n_tree_expansions", 0)
            s["n_llm_calls"].append(n_llm)
            s["n_tool_calls_measured"].append(r.get("n_tool_calls", 0))
            s["n_reflections"].append(n_refl)
            s["n_tree_expansions"].append(n_tree)
            s["correct"].append(1 if r.get("correct") else 0)

            # Per-row tool-call estimate using structural models.
            if a == "react":
                est = r.get("n_tool_calls", 0)  # already measured
                method = "measured"
            elif a == "reflexion":
                # Reflexion = (reflections+1) trials, each trial ends with a
                # "finish" LLM call that has no tool. Reflection LLM calls
                # also have no tool. So:
                #   tools ≈ n_llm − n_reflections − (n_reflections + 1)
                trials = n_refl + 1
                est = max(0, n_llm - n_refl - trials)
                method = "n_llm − n_reflections − (n_reflections + 1)"
            elif a == "lats":
                # LATS expands tree nodes. Each expansion samples 5 children
                # (n_generate_sample=5). Each child trajectory executes a few
                # tool actions before being evaluated. Empirical reading of
                # src/agents/LATS/hotpotqa/search.py shows actions are tightly
                # bound to expansions, not raw LLM calls.
                est = round(n_tree * LATS_N_GENERATE * LATS_ACTIONS_PER_CHILD)
                method = f"n_tree_expansions × {LATS_N_GENERATE} children × {LATS_ACTIONS_PER_CHILD} actions"
            elif a == "llmcompiler":
                # LLMCompiler made only 1 LLM call on average → the planner
                # ran but the DAG executor did not produce a final answer in
                # most cases (final_answer is empty). Tool count cannot be
                # recovered from structural fields alone.
                est = None
                method = "UNKNOWN — planner ran but executor broken (final_answer empty in 41/50)"
            else:
                est = None
                method = "unknown agent"

            s["n_tool_calls_estimated"].append(est)
            if not s["estimation_method"]:
                s["estimation_method"] = method

    # Summarize.
    summary = {}
    for a, s in by_agent.items():
        valid_est = [e for e in s["n_tool_calls_estimated"] if e is not None]
        summary[a] = {
            "rows": s["rows"],
            "n_llm_calls_mean": round(statistics.mean(s["n_llm_calls"]), 2),
            "n_tool_calls_measured_mean": round(statistics.mean(s["n_tool_calls_measured"]), 2),
            "n_tool_calls_estimated_mean": (
                round(statistics.mean(valid_est), 2) if valid_est else None
            ),
            "estimation_method": s["estimation_method"],
            "accuracy_pct": round(statistics.mean(s["correct"]) * 100, 1),
            "n_reflections_mean": round(statistics.mean(s["n_reflections"]), 2),
            "n_tree_expansions_mean": round(statistics.mean(s["n_tree_expansions"]), 2),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Written: {OUT}")
    print()
    print(f"{'agent':12} {'rows':>5} {'n_llm':>8} {'n_tool(measured)':>17} {'n_tool(estimated)':>18}  method")
    print("-" * 110)
    for a in ("react", "reflexion", "lats", "llmcompiler"):
        if a not in summary:
            continue
        s = summary[a]
        est_str = f"{s['n_tool_calls_estimated_mean']}" if s["n_tool_calls_estimated_mean"] is not None else "—"
        print(f"{a:12} {s['rows']:>5} {s['n_llm_calls_mean']:>8} {s['n_tool_calls_measured_mean']:>17} {est_str:>18}  {s['estimation_method']}")


if __name__ == "__main__":
    recover()
