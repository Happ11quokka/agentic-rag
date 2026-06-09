"""Fig 4: mean LLM calls and tool calls per agent on HotpotQA.

Tool calls for ReAct are measured via LangChain callbacks.
For Reflexion/LATS/LLMCompiler the agents bypass standard callbacks
(see design spec §11 L4), so tool counts are recovered structurally
from n_llm_calls / n_reflections / n_tree_expansions
(see repro/analysis/recover_tool_calls.py).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR


RECOVERED = (
    Path(__file__).resolve().parents[1]
    / "results" / "aggregated" / "tool_calls_recovered.json"
)


def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type")[["n_llm_calls", "n_tool_calls"]].mean()
    order = ["react", "reflexion", "llmcompiler", "lats"]
    agg = agg.reindex(order)

    # Load recovered tool-call estimates.
    if not RECOVERED.exists():
        raise SystemExit(
            f"Missing {RECOVERED}. Run: python -m analysis.recover_tool_calls"
        )
    recovered = json.loads(RECOVERED.read_text())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = list(range(len(agg)))
    w = 0.28

    # LLM calls (always measured).
    llm_vals = agg["n_llm_calls"].tolist()
    ax.bar([i - w for i in x], llm_vals, width=w,
           label="LLM (red)", color="#d62728")

    # Tool calls — split into measured vs recovered.
    measured_vals, estimated_vals, est_unknown = [], [], []
    for a in order:
        m = recovered[a]["n_tool_calls_measured_mean"]
        e = recovered[a]["n_tool_calls_estimated_mean"]
        if a == "react":
            measured_vals.append(m)
            estimated_vals.append(0)
            est_unknown.append(False)
        elif e is None:
            measured_vals.append(0)
            estimated_vals.append(0)
            est_unknown.append(True)  # plot a marker instead
        else:
            measured_vals.append(0)
            estimated_vals.append(e)
            est_unknown.append(False)

    ax.bar(x, measured_vals, width=w,
           label="Tool (black) — measured", color="black")
    ax.bar(x, estimated_vals, width=w,
           label="Tool (black hatched) — estimated",
           color="black", hatch="///", edgecolor="white", linewidth=0, alpha=0.85)

    # Numeric labels above each bar.
    for i, v in enumerate(llm_vals):
        ax.text(i - w, v + max(llm_vals) * 0.01, f"{v:.1f}", ha="center", fontsize=9, color="#d62728")
    for i, (m, e) in enumerate(zip(measured_vals, estimated_vals)):
        v = m if m else e
        if v > 0:
            ax.text(i, v + max(llm_vals) * 0.01, f"{v:.1f}", ha="center", fontsize=9, color="black")
        elif est_unknown[i]:
            ax.text(i, max(llm_vals) * 0.04, "?",
                    ha="center", fontsize=18, color="#aa3333", fontweight="bold")
            ax.text(i + w, max(llm_vals) * 0.10, "exec\nbroken",
                    ha="center", fontsize=7, color="#aa3333")

    # Display names match paper convention (capitalized).
    DISPLAY = {
        "react": "ReAct", "reflexion": "Reflexion",
        "llmcompiler": "LLMCompiler\n(planner ran,\nexecutor failed)",
        "lats": "LATS",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[a] for a in order])
    ax.set_ylabel("Mean per query")
    ax.set_title("Fig 4: LLM and Tool calls per request (HotpotQA, 8B Q4_K_M)")
    ax.legend(loc="upper center", fontsize=9)
    ax.set_yscale("log")
    ax.set_ylim(0.8, max(llm_vals) * 1.6)

    # Footnote.
    fig.text(0.5, -0.02,
             "Tool counts for Reflexion/LATS recovered from n_reflections / "
             "n_tree_expansions × 5 children × 3 actions (L4 limit). "
             "LLMCompiler tool count not recoverable — planner made 1 LLM call but DAG executor produced empty answers in 41/50 queries.",
             ha="center", fontsize=7.5, color="#555", wrap=True)

    out = FIGURES_DIR / "fig4_calls.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification criterion (spec §12).
    react_mean = agg.loc["react", "n_llm_calls"]
    lats_mean = agg.loc["lats", "n_llm_calls"]
    ratio = lats_mean / react_mean if react_mean else 0
    print(f"LATS/ReAct LLM call ratio: {ratio:.2f} (must be ≥ 5.0)")
    assert ratio >= 5.0, f"FAIL: ratio {ratio:.2f} < 5.0"
    print("Fig 4 verification: PASS")


if __name__ == "__main__":
    main()
