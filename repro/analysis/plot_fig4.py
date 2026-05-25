"""Fig 4: mean LLM calls and tool calls per agent on HotpotQA."""
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, AGENT_COLORS

def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type")[["n_llm_calls", "n_tool_calls"]].mean()
    order = ["react", "reflexion", "llmcompiler", "lats"]
    agg = agg.reindex(order)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(agg))
    w = 0.35
    ax.bar([i - w/2 for i in x], agg["n_llm_calls"], width=w,
           label="LLM calls", color="#d62728")
    ax.bar([i + w/2 for i in x], agg["n_tool_calls"], width=w,
           label="Tool calls", color="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(agg.index)
    ax.set_ylabel("Mean per query")
    ax.set_title("Fig 4: LLM and Tool calls per request (HotpotQA)")
    ax.legend()
    out = FIGURES_DIR / "fig4_calls.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification criterion (spec §12)
    if "lats" not in agg.index or "react" not in agg.index:
        raise SystemExit(
            f"FAIL: missing agent rows. Have: {list(agg.index)}; expected react+lats"
        )
    react_mean = agg.loc["react", "n_llm_calls"]
    lats_mean = agg.loc["lats", "n_llm_calls"]
    ratio = lats_mean / react_mean if react_mean else 0
    print(f"LATS/ReAct LLM call ratio: {ratio:.2f} (must be ≥ 5.0)")
    assert ratio >= 5.0, f"FAIL: ratio {ratio:.2f} < 5.0"
    print("Fig 4 verification: PASS")

if __name__ == "__main__":
    main()
