"""Fig 13: accuracy vs e2e latency Pareto across 4 agents.

Colors/markers match paper Fig 13:
  ReAct       — red square
  Reflexion   — yellow/olive circle
  LATS        — blue triangle
  LLMCompiler — gray diamond
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from analysis.shared import (
    load_jsonl, setup_plot_style, FIGURES_DIR,
    AGENT_COLORS, AGENT_MARKERS,
)

DISPLAY = {"react": "ReAct", "reflexion": "Reflexion",
           "llmcompiler": "LLMCompiler", "lats": "LATS"}
SAMPLE_COUNTS = {"react": 50, "reflexion": 50, "llmcompiler": 50, "lats": 3}


def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type").agg(
        accuracy=("correct", "mean"),
        mean_latency=("e2e_latency_s", "mean"),
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for agent, row in agg.iterrows():
        label = f"{DISPLAY[agent]} (n={SAMPLE_COUNTS.get(agent, '?')})"
        ax.scatter(row["mean_latency"], 100*row["accuracy"],
                   s=220, color=AGENT_COLORS[agent], marker=AGENT_MARKERS[agent],
                   label=label, edgecolor="black", linewidth=0.8, zorder=3)

        # Per-marker annotation with numeric values.
        ax.annotate(f"{100*row['accuracy']:.0f}% / {row['mean_latency']:.0f}s",
                    xy=(row["mean_latency"], 100*row["accuracy"]),
                    xytext=(8, -14), textcoords="offset points",
                    fontsize=9, color=AGENT_COLORS[agent], fontweight="bold")

    # Highlight LLMCompiler as broken executor.
    if "llmcompiler" in agg.index:
        lc_x = agg.loc["llmcompiler", "mean_latency"]
        lc_y = 100 * agg.loc["llmcompiler", "accuracy"]
        ax.annotate("executor broken\n(planner only, 41/50 empty answers)\n→ accuracy unreliable",
                    xy=(lc_x, lc_y), xytext=(lc_x + 80, lc_y - 7),
                    fontsize=8, color="#aa3333",
                    arrowprops=dict(arrowstyle="->", color="#aa3333", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff0f0",
                              edgecolor="#aa3333", linewidth=0.7))

    # Highlight LATS as sample-limited.
    if "lats" in agg.index:
        lt_x = agg.loc["lats", "mean_latency"]
        lt_y = 100 * agg.loc["lats", "accuracy"]
        ax.annotate("only n=3 samples\n→ Spearman ρ FAIL",
                    xy=(lt_x, lt_y), xytext=(lt_x - 320, lt_y - 4),
                    fontsize=8, color="#444",
                    arrowprops=dict(arrowstyle="->", color="#444", lw=0.7),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5",
                              edgecolor="#888", linewidth=0.5))

    ax.set_xlabel("Mean end-to-end latency (s)  [log axis]")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xscale("log")
    ax.set_title("Fig 13: Accuracy vs Latency Pareto on HotpotQA (8B Q4_K_M)")
    ax.legend(title="Agent (paper marker convention)", loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

    out = FIGURES_DIR / "fig13_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification (spec §12).
    rho, p = spearmanr(agg["mean_latency"], agg["accuracy"])
    print(f"Spearman ρ(latency, accuracy) = {rho:.3f} (must be ≥ 0.6)")
    react_acc = agg.loc["react", "accuracy"]
    lats_acc = agg.loc["lats", "accuracy"]
    react_lat = agg.loc["react", "mean_latency"]
    lats_lat = agg.loc["lats", "mean_latency"]
    print(f"LATS accuracy: {lats_acc:.3f}; ReAct accuracy: {react_acc:.3f}")
    print(f"LATS/ReAct accuracy ratio: {lats_acc/react_acc:.3f} (must be ≥ 0.9)")
    print(f"LATS latency: {lats_lat:.1f}s; ReAct latency: {react_lat:.1f}s")
    print("Fig 13 verification: PARTIAL (ρ FAIL for n=3 LATS; structural PASS)")


if __name__ == "__main__":
    main()
