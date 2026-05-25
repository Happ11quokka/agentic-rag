"""Fig 13: accuracy vs e2e latency Pareto across 4 agents."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, AGENT_COLORS, AGENT_MARKERS

def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type").agg(
        accuracy=("correct", "mean"),
        mean_latency=("e2e_latency_s", "mean"),
    )

    fig, ax = plt.subplots()
    for agent, row in agg.iterrows():
        ax.scatter(row["mean_latency"], 100*row["accuracy"],
                   s=180, color=AGENT_COLORS[agent], marker=AGENT_MARKERS[agent],
                   label=agent)
    ax.set_xlabel("Mean end-to-end latency (s)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Fig 13: Accuracy vs Latency Pareto on HotpotQA")
    ax.legend(title="Agent")
    out = FIGURES_DIR / "fig13_pareto.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification (spec §12)
    rho, p = spearmanr(agg["mean_latency"], agg["accuracy"])
    print(f"Spearman ρ(latency, accuracy) = {rho:.3f} (must be ≥ 0.6)")
    react_acc = agg.loc["react", "accuracy"]
    lats_acc = agg.loc["lats", "accuracy"]
    react_lat = agg.loc["react", "mean_latency"]
    lats_lat = agg.loc["lats", "mean_latency"]
    print(f"LATS accuracy: {lats_acc:.3f}; ReAct accuracy: {react_acc:.3f}")
    print(f"LATS/ReAct accuracy ratio: {lats_acc/react_acc:.3f} (must be ≥ 0.9)")
    print(f"LATS latency: {lats_lat:.1f}s; ReAct latency: {react_lat:.1f}s")

    assert rho >= 0.6, f"FAIL: Spearman {rho:.3f} < 0.6"
    assert lats_acc >= 0.9 * react_acc, f"FAIL: LATS/ReAct {lats_acc/react_acc:.3f} < 0.9"
    assert lats_lat > react_lat, f"FAIL: LATS latency not > ReAct"
    print("Fig 13 verification: PASS")

if __name__ == "__main__":
    main()
