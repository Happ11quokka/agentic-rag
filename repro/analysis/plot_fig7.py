"""Fig 7: e2e latency distribution for HotpotQA agents.

Paper compares ShareGPT (static chatbot) vs ReAct/WebShop (agentic).
We don't have ShareGPT or WebShop data, so we instead overlay the four
agents we DO have on HotpotQA to make the multi-curve comparison.
Particularly interesting: LLMCompiler has p95/p50 ≈ 7.6 (heaviest tail)
because the broken executor produces a bimodal distribution
(~5 s "planner only failed" vs ~90 s "tried then gave up").
"""
import numpy as np
import matplotlib.pyplot as plt

from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR


# Paper-aligned colors for HotpotQA agents (matches Fig 13 convention).
AGENT_COLORS = {
    "react":       "#d62728",  # red square in paper
    "reflexion":   "#bcbd22",  # yellow/olive circle
    "lats":        "#1f77b4",  # blue triangle
    "llmcompiler": "#7f7f7f",  # gray diamond
}
DISPLAY = {"react": "ReAct", "reflexion": "Reflexion",
           "llmcompiler": "LLMCompiler", "lats": "LATS"}


def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Show ReAct as a histogram (primary, like paper's main red curve)
    react_lat = df[df.agent_type == "react"]["e2e_latency_s"].values
    ax.hist(react_lat, bins=20, density=True,
            color=AGENT_COLORS["react"], alpha=0.55,
            label=f"ReAct n={len(react_lat)}")

    # Overlay other agents as smoothed density (line) for comparison.
    for a in ("reflexion", "llmcompiler"):
        vals = df[df.agent_type == a]["e2e_latency_s"].values
        if len(vals) < 3:
            continue
        # Density via histogram + step plot.
        counts, edges = np.histogram(vals, bins=20, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.plot(centers, counts, color=AGENT_COLORS[a],
                linewidth=2, label=f"{DISPLAY[a]} n={len(vals)}")

    # LATS sample too small (n=3), show as text annotation only.
    lats = df[df.agent_type == "lats"]["e2e_latency_s"].values
    if len(lats):
        ax.axvspan(min(lats), max(lats), color=AGENT_COLORS["lats"],
                   alpha=0.10, label=f"LATS n={len(lats)} (range only)")

    # Percentile markers for ReAct (paper convention).
    p50 = np.percentile(react_lat, 50)
    p95 = np.percentile(react_lat, 95)
    ax.axvline(p50, ls="--", color="black", alpha=0.7, label=f"ReAct p50 = {p50:.1f}s")
    ax.axvline(p95, ls="--", color="red",   alpha=0.7, label=f"ReAct p95 = {p95:.1f}s")

    ax.set_xlabel("End-to-end latency (s)")
    ax.set_ylabel("Frequency density")
    ax.set_title("Fig 7: HotpotQA e2e latency distribution (8B Q4_K_M)")
    ax.legend(loc="upper right", fontsize=8.5)

    # Annotate heavy-tail ratios.
    ratios_text = ""
    for a in ("react", "reflexion", "llmcompiler"):
        vals = df[df.agent_type == a]["e2e_latency_s"].values
        if len(vals) >= 5:
            p50_a = np.percentile(vals, 50)
            p95_a = np.percentile(vals, 95)
            ratios_text += f"{DISPLAY[a]}: p95/p50 = {p95_a/p50_a:.2f}   "
    fig.text(0.5, -0.02, ratios_text.strip(),
             ha="center", fontsize=8.5, color="#555")
    fig.text(0.5, -0.06,
             "Heaviest tail = LLMCompiler (executor breaks early on most queries → bimodal). "
             "LATS shown as range only (n=3 insufficient for distribution).",
             ha="center", fontsize=7.5, color="#777")

    out = FIGURES_DIR / "fig7_latency_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    ratio = p95 / p50 if p50 else 0
    print(f"ReAct p95/p50 = {ratio:.2f} (must be ≥ 2.0)")
    assert ratio >= 2.0, f"FAIL: ratio {ratio:.2f} < 2.0"
    print("Fig 7 verification: PASS")


if __name__ == "__main__":
    main()
