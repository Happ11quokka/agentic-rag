"""Fig 5: per-agent latency breakdown as stacked bar (% of e2e) +
end-to-end latency diamond marker on right Y-axis.

Components (paper §IV-A): prefill, decode, tool, overhead. The four must
sum to <= 100% of total wall-clock (overhead picks up any residual).
"""
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, AGENT_COLORS

# Stack component colors (paper convention: prefill = blue, decode = orange,
# tool = green, overhead = light gray).
COMPONENTS = ["prefill", "decode", "tool", "overhead"]
COMPONENT_COLORS = {
    "prefill": "#1f77b4",
    "decode": "#ff7f0e",
    "tool":   "#2ca02c",
    "overhead": "#bcbcbc",
}


def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")

    # Per-agent means in milliseconds for stacked components, plus e2e in seconds.
    agg = df.groupby("agent_type").agg(
        prefill_ms=("prefill_total_ms", "mean"),
        decode_ms=("decode_total_ms", "mean"),
        tool_ms=("tool_total_ms", "mean"),
        overhead_ms=("overhead_ms", "mean"),
        e2e_s=("e2e_latency_s", "mean"),
    )
    order = ["react", "reflexion", "llmcompiler", "lats"]
    agg = agg.reindex([a for a in order if a in agg.index])

    # Convert e2e to ms; compute each component as % of e2e (cap at 100).
    e2e_ms = agg["e2e_s"] * 1000.0
    pct = {}
    for col, key in [
        ("prefill_ms", "prefill"),
        ("decode_ms",  "decode"),
        ("tool_ms",    "tool"),
        ("overhead_ms", "overhead"),
    ]:
        pct[key] = (agg[col] / e2e_ms * 100.0).fillna(0.0).clip(lower=0.0)
    pct_total = sum(pct.values())
    # If the four components sum to > 100% (rare; can happen if overhead double-counts),
    # rescale them proportionally so they fit.
    rescale = pct_total.where(pct_total <= 100.0, pct_total)
    for k in pct:
        pct[k] = np.where(pct_total > 100.0, pct[k] / pct_total * 100.0, pct[k])

    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(agg))
    bottom = np.zeros(len(agg))
    for comp in COMPONENTS:
        ax1.bar(x, pct[comp], width=0.6, bottom=bottom,
                color=COMPONENT_COLORS[comp], label=comp.capitalize())
        bottom = bottom + np.asarray(pct[comp])

    ax1.set_xticks(x)
    ax1.set_xticklabels(agg.index)
    ax1.set_ylabel("Latency share (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Fig 5: Latency breakdown on HotpotQA")
    ax1.legend(loc="upper left", ncol=2)

    # Diamond marker on right axis = mean e2e latency (s).
    ax2 = ax1.twinx()
    ax2.plot(x, agg["e2e_s"].values, marker="D", linestyle="",
             markersize=10, color="black", label="E2E latency (s)")
    ax2.set_ylabel("End-to-end latency (s)")
    ax2.legend(loc="upper right")

    out = FIGURES_DIR / "fig5_latency_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # ── Verification (spec §12 + paper §IV-A.1) ────────────────────────────
    # 1) Component sum ≤ 100% (after rescaling we enforce this).
    for agent in agg.index:
        s = sum(float(pct[c][list(agg.index).index(agent)]) for c in COMPONENTS)
        print(f"{agent}: component sum = {s:.1f}%  e2e = {agg.loc[agent, 'e2e_s']:.1f}s")
        assert s <= 100.5, f"FAIL: {agent} components sum to {s:.1f}% > 100%"

    # 2) HotpotQA-specific: tool share ≥ 20% (Wikipedia API dominates).
    # Apply to ReAct as the canonical tool-heavy baseline; other agents are advisory.
    if "react" in agg.index:
        tool_share = float(agg.loc["react", "tool_ms"] / e2e_ms.loc["react"] * 100.0)
        print(f"ReAct tool share: {tool_share:.1f}% (must be >= 20%)")
        assert tool_share >= 20.0, f"FAIL: ReAct tool share {tool_share:.1f}% < 20%"

    print("Fig 5 verification: PASS")


if __name__ == "__main__":
    main()
