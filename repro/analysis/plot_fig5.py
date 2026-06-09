"""Fig 5: per-agent latency breakdown as stacked bar (% of e2e) +
end-to-end latency diamond marker on right Y-axis.

Components (paper §IV-A): prefill, decode, tool, overhead.

For ReAct, all four components are directly measured.
For Reflexion/LATS, tool_ms is 0 in the raw data (L4 callback limit),
so we estimate tool latency as recovered_tool_count × ReAct per-call rate
(~2.08 s/Wikipedia call). The estimated portion is rendered with hatched
fill to distinguish from measured data. LLMCompiler's tool wait went
into the 'overhead' bucket (executor broken) and is not separately
estimable.
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

RECOVERED = (
    Path(__file__).resolve().parents[1]
    / "results" / "aggregated" / "tool_calls_recovered.json"
)

# Paper §IV-A color convention:
#   red = LLM (prefill + decode combined), black = Tool,
#   gray = Others/Overhead, green diamond = e2e latency.
# We keep prefill vs decode visually distinct via shade so our detail is
# preserved while the overall palette matches the paper.
COMPONENTS = ["prefill", "decode", "tool", "tool_est", "overhead"]
COMPONENT_COLORS = {
    "prefill":  "#f4a3a3",   # light red (LLM prefill)
    "decode":   "#d62728",   # red       (LLM decode)
    "tool":     "#000000",   # black     (Tool, measured)
    "tool_est": "#000000",   # black hatched (Tool, estimated)
    "overhead": "#bcbcbc",   # gray
}
COMPONENT_HATCH = {"tool_est": "///"}


def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    agg = df.groupby("agent_type").agg(
        prefill_ms=("prefill_total_ms", "mean"),
        decode_ms=("decode_total_ms", "mean"),
        tool_ms=("tool_total_ms", "mean"),
        overhead_ms=("overhead_ms", "mean"),
        e2e_s=("e2e_latency_s", "mean"),
    )
    order = ["react", "reflexion", "llmcompiler", "lats"]
    agg = agg.reindex([a for a in order if a in agg.index])

    # Estimate tool latency for agents where callbacks missed tool calls.
    react_tool_ms = float(agg.loc["react", "tool_ms"])
    react_n_tools = float(
        df[df["agent_type"] == "react"]["n_tool_calls"].mean()
    )
    per_call_ms = react_tool_ms / react_n_tools if react_n_tools else 0.0

    recovered = json.loads(RECOVERED.read_text()) if RECOVERED.exists() else {}
    tool_est_ms = {}
    for a in agg.index:
        if a == "react":
            tool_est_ms[a] = 0.0
            continue
        info = recovered.get(a, {})
        n_est = info.get("n_tool_calls_estimated_mean")
        tool_est_ms[a] = float(n_est) * per_call_ms if n_est else 0.0

    e2e_ms = agg["e2e_s"] * 1000.0
    raw = {
        "prefill":  agg["prefill_ms"].fillna(0.0),
        "decode":   agg["decode_ms"].fillna(0.0),
        "tool":     agg["tool_ms"].fillna(0.0),
        "tool_est": agg.index.map(lambda a: tool_est_ms[a]),
        "overhead": agg["overhead_ms"].fillna(0.0),
    }
    pct = {k: (np.asarray(v, dtype=float) / e2e_ms.values * 100.0).clip(min=0.0)
           for k, v in raw.items()}
    total_pct = sum(pct.values())
    # If components sum > 100% of e2e (polling double-counting under
    # concurrency), rescale proportionally so they fit visually.
    for k in pct:
        pct[k] = np.where(total_pct > 100.0,
                          pct[k] / total_pct * 100.0,
                          pct[k])

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(agg))
    bottom = np.zeros(len(agg))

    # Paper uses {LLM, LLM+Tool, Tool, Others}. We expose LLM as
    # prefill+decode for diagnostic detail; legend keeps the paper's vocab.
    for comp in COMPONENTS:
        label = {"prefill":  "LLM — prefill",
                 "decode":   "LLM — decode",
                 "tool":     "Tool — measured (ReAct only)",
                 "tool_est": "Tool — estimated (recovered)",
                 "overhead": "Others"}.get(comp, comp)
        edge = "white" if comp == "tool_est" else "none"
        ax1.bar(x, pct[comp], width=0.6, bottom=bottom,
                color=COMPONENT_COLORS[comp],
                hatch=COMPONENT_HATCH.get(comp, None),
                edgecolor=edge,
                linewidth=0,
                label=label)
        bottom = bottom + np.asarray(pct[comp])

    # Annotate raw-sum overflow for each agent (data quality warning).
    for i, a in enumerate(agg.index):
        raw_total = float(total_pct[i])
        if raw_total > 105:
            ax1.text(i, 102, f"raw sum {raw_total:.0f}%\n(over 100% =\nconcurrent\nattribution)",
                     ha="center", fontsize=7, color="#aa3333")

    # Paper-convention agent names (capitalized).
    DISPLAY = {"react": "ReAct", "reflexion": "Reflexion",
               "llmcompiler": "LLMCompiler", "lats": "LATS"}
    ax1.set_xticks(x)
    ax1.set_xticklabels([DISPLAY.get(a, a) for a in agg.index])
    ax1.set_ylabel("Latency share (%)")
    ax1.set_ylim(0, 118)
    ax1.set_title("Fig 5: Latency breakdown on HotpotQA (8B Q4_K_M)")
    ax1.legend(loc="upper left", ncol=2, fontsize=8.5)

    # Diamond markers on right axis = mean e2e latency (s).
    # Paper uses GREEN diamonds for e2e.
    ax2 = ax1.twinx()
    ax2.plot(x, agg["e2e_s"].values, marker="D", linestyle="",
             markersize=11, markerfacecolor="#2ca02c", markeredgecolor="#1a6020",
             markeredgewidth=1.2, label="E2E latency (green ◆)")
    ax2.set_ylabel("End-to-end latency (s) — green ◆")
    ax2.legend(loc="upper right", fontsize=8.5)

    # Footnote.
    fig.text(0.5, -0.04,
             f"Tool estimate = recovered tool count × ReAct per-call rate "
             f"({per_call_ms:.0f} ms/call). Reflexion ≈ {tool_est_ms.get('reflexion',0)/1000:.0f}s, "
             f"LATS ≈ {tool_est_ms.get('lats',0)/1000:.0f}s. "
             f"LLMCompiler tool wait went into the 'overhead' bucket (executor broken).",
             ha="center", fontsize=7.5, color="#555", wrap=True)

    out = FIGURES_DIR / "fig5_latency_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification (spec §12).
    for agent in agg.index:
        idx = list(agg.index).index(agent)
        s = sum(float(pct[c][idx]) for c in COMPONENTS)
        print(f"{agent}: drawn component sum = {s:.1f}% (raw {total_pct[idx]:.1f}%), "
              f"e2e = {agg.loc[agent, 'e2e_s']:.1f}s")
        assert s <= 100.5, f"FAIL: {agent} drawn sum {s:.1f}% > 100%"

    if "react" in agg.index:
        tool_share = float(agg.loc["react", "tool_ms"] / e2e_ms.loc["react"] * 100.0)
        print(f"ReAct tool share: {tool_share:.1f}% (must be >= 20%)")
        assert tool_share >= 20.0, f"FAIL: ReAct tool share {tool_share:.1f}% < 20%"

    print("Fig 5 verification: PASS")


if __name__ == "__main__":
    main()
