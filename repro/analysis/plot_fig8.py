"""Fig 8: per-agent input-token decomposition stacked bar.

For each agent, sum tokens_by_role across queries (system + human + ai + tool)
then stack tokens_output_total on top. The paper's Fig 8 shows that AI history
and tool history dominate for agentic workloads.

Data: fig13_pareto.jsonl. tokens_by_role is best-effort (see chat_wrapper.py),
so if buckets are mostly empty we still produce a plot — annotated as partial.
"""
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, AGENT_COLORS

ROLES = ["system", "human", "ai", "tool"]
ROLE_COLORS = {
    "system": "#999999",
    "human":  "#1f77b4",
    "ai":     "#d62728",
    "tool":   "#2ca02c",
    "output": "#ff7f0e",
}


def _sum_by_role(df) -> dict[str, float]:
    """Aggregate dict-typed tokens_by_role column across rows."""
    out = {r: 0 for r in ROLES}
    for d in df.get("tokens_by_role", []):
        if not isinstance(d, dict):
            continue
        for r in ROLES:
            out[r] += int(d.get(r, 0) or 0)
    return out


def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")

    # Defensive: tokens_by_role may not be present in pre-Fig8 data.
    if "tokens_by_role" not in df.columns:
        df["tokens_by_role"] = [{} for _ in range(len(df))]

    order = ["react", "reflexion", "llmcompiler", "lats"]
    agents = [a for a in order if a in df["agent_type"].unique()]

    # Per-agent role totals + output totals.
    role_totals = {}
    output_totals = {}
    total_tokens_by_role_nonempty = 0
    for agent in agents:
        sub = df[df.agent_type == agent]
        role_totals[agent] = _sum_by_role(sub)
        output_totals[agent] = int(sub["tokens_output_total"].sum())
        total_tokens_by_role_nonempty += sum(role_totals[agent].values())

    partial_data = total_tokens_by_role_nonempty == 0

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(agents))
    bottom = np.zeros(len(agents))

    # Stack the four input roles (in display order).
    for role in ROLES:
        vals = np.array([role_totals[a][role] for a in agents], dtype=float)
        ax.bar(x, vals, width=0.6, bottom=bottom,
               color=ROLE_COLORS[role], label=f"input/{role}")
        bottom = bottom + vals

    # Stack output tokens on top.
    out_vals = np.array([output_totals[a] for a in agents], dtype=float)
    ax.bar(x, out_vals, width=0.6, bottom=bottom,
           color=ROLE_COLORS["output"], label="output")

    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Token total across queries")
    title = "Fig 8: Token decomposition by message role on HotpotQA"
    if partial_data:
        title += " (partial data: tokens_by_role empty)"
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2)

    if partial_data:
        ax.text(0.5, 0.5,
                "tokens_by_role not populated in this dataset\n"
                "(measurement layer fallback failed or data is pre-Fig8)",
                transform=ax.transAxes, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.85), fontsize=10)

    out = FIGURES_DIR / "fig8_token_decomp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # ── Verification (spec §12) ────────────────────────────────────────────
    print(f"partial_data = {partial_data}")
    for agent in agents:
        rt = role_totals[agent]
        ot = output_totals[agent]
        total_in = sum(rt.values())
        print(f"{agent}: input{{system={rt['system']}, human={rt['human']}, "
              f"ai={rt['ai']}, tool={rt['tool']}}}  output={ot}")

        if partial_data:
            # Don't enforce verification when we have no data.
            continue

        # AI + tool history must contribute > 0 (the whole point of Fig 8).
        ai_plus_tool = rt["ai"] + rt["tool"]
        assert ai_plus_tool > 0, (
            f"FAIL ({agent}): ai+tool tokens = 0 — agent is not running tools "
            "or callback fired before tool messages were added"
        )

        # HotpotQA-specific: tool share ≥ 10% of input.
        if total_in > 0:
            tool_share = rt["tool"] / total_in
            print(f"  tool input share: {tool_share*100:.1f}% (must be >= 10%)")
            assert tool_share >= 0.10, (
                f"FAIL ({agent}): tool tokens are {tool_share*100:.1f}% of input "
                "(< 10%); HotpotQA tool history should dominate"
            )

    print("Fig 8 verification: " + ("SKIPPED (partial data)" if partial_data else "PASS"))


if __name__ == "__main__":
    main()
