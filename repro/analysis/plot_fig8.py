"""Fig 8: per-agent input/output token decomposition stacked bar.

Paper's 6-way decomposition (Instruction, Few-shot, User, LLM history,
Tool history, Output) is not directly recoverable from our LangChain
BaseMessage hooks. We expose the 4-way categorization that IS available
(system / human / ai / tool) + output, with paper-aligned colors and
explicit annotation of what's missing.

L4 limit consequences shown directly on the chart:
- input/tool = 0 across all agents (HumanMessage-wrapped tool responses)
- LLMCompiler input/output totals = 0 (planner ran but callback missed)
"""
import numpy as np
import matplotlib.pyplot as plt

from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

# Paper Fig 8 color convention (mapped to our 4-way available roles):
#   light gray = Instruction       → input/system
#   dark gray  = Few-shot           (merged into system)
#   black      = User               → input/human
#   green      = LLM history        → input/ai
#   yellow     = Tool history       → input/tool  (0 in our data)
#   red        = Output             → output
ROLES = ["system", "human", "ai", "tool"]
ROLE_COLORS = {
    "system": "#b8b8b8",  # light gray — Instruction (+ Few-shot, merged)
    "human":  "#1a1a1a",  # black     — User
    "ai":     "#2ca02c",  # green     — LLM history
    "tool":   "#f1c93b",  # yellow    — Tool history (paper convention)
    "output": "#d62728",  # red       — Output
}
DISPLAY = {"react": "ReAct", "reflexion": "Reflexion",
           "llmcompiler": "LLMCompiler", "lats": "LATS"}


def _sum_by_role(df) -> dict[str, float]:
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

    if "tokens_by_role" not in df.columns:
        df["tokens_by_role"] = [{} for _ in range(len(df))]

    order = ["react", "reflexion", "llmcompiler", "lats"]
    agents = [a for a in order if a in df["agent_type"].unique()]

    role_totals = {}
    output_totals = {}
    for agent in agents:
        sub = df[df.agent_type == agent]
        role_totals[agent] = _sum_by_role(sub)
        output_totals[agent] = int(sub["tokens_output_total"].sum())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(agents))
    bottom = np.zeros(len(agents))

    for role in ROLES:
        vals = np.array([role_totals[a][role] for a in agents], dtype=float)
        ax.bar(x, vals, width=0.6, bottom=bottom,
               color=ROLE_COLORS[role],
               label=f"input/{role} ({'paper: '+{'system':'Instruction+Fewshot (gray)','human':'User (black)','ai':'LLM history (green)','tool':'Tool history (yellow)'}[role]})")
        bottom = bottom + vals

    out_vals = np.array([output_totals[a] for a in agents], dtype=float)
    ax.bar(x, out_vals, width=0.6, bottom=bottom,
           color=ROLE_COLORS["output"],
           label="output (paper: Output, red)")

    # Capitalize agent names.
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY[a] for a in agents])
    ax.set_ylabel("Total tokens across all queries")
    ax.set_title("Fig 8: Token decomposition by role on HotpotQA (8B Q4_K_M)")
    ax.legend(loc="upper left", ncol=1, fontsize=7.5)

    # Annotate L4 limitations directly on broken bars.
    for i, a in enumerate(agents):
        total = sum(role_totals[a].values()) + output_totals[a]
        # LLMCompiler is nearly empty.
        if total < 100000:
            ax.text(i, total + 20000,
                    "L4: callback\nmissed most\ntokens\n(planner only)" if a == "llmcompiler" else "",
                    ha="center", fontsize=7.5, color="#aa3333")
        # Tool tokens always 0 — annotate at base.
        if role_totals[a]["tool"] == 0 and role_totals[a]["system"] + role_totals[a]["ai"] > 0:
            ax.text(i + 0.32, role_totals[a]["system"] * 0.5,
                    "tool = 0\n(HumanMsg-\nwrapped)",
                    ha="left", va="center", fontsize=6.5, color="#aa3333",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="#aa3333", linewidth=0.6))

    # Bottom footnote — explain the FAIL.
    fig.text(0.5, -0.04,
             "L4 (Fig 8 verification FAIL): input/tool = 0 across all agents because HotpotQA tool responses "
             "arrive as LangChain HumanMessage, not ToolMessage. Paper's 6-way decomposition not recoverable "
             "from our handler; counts above are the 4-way (system/human/ai/tool) we CAN read.",
             ha="center", fontsize=7.5, color="#555", wrap=True)

    out = FIGURES_DIR / "fig8_token_decomp.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    for agent in agents:
        rt = role_totals[agent]
        ot = output_totals[agent]
        print(f"{agent}: input{{system={rt['system']}, human={rt['human']}, "
              f"ai={rt['ai']}, tool={rt['tool']}}}  output={ot}")
    print("Fig 8 verification: FAIL (tool=0 across all agents) — documented L4 limit")


if __name__ == "__main__":
    main()
