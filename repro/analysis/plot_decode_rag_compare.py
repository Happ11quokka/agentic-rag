"""decode-RAG baseline comparison: live Wikipedia ReAct vs Cohere vector-DB ReAct.

Compares the original reproduction's retrieval backend (live Wikipedia keyword
search, from fig13_pareto ReAct rows) against the current experiment's backend
(Cohere dense vector DB over a HotpotQA-targeted corpus, react_vectordb).

The headline metric is the *decode-RAG ratio* = tool_total_ms / e2e_ms, i.e. the
wall-clock fraction of a query spent waiting on retrieval — the slack a
decode-time RAG prefetch/overlap scheme could in principle hide.

Generates four figures into results/figures/decode_rag/:
  A  decode_rag_ratio.png        — ratio (tool/e2e), mean & median, per backend
  B  decode_rag_ratio_dist.png   — per-query ratio distribution (box + jitter)
  C  latency_breakdown.png       — mean latency composition (LLM/tool/overhead), s
  D  baseline_overview.png       — 4-panel: EM acc, e2e median, searches/q, ms/search

Labels are English on purpose (matplotlib Korean glyphs are unreliable); the
surrounding report supplies Korean narration.
"""
import numpy as np
import matplotlib.pyplot as plt

from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR
from measurement.eval import hotpotqa_f1

OUT_DIR = FIGURES_DIR / "decode_rag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# live Wikipedia = old/baseline (amber), Cohere vector DB = current/new (green)
WIKI = "#e0892e"
VDB = "#2ca02c"
WIKI_LABEL = "Live Wikipedia\n(ReAct, existing repro)"
VDB_LABEL = "Cohere vector DB\n(ReAct, current exp.)"
SHORT = ["Live Wikipedia", "Cohere vector DB"]


def backend_frames():
    """Return (wiki_df, vdb_df) — ReAct-only Wikipedia rows + vector-DB rows."""
    pareto = load_jsonl("fig13_pareto")
    wiki = pareto[pareto["agent_type"] == "react"].copy()
    vdb = load_jsonl("react_vectordb").copy()
    for df in (wiki, vdb):
        df["e2e_ms"] = df["e2e_latency_s"] * 1000.0
        df["tool_total_ms"] = df["tool_total_ms"].fillna(0.0)
        df["ratio"] = (df["tool_total_ms"] / df["e2e_ms"]).clip(lower=0.0)
        # F1 computed post-hoc from stored answers (older jsonl has no f1 column).
        df["f1"] = (df["f1"] if "f1" in df.columns else
                    [hotpotqa_f1(p, g) for p, g in
                     zip(df["final_answer"], df["expected_answer"])])
    return wiki, vdb


def per_search_ms(df):
    m = df["n_tool_calls"] > 0
    return float((df.loc[m, "tool_total_ms"] / df.loc[m, "n_tool_calls"]).mean())


def _annotate(ax, bars, fmt="{:.1f}", dy=0.0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=10, fontweight="bold")


# ---------------------------------------------------------------- Fig A
def fig_ratio(wiki, vdb):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    w = 0.34
    means = [wiki["ratio"].mean() * 100, vdb["ratio"].mean() * 100]
    meds = [wiki["ratio"].median() * 100, vdb["ratio"].median() * 100]
    b1 = ax.bar(x - w / 2, means, w, color=[WIKI, VDB], alpha=0.95, label="mean")
    b2 = ax.bar(x + w / 2, meds, w, color=[WIKI, VDB], alpha=0.5, label="median",
                hatch="//", edgecolor="white")
    _annotate(ax, b1, "{:.1f}%", dy=0.4)
    _annotate(ax, b2, "{:.1f}%", dy=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("decode-RAG ratio = tool / e2e  (%)")
    ax.set_ylim(0, max(means + meds) * 1.25)
    ax.set_title("Fig A — decode-RAG ratio (retrieval-time fraction)")
    # legend that distinguishes mean (solid) vs median (hatched)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#888", label="mean"),
                       Patch(facecolor="#888", alpha=0.5, hatch="//",
                             edgecolor="white", label="median")],
              loc="upper right")
    fig.text(0.5, -0.02,
             "Faster dense retrieval (415 ms vs ~2 s/search) lowers the ratio, "
             "but the backend is reproducible (fixed 2023-11 snapshot).",
             ha="center", fontsize=8, color="#555", wrap=True)
    out = OUT_DIR / "decode_rag_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


# ---------------------------------------------------------------- Fig B
def fig_ratio_dist(wiki, vdb):
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    data = [wiki["ratio"].values * 100, vdb["ratio"].values * 100]
    bp = ax.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", linewidth=2))
    for patch, c in zip(bp["boxes"], [WIKI, VDB]):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
    for i, (vals, c) in enumerate(zip(data, [WIKI, VDB]), start=1):
        jx = rng.normal(i, 0.05, size=len(vals))
        ax.scatter(jx, vals, s=22, color=c, alpha=0.8, edgecolor="white",
                   linewidth=0.5, zorder=3)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("per-query decode-RAG ratio  (%)")
    ax.set_title("Fig B — decode-RAG ratio distribution (per query, n=50 each)")
    ax.set_ylim(0, max(np.max(data[0]), np.max(data[1])) * 1.1)
    out = OUT_DIR / "decode_rag_ratio_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


# ---------------------------------------------------------------- Fig C
def fig_latency_breakdown(wiki, vdb):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(2)
    llm = np.array([wiki["llm_total_ms"].mean(), vdb["llm_total_ms"].mean()]) / 1000
    tool = np.array([wiki["tool_total_ms"].mean(), vdb["tool_total_ms"].mean()]) / 1000
    over = np.array([wiki["overhead_ms"].fillna(0).mean(),
                     vdb["overhead_ms"].fillna(0).mean()]) / 1000
    e2e = np.array([wiki["e2e_ms"].mean(), vdb["e2e_ms"].mean()]) / 1000
    ax.bar(x, llm, 0.5, color="#d62728", label="LLM (prefill+decode)")
    ax.bar(x, tool, 0.5, bottom=llm, color="#000000", label="Tool / retrieval wait")
    ax.bar(x, over, 0.5, bottom=llm + tool, color="#bcbcbc", label="Overhead")
    for i in range(2):
        ax.text(i, e2e[i] + 0.6, f"e2e {e2e[i]:.1f}s", ha="center",
                fontsize=10, fontweight="bold")
        ax.text(i, llm[i] / 2, f"{llm[i]:.1f}s", ha="center", va="center",
                color="white", fontsize=9)
        ax.text(i, llm[i] + tool[i] / 2, f"{tool[i]:.1f}s", ha="center",
                va="center", color="white", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("mean latency  (s)")
    ax.set_ylim(0, max(e2e) * 1.2)
    ax.set_title("Fig C — mean latency breakdown (lower = faster overall)")
    ax.legend(loc="upper right")
    out = OUT_DIR / "latency_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


# ---------------------------------------------------------------- Fig D
def fig_overview(wiki, vdb):
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    axf = axes.ravel()
    # Panel 0 — accuracy: EM (strict, solid) + F1 (partial credit, hatched).
    ax = axf[0]
    x = np.arange(2)
    w = 0.36
    em = [wiki["correct"].mean() * 100, vdb["correct"].mean() * 100]
    f1 = [wiki["f1"].mean() * 100, vdb["f1"].mean() * 100]
    b1 = ax.bar(x - w / 2, em, w, color=[WIKI, VDB], alpha=0.95)
    b2 = ax.bar(x + w / 2, f1, w, color=[WIKI, VDB], alpha=0.45,
                hatch="//", edgecolor="white")
    _annotate(ax, b1, "{:.1f}", dy=0.5)
    _annotate(ax, b2, "{:.1f}", dy=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(SHORT, fontsize=9)
    ax.set_title("accuracy — EM (solid) / F1 (hatched)  %", fontsize=12)
    ax.set_ylim(0, max(em + f1) * 1.28)
    # Panels 1-3 — single-value comparisons.
    panels = [
        ("e2e latency — median (s)",
         [wiki["e2e_ms"].median() / 1000, vdb["e2e_ms"].median() / 1000], "{:.1f}"),
        ("searches / query (mean)",
         [wiki["n_tool_calls"].mean(), vdb["n_tool_calls"].mean()], "{:.2f}"),
        ("time / search (ms)",
         [per_search_ms(wiki), per_search_ms(vdb)], "{:.0f}"),
    ]
    for ax, (title, vals, fmt) in zip(axf[1:], panels):
        bars = ax.bar([0, 1], vals, 0.55, color=[WIKI, VDB], alpha=0.95)
        _annotate(ax, bars, fmt, dy=max(vals) * 0.01)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(SHORT, fontsize=9)
        ax.set_title(title, fontsize=12)
        ax.set_ylim(0, max(vals) * 1.2)
    fig.suptitle("Fig D — baseline overview: Live Wikipedia vs Cohere vector DB "
                 "(ReAct, HotpotQA, 8B Q4)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "baseline_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


def main():
    setup_plot_style()
    wiki, vdb = backend_frames()
    print(f"wiki ReAct rows={len(wiki)}  vdb rows={len(vdb)}")
    print(f"ratio  wiki: mean={wiki['ratio'].mean()*100:.1f}% "
          f"median={wiki['ratio'].median()*100:.1f}%   "
          f"vdb: mean={vdb['ratio'].mean()*100:.1f}% "
          f"median={vdb['ratio'].median()*100:.1f}%")
    print(f"EM     wiki={wiki['correct'].mean()*100:.1f}%  "
          f"vdb={vdb['correct'].mean()*100:.1f}%")
    print(f"F1     wiki={wiki['f1'].mean()*100:.1f}%  "
          f"vdb={vdb['f1'].mean()*100:.1f}%")
    print(f"per-search ms  wiki={per_search_ms(wiki):.0f}  vdb={per_search_ms(vdb):.0f}")
    fig_ratio(wiki, vdb)
    fig_ratio_dist(wiki, vdb)
    fig_latency_breakdown(wiki, vdb)
    fig_overview(wiki, vdb)
    print("done.")


if __name__ == "__main__":
    main()
