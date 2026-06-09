"""Three-way comparison: paper vs 1st experiment (Live Wikipedia ReAct) vs
current experiment (Cohere vector-DB ReAct).

The PAPER column is the original arXiv screenshot, embedded directly by the
HTML report (three_way_compare.html) — this script only generates the two
"ours" columns:
  - 1st  = Live Wikipedia ReAct  (fig13_pareto.jsonl, agent_type=="react")
  - cur  = Cohere vector DB ReAct (react_vectordb.jsonl)
One PNG per backend per metric, with SHARED axes so the two columns compare
fairly (the single most important requirement here).

Metrics are ratio/shape only — absolute seconds are NOT comparable across
A100/vLLM (paper) ↔ M3 Pro/llama.cpp Q4 (ours):
  m1 retrieval/tool fraction (tool/e2e)   — box + jitter (distribution)
  m2 latency composition (LLM/tool/over)  — stacked 100% bar
  m3 calls per query (LLM, tool)          — grouped bar
  m4 EM accuracy                          — single bar
  m5 latency tail (p50/p95)               — histogram + p50/p95 vlines
  m6 prefill/decode split                 — stacked 100% bar (estimated)

Labels are English on purpose (matplotlib Korean glyphs unreliable); the
HTML report supplies the Korean narration.
"""
import numpy as np
import matplotlib.pyplot as plt

from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR
from measurement.eval import hotpotqa_f1

OUT_DIR = FIGURES_DIR / "three_way"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Backend identity colors (match plot_decode_rag_compare.py).
WIKI = "#e0892e"   # amber — 1st / Live Wikipedia
VDB = "#2ca02c"    # green — current / Cohere vector DB
# Component colors (match plot_fig5.py palette).
LLM_C = "#d62728"      # red
TOOL_C = "#000000"     # black
OVER_C = "#bcbcbc"     # gray
PREFILL_C = "#f4a3a3"  # light red
DECODE_C = "#d62728"   # red


def backend_frames():
    """ReAct-only Live-Wikipedia rows + Cohere vector-DB rows, with derived
    latency columns. Mirrors plot_decode_rag_compare.backend_frames()."""
    pareto = load_jsonl("fig13_pareto")
    wiki = pareto[pareto["agent_type"] == "react"].copy()
    # "현재" backend is the rerank run (Cohere VDB + local cross-encoder rerank),
    # scaled past the 50-sample dense baseline. See three_way_compare.html ①~⑥.
    vdb = load_jsonl("react_vectordb_rerank").copy()
    for df in (wiki, vdb):
        df["e2e_ms"] = df["e2e_latency_s"] * 1000.0
        for col in ("tool_total_ms", "llm_total_ms", "overhead_ms",
                    "prefill_total_ms", "decode_total_ms"):
            df[col] = df[col].fillna(0.0)
        df["ratio"] = (df["tool_total_ms"] / df["e2e_ms"]).clip(lower=0.0)
        # F1 computed post-hoc from stored answers (older jsonl has no f1 column).
        df["f1"] = (df["f1"] if "f1" in df.columns else
                    [hotpotqa_f1(p, g) for p, g in
                     zip(df["final_answer"], df["expected_answer"])])
    return wiki, vdb


def _sources(wiki, vdb):
    return [("wiki", wiki, WIKI, "Live Wikipedia"),
            ("vdb", vdb, VDB, "Cohere VDB + rerank")]


def _save(fig, name):
    out = OUT_DIR / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


# ----------------------------------------------------------------- m1
def m1_toolfrac(wiki, vdb):
    """decode-RAG ratio (tool/e2e) distribution — box + jitter, shared y."""
    rng = np.random.default_rng(42)
    wv = wiki["ratio"].values * 100
    vv = vdb["ratio"].values * 100
    ymax = max(wv.max(), vv.max()) * 1.15
    for tag, vals, color, short in [("wiki", wv, WIKI, "Live Wikipedia"),
                                    ("vdb", vv, VDB, "Cohere VDB + rerank")]:
        fig, ax = plt.subplots(figsize=(4.6, 5))
        bp = ax.boxplot([vals], positions=[1], widths=0.5, patch_artist=True,
                        showfliers=False,
                        medianprops=dict(color="black", linewidth=2))
        bp["boxes"][0].set_facecolor(color)
        bp["boxes"][0].set_alpha(0.35)
        jx = rng.normal(1, 0.05, size=len(vals))
        ax.scatter(jx, vals, s=24, color=color, alpha=0.8,
                   edgecolor="white", linewidth=0.5, zorder=3)
        ax.set_ylim(0, ymax)
        ax.set_xticks([1])
        ax.set_xticklabels([short])
        ax.set_ylabel("decode-RAG ratio = tool / e2e  (%)")
        ax.set_title(f"{short}\nmean {vals.mean():.1f}%  ·  median {np.median(vals):.1f}%",
                     fontsize=11)
        _save(fig, f"m1_toolfrac_{tag}.png")


# ----------------------------------------------------------------- m2
def m2_latcomp(wiki, vdb):
    """Latency composition LLM/tool/overhead — stacked 100% bar (y fixed 0-100)."""
    for tag, df, _color, short in _sources(wiki, vdb):
        llm = (df["llm_total_ms"] / df["e2e_ms"] * 100).mean()
        tool = (df["tool_total_ms"] / df["e2e_ms"] * 100).mean()
        over = (df["overhead_ms"] / df["e2e_ms"] * 100).mean()
        fig, ax = plt.subplots(figsize=(4.6, 5))
        b = 0.0
        # Inline labels inside each segment (no legend — it overlaps the bar).
        for val, c, lab in [(llm, LLM_C, "LLM"),
                            (tool, TOOL_C, "Tool"),
                            (over, OVER_C, "Overhead")]:
            ax.bar(0, val, 0.6, bottom=b, color=c)
            if val >= 8:
                ax.text(0, b + val / 2, f"{lab}\n{val:.0f}%", ha="center",
                        va="center", color="white", fontsize=10, fontweight="bold")
            elif val >= 3.5:
                ax.text(0, b + val / 2, f"{val:.0f}%", ha="center", va="center",
                        color="white", fontsize=9, fontweight="bold")
            b += val
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([0])
        ax.set_xticklabels([short])
        ax.set_ylabel("share of e2e latency  (%)")
        ax.set_title(f"{short}\nlatency composition", fontsize=11)
        _save(fig, f"m2_latcomp_{tag}.png")


# ----------------------------------------------------------------- m3
def m3_calls(wiki, vdb):
    """Calls per query (LLM vs tool) — grouped bar, shared y."""
    wl, wt = wiki["n_llm_calls"].mean(), wiki["n_tool_calls"].mean()
    vl, vt = vdb["n_llm_calls"].mean(), vdb["n_tool_calls"].mean()
    ymax = max(wl, wt, vl, vt) * 1.28
    for tag, lc, tc, color, short in [("wiki", wl, wt, WIKI, "Live Wikipedia"),
                                      ("vdb", vl, vt, VDB, "Cohere VDB + rerank")]:
        fig, ax = plt.subplots(figsize=(4.6, 5))
        bars = ax.bar([0, 1], [lc, tc], 0.6, color=color, alpha=0.95)
        bars[1].set_alpha(0.5)
        bars[1].set_hatch("//")
        for x, v in zip([0, 1], [lc, tc]):
            ax.text(x, v + ymax * 0.02, f"{v:.2f}", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_ylim(0, ymax)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["LLM calls", "tool calls"])
        ax.set_ylabel("calls per query (mean)")
        ax.set_title(f"{short}\ncalls / query", fontsize=11)
        _save(fig, f"m3_calls_{tag}.png")


# ----------------------------------------------------------------- m4
def m4_em(wiki, vdb):
    """Accuracy — EM (strict) + F1 (partial credit), grouped bar, y fixed 0-100."""
    for tag, df, color, short in _sources(wiki, vdb):
        em = df["correct"].mean() * 100
        f1 = df["f1"].mean() * 100
        emn = int(df["correct"].sum())
        tot = len(df)
        fig, ax = plt.subplots(figsize=(4.6, 5))
        bars = ax.bar([0, 1], [em, f1], 0.55, color=color, alpha=0.95)
        bars[1].set_alpha(0.55)
        bars[1].set_hatch("//")
        ax.text(0, em + 1.5, f"{em:.1f}%\n({emn}/{tot})", ha="center",
                fontsize=10, fontweight="bold")
        ax.text(1, f1 + 1.5, f"{f1:.1f}%", ha="center", fontsize=10, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, 1.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["EM (strict)", "F1 (partial)"])
        ax.set_ylabel("accuracy  (%)")
        ax.set_title(f"{short}\nEM vs F1 accuracy", fontsize=11)
        _save(fig, f"m4_em_{tag}.png")


# ----------------------------------------------------------------- m5
def m5_lattail(wiki, vdb):
    """e2e latency distribution — histogram + p50/p95 vlines, shared x & y."""
    wl = wiki["e2e_latency_s"].values
    vl = vdb["e2e_latency_s"].values
    xmax = max(wl.max(), vl.max()) * 1.03
    bins = np.linspace(0, xmax, 22)
    wd, _ = np.histogram(wl, bins=bins, density=True)
    vd, _ = np.histogram(vl, bins=bins, density=True)
    ymax = max(wd.max(), vd.max()) * 1.18
    for tag, vals, color, short in [("wiki", wl, WIKI, "Live Wikipedia"),
                                    ("vdb", vl, VDB, "Cohere VDB + rerank")]:
        fig, ax = plt.subplots(figsize=(4.6, 5))
        ax.hist(vals, bins=bins, density=True, color=color, alpha=0.55)
        p50 = np.percentile(vals, 50)
        p95 = np.percentile(vals, 95)
        ax.axvline(p50, ls="--", color="black", alpha=0.8, label=f"p50 = {p50:.1f}s")
        ax.axvline(p95, ls="--", color="red", alpha=0.8, label=f"p95 = {p95:.1f}s")
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, ymax)
        ax.set_xlabel("end-to-end latency (s)")
        ax.set_ylabel("frequency density")
        ax.set_title(f"{short}\np95/p50 = {p95 / p50:.2f}", fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
        _save(fig, f"m5_lattail_{tag}.png")


# ----------------------------------------------------------------- m6
def m6_prefdec(wiki, vdb):
    """Prefill/decode split — stacked 100% bar (estimated), y fixed 0-100."""
    for tag, df, _color, short in _sources(wiki, vdb):
        denom = df["prefill_total_ms"] + df["decode_total_ms"]
        mask = denom > 0
        pre = (df.loc[mask, "prefill_total_ms"] / denom[mask] * 100).mean()
        dec = (df.loc[mask, "decode_total_ms"] / denom[mask] * 100).mean()
        fig, ax = plt.subplots(figsize=(4.6, 5))
        b = 0.0
        # Inline labels inside each segment (no legend — it overlaps the bar).
        for val, c, lab, txt in [(pre, PREFILL_C, "prefill", "#5a1a1a"),
                                 (dec, DECODE_C, "decode", "white")]:
            ax.bar(0, val, 0.6, bottom=b, color=c)
            ax.text(0, b + val / 2, f"{lab}\n{val:.0f}%", ha="center", va="center",
                    color=txt, fontsize=10, fontweight="bold")
            b += val
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([0])
        ax.set_xticklabels([short])
        ax.set_ylabel("share of active LLM time  (%)")
        ax.set_title(f"{short}\nprefill / decode (estimated)", fontsize=11)
        _save(fig, f"m6_prefdec_{tag}.png")


def main():
    setup_plot_style()
    wiki, vdb = backend_frames()
    assert len(wiki) == 50, f"expected 50 wiki ReAct rows, got {len(wiki)}"
    assert len(vdb) >= 50, f"expected >=50 vdb(rerank) rows, got {len(vdb)}"
    print(f"wiki ReAct rows={len(wiki)}  vdb rows={len(vdb)}")
    # Regression-check prints (must match progress_report.html PART 3).
    print(f"m1 tool fraction  wiki mean={wiki['ratio'].mean() * 100:.1f}% "
          f"median={wiki['ratio'].median() * 100:.1f}%   "
          f"vdb mean={vdb['ratio'].mean() * 100:.1f}% "
          f"median={vdb['ratio'].median() * 100:.1f}%")
    print(f"m4 EM  wiki={wiki['correct'].mean() * 100:.1f}%  "
          f"vdb={vdb['correct'].mean() * 100:.1f}%")
    print(f"m4 F1  wiki={wiki['f1'].mean() * 100:.1f}%  "
          f"vdb={vdb['f1'].mean() * 100:.1f}%")
    m1_toolfrac(wiki, vdb)
    m2_latcomp(wiki, vdb)
    m3_calls(wiki, vdb)
    m4_em(wiki, vdb)
    m5_lattail(wiki, vdb)
    m6_prefdec(wiki, vdb)
    print("done.")


if __name__ == "__main__":
    main()
