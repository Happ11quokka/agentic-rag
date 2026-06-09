"""Per-hop retrieval vs decode balance — current Cohere vector-DB backend.

For decode-RAG, what matters per hop is retrieval_ms vs the LLM's working time
that a prefetch/overlap scheme could hide it behind. This script pairs each
search with the LLM call that immediately follows it and measures the balance,
purely from existing traces (react_vectordb.jsonl) — no sweep rerun.

"decode" is measured TWO ways so the comparison is honest about what can be
overlapped:
  - LLM total  = prefill + decode wall-clock (t_end - t_start), reliable for every hop
  - decode only = decode_ms_estimate, used only for hops with coarse_attribution=False

Outputs to results/figures/hop_balance/:
  hop_scatter.png     — retrieval vs decode per hop, with the 1:1 reference line
  hop_ratio_hist.png  — distribution of per-hop retrieval/decode ratio (target = 1.0)

Labels are English (matplotlib Korean glyphs unreliable); the HTML report narrates.
"""
import json

import numpy as np
import matplotlib.pyplot as plt

from analysis.shared import RESULTS_DIR, setup_plot_style, FIGURES_DIR

OUT_DIR = FIGURES_DIR / "hop_balance"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VDB = "#2ca02c"    # current backend (green)
RET = "#1f77b4"    # retrieval/LLM-total ratio (blue)
DEC = "#d62728"    # retrieval/decode-only ratio (red)

# Tolerance (seconds) for "the LLM call that starts right after this search".
PAIR_TOL_S = 0.05


def load_hops(name: str = "react_vectordb"):
    """Pair each search with the immediately-following LLM call.

    Returns a list of dicts: retrieval_ms, llm_total_ms, decode_ms (or None when
    the decode estimate is unreliable / coarse_attribution).
    """
    path = RESULTS_DIR / "raw" / f"{name}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    hops = []
    for r in rows:
        llms = sorted(r.get("llm_calls", []), key=lambda c: c["t_start"])
        tools = sorted(r.get("tool_calls", []), key=lambda c: c["t_start"])
        for t in tools:
            if t.get("tool_name") and t["tool_name"] != "search":
                continue
            ret_ms = (t["t_end"] - t["t_start"]) * 1000.0
            nxt = next((c for c in llms if c["t_start"] >= t["t_end"] - PAIR_TOL_S), None)
            if nxt is None:
                continue
            llm_total = (nxt["t_end"] - nxt["t_start"]) * 1000.0
            dec = nxt.get("decode_ms_estimate", 0.0)
            coarse = nxt.get("coarse_attribution", False)
            hops.append({
                "retrieval_ms": ret_ms,
                "llm_total_ms": llm_total,
                "decode_ms": dec if (not coarse and dec > 0) else None,
            })
    return hops


def fig_scatter(hops):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    specs = [("llm_total_ms", "LLM total (prefill+decode)", axes[0]),
             ("decode_ms", "decode only (estimate)", axes[1])]
    for key, label, ax in specs:
        pairs = [(h[key], h["retrieval_ms"]) for h in hops if h[key]]
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        ax.scatter(xs, ys, s=28, color=VDB, alpha=0.6,
                   edgecolor="white", linewidth=0.5, zorder=3)
        m = max(max(xs), max(ys)) * 1.05
        ax.plot([0, m], [0, m], ls="--", color="black", alpha=0.6,
                label="1:1 (retrieval = decode)")
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        ax.set_xlabel(f"{label}  (ms)")
        ax.set_ylabel("retrieval  (ms)")
        med = np.median([y / x for x, y in pairs if x > 0])
        ax.set_title(f"vs {label}\nmedian ratio {med:.2f}   (n={len(xs)} hops)",
                     fontsize=11)
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Per-hop balance: retrieval vs decode  (current Cohere vector DB)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "hop_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def fig_hist(hops):
    fig, ax = plt.subplots(figsize=(8, 5))
    r_llm = [h["retrieval_ms"] / h["llm_total_ms"] for h in hops if h["llm_total_ms"] > 0]
    r_dec = [h["retrieval_ms"] / h["decode_ms"] for h in hops if h["decode_ms"]]
    hi = max(max(r_llm), max(r_dec)) * 1.05
    bins = np.linspace(0, hi, 30)
    ax.hist(r_llm, bins=bins, alpha=0.55, color=RET,
            label=f"retrieval / LLM-total  (median {np.median(r_llm):.2f})")
    ax.hist(r_dec, bins=bins, alpha=0.55, color=DEC,
            label=f"retrieval / decode-only  (median {np.median(r_dec):.2f})")
    ax.axvline(1.0, ls="--", color="black", alpha=0.8, label="1:1 target")
    ax.set_xlabel("per-hop ratio  (retrieval / decode)")
    ax.set_ylabel("hop count")
    ax.set_title("Per-hop retrieval/decode ratio distribution  (target = 1.0)",
                 fontsize=12)
    ax.legend(fontsize=9)
    out = OUT_DIR / "hop_ratio_hist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def fig_compare(before, after):
    """rerank before/after: per-hop retrieval/decode ratio distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    specs = [(axes[0], "decode_ms", "retrieval / decode-only"),
             (axes[1], "llm_total_ms", "retrieval / LLM-total")]
    for ax, key, label in specs:
        rb = [h["retrieval_ms"] / h[key] for h in before if h[key]]
        ra = [h["retrieval_ms"] / h[key] for h in after if h[key]]
        hi = min(max(max(rb), max(ra)) * 1.05, 3.0)
        bins = np.linspace(0, hi, 28)
        ax.hist(rb, bins=bins, alpha=0.55, color="#888888",
                label=f"before rerank (med {np.median(rb):.2f})")
        ax.hist(ra, bins=bins, alpha=0.6, color="#2ca02c",
                label=f"after rerank (med {np.median(ra):.2f})")
        ax.axvline(1.0, ls="--", color="black", alpha=0.8, label="1:1 target")
        ax.set_xlabel(f"{label} ratio")
        ax.set_ylabel("hop count")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8.5)
    fig.suptitle("Per-hop retrieval / decode ratio — before vs after rerank",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = OUT_DIR / "hop_compare.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


def main():
    setup_plot_style()
    hops = load_hops()
    n = len(hops)
    nd = sum(1 for h in hops if h["decode_ms"])
    ret = np.array([h["retrieval_ms"] for h in hops])
    llm = np.array([h["llm_total_ms"] for h in hops])
    dec = np.array([h["decode_ms"] for h in hops if h["decode_ms"]])
    r_llm = ret / llm
    r_dec = np.array([h["retrieval_ms"] / h["decode_ms"] for h in hops if h["decode_ms"]])
    print(f"hops paired: {n}  (decode-estimate reliable: {nd})")
    print(f"retrieval ms     median={np.median(ret):.0f}")
    print(f"LLM-total ms     median={np.median(llm):.0f}")
    print(f"decode-only ms   median={np.median(dec):.0f}")
    print(f"ratio retrieval/LLM-total   median={np.median(r_llm):.2f}  "
          f"-> need ~{1/np.median(r_llm):.1f}x retrieval for 1:1")
    print(f"ratio retrieval/decode-only median={np.median(r_dec):.2f}  "
          f"-> need ~{1/np.median(r_dec):.1f}x retrieval for 1:1")
    fig_scatter(hops)
    fig_hist(hops)
    # rerank before/after comparison
    after = load_hops("react_vectordb_rerank")
    rd_b = np.median([h["retrieval_ms"] / h["decode_ms"] for h in hops if h["decode_ms"]])
    rd_a = np.median([h["retrieval_ms"] / h["decode_ms"] for h in after if h["decode_ms"]])
    print(f"rerank: hop retrieval/decode-only median {rd_b:.2f} -> {rd_a:.2f}  (target 1.0)")
    fig_compare(hops, after)
    print("done.")


if __name__ == "__main__":
    main()
