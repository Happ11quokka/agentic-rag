"""Fig 9: prefix-caching effect on prefill latency.

Side-by-side bars comparing per-agent mean prefill_total_ms between two
sweep runs: one with llama-server --cache-reuse OFF (default) and one ON.

Verification (paper §IV-B): mean prefill latency drops with caching on.
The paper reports ~60% reduction; we require cache_on < 0.7 * cache_off
to allow for measurement noise.

If the cache-on JSONL doesn't exist yet, the script prints an informative
message and exits cleanly (so the analysis pipeline doesn't bail).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR, RESULTS_DIR, AGENT_COLORS


CACHE_OFF_RUN = "fig13_pareto"
CACHE_ON_RUN = "fig13_pareto_cache_on"


def main():
    setup_plot_style()

    off_path = RESULTS_DIR / "raw" / f"{CACHE_OFF_RUN}.jsonl"
    on_path  = RESULTS_DIR / "raw" / f"{CACHE_ON_RUN}.jsonl"

    if not on_path.exists():
        print(f"Fig 9: cache-on JSONL not found at {on_path}", file=sys.stderr)
        print("       run `sweep/run_full.sh fig13_pareto_cache_on.yaml cache_on` first",
              file=sys.stderr)
        print("Fig 9 verification: SKIPPED (no cache_on data)")
        return

    if not off_path.exists():
        print(f"Fig 9: cache-off JSONL not found at {off_path}", file=sys.stderr)
        print("       run `sweep/run_full.sh fig13_pareto.yaml` first", file=sys.stderr)
        print("Fig 9 verification: SKIPPED (no cache_off data)")
        return

    df_off = load_jsonl(CACHE_OFF_RUN)
    df_on  = load_jsonl(CACHE_ON_RUN)

    agg_off = df_off.groupby("agent_type")["prefill_total_ms"].mean()
    agg_on  = df_on.groupby("agent_type")["prefill_total_ms"].mean()

    order = ["react", "reflexion", "llmcompiler", "lats"]
    agents = [a for a in order if a in agg_off.index and a in agg_on.index]
    off_vals = np.array([agg_off[a] for a in agents])
    on_vals  = np.array([agg_on[a]  for a in agents])

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(agents))
    w = 0.38
    ax.bar(x - w/2, off_vals, width=w, color="#d62728", label="cache OFF")
    ax.bar(x + w/2, on_vals,  width=w, color="#1f77b4", label="cache ON (--cache-reuse 256)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_ylabel("Mean prefill total (ms)")
    ax.set_title("Fig 9: Prefix-caching effect on prefill latency (HotpotQA)")
    ax.legend()

    # Annotate the % reduction over each pair.
    for i, agent in enumerate(agents):
        if off_vals[i] > 0:
            reduction = (1.0 - on_vals[i] / off_vals[i]) * 100.0
            ax.text(i, max(off_vals[i], on_vals[i]) * 1.02,
                    f"{reduction:+.0f}%", ha="center", va="bottom", fontsize=9)

    out = FIGURES_DIR / "fig9_prefix_cache.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # ── Verification (paper §IV-B Figure 9) ────────────────────────────────
    # Per-agent: cache_on < 0.7 * cache_off (allow noise; paper claim ~60%).
    overall_off = float(np.mean(off_vals)) if len(off_vals) else 0.0
    overall_on  = float(np.mean(on_vals))  if len(on_vals)  else 0.0
    print(f"overall mean prefill: OFF={overall_off:.1f}ms  ON={overall_on:.1f}ms")
    assert overall_off > 0, "FAIL: cache-off prefill is zero — no data?"
    assert overall_on < 0.7 * overall_off, (
        f"FAIL: cache-on prefill {overall_on:.1f}ms is not < 0.7 * "
        f"cache-off {overall_off:.1f}ms (= {0.7 * overall_off:.1f}ms)"
    )
    print("Fig 9 verification: PASS")


if __name__ == "__main__":
    main()
