"""Fig 7: e2e latency distribution for ReAct on HotpotQA."""
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

def main():
    setup_plot_style()
    df = load_jsonl("fig13_pareto")
    react_lat = df[df.agent_type == "react"]["e2e_latency_s"].values

    fig, ax = plt.subplots()
    ax.hist(react_lat, bins=20, density=True, color="#d62728", alpha=0.8)
    p50 = np.percentile(react_lat, 50)
    p95 = np.percentile(react_lat, 95)
    ax.axvline(p50, ls="--", color="black", label=f"p50 = {p50:.1f}s")
    ax.axvline(p95, ls="--", color="red", label=f"p95 = {p95:.1f}s")
    ax.set_xlabel("End-to-end latency (s)")
    ax.set_ylabel("Frequency density")
    ax.set_title("Fig 7: HotpotQA ReAct latency distribution")
    ax.legend()
    out = FIGURES_DIR / "fig7_latency_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    ratio = p95 / p50 if p50 else 0
    print(f"p95/p50 = {ratio:.2f} (must be ≥ 2.0)")
    assert ratio >= 2.0, f"FAIL: ratio {ratio:.2f} < 2.0"
    print("Fig 7 verification: PASS")

if __name__ == "__main__":
    main()
