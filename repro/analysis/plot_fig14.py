"""Fig 14: iteration budget sweep (mean & p95 latency, accuracy) for ReAct."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

def main():
    setup_plot_style()
    df = load_jsonl("fig14_iteration_sweep")
    agg = df.groupby("iteration_limit").agg(
        mean_lat=("e2e_latency_s", "mean"),
        p95_lat=("e2e_latency_s", lambda x: np.percentile(x, 95)),
        accuracy=("correct", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots()
    ax1.plot(agg.index, agg["mean_lat"], "o-", color="black", label="mean latency")
    ax1.plot(agg.index, agg["p95_lat"], "s--", color="gray", label="p95 latency")
    ax1.set_xlabel("Iteration budget")
    ax1.set_ylabel("Latency (s)")
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(agg.index, 100*agg["accuracy"], "D-", color="red", label="accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="upper right")
    ax1.set_title("Fig 14: ReAct iteration budget sweep on HotpotQA")
    out = FIGURES_DIR / "fig14_iteration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification: mean plateaus, p95 monotone Spearman ρ ≥ 0.8
    vals = sorted(agg.index)
    mean_lat = agg["mean_lat"].values
    p95_lat = agg["p95_lat"].values
    n = len(vals)
    lower_slope = (mean_lat[n//2] - mean_lat[0]) / (n//2) if n//2 else 0
    upper_slope = (mean_lat[-1] - mean_lat[n//2]) / max(n - n//2 - 1, 1)
    print(f"mean latency lower-half slope: {lower_slope:.2f}, upper-half slope: {upper_slope:.2f}")
    assert upper_slope < 0.25 * abs(lower_slope) + 0.5, "FAIL: mean latency did not plateau"

    rho, _ = spearmanr(vals, p95_lat)
    print(f"Spearman ρ(iter, p95_lat) = {rho:.3f} (must be ≥ 0.8)")
    assert rho >= 0.8, f"FAIL: p95 not monotone"
    print("Fig 14 verification: PASS")

if __name__ == "__main__":
    main()
