"""Fig 15: few-shot count sweep (accuracy & latency) for ReAct."""
import matplotlib.pyplot as plt
from analysis.shared import load_jsonl, setup_plot_style, FIGURES_DIR

def main():
    setup_plot_style()
    df = load_jsonl("fig15_fewshot_sweep")
    agg = df.groupby("fewshot").agg(
        mean_lat=("e2e_latency_s", "mean"),
        accuracy=("correct", "mean"),
    ).sort_index()

    fig, ax1 = plt.subplots()
    ax1.plot(agg.index, agg["mean_lat"], "o-", color="black", label="mean latency")
    ax1.set_xlabel("Few-shot count")
    ax1.set_ylabel("Latency (s)")
    ax2 = ax1.twinx()
    ax2.plot(agg.index, 100*agg["accuracy"], "D-", color="red", label="accuracy")
    ax2.set_ylabel("Accuracy (%)")
    ax1.set_title("Fig 15: ReAct few-shot sweep on HotpotQA")
    fig.legend(loc="upper right")
    out = FIGURES_DIR / "fig15_fewshot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    # Verification: accuracy is non-monotone OR plateaus by fewshot=5
    acc = agg["accuracy"].values
    n = len(acc)
    monotone_increasing = all(acc[i] <= acc[i+1] for i in range(n-1))
    plateau_in_upper_half = (acc[-1] - acc[n//2]) < 0.005 * (n - n//2)   # < 0.5%-pt per step
    non_monotone = (acc[-1] < max(acc))
    print(f"accuracies: {acc}")
    print(f"monotone increasing: {monotone_increasing}")
    print(f"non-monotone (max not at last): {non_monotone}")
    print(f"plateau in upper half: {plateau_in_upper_half}")
    assert non_monotone or plateau_in_upper_half, "FAIL: accuracy strictly increases"
    print("Fig 15 verification: PASS")

if __name__ == "__main__":
    main()
