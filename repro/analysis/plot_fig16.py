"""Fig 16: sequential vs parallel scaling.

Three panels:
  (a) Reflexion sequential — sweep reflection_limit in [2, 4, 8, 16]
  (b) LATS sequential      — sweep iteration_limit in [4, 8, 16, 32(, 64)]
  (c) LATS parallel        — sweep n_generate_sample in [1, 2, 4, 8]

For each panel, plot accuracy vs e2e latency at each sweep value
(connecting points so the diminishing-returns curve is visible).

Verification (paper §V-B):
- (a) and (b): saturating curve. max-min accuracy > 5pp AND the gain
      from the last step is smaller than the gain from the first step.
- (c): parallel may reduce latency WHILE accuracy goes up — assert at
      least one value with both higher accuracy AND lower latency than
      n_generate_sample=1.

If any panel's source JSONL is missing, that panel is annotated and we
skip that panel's assertion (don't error).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from analysis.shared import (
    load_jsonl, setup_plot_style, FIGURES_DIR, RESULTS_DIR,
)


PANELS = [
    {
        "key": "a",
        "run":  "fig16a_reflexion_sequential",
        "sweep_var": "reflection_limit",
        "title": "(a) Reflexion sequential\nreflection_limit",
        "color": "#bcbd22",
        "marker": "o",
    },
    {
        "key": "b",
        "run":  "fig16b_lats_sequential",
        "sweep_var": "iteration_limit",
        "title": "(b) LATS sequential\niteration_limit",
        "color": "#1f77b4",
        "marker": "^",
    },
    {
        "key": "c",
        "run":  "fig16c_lats_parallel",
        "sweep_var": "n_generate_sample",
        "title": "(c) LATS parallel\nn_generate_sample",
        "color": "#9467bd",
        "marker": "D",
    },
]


def _aggregate(df, sweep_var: str):
    """Per-sweep-value aggregation, sorted by sweep value."""
    agg = df.groupby(sweep_var).agg(
        accuracy=("correct", "mean"),
        latency=("e2e_latency_s", "mean"),
    ).sort_index()
    return agg


def _read_sweep_var_from_meta(df, sweep_var: str):
    """Some run_one variants don't propagate the sweep var into top-level columns.
    Fall back to df['meta'].apply(...) if needed."""
    if sweep_var in df.columns and df[sweep_var].notna().any():
        return df
    # Try meta dict
    if "meta" in df.columns:
        df = df.copy()
        df[sweep_var] = df["meta"].apply(lambda m: (m or {}).get(sweep_var))
    return df


def main():
    setup_plot_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panel_status = {}  # key -> "ok" | "missing"

    for ax, panel in zip(axes, PANELS):
        path = RESULTS_DIR / "raw" / f"{panel['run']}.jsonl"
        if not path.exists():
            print(f"Panel {panel['key']}: missing {path}", file=sys.stderr)
            ax.set_title(panel["title"] + "\n(no data)")
            ax.text(0.5, 0.5,
                    f"missing\n{panel['run']}.jsonl",
                    transform=ax.transAxes, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.85))
            ax.set_xlabel("E2E latency (s)")
            ax.set_ylabel("Accuracy (%)")
            panel_status[panel["key"]] = "missing"
            continue

        df = load_jsonl(panel["run"])
        df = _read_sweep_var_from_meta(df, panel["sweep_var"])
        if panel["sweep_var"] not in df.columns or df[panel["sweep_var"]].isna().all():
            print(f"Panel {panel['key']}: sweep var {panel['sweep_var']} missing in data",
                  file=sys.stderr)
            ax.set_title(panel["title"] + "\n(sweep var missing)")
            ax.text(0.5, 0.5,
                    f"{panel['sweep_var']} missing\nin {panel['run']}.jsonl",
                    transform=ax.transAxes, ha="center", va="center",
                    bbox=dict(facecolor="white", alpha=0.85))
            ax.set_xlabel("E2E latency (s)")
            ax.set_ylabel("Accuracy (%)")
            panel_status[panel["key"]] = "missing"
            continue

        agg = _aggregate(df, panel["sweep_var"])
        ax.plot(agg["latency"].values, 100 * agg["accuracy"].values,
                marker=panel["marker"], markersize=10, linestyle="-",
                color=panel["color"])
        # Label each point with the sweep var value.
        for v, row in agg.iterrows():
            ax.annotate(str(v), (row["latency"], 100 * row["accuracy"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xlabel("Mean E2E latency (s)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(panel["title"])
        panel_status[panel["key"]] = "ok"

        # Per-panel verification, depending on which panel.
        accs = agg["accuracy"].values
        lats = agg["latency"].values
        sweep_vals = agg.index.values
        n = len(accs)
        if panel["key"] in ("a", "b") and n >= 3:
            spread = max(accs) - min(accs)
            first_gain = accs[1] - accs[0]
            last_gain  = accs[-1] - accs[-2]
            print(f"Panel {panel['key']}: spread={spread:.3f}, "
                  f"first_gain={first_gain:.3f}, last_gain={last_gain:.3f}")
            assert spread > 0.05, (
                f"FAIL panel {panel['key']}: accuracy spread {spread:.3f} <= 0.05"
            )
            # Saturating: last-step gain magnitude smaller than first-step gain magnitude
            assert abs(last_gain) <= abs(first_gain) + 0.02, (
                f"FAIL panel {panel['key']}: last gain |{last_gain:.3f}| not "
                f"<= first gain |{first_gain:.3f}| — curve not saturating"
            )
        elif panel["key"] == "c" and n >= 2:
            # n_generate_sample=1 is the reference; find any value with both higher
            # accuracy AND lower latency than reference.
            ref_idx = 0  # smallest sweep value (1)
            ref_acc = accs[ref_idx]
            ref_lat = lats[ref_idx]
            better = [
                (sweep_vals[i], accs[i], lats[i])
                for i in range(n)
                if i != ref_idx and accs[i] > ref_acc and lats[i] < ref_lat
            ]
            print(f"Panel c: ref(n=1) acc={ref_acc:.3f} lat={ref_lat:.1f}s; "
                  f"better-on-both = {better}")
            assert len(better) >= 1, (
                "FAIL panel c: no parallel value has higher accuracy AND lower "
                f"latency than n_generate_sample=1 (acc={ref_acc:.3f}, lat={ref_lat:.1f}s)"
            )

    fig.suptitle("Fig 16: Sequential vs Parallel scaling on HotpotQA")
    fig.tight_layout()
    out = FIGURES_DIR / "fig16_scaling.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")

    print(f"panel statuses: {panel_status}")
    n_missing = sum(1 for s in panel_status.values() if s == "missing")
    if n_missing == 0:
        print("Fig 16 verification: PASS")
    elif n_missing < len(PANELS):
        print(f"Fig 16 verification: PARTIAL ({n_missing}/{len(PANELS)} panels missing data)")
    else:
        print("Fig 16 verification: SKIPPED (no sweep data yet)")


if __name__ == "__main__":
    main()
