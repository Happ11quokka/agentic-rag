"""Common helpers for figure generation: load JSONL, plotting style."""
import json
from pathlib import Path

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/Users/imdonghyeon/agentic_rag/repro/results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(name: str) -> pd.DataFrame:
    """Load one of fig13_pareto / fig14_iteration_sweep / fig15_fewshot_sweep."""
    path = RESULTS_DIR / "raw" / f"{name}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return pd.DataFrame(rows)


def setup_plot_style() -> None:
    mpl.rcParams.update({
        "figure.figsize": (8, 5),
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
    })


AGENT_COLORS = {
    "react": "#d62728",       # red
    "reflexion": "#bcbd22",   # olive
    "lats": "#1f77b4",        # blue
    "llmcompiler": "#7f7f7f", # gray
}
AGENT_MARKERS = {
    "react": "s", "reflexion": "o", "lats": "^", "llmcompiler": "D",
}
