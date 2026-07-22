from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

from . import orchestrate

SETUP_DIR = Path(__file__).resolve().parent


def available_experiments(setup_dir: Path = SETUP_DIR) -> list[Path]:
    return sorted(setup_dir.glob("*.toml"))


def choose_experiment(
    experiments: list[Path],
    *,
    input_fn: Callable[[str], str] = input,
) -> Path:
    if not experiments:
        raise SystemExit(f"No experiment TOML files found in {SETUP_DIR}")

    print("Available experiments:")
    for index, path in enumerate(experiments, start=1):
        description = orchestrate.load_config(path).get("description", "")
        suffix = f" — {description}" if description else ""
        print(f"  {index}. {path.stem}{suffix}")

    while True:
        try:
            answer = input_fn("Select experiment by number or name: ").strip()
        except EOFError:
            raise SystemExit("Experiment selection requires interactive input") from None
        if answer.isdigit():
            index = int(answer)
            if 1 <= index <= len(experiments):
                return experiments[index - 1]
        for path in experiments:
            if answer == path.stem:
                return path
        print("Invalid experiment selection.")


def download_models() -> None:
    config = orchestrate.load_config()
    models_dir = orchestrate.ROOT / "experiment" / "models"
    for role in ("main", "draft"):
        path, timing = orchestrate.resolve_model(role, config["models"][role], models_dir)
        print(f"{role}: {path} ({timing['duration_ms']:.0f} ms)")


def run_experiment() -> None:
    experiment = choose_experiment(available_experiments())
    orchestrate.main(["--config", str(experiment), *sys.argv[1:]])
