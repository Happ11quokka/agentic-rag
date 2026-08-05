from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass

from . import independent_run, parallel, prefetched_toolcall, tooluse, vectordb
from .common import ExperimentError, run_main


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    description: str
    prompt_and_run: Callable[..., None]


EXPERIMENTS = (
    Experiment(
        "parallel", "simultaneous main/draft inference", parallel.prompt_and_run
    ),
    Experiment(
        "independent-run",
        "isolated main/draft inference",
        independent_run.prompt_and_run,
    ),
    Experiment(
        "tooluse", "tool-use query tokens and latency", tooluse.prompt_and_run
    ),
    Experiment(
        "prefetched-toolcall",
        "synchronized draft query prefetch latency",
        prefetched_toolcall.prompt_and_run,
    ),
    Experiment(
        "vectordb", "Qdrant cache hit and non-hit latency", vectordb.prompt_and_run
    ),
)


def choose_experiment(*, input_fn: Callable[[str], str] = input) -> Experiment:
    print("Available experiments:")
    for index, experiment in enumerate(EXPERIMENTS, start=1):
        print(f"  {index}. {experiment.name} — {experiment.description}")
    while True:
        try:
            answer = input_fn("Select experiment by number or name: ").strip()
        except EOFError:
            raise ExperimentError(
                "experiment selection requires interactive input"
            ) from None
        if answer.isdigit() and 1 <= int(answer) <= len(EXPERIMENTS):
            return EXPERIMENTS[int(answer) - 1]
        for experiment in EXPERIMENTS:
            if answer == experiment.name:
                return experiment
        print("Invalid experiment selection.")


def run_experiment(*, input_fn: Callable[[str], str] = input) -> None:
    if len(sys.argv) > 1:
        raise ExperimentError(
            "run-experiment is interactive and accepts no arguments; select an experiment "
            "and enter counts at its prompts"
        )
    selected = choose_experiment(input_fn=input_fn)
    selected.prompt_and_run(input_fn=input_fn)


def main() -> None:
    run_main(run_experiment)
