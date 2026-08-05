from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from . import (
    independent_run,
    parallel,
    parallel_scheduled,
    prefetched_toolcall,
    tooluse,
    vectordb,
)
from .common import (
    DEFAULT_MODEL_KEYS,
    MODEL_SPECS,
    ExperimentError,
    ModelSpec,
    run_main,
)


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    description: str
    prompt_and_run: Callable[..., None]
    uses_models: bool = True


EXPERIMENTS = (
    Experiment(
        "parallel", "simultaneous main/draft inference", parallel.prompt_and_run
    ),
    Experiment(
        "parallel-scheduled",
        "single-process time-sliced main/draft inference",
        parallel_scheduled.prompt_and_run,
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
        "vectordb",
        "Qdrant cache hit and non-hit latency",
        vectordb.prompt_and_run,
        uses_models=False,
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


def choose_model(
    role: str,
    *,
    input_fn: Callable[[str], str] = input,
) -> ModelSpec:
    label = "target" if role == "main" else role
    specs = tuple(MODEL_SPECS.values())
    default_key = DEFAULT_MODEL_KEYS[role]
    print(f"Available {label} models:")
    for index, spec in enumerate(specs, start=1):
        suffix = " (default)" if spec.key == default_key else ""
        print(f"  {index}. {spec.label} [{spec.key}]{suffix}")
    while True:
        try:
            answer = input_fn(f"Select {label} model by number or key: ").strip()
        except EOFError:
            raise ExperimentError("model selection requires interactive input") from None
        if not answer:
            return MODEL_SPECS[default_key]
        if answer.isdigit() and 1 <= int(answer) <= len(specs):
            return specs[int(answer) - 1]
        if answer in MODEL_SPECS:
            return MODEL_SPECS[answer]
        print("Invalid model selection.")


def choose_model_specs(
    *, input_fn: Callable[[str], str] = input
) -> Mapping[str, ModelSpec]:
    return {
        role: choose_model(role, input_fn=input_fn) for role in ("main", "draft")
    }


def run_experiment(*, input_fn: Callable[[str], str] = input) -> None:
    if len(sys.argv) > 1:
        raise ExperimentError(
            "run-experiment is interactive and accepts no arguments; select an experiment "
            "and enter counts at its prompts"
        )
    selected = choose_experiment(input_fn=input_fn)
    if selected.uses_models:
        model_specs = choose_model_specs(input_fn=input_fn)
        selected.prompt_and_run(input_fn=input_fn, model_specs=model_specs)
    else:
        selected.prompt_and_run(input_fn=input_fn)


def main() -> None:
    run_main(run_experiment)
