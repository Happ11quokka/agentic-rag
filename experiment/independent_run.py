from __future__ import annotations

from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from typing import Any

from agent.runner import LlamaCppClient

from .common import (
    DRAFT_PORT,
    FIXED_LLM_WARMUP,
    GENERATION,
    MAIN_PORT,
    MODEL_SPECS,
    REQUEST_TIMEOUT_SECONDS,
    ExperimentError,
    ModelServer,
    Progress,
    llama_server_binary,
    print_table,
    prompt_positive_int,
    require_models,
    select_benchmark_questions,
)
from .inference import (
    DEFAULT_REPETITIONS,
    DEFAULT_TASKS,
    benchmark_messages,
    role_metric_rows,
)

DESCRIPTION = "Measure isolated main/draft TTFT and decode throughput"


def run_phases(
    roles: Iterable[str],
    server_factory: Callable[[str], AbstractContextManager[Any]],
    run_role: Callable[[str, Any], None],
) -> None:
    """Run each model only after the previous model server has stopped."""
    for role in roles:
        with server_factory(role) as server:
            run_role(role, server)


def render_report(calls_by_role: dict[str, list[dict[str, Any]]]) -> None:
    print("\nIndependent-run experiment metrics")
    print_table(
        ("role", "metric", "n", "mean", "median", "p95-worst"),
        role_metric_rows(calls_by_role),
    )
    print(
        "p95-worst is numeric p95 for TTFT and numeric p5 for throughput "
        "(95% of runs meet or exceed it)."
    )


def run(task_count: int, repetitions: int) -> None:
    questions = select_benchmark_questions(task_count)
    model_paths = require_models()
    binary, version = llama_server_binary()
    print(
        f"setup: llama.cpp build={version['build']}; "
        f"models={MODEL_SPECS['main'].filename}, {MODEL_SPECS['draft'].filename}"
    )

    ports = {"main": MAIN_PORT, "draft": DRAFT_PORT}
    calls_by_role: dict[str, list[dict[str, Any]]] = {"main": [], "draft": []}
    total = task_count * repetitions * 2
    progress = Progress("independent-run", total)
    completed = 0

    def server_factory(role: str) -> ModelServer:
        return ModelServer(role, binary, model_paths[role], ports[role])

    def run_role(role: str, server: ModelServer) -> None:
        nonlocal completed
        client = LlamaCppClient(
            server.base_url, timeout_seconds=REQUEST_TIMEOUT_SECONDS
        )
        try:
            print(f"setup: warming {role} model server", flush=True)
            client.stream_completion(
                [{"role": "user", "content": FIXED_LLM_WARMUP}],
                {**GENERATION, "max_tokens": 32, "temperature": 0},
            )
            for repetition in range(repetitions):
                for question in questions:
                    call = client.stream_completion(
                        benchmark_messages(question), GENERATION
                    )
                    calls_by_role[role].append(call)
                    completed += 1
                    progress.update(
                        completed,
                        (
                            f"role={role} round={repetition + 1} "
                            f"question={question.id}"
                        ),
                    )
        finally:
            client.close()

    run_phases(("main", "draft"), server_factory, run_role)

    if any(not values for values in calls_by_role.values()):
        raise ExperimentError(
            "independent-run experiment produced no complete model calls"
        )
    render_report(calls_by_role)


def prompt_and_run(*, input_fn: Callable[[str], str] = input) -> None:
    task_count = prompt_positive_int(
        "FanOutQA task count", DEFAULT_TASKS, input_fn=input_fn
    )
    repetitions = prompt_positive_int(
        "Repetitions per task", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(task_count, repetitions)
