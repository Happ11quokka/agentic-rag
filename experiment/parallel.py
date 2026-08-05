from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from typing import Any

from agent.runner import LlamaCppClient

from .artifacts import RunArtifacts, run_metadata
from .common import (
    DRAFT_PORT,
    FIXED_LLM_WARMUP,
    GENERATION,
    MAIN_PORT,
    REQUEST_TIMEOUT_SECONDS,
    ExperimentError,
    ModelSpec,
    ModelServer,
    Progress,
    format_metric,
    llama_server_binary,
    print_table,
    prompt_positive_int,
    require_models,
    resolve_model_specs,
    select_benchmark_questions,
    summarize,
)
from .inference import (
    DEFAULT_REPETITIONS,
    DEFAULT_TASKS,
    benchmark_messages,
    role_metric_rows,
    role_metric_summary,
)

DESCRIPTION = "Measure simultaneous main/draft TTFT and decode throughput"


def paired_completion(
    clients: dict[str, Any],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Release both client requests from one barrier and return both calls."""
    barrier = threading.Barrier(len(clients) + 1)

    def invoke(client: Any) -> dict[str, Any]:
        barrier.wait()
        return client.stream_completion(messages, generation)

    pool = ThreadPoolExecutor(
        max_workers=len(clients), thread_name_prefix="parallel-model"
    )
    futures: dict[str, Any] = {}
    try:
        futures = {
            role: pool.submit(invoke, client) for role, client in clients.items()
        }
        barrier.wait()
        result = {role: future.result() for role, future in futures.items()}
    except BaseException:
        for future in futures.values():
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    pool.shutdown()
    return result


def _combined_throughput(calls: dict[str, dict[str, Any]]) -> float | None:
    starts = [call.get("first_decode_ns") for call in calls.values()]
    if any(not isinstance(value, int) for value in starts):
        return None
    end = max(int(call["request_end_ns"]) for call in calls.values())
    start = min(int(value) for value in starts)
    duration = (end - start) / 1_000_000_000
    if duration <= 0:
        return None
    tokens = sum(
        int(
            call.get("timing", {}).get("server_decode_tokens")
            or call.get("usage", {}).get("completion_tokens")
            or 0
        )
        for call in calls.values()
    )
    return tokens / duration if tokens else None


def render_report(
    calls_by_role: dict[str, list[dict[str, Any]]],
    combined_throughput: list[float],
) -> None:
    print("\nParallel experiment metrics")
    rows = role_metric_rows(calls_by_role)
    combined = summarize(combined_throughput, higher_is_better=True)
    rows.append(
        [
            "both",
            "combined (tok/s)",
            combined.count,
            format_metric(combined.mean),
            format_metric(combined.median),
            format_metric(combined.p95_worst),
        ]
    )
    print_table(("role", "metric", "n", "mean", "median", "p95-worst"), rows)
    print(
        "p95-worst is numeric p95 for TTFT and numeric p5 for throughput "
        "(95% of runs meet or exceed it)."
    )


def run(
    task_count: int,
    repetitions: int,
    *,
    model_specs: Mapping[str, ModelSpec] | None = None,
) -> None:
    selected_models = resolve_model_specs(model_specs)
    questions = select_benchmark_questions(task_count)
    model_paths = require_models(selected_models)
    binary, version = llama_server_binary()
    print(
        f"setup: llama.cpp build={version['build']}; "
        f"models={selected_models['main'].filename}, "
        f"{selected_models['draft'].filename}"
    )

    ports = {"main": MAIN_PORT, "draft": DRAFT_PORT}
    calls_by_role: dict[str, list[dict[str, Any]]] = {"main": [], "draft": []}
    combined: list[float] = []
    metadata = run_metadata(
        task_count=task_count,
        repetitions=repetitions,
        questions=questions,
        model_specs=selected_models,
        model_paths=model_paths,
        llama_cpp=version,
        generation=GENERATION,
        parameters={"execution": "simultaneous paired inference"},
    )
    with RunArtifacts("parallel", metadata) as artifacts:
        with ExitStack() as stack:
            servers = {
                role: stack.enter_context(
                    ModelServer(role, binary, model_paths[role], ports[role])
                )
                for role in ("main", "draft")
            }
            clients = {
                role: LlamaCppClient(
                    server.base_url, timeout_seconds=REQUEST_TIMEOUT_SECONDS
                )
                for role, server in servers.items()
            }

            def close_clients() -> None:
                for client in clients.values():
                    client.close()

            stack.callback(close_clients)
            print("setup: warming both model servers concurrently", flush=True)
            paired_completion(
                clients,
                [{"role": "user", "content": FIXED_LLM_WARMUP}],
                {**GENERATION, "max_tokens": 32, "temperature": 0},
            )

            total = task_count * repetitions
            progress = Progress("parallel", total)
            completed = 0
            for repetition in range(repetitions):
                for question in questions:
                    messages = benchmark_messages(question)
                    calls = paired_completion(clients, messages, GENERATION)
                    for role, call in calls.items():
                        calls_by_role[role].append(call)
                    value = _combined_throughput(calls)
                    if value is not None:
                        combined.append(value)
                    artifacts.append(
                        {
                            "record_type": "paired_completion",
                            "repetition": repetition + 1,
                            "question": question.agent_value(),
                            "model_calls": calls,
                            "combined_decode_tokens_per_second": value,
                        }
                    )
                    completed += 1
                    progress.update(
                        completed,
                        f"round={repetition + 1} question={question.id}",
                    )

        if any(not values for values in calls_by_role.values()):
            raise ExperimentError(
                "parallel experiment produced no complete paired calls"
            )
        artifacts.write_summary(
            {
                "metrics": {
                    "by_role": role_metric_summary(calls_by_role),
                    "combined_decode_tokens_per_second": asdict(
                        summarize(combined, higher_is_better=True)
                    ),
                }
            }
        )
    render_report(calls_by_role, combined)


def prompt_and_run(
    *,
    input_fn: Callable[[str], str] = input,
    model_specs: Mapping[str, ModelSpec] | None = None,
) -> None:
    task_count = prompt_positive_int(
        "FanOutQA task count", DEFAULT_TASKS, input_fn=input_fn
    )
    repetitions = prompt_positive_int(
        "Repetitions per task", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(task_count, repetitions, model_specs=model_specs)
