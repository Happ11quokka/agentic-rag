from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, Callable

from agent.scheduling import LlamaConfig, ScheduledPairRunner
from agent.scheduling.build import NativeBuildError, resolve_native_engine
from agent.scheduling.config import (
    DECODE_ORDER,
    PREFILL_CHUNK_TOKENS,
    PREFILL_ORDER,
    TIME_QUANTUM_NS,
)
from .artifacts import RunArtifacts, run_metadata
from .common import (
    FIXED_LLM_WARMUP,
    GENERATION,
    ExperimentError,
    ModelSpec,
    Progress,
    format_metric,
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

DESCRIPTION = "single-process time-sliced main/draft inference"


def combined_throughput(calls: dict[str, dict[str, Any]]) -> float | None:
    starts = [call.get("first_decode_ns") for call in calls.values()]
    if any(not isinstance(value, int) for value in starts):
        return None
    start = min(int(value) for value in starts)
    end = max(int(call["request_end_ns"]) for call in calls.values())
    seconds = (end - start) / 1_000_000_000
    tokens = sum(int(call.get("usage", {}).get("completion_tokens") or 0) for call in calls.values())
    return tokens / seconds if seconds > 0 and tokens else None


def build_summary(
    calls_by_role: dict[str, list[dict[str, Any]]],
    combined: list[float],
    schedulers: list[dict[str, Any]],
) -> dict[str, Any]:
    role_scheduler: dict[str, Any] = {}
    for role in ("main", "draft"):
        metrics = [item["role_metrics"][role] for item in schedulers]
        role_scheduler[role] = {
            "active_decode_tokens_per_second": asdict(
                summarize(
                    (
                        value["active_tokens_per_second"]
                        for value in metrics
                        if value["active_tokens_per_second"] is not None
                    ),
                    higher_is_better=True,
                )
            ),
            "queued_decode_ms": asdict(
                summarize(value["queued_decode_ms"] for value in metrics)
            ),
            "tokens_per_slice": asdict(
                summarize(value["tokens_per_slice"] for value in metrics)
            ),
            "slice_count": asdict(
                summarize(value["slice_count"] for value in metrics)
            ),
            "both_active_time_share": asdict(
                summarize(
                    value["both_active_time_share"]
                    for value in metrics
                    if value["both_active_time_share"] is not None
                )
            ),
        }
    overshoot_ms = [
        turn["overshoot_ns"] / 1_000_000
        for item in schedulers
        for turn in item["turns"]
    ]
    return {
        "metrics": {
            "by_role": role_metric_summary(calls_by_role),
            "combined_decode_tokens_per_second": asdict(
                summarize(combined, higher_is_better=True)
            ),
            "scheduler_by_role": role_scheduler,
            "pair_slice_count": asdict(
                summarize(item["slice_count"] for item in schedulers)
            ),
            "role_switch_count": asdict(
                summarize(item["role_switch_count"] for item in schedulers)
            ),
            "quota_overshoot_ms": asdict(summarize(overshoot_ms)),
        }
    }


def render_report(
    calls_by_role: dict[str, list[dict[str, Any]]],
    combined: list[float],
    schedulers: list[dict[str, Any]],
) -> None:
    print("\nParallel-scheduled experiment metrics")
    rows = role_metric_rows(calls_by_role)
    combined_stats = summarize(combined, higher_is_better=True)
    rows.append(
        [
            "both",
            "combined (tok/s)",
            combined_stats.count,
            format_metric(combined_stats.mean),
            format_metric(combined_stats.median),
            format_metric(combined_stats.p95_worst),
        ]
    )
    print_table(("role", "metric", "n", "mean", "median", "p95-worst"), rows)
    if schedulers:
        switches = summarize(item["role_switch_count"] for item in schedulers)
        print(
            f"scheduler: quantum={TIME_QUANTUM_NS / 1_000_000:g}ms; "
            f"mean role switches={format_metric(switches.mean)}"
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
    try:
        native = resolve_native_engine()
    except NativeBuildError as exc:
        raise ExperimentError(str(exc)) from exc
    llama_config = LlamaConfig().resolved()
    print(
        f"setup: native llama.cpp build={native.version['build']}; "
        f"models={selected_models['main'].filename}, "
        f"{selected_models['draft'].filename}; "
        f"threads={llama_config.n_threads} ({llama_config.thread_source})"
    )
    calls_by_role: dict[str, list[dict[str, Any]]] = {"main": [], "draft": []}
    combined: list[float] = []
    schedulers: list[dict[str, Any]] = []
    scheduler_config = {
        "policy": "equal-time-round-robin",
        "time_quantum_ns": TIME_QUANTUM_NS,
        "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
        "prefill_order": list(PREFILL_ORDER),
        "decode_order": list(DECODE_ORDER),
    }
    metadata = run_metadata(
        task_count=task_count,
        repetitions=repetitions,
        questions=questions,
        model_specs=selected_models,
        model_paths=model_paths,
        llama_cpp=native.version,
        generation=GENERATION,
        parameters={
            "execution": "single-process cooperative model scheduling",
            "scheduler": scheduler_config,
            "llama_config": asdict(llama_config),
        },
    )
    with RunArtifacts("parallel-scheduled", metadata) as artifacts:
        try:
            with ScheduledPairRunner(
                model_paths=model_paths,
                llama_config=llama_config,
                native_engine=native,
            ) as runner:
                print("setup: warming resident model pair", flush=True)
                runner.run_pair(
                    [{"role": "user", "content": FIXED_LLM_WARMUP}],
                    {**GENERATION, "max_tokens": 32, "temperature": 0},
                )
                total = task_count * repetitions
                progress = Progress("parallel-scheduled", total)
                completed = 0
                for repetition in range(repetitions):
                    for question in questions:
                        result = runner.run_pair(
                            benchmark_messages(question), GENERATION
                        )
                        for role, call in result.calls.items():
                            calls_by_role[role].append(call)
                        value = combined_throughput(result.calls)
                        if value is not None:
                            combined.append(value)
                        schedulers.append(result.scheduler)
                        artifacts.append(
                            {
                                "record_type": "scheduled_paired_completion",
                                "repetition": repetition + 1,
                                "question": question.agent_value(),
                                "model_calls": result.calls,
                                "combined_decode_tokens_per_second": value,
                                "scheduler": result.scheduler,
                            }
                        )
                        completed += 1
                        progress.update(
                            completed,
                            f"round={repetition + 1} question={question.id}",
                        )
        except (RuntimeError, OSError, ValueError) as exc:
            raise ExperimentError(str(exc)) from exc
        if any(not values for values in calls_by_role.values()):
            raise ExperimentError(
                "parallel-scheduled experiment produced no complete paired calls"
            )
        artifacts.write_summary(build_summary(calls_by_role, combined, schedulers))
    render_report(calls_by_role, combined, schedulers)


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
