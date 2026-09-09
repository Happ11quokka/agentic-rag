"""Measure resident solo/parallel contention in fixed-duration capture windows."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
import os
import signal
import threading
import time
from typing import Any

import httpx

from agent.runner import LlamaCppClient

from .artifacts import RunArtifacts, run_metadata
from .common import (
    DRAFT_PORT,
    FIXED_LLM_WARMUP,
    GENERATION,
    MAIN_PORT,
    REQUEST_TIMEOUT_SECONDS,
    ExperimentError,
    ModelServer,
    ModelSpec,
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
from .inference import benchmark_messages
from .metal_counters import (
    CAPTURE_SECONDS,
    MAX_CAPTURE_SECONDS,
    REQUIRED_COUNTERS,
    MetalRecording,
    collector_metadata,
)
from .parallel import paired_completion

DESCRIPTION = "Compare resident solo/parallel inference with Metal GPU counters"
DEFAULT_TASKS = 1
DEFAULT_REPETITIONS = 1
CONDITIONS = ("main-only", "draft-only", "simultaneous")


def condition_order(unit: int) -> tuple[str, ...]:
    offset = unit % len(CONDITIONS)
    return CONDITIONS[offset:] + CONDITIONS[:offset]


def _kill_servers(servers: Mapping[str, ModelServer]) -> None:
    # These are the process groups created by ModelServer, never existing services.
    # Signal every group before waiting, so one slow shutdown cannot delay another.
    for server in servers.values():
        process = server.process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for server in servers.values():
        server.stop()


def fixed_duration_calls(
    clients: Mapping[str, LlamaCppClient],
    servers: Mapping[str, ModelServer],
    messages: list[dict[str, str]],
    generation: dict[str, Any],
    duration_seconds: int,
) -> tuple[dict[str, dict[str, Any]], tuple[int, int]]:
    """Keep partial streams, then kill both servers at the capture deadline."""
    cancel = threading.Event()
    failed = threading.Event()
    barrier = threading.Barrier(len(clients) + 1)
    states: dict[str, dict[str, Any]] = {role: {"chunks": []} for role in clients}
    deadline_ns = 0

    def invoke(role: str, client: LlamaCppClient) -> None:
        state = states[role]
        barrier.wait()

        def started(timestamp: int) -> None:
            state["start_ns"] = timestamp

        def chunk(item: dict[str, Any]) -> None:
            if item["received_ns"] <= deadline_ns:
                state["chunks"].append(dict(item))

        try:
            call = client.stream_completion(
                messages,
                generation,
                cancel_event=cancel,
                on_request_start=started,
                on_chunk=chunk,
            )
            state["completed"] = not call.get("cancelled", False)
            state["end_ns"] = call["request_end_ns"]
        except httpx.HTTPError as error:
            if not cancel.is_set():
                state["error"] = error
                failed.set()
        except BaseException as error:
            state["error"] = error
            failed.set()

    pool = ThreadPoolExecutor(
        max_workers=len(clients), thread_name_prefix="contention-stream"
    )
    futures = []
    try:
        futures = [
            pool.submit(invoke, role, client) for role, client in clients.items()
        ]
        start_ns = time.perf_counter_ns()
        deadline_ns = start_ns + duration_seconds * 1_000_000_000
        barrier.wait()
        remaining = max(0.0, (deadline_ns - time.perf_counter_ns()) / 1_000_000_000)
        failed.wait(remaining)
    finally:
        cancel.set()
        barrier.abort()
        _kill_servers(servers)
        _, pending = wait(futures, timeout=5)
        pool.shutdown(wait=not pending, cancel_futures=True)
    if pending:
        raise ExperimentError(
            "LLM stream workers did not exit after server termination"
        )
    calls = {}
    for role, state in states.items():
        if error := state.get("error"):
            raise error
        if "start_ns" not in state:
            raise ExperimentError(
                f"{role} request did not start within the capture window"
            )
        chunks = state["chunks"]
        first = chunks[0]["received_ns"] if chunks else None
        span = chunks[-1]["received_ns"] - first if len(chunks) >= 2 else 0
        completed = bool(
            state.get("completed")
            and state.get("end_ns", deadline_ns + 1) <= deadline_ns
        )
        calls[role] = {
            "request": {"messages": deepcopy(messages), "stream": True, **generation},
            "request_start_ns": state["start_ns"],
            "request_end_ns": min(state.get("end_ns", deadline_ns), deadline_ns),
            "first_decode_ns": first,
            "chunks": chunks,
            "reasoning": "".join(
                c.get("text", "") for c in chunks if c.get("channel") == "reasoning"
            ),
            "content": "".join(
                c.get("text", "") for c in chunks if c.get("channel") == "content"
            ),
            "cancelled": not completed,
            "stop_reason": "completed" if completed else "capture_deadline",
            "timing": {
                "ttft_ms": (first - state["start_ns"]) / 1_000_000
                if first is not None
                else None,
                "stream_chunks_per_second": (len(chunks) - 1) * 1_000_000_000 / span
                if span > 0
                else None,
                "observed_chunks": len(chunks),
            },
        }
    return calls, (start_ns, deadline_ns)


def measurement_windows(
    condition: str,
    calls: Mapping[str, dict[str, Any]],
) -> dict[str, tuple[int, int]]:
    windows = {
        f"{condition}/request": (
            min(call["request_start_ns"] for call in calls.values()),
            max(call["request_end_ns"] for call in calls.values()),
        )
    }
    decodes: list[tuple[int, int]] = []
    for role, call in calls.items():
        first = call.get("first_decode_ns")
        if not isinstance(first, int):
            continue
        windows[f"{condition}/{role}/before-first-chunk"] = (
            call["request_start_ns"],
            first,
        )
        chunks = call.get("chunks", [])
        last = chunks[-1]["received_ns"] if chunks else first
        if last > first:
            windows[f"{condition}/{role}/decode"] = (first, last)
            decodes.append((first, last))
    if len(calls) == 2 and len(decodes) == 2:
        overlap = (max(start for start, _ in decodes), min(end for _, end in decodes))
        if overlap[1] > overlap[0]:
            windows[f"{condition}/decode-overlap"] = overlap
            tail = (overlap[1], max(end for _, end in decodes))
            if tail[1] > tail[0]:
                windows[f"{condition}/decode-tail"] = tail
    return windows


def matched_slowdowns(
    conditions: Mapping[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    result = {}
    for role in ("main", "draft"):
        solo = conditions[f"{role}-only"][role]["timing"]
        parallel = conditions["simultaneous"][role]["timing"]
        row = {}
        for name, numerator, denominator in (
            ("ttft_ratio", parallel.get("ttft_ms"), solo.get("ttft_ms")),
            (
                "stream_rate_ratio",
                solo.get("stream_chunks_per_second"),
                parallel.get("stream_chunks_per_second"),
            ),
        ):
            row[name] = (
                numerator / denominator
                if numerator is not None and denominator and denominator > 0
                else None
            )
        result[role] = row
    return result


def render_counter_report(result: dict[str, Any]) -> None:
    rows = []
    for name, window in result["windows"].items():
        if name.endswith("before-first-chunk"):
            continue
        counters = window["counters"]

        def mean(counter: str) -> str:
            return format_metric(counters.get(counter, {}).get("mean"))

        def coverage(counter: str) -> str:
            value = counters.get(counter, {}).get("coverage_fraction")
            return "n/a" if value is None else f"{value:.0%}"

        ranked = window["ranked_limiters"][:3]
        leaders = ", ".join(
            f"{key.removesuffix(' Limiter')} {mean(key)}%" for key in ranked
        )
        rows.append(
            [
                name,
                format_metric((window["end_ns"] - window["start_ns"]) / 1e6),
                mean("GPU Read Bandwidth"),
                mean("GPU Write Bandwidth"),
                mean("ALU Utilization"),
                f"{coverage('GPU Read Bandwidth')}/{coverage('ALU Utilization')}",
                leaders or "unavailable",
            ]
        )
    print_table(
        (
            "window",
            "ms",
            "read GB/s",
            "write GB/s",
            "ALU %",
            "coverage BW/ALU",
            "highest limiters",
        ),
        rows,
    )
    if "simultaneous/decode-overlap" not in result["windows"]:
        print("decode overlap: unavailable (no shared client-observed decode window)")
    if invalid := result.get("invalid_samples_by_counter"):
        print(
            f"Excluded {sum(invalid.values())} invalid counter samples; counts saved in counter summary."
        )


def render_report(metrics: dict[str, Any]) -> None:
    print("\nContention experiment metrics")
    rows = []
    for role in ("main", "draft"):
        solo = metrics["by_condition"][f"{role}-only"][role]
        parallel = metrics["by_condition"]["simultaneous"][role]
        for label, key, ratio in (
            ("TTFT (ms)", "ttft_ms", "ttft_ratio"),
            ("stream (chunks/s)", "stream_chunks_per_second", "stream_rate_ratio"),
        ):
            rows.append(
                [
                    role,
                    label,
                    format_metric(solo[key]["median"]),
                    format_metric(parallel[key]["median"]),
                    format_metric(metrics["matched_slowdowns"][role][ratio]["median"]),
                ]
            )
    print_table(
        ("role", "metric", "solo median", "parallel median", "matched slowdown"), rows
    )
    print(
        "Slowdown >1 means worse in parallel. Stream chunks are not tokenizer tokens."
    )
    print(
        "Requests stop at the fixed deadline. Timings cover only the recorded window."
    )
    print(
        "Counters are GPU-wide. Means cover sampled intervals; client chunk boundaries approximate decode."
    )
    print("Ranked limiters are evidence, not an automatic memory/compute diagnosis.")


def run(
    task_count: int,
    repetitions: int,
    *,
    model_specs: Mapping[str, ModelSpec] | None = None,
    recording_seconds: int = CAPTURE_SECONDS,
) -> None:
    if (
        not isinstance(recording_seconds, int)
        or not 1 <= recording_seconds <= MAX_CAPTURE_SECONDS
    ):
        raise ExperimentError(
            f"recording duration must be 1–{MAX_CAPTURE_SECONDS} seconds"
        )
    selected = resolve_model_specs(model_specs)
    collector = collector_metadata()
    collector["capture_seconds"] = recording_seconds
    questions = select_benchmark_questions(task_count)
    paths = require_models(selected)
    binary, version = llama_server_binary()
    metadata = run_metadata(
        task_count=task_count,
        repetitions=repetitions,
        questions=questions,
        model_specs=selected,
        model_paths=paths,
        llama_cpp=version,
        generation=GENERATION,
        parameters={
            "execution": "fixed-duration resident solo and simultaneous inference",
            "condition_order": "rotate main-only/draft-only/simultaneous per unit",
            "server_lifecycle": "fresh warmed resident pair per condition; killed at deadline",
            "collector": collector,
        },
    )
    by_condition: dict[str, dict[str, list[dict[str, Any]]]] = {
        name: {"main": [], "draft": []} for name in CONDITIONS
    }
    ratios: dict[str, dict[str, list[float]]] = {
        role: {"ttft_ratio": [], "stream_rate_ratio": []} for role in ("main", "draft")
    }
    counter_units = []
    with RunArtifacts("contention-measure", metadata) as artifacts:
        progress = Progress("contention-measure", task_count * repetitions * 3)
        unit = 0
        for repetition in range(repetitions):
            for question in questions:
                order = condition_order(unit)
                directory = artifacts.run_dir / f"counters-{unit + 1:04d}"
                conditions = {}
                counters = {
                    "windows": {},
                    "recordings": {},
                    "invalid_samples_by_counter": {},
                }
                for condition in order:
                    print(
                        f"\nsetup: {condition}; {recording_seconds}s capture",
                        flush=True,
                    )
                    with ExitStack() as stack:
                        servers = {
                            role: stack.enter_context(
                                ModelServer(role, binary, paths[role], port)
                            )
                            for role, port in (
                                ("main", MAIN_PORT),
                                ("draft", DRAFT_PORT),
                            )
                        }
                        warm_clients = {}
                        for role, server in servers.items():
                            warm_clients[role] = LlamaCppClient(
                                server.base_url, timeout_seconds=REQUEST_TIMEOUT_SECONDS
                            )
                            stack.callback(warm_clients[role].close)
                        print("setup: warming both resident servers", flush=True)
                        paired_completion(
                            warm_clients,
                            [{"role": "user", "content": FIXED_LLM_WARMUP}],
                            {**GENERATION, "max_tokens": 32, "temperature": 0},
                        )
                        for client in warm_clients.values():
                            client.close()
                        roles = (
                            ("main", "draft")
                            if condition == "simultaneous"
                            else (condition.removesuffix("-only"),)
                        )
                        clients = {}
                        for role in roles:
                            clients[role] = LlamaCppClient(
                                servers[role].base_url,
                                timeout_seconds=recording_seconds + 2,
                            )
                            stack.callback(clients[role].close)
                        phase_directory = directory / condition
                        with MetalRecording(
                            phase_directory, duration_seconds=recording_seconds
                        ) as recording:
                            calls, interval = fixed_duration_calls(
                                clients,
                                servers,
                                benchmark_messages(question),
                                GENERATION,
                                recording_seconds,
                            )
                            conditions[condition] = calls
                            for role, call in calls.items():
                                by_condition[condition][role].append(call)
                            artifacts.append(
                                {
                                    "record_type": "contention_completion",
                                    "unit": unit + 1,
                                    "repetition": repetition + 1,
                                    "question": question.agent_value(),
                                    "condition": condition,
                                    "condition_order": order,
                                    "model_calls": calls,
                                    "capture_interval_ns": interval,
                                    "counter_directory": str(
                                        phase_directory.relative_to(artifacts.run_dir)
                                    ),
                                }
                            )
                            print(
                                "model servers stopped; finalizing bounded GPU trace",
                                flush=True,
                            )
                    # Model processes and HTTP clients are gone before export.
                    print("exporting GPU counters", flush=True)
                    windows = measurement_windows(condition, calls)
                    windows[f"{condition}/capture"] = interval
                    phase_counters = recording.export(windows)
                    observed = phase_counters["windows"][f"{condition}/capture"][
                        "counters"
                    ]
                    if missing := REQUIRED_COUNTERS - observed.keys():
                        raise ExperimentError(
                            f"{condition} has no samples for required counters: {', '.join(sorted(missing))}"
                        )
                    counters["windows"].update(phase_counters["windows"])
                    counters["recordings"][condition] = {
                        "directory": str(
                            phase_directory.relative_to(artifacts.run_dir)
                        ),
                        **{
                            key: value
                            for key, value in phase_counters.items()
                            if key != "windows"
                        },
                    }
                    for name, count in phase_counters.get(
                        "invalid_samples_by_counter", {}
                    ).items():
                        counters["invalid_samples_by_counter"][name] = (
                            counters["invalid_samples_by_counter"].get(name, 0) + count
                        )
                    progress.update(unit * 3 + len(conditions), condition)
                slowdown = matched_slowdowns(conditions)
                for role, row in slowdown.items():
                    for key, value in row.items():
                        if value is not None:
                            ratios[role][key].append(value)
                counter_units.append(
                    {
                        "unit": unit + 1,
                        "directory": directory.name,
                        "measurements": counters,
                    }
                )
                artifacts.append(
                    {
                        "record_type": "contention_counters",
                        "unit": unit + 1,
                        "counter_directory": directory.name,
                        "matched_slowdowns": slowdown,
                        "measurements": counters,
                    }
                )
                render_counter_report(counters)
                unit += 1
        metrics = {
            "by_condition": {
                condition: {
                    role: {
                        key: asdict(
                            summarize(
                                (
                                    call["timing"][key]
                                    for call in calls
                                    if call["timing"].get(key) is not None
                                ),
                                higher_is_better=key == "stream_chunks_per_second",
                            )
                        )
                        for key in ("ttft_ms", "stream_chunks_per_second")
                    }
                    for role, calls in roles.items()
                }
                for condition, roles in by_condition.items()
            },
            "matched_slowdowns": {
                role: {key: asdict(summarize(values)) for key, values in row.items()}
                for role, row in ratios.items()
            },
        }
        artifacts.write_summary(
            {
                "metrics": metrics,
                "collector": collector,
                "measured_units": unit,
                "counter_units": counter_units,
            }
        )
    render_report(metrics)


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
    while True:
        recording_seconds = prompt_positive_int(
            f"Recording seconds per condition (1–{MAX_CAPTURE_SECONDS})",
            CAPTURE_SECONDS,
            input_fn=input_fn,
        )
        if recording_seconds <= MAX_CAPTURE_SECONDS:
            break
        print(
            f"Use at most {MAX_CAPTURE_SECONDS}s to bound full-resolution trace size."
        )
    run(
        task_count,
        repetitions,
        model_specs=model_specs,
        recording_seconds=recording_seconds,
    )
