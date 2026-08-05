from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from agent.retrieval import TimedRetriever
from agent.runner import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    AgentRunner,
    LlamaCppClient,
)
from wikipedia.encoder import Encoder
from wikipedia.qdrant import QdrantVectorDB

from .artifacts import RunArtifacts, run_metadata
from .common import (
    FIXED_LLM_WARMUP,
    FIXED_RETRIEVAL_WARMUP,
    GENERATION,
    MAIN_PORT,
    MAX_SEARCHES,
    REQUEST_TIMEOUT_SECONDS,
    RETRIEVAL_MAX_CHARS,
    RETRIEVAL_TOP_K,
    ExperimentError,
    ModelSpec,
    ModelServer,
    Progress,
    format_metric,
    llama_server_binary,
    prepare_wikipedia,
    print_histograms,
    print_table,
    prompt_positive_int,
    require_models,
    resolve_model_specs,
    require_restartable_qdrant,
    restart_qdrant,
    select_benchmark_questions,
    summarize,
)

DESCRIPTION = "Measure tool-use query tokens and latency"
DEFAULT_TASKS = 5
DEFAULT_REPETITIONS = 3
REASONING_BUDGET_TOKENS = 512
QUESTION_TIMEOUT_SECONDS = 600.0
VALID_TERMINAL_STATUSES = frozenset({"final", "search_limit"})


def query_token_metrics(outcome: dict[str, Any]) -> tuple[int | None, list[int]]:
    counts: list[int] = []
    for call in outcome["llm_calls"]:
        tool_calls = call.get("tool_calls") or []
        if not any(
            tool_call.get("function", {}).get("name") == "search"
            for tool_call in tool_calls
        ):
            continue
        completion_tokens = call.get("usage", {}).get("completion_tokens")
        if (
            not isinstance(completion_tokens, int)
            or isinstance(completion_tokens, bool)
            or completion_tokens < 0
        ):
            raise ExperimentError(
                "llama.cpp did not report completion_tokens for a search call"
            )
        counts.append(completion_tokens)
    return (counts[0] if counts else None, counts[1:])


def _new_role_result() -> dict[str, Any]:
    return {
        "queries": [],
        "first": [],
        "between": [],
        "incomplete_queries": [],
        "incomplete_first": [],
        "incomplete_between": [],
        "end_to_end": [],
        "incomplete_end_to_end": [],
        "retrieval": {"encode": [], "qdrant": [], "total": []},
        "valid": 0,
        "failed": 0,
        "failure_statuses": {},
        "zero_query": 0,
        "one_query": 0,
    }


def _record_outcome(selected: dict[str, Any], outcome: dict[str, Any]) -> None:
    search_count = int(outcome["search_count"])
    if search_count == 0:
        selected["zero_query"] += 1
    elif search_count == 1:
        selected["one_query"] += 1

    timing = outcome.get("timing") or {}
    end_to_end = timing.get("end_to_end_ms")
    valid = outcome["terminal_status"] in VALID_TERMINAL_STATUSES
    if isinstance(end_to_end, (int, float)) and not isinstance(end_to_end, bool):
        key = "end_to_end" if valid else "incomplete_end_to_end"
        selected[key].append(float(end_to_end))
    for retrieval in outcome.get("retrieval_calls") or []:
        encode = float(retrieval["encode_duration_ms"])
        qdrant = float(retrieval["qdrant_duration_ms"])
        selected["retrieval"]["encode"].append(encode)
        selected["retrieval"]["qdrant"].append(qdrant)
        selected["retrieval"]["total"].append(encode + qdrant)

    if valid:
        first, between = query_token_metrics(outcome)
        selected["valid"] += 1
        selected["queries"].append(float(search_count))
        if first is not None:
            selected["first"].append(float(first))
            selected["between"].extend(float(value) for value in between)
        return

    selected["failed"] += 1
    status = str(outcome["terminal_status"])
    selected["failure_statuses"][status] = (
        selected["failure_statuses"].get(status, 0) + 1
    )
    selected["incomplete_queries"].append(float(search_count))
    try:
        first, between = query_token_metrics(outcome)
    except ExperimentError:
        return
    if first is not None:
        selected["incomplete_first"].append(float(first))
        selected["incomplete_between"].extend(float(value) for value in between)


def _metric_rows(
    results: dict[str, dict[str, Any]], *, prefix: str = ""
) -> list[list[object]]:
    rows: list[list[object]] = []
    for role in ("main", "draft"):
        result = results[role]
        metrics = (
            ("queries / run", summarize(result[f"{prefix}queries"])),
            ("tokens to first query", summarize(result[f"{prefix}first"])),
            ("tokens between queries", summarize(result[f"{prefix}between"])),
        )
        for label, stats in metrics:
            rows.append(
                [
                    role,
                    label,
                    stats.count,
                    format_metric(stats.mean),
                    format_metric(stats.median),
                    format_metric(stats.p95_worst),
                ]
            )
    return rows


def render_report(results: dict[str, dict[str, Any]]) -> None:
    print("\nTool-use latency metrics")
    latency_rows: list[list[object]] = []
    for role in ("main", "draft"):
        values = results[role]
        for label, selected in (
            ("end-to-end (ms)", values["end_to_end"]),
            ("encode (ms)", values["retrieval"]["encode"]),
            ("Qdrant (ms)", values["retrieval"]["qdrant"]),
            ("retrieval total (ms)", values["retrieval"]["total"]),
        ):
            stats = summarize(selected)
            latency_rows.append(
                [
                    role,
                    label,
                    stats.count,
                    format_metric(stats.mean),
                    format_metric(stats.median),
                    format_metric(stats.p95_worst),
                ]
            )
    headers = ("role", "metric", "n", "mean", "median", "p95-worst")
    print_table(headers, latency_rows)
    print_histograms(
        "End-to-end latency histogram (ms; valid runs)",
        {role: results[role]["end_to_end"] for role in ("main", "draft")},
    )
    print_histograms(
        "Qdrant RPC latency histogram (ms; completed queries)",
        {
            role: results[role]["retrieval"]["qdrant"]
            for role in ("main", "draft")
        },
    )

    print("\nTool-use query metrics (generated tokens only)")
    print_table(headers, _metric_rows(results))
    print("p95-worst is numeric p95; lower is treated as better for query/token cost.")

    print("\nIncomplete-run metrics (diagnostic only)")
    print_table(
        headers,
        [
            [
                role,
                "end-to-end (ms)",
                stats.count,
                format_metric(stats.mean),
                format_metric(stats.median),
                format_metric(stats.p95_worst),
            ]
            for role in ("main", "draft")
            for stats in [summarize(results[role]["incomplete_end_to_end"])]
        ]
        + _metric_rows(results, prefix="incomplete_"),
    )

    print("\nRun outcomes")
    print_table(
        ("role", "valid", "failed", "zero-query", "one-query"),
        [
            [
                role,
                values["valid"],
                values["failed"],
                values["zero_query"],
                values["one_query"],
            ]
            for role, values in results.items()
        ],
    )
    print("\nFailure status breakdown")
    print_table(
        ("role", "status", "count"),
        [
            [role, status, count]
            for role in ("main", "draft")
            for status, count in (
                sorted(results[role]["failure_statuses"].items()) or [("none", 0)]
            )
        ],
    )


def build_summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"roles": {}}
    for role in ("main", "draft"):
        result = results[role]
        summary["roles"][role] = {
            "valid_metrics": {
                "end_to_end_ms": asdict(summarize(result["end_to_end"])),
                "queries_per_run": asdict(summarize(result["queries"])),
                "tokens_to_first_query": asdict(summarize(result["first"])),
                "tokens_between_queries": asdict(summarize(result["between"])),
            },
            "incomplete_metrics": {
                "end_to_end_ms": asdict(
                    summarize(result["incomplete_end_to_end"])
                ),
                "queries_per_run": asdict(summarize(result["incomplete_queries"])),
                "tokens_to_first_query": asdict(
                    summarize(result["incomplete_first"])
                ),
                "tokens_between_queries": asdict(
                    summarize(result["incomplete_between"])
                ),
            },
            "retrieval_metrics": {
                "encode_duration_ms": asdict(
                    summarize(result["retrieval"]["encode"])
                ),
                "qdrant_duration_ms": asdict(
                    summarize(result["retrieval"]["qdrant"])
                ),
                "total_duration_ms": asdict(
                    summarize(result["retrieval"]["total"])
                ),
            },
            "outcomes": {
                "valid": result["valid"],
                "failed": result["failed"],
                "zero_query": result["zero_query"],
                "one_query": result["one_query"],
                "failure_statuses": result["failure_statuses"],
            },
        }
    return summary


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
        f"target={selected_models['main'].filename}, "
        f"draft={selected_models['draft'].filename}"
    )
    results = {role: _new_role_result() for role in ("main", "draft")}

    with prepare_wikipedia(require_idle=True) as environment:
        require_restartable_qdrant(environment)
        print(f"setup: loading BGE-M3 from {environment.paths.model_dir}", flush=True)
        try:
            encoder = Encoder(environment.paths.bundle_dir, require_complete=False)
            print("setup: warming BGE-M3 encoder", flush=True)
            encoder.encode(FIXED_RETRIEVAL_WARMUP)
        except Exception as exc:
            raise ExperimentError(f"BGE-M3 could not be loaded: {exc}") from exc

        wiki_manifest = environment.manifest
        metadata = run_metadata(
            task_count=task_count,
            repetitions=repetitions,
            questions=questions,
            model_specs=selected_models,
            model_paths=model_paths,
            llama_cpp=version,
            generation=GENERATION,
            parameters={
                "execution": "isolated non-overlapping model phases",
                "agent_prompt": {
                    "text": SYSTEM_PROMPT,
                    "sha256": SYSTEM_PROMPT_SHA256,
                },
                "reasoning_budget_tokens": REASONING_BUDGET_TOKENS,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
                "question_timeout_seconds": QUESTION_TIMEOUT_SECONDS,
                "max_searches": MAX_SEARCHES,
                "retrieval": {
                    "top_k": RETRIEVAL_TOP_K,
                    "max_chars_per_result": RETRIEVAL_MAX_CHARS,
                    "database_reset": "docker_restart_per_benchmark_unit",
                    "host_page_cache_evicted": False,
                    "reset_included_in_end_to_end": False,
                },
                "wikipedia": {
                    "collection": environment.database.config.collection_name,
                    "endpoint": environment.database.config.endpoint,
                    "points_count": environment.points_count,
                    "incomplete": environment.incomplete,
                    "dataset_revision": wiki_manifest.get("dataset", {}).get(
                        "resolved_revision"
                    ),
                    "model_revision": wiki_manifest.get("model", {}).get(
                        "resolved_revision"
                    ),
                    "manifest_sha256": hashlib.sha256(
                        json.dumps(
                            wiki_manifest, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            },
        )
        with RunArtifacts("tooluse", metadata) as artifacts:
            total = task_count * repetitions * 2
            progress = Progress("tooluse", total)
            completed = 0
            for role in ("main", "draft"):
                with ModelServer(
                    role,
                    binary,
                    model_paths[role],
                    MAIN_PORT,
                    reasoning_budget=REASONING_BUDGET_TOKENS,
                ) as server:
                    client = LlamaCppClient(
                        server.base_url, timeout_seconds=REQUEST_TIMEOUT_SECONDS
                    )
                    try:
                        client.stream_completion(
                            [{"role": "user", "content": FIXED_LLM_WARMUP}],
                            {**GENERATION, "max_tokens": 32, "temperature": 0},
                        )
                        for repetition in range(repetitions):
                            for question in questions:
                                restart_qdrant(environment)
                                with QdrantVectorDB(
                                    environment.database.config
                                ) as database:
                                    retriever = TimedRetriever(
                                        encoder,
                                        database,
                                        top_k=RETRIEVAL_TOP_K,
                                        max_chars_per_result=RETRIEVAL_MAX_CHARS,
                                    )
                                    runner = AgentRunner(
                                        client,
                                        retriever,
                                        generation=GENERATION,
                                        max_searches=MAX_SEARCHES,
                                        question_timeout_seconds=(
                                            QUESTION_TIMEOUT_SECONDS
                                        ),
                                    )
                                    outcome = runner.run(question)
                                artifacts.append(
                                    {
                                        "record_type": "agent_run",
                                        "model_role": role,
                                        "repetition": repetition + 1,
                                        "question": question.agent_value(),
                                        "outcome": outcome,
                                    }
                                )
                                selected = results[role]
                                _record_outcome(selected, outcome)
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

            if sum(result["valid"] for result in results.values()) == 0:
                raise ExperimentError("tool-use experiment produced no valid runs")
            artifacts.write_summary(build_summary(results))
    render_report(results)


def prompt_and_run(
    *,
    input_fn: Callable[[str], str] = input,
    model_specs: Mapping[str, ModelSpec] | None = None,
) -> None:
    task_count = prompt_positive_int(
        "FanOutQA task count", DEFAULT_TASKS, input_fn=input_fn
    )
    repetitions = prompt_positive_int(
        "Repetitions per task and model", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(task_count, repetitions, model_specs=model_specs)
