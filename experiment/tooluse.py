from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agent.retrieval import TimedRetriever
from agent.runner import AgentRunner, LlamaCppClient
from wikipedia.encoder import Encoder

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
    ModelServer,
    Progress,
    format_metric,
    llama_server_binary,
    prepare_wikipedia,
    print_table,
    prompt_positive_int,
    require_models,
    select_benchmark_questions,
    summarize,
)

DESCRIPTION = "Measure generated tokens around Qdrant tool queries"
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

    if outcome["terminal_status"] in VALID_TERMINAL_STATUSES:
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
    print("\nTool-use experiment metrics (generated tokens only)")
    headers = ("role", "metric", "n", "mean", "median", "p95-worst")
    print_table(headers, _metric_rows(results))
    print("p95-worst is numeric p95; lower is treated as better for query/token cost.")

    print("\nIncomplete-run query metrics (diagnostic only)")
    print_table(headers, _metric_rows(results, prefix="incomplete_"))

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


def run(task_count: int, repetitions: int) -> None:
    questions = select_benchmark_questions(task_count)
    model_paths = require_models()
    binary, version = llama_server_binary()
    print(f"setup: llama.cpp build={version['build']}")
    results = {role: _new_role_result() for role in ("main", "draft")}

    with prepare_wikipedia(require_idle=True) as environment:
        print(f"setup: loading BGE-M3 from {environment.paths.model_dir}", flush=True)
        try:
            encoder = Encoder(environment.paths.bundle_dir, require_complete=False)
        except Exception as exc:
            raise ExperimentError(f"BGE-M3 could not be loaded: {exc}") from exc
        retriever = TimedRetriever(
            encoder,
            environment.database,
            top_k=RETRIEVAL_TOP_K,
            max_chars_per_result=RETRIEVAL_MAX_CHARS,
        )
        print("setup: warming retrieval", flush=True)
        retriever.search(FIXED_RETRIEVAL_WARMUP)

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
                    runner = AgentRunner(
                        client,
                        retriever,
                        generation=GENERATION,
                        max_searches=MAX_SEARCHES,
                        question_timeout_seconds=QUESTION_TIMEOUT_SECONDS,
                    )
                    for repetition in range(repetitions):
                        for question in questions:
                            outcome = runner.run(question)
                            selected = results[role]
                            _record_outcome(selected, outcome)
                            completed += 1
                            progress.update(
                                completed,
                                f"role={role} round={repetition + 1} question={question.id}",
                            )
                finally:
                    client.close()

    if sum(result["valid"] for result in results.values()) == 0:
        raise ExperimentError("tool-use experiment produced no valid runs")
    render_report(results)


def prompt_and_run(*, input_fn: Callable[[str], str] = input) -> None:
    task_count = prompt_positive_int(
        "FanOutQA task count", DEFAULT_TASKS, input_fn=input_fn
    )
    repetitions = prompt_positive_int(
        "Repetitions per task and model", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(task_count, repetitions)
