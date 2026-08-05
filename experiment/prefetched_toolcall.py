from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import httpx

from agent.retrieval import TimedRetriever
from agent.runner import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHA256,
    AgentRunner,
    LlamaCppClient,
    _parse_search_tool_call,
    initial_messages,
    prompt_hash,
    tool_generation,
)
from wikipedia.encoder import Encoder
from wikipedia.qdrant import QdrantVectorDB

from .artifacts import RunArtifacts, run_metadata
from .common import (
    DRAFT_PORT,
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
from .parallel import paired_completion
from .tooluse import (
    DEFAULT_REPETITIONS,
    DEFAULT_TASKS,
    QUESTION_TIMEOUT_SECONDS,
    REASONING_BUDGET_TOKENS,
    VALID_TERMINAL_STATUSES,
)

DESCRIPTION = "Measure synchronized draft query prefetch latency"


class DraftCoordinator:
    """Run one draft query for each latest target transcript snapshot."""

    def __init__(
        self,
        client: Any,
        retriever: Any,
        *,
        generation: dict[str, Any],
        start_barrier: threading.Barrier,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        self.client = client
        self.retriever = retriever
        self.generation = tool_generation(generation)
        self.start_barrier = start_barrier
        self.clock_ns = clock_ns
        self.condition = threading.Condition()
        self.latest_snapshot: dict[str, Any] | None = None
        self.consumed_sync_id = -1
        self.current_cancel: threading.Event | None = None
        self.stopping = False
        self.sync_events: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.fatal_error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="prefetched-toolcall-draft", daemon=True
        )

    def publish(
        self,
        messages: list[dict[str, Any]],
        *,
        source: str,
        target_retrieval_index: int | None = None,
    ) -> int:
        published_ns = self.clock_ns()
        with self.condition:
            sync_id = len(self.sync_events)
            event = {
                "sync_id": sync_id,
                "source": source,
                "target_retrieval_index": target_retrieval_index,
                "published_ns": published_ns,
                "prompt_hash": prompt_hash(messages),
                "message_count": len(messages),
            }
            self.sync_events.append(event)
            self.latest_snapshot = {
                **event,
                "messages": deepcopy(messages),
            }
            if self.current_cancel is not None:
                self.current_cancel.set()
            self.condition.notify_all()
        return sync_id

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.start_barrier.abort()
        with self.condition:
            self.stopping = True
            if self.current_cancel is not None:
                self.current_cancel.set()
            self.condition.notify_all()
        self.thread.join()
        if self.fatal_error is not None:
            raise ExperimentError(
                "draft coordinator failed: "
                f"{type(self.fatal_error).__name__}: {self.fatal_error}"
            ) from self.fatal_error
        return deepcopy(self.attempts), deepcopy(self.sync_events)

    def _next_snapshot(self) -> tuple[dict[str, Any], threading.Event] | None:
        with self.condition:
            while not self.stopping and (
                self.latest_snapshot is None
                or self.latest_snapshot["sync_id"] <= self.consumed_sync_id
            ):
                self.condition.wait()
            if self.stopping:
                return None
            snapshot = deepcopy(self.latest_snapshot)
            self.consumed_sync_id = int(snapshot["sync_id"])
            cancel = threading.Event()
            self.current_cancel = cancel
            return snapshot, cancel

    def _superseded_by(self, sync_id: int) -> int | None:
        with self.condition:
            latest = self.latest_snapshot
            if latest is not None and int(latest["sync_id"]) > sync_id:
                return int(latest["sync_id"])
            return None

    def _is_stopping(self) -> bool:
        with self.condition:
            return self.stopping

    def _record(self, attempt: dict[str, Any], cancel: threading.Event) -> None:
        with self.condition:
            self.attempts.append(attempt)
            if self.current_cancel is cancel:
                self.current_cancel = None

    def _run(self) -> None:
        try:
            try:
                self.start_barrier.wait()
            except threading.BrokenBarrierError:
                if self._is_stopping():
                    return
                raise
            while selected := self._next_snapshot():
                snapshot, cancel = selected
                attempt = self._attempt(snapshot, cancel)
                self._record(attempt, cancel)
        except BaseException as exc:
            self.fatal_error = exc

    def _attempt(
        self, snapshot: dict[str, Any], cancel: threading.Event
    ) -> dict[str, Any]:
        sync_id = int(snapshot["sync_id"])
        attempt: dict[str, Any] = {
            "sync_id": sync_id,
            "sync_source": snapshot["source"],
            "sync_published_ns": snapshot["published_ns"],
            "sync_prompt_hash": snapshot["prompt_hash"],
            "status": "server_error",
            "error": None,
            "model_call": None,
            "retrieval_call": None,
            "superseded_by_sync_id": None,
        }
        if cancel.is_set():
            superseded_by = self._superseded_by(sync_id)
            attempt["superseded_by_sync_id"] = superseded_by
            attempt["status"] = (
                "superseded" if superseded_by is not None else "target_finished"
            )
            return attempt
        try:
            call = self.client.stream_completion(
                snapshot["messages"],
                self.generation,
                cancel_event=cancel,
                stop_after_complete_tool_call=True,
            )
        except httpx.TimeoutException as exc:
            attempt.update(status="timeout", error=str(exc))
            return attempt
        except (httpx.HTTPError, OSError, ValueError, json.JSONDecodeError) as exc:
            attempt.update(status="server_error", error=str(exc))
            return attempt

        attempt["model_call"] = call
        superseded_by = self._superseded_by(sync_id)
        attempt["superseded_by_sync_id"] = superseded_by
        if call.get("cancelled"):
            attempt["status"] = (
                "superseded" if superseded_by is not None else "target_finished"
            )
            return attempt
        if superseded_by is not None:
            attempt["status"] = "superseded"
            return attempt

        tool_calls = call.get("tool_calls") or []
        if not tool_calls:
            if call.get("content", "").strip():
                attempt["status"] = "final"
            else:
                attempt.update(
                    status="protocol_error",
                    error="draft returned neither a search tool call nor content",
                )
            return attempt
        try:
            _, query = _parse_search_tool_call(call)
        except ValueError as exc:
            attempt.update(status="protocol_error", error=str(exc))
            return attempt

        try:
            retrieval = self.retriever.search(query)
        except Exception as exc:
            attempt.update(status="retrieval_error", error=str(exc))
            return attempt
        attempt["retrieval_call"] = retrieval
        attempt["superseded_by_sync_id"] = self._superseded_by(sync_id)
        if attempt["superseded_by_sync_id"] is not None:
            attempt["status"] = "retrieved_stale"
        elif self._is_stopping():
            attempt["status"] = "retrieved_after_target"
        else:
            attempt["status"] = "retrieved"
        return attempt


def run_benchmark(
    question: Any,
    target_client: Any,
    draft_client: Any,
    target_retriever: Any,
    draft_retriever: Any,
    *,
    generation: dict[str, Any] = GENERATION,
    max_searches: int = MAX_SEARCHES,
    question_timeout_seconds: float = QUESTION_TIMEOUT_SECONDS,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    barrier = threading.Barrier(2)
    coordinator = DraftCoordinator(
        draft_client,
        draft_retriever,
        generation=generation,
        start_barrier=barrier,
        clock_ns=clock_ns,
    )
    coordinator.publish(initial_messages(question), source="initial")
    target_retrieval_index = 0

    def synchronize(messages: list[dict[str, Any]], _retrieval: dict[str, Any]) -> None:
        nonlocal target_retrieval_index
        target_retrieval_index += 1
        coordinator.publish(
            messages,
            source="target_retrieval",
            target_retrieval_index=target_retrieval_index,
        )

    target = AgentRunner(
        target_client,
        target_retriever,
        generation=generation,
        max_searches=max_searches,
        question_timeout_seconds=question_timeout_seconds,
        clock_ns=clock_ns,
        on_retrieval_complete=synchronize,
    )
    coordinator.start()
    target_outcome: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = []
    sync_events: list[dict[str, Any]] = []
    try:
        barrier.wait()
        target_outcome = target.run(question)
    finally:
        attempts, sync_events = coordinator.stop()
    if target_outcome is None:
        raise ExperimentError("target model produced no outcome")
    return {
        "target_outcome": target_outcome,
        "draft_attempts": attempts,
        "sync_events": sync_events,
        "query_pairs": pair_retrievals(target_outcome, attempts),
    }


def _retrieval_total_ms(retrieval: dict[str, Any]) -> float:
    start = retrieval.get("encode_start_ns")
    end = retrieval.get("qdrant_end_ns")
    if isinstance(start, int) and isinstance(end, int):
        return (end - start) / 1_000_000
    return float(retrieval.get("encode_duration_ms", 0)) + float(
        retrieval.get("qdrant_duration_ms", 0)
    )


def _result_source_overlap(
    target: dict[str, Any], draft: dict[str, Any]
) -> float | None:
    target_ids = {
        str(item.get("source_id"))
        for item in target.get("results", [])
        if item.get("source_id") is not None
    }
    draft_ids = {
        str(item.get("source_id"))
        for item in draft.get("results", [])
        if item.get("source_id") is not None
    }
    if not target_ids:
        return None
    return len(target_ids & draft_ids) / len(target_ids)


def pair_retrievals(
    target_outcome: dict[str, Any], draft_attempts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_sync = {int(attempt["sync_id"]): attempt for attempt in draft_attempts}
    pairs: list[dict[str, Any]] = []
    for index, target in enumerate(target_outcome["retrieval_calls"], start=1):
        attempt = by_sync.get(index - 1)
        draft = attempt and attempt.get("retrieval_call")
        target_start = target.get("qdrant_start_ns")
        draft_end = draft and draft.get("qdrant_end_ns")
        ready = (
            isinstance(target_start, int)
            and isinstance(draft_end, int)
            and draft_end <= target_start
        )
        exact = bool(draft and draft.get("query") == target.get("query"))
        pairs.append(
            {
                "target_retrieval_index": index,
                "draft_sync_id": index - 1,
                "draft_status": attempt and attempt.get("status"),
                "target_query": target.get("query"),
                "draft_query": draft and draft.get("query"),
                "draft_retrieval_present": draft is not None,
                "draft_finished_before_target_qdrant": ready,
                "query_exact_match": exact,
                "exact_warm_ready": ready and exact,
                "lead_time_ms": (
                    (target_start - draft_end) / 1_000_000
                    if isinstance(target_start, int) and isinstance(draft_end, int)
                    else None
                ),
                "result_source_overlap": (
                    _result_source_overlap(target, draft) if draft else None
                ),
                "target_qdrant_duration_ms": target.get("qdrant_duration_ms"),
                "draft_qdrant_duration_ms": (
                    draft.get("qdrant_duration_ms") if draft else None
                ),
            }
        )
    return pairs


def _new_results() -> dict[str, Any]:
    return {
        "target_e2e": [],
        "incomplete_target_e2e": [],
        "target_statuses": {},
        "draft_statuses": {},
        "retrieval": {
            role: {"encode": [], "qdrant": [], "total": []}
            for role in ("target", "draft")
        },
        "llm": {
            role: {"ttft": [], "prefill": [], "decode_rate": []}
            for role in ("target", "draft")
        },
        "target_query_count": 0,
        "draft_query_count": 0,
        "paired_draft_query_count": 0,
        "warm_ready_count": 0,
        "exact_match_count": 0,
        "exact_warm_ready_count": 0,
        "lead_time": [],
        "result_overlap": [],
        "sync_delay": [],
        "target_qdrant_exact_warm": [],
        "target_qdrant_other": [],
        "cancelled_draft_generations": 0,
        "draft_tool_call_early_stops": 0,
        "stale_draft_queries": 0,
    }


def _append_call_metrics(
    selected: dict[str, list[float]], call: dict[str, Any]
) -> None:
    if call.get("cancelled"):
        return
    timing = call.get("timing", {})
    for key, source in (
        ("ttft", "ttft_ms"),
        ("prefill", "server_prompt_ms"),
        ("decode_rate", "decode_tokens_per_second"),
    ):
        value = timing.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            selected[key].append(float(value))


def _append_retrieval_metrics(
    selected: dict[str, list[float]], retrieval: dict[str, Any]
) -> None:
    selected["encode"].append(float(retrieval["encode_duration_ms"]))
    selected["qdrant"].append(float(retrieval["qdrant_duration_ms"]))
    selected["total"].append(_retrieval_total_ms(retrieval))


def record_result(results: dict[str, Any], result: dict[str, Any]) -> None:
    target = result["target_outcome"]
    status = str(target["terminal_status"])
    results["target_statuses"][status] = results["target_statuses"].get(status, 0) + 1
    selected_e2e = (
        results["target_e2e"]
        if status in VALID_TERMINAL_STATUSES
        else results["incomplete_target_e2e"]
    )
    selected_e2e.append(float(target["timing"]["end_to_end_ms"]))
    for call in target["llm_calls"]:
        _append_call_metrics(results["llm"]["target"], call)
    for retrieval in target["retrieval_calls"]:
        _append_retrieval_metrics(results["retrieval"]["target"], retrieval)

    for attempt in result["draft_attempts"]:
        draft_status = str(attempt["status"])
        results["draft_statuses"][draft_status] = (
            results["draft_statuses"].get(draft_status, 0) + 1
        )
        call = attempt.get("model_call")
        if call:
            if call.get("cancelled"):
                results["cancelled_draft_generations"] += 1
            if call.get("client_stop_reason") == "complete_tool_call":
                results["draft_tool_call_early_stops"] += 1
            _append_call_metrics(results["llm"]["draft"], call)
            delay = (
                call["request_start_ns"] - attempt["sync_published_ns"]
            ) / 1_000_000
            results["sync_delay"].append(max(0.0, delay))
        retrieval = attempt.get("retrieval_call")
        if retrieval:
            results["draft_query_count"] += 1
            _append_retrieval_metrics(results["retrieval"]["draft"], retrieval)
            if attempt["status"] in {"retrieved_stale", "retrieved_after_target"}:
                results["stale_draft_queries"] += 1

    results["target_query_count"] += len(result["query_pairs"])
    for pair in result["query_pairs"]:
        qdrant = pair.get("target_qdrant_duration_ms")
        if pair["draft_retrieval_present"]:
            results["paired_draft_query_count"] += 1
        if pair["draft_finished_before_target_qdrant"]:
            results["warm_ready_count"] += 1
        if pair["query_exact_match"]:
            results["exact_match_count"] += 1
        if pair["exact_warm_ready"]:
            results["exact_warm_ready_count"] += 1
            if qdrant is not None:
                results["target_qdrant_exact_warm"].append(float(qdrant))
        elif qdrant is not None:
            results["target_qdrant_other"].append(float(qdrant))
        if pair["lead_time_ms"] is not None:
            results["lead_time"].append(float(pair["lead_time_ms"]))
        if pair["result_source_overlap"] is not None:
            results["result_overlap"].append(float(pair["result_source_overlap"]))


def _stats(values: list[float], *, higher_is_better: bool = False) -> dict[str, Any]:
    return asdict(summarize(values, higher_is_better=higher_is_better))


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def build_summary(results: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": {
            "target_end_to_end_ms": _stats(results["target_e2e"]),
            "incomplete_target_end_to_end_ms": _stats(results["incomplete_target_e2e"]),
            "retrieval_by_role": {
                role: {
                    "encode_duration_ms": _stats(values["encode"]),
                    "qdrant_duration_ms": _stats(values["qdrant"]),
                    "total_duration_ms": _stats(values["total"]),
                }
                for role, values in results["retrieval"].items()
            },
            "llm_by_role": {
                role: {
                    "ttft_ms": _stats(values["ttft"]),
                    "server_prefill_ms": _stats(values["prefill"]),
                    "decode_tokens_per_second": _stats(
                        values["decode_rate"], higher_is_better=True
                    ),
                }
                for role, values in results["llm"].items()
            },
            "prefetch": {
                "target_query_count": results["target_query_count"],
                "draft_query_count": results["draft_query_count"],
                "paired_draft_query_count": results["paired_draft_query_count"],
                "coverage_rate": _rate(
                    results["paired_draft_query_count"],
                    results["target_query_count"],
                ),
                "warm_ready_rate": _rate(
                    results["warm_ready_count"], results["target_query_count"]
                ),
                "exact_match_rate": _rate(
                    results["exact_match_count"], results["target_query_count"]
                ),
                "exact_warm_ready_rate": _rate(
                    results["exact_warm_ready_count"],
                    results["target_query_count"],
                ),
                "lead_time_ms": _stats(results["lead_time"], higher_is_better=True),
                "result_source_overlap": _stats(
                    results["result_overlap"], higher_is_better=True
                ),
                "sync_to_draft_request_ms": _stats(results["sync_delay"]),
                "target_qdrant_ms_exact_warm_ready": _stats(
                    results["target_qdrant_exact_warm"]
                ),
                "target_qdrant_ms_other": _stats(results["target_qdrant_other"]),
                "cancelled_draft_generations": results["cancelled_draft_generations"],
                "draft_tool_call_early_stops": results[
                    "draft_tool_call_early_stops"
                ],
                "stale_draft_queries": results["stale_draft_queries"],
            },
        },
        "outcomes": {
            "target": results["target_statuses"],
            "draft_attempts": results["draft_statuses"],
        },
    }


def render_report(results: dict[str, Any]) -> None:
    summary = build_summary(results)
    metrics = summary["metrics"]
    e2e = metrics["target_end_to_end_ms"]
    print("\nPrefetched tool-call target latency")
    print_table(
        ("metric", "n", "mean", "median", "p95-worst"),
        [
            [
                "end-to-end (ms)",
                e2e["count"],
                format_metric(e2e["mean"]),
                format_metric(e2e["median"]),
                format_metric(e2e["p95_worst"]),
            ]
        ],
    )
    print_histograms(
        "End-to-end latency histogram (ms; valid target runs)",
        {"target": results["target_e2e"]},
    )

    incomplete_e2e = metrics["incomplete_target_end_to_end_ms"]
    print("\nIncomplete target latency (diagnostic only)")
    print_table(
        ("metric", "n", "mean", "median", "p95-worst"),
        [_summary_row("end-to-end (ms)", incomplete_e2e)],
    )

    print("\nObserved retrieval latency")
    rows: list[list[object]] = []
    for role in ("target", "draft"):
        for label, key in (
            ("encode (ms)", "encode_duration_ms"),
            ("Qdrant (ms)", "qdrant_duration_ms"),
            ("total (ms)", "total_duration_ms"),
        ):
            stats = metrics["retrieval_by_role"][role][key]
            rows.append(
                [
                    role,
                    label,
                    stats["count"],
                    format_metric(stats["mean"]),
                    format_metric(stats["median"]),
                    format_metric(stats["p95_worst"]),
                ]
            )
    print_table(("role", "metric", "n", "mean", "median", "p95-worst"), rows)
    print_histograms(
        "Qdrant RPC latency histogram (ms; completed queries)",
        {
            role: results["retrieval"][role]["qdrant"]
            for role in ("target", "draft")
        },
    )

    print("\nConcurrent LLM diagnostics")
    rows = []
    for role in ("target", "draft"):
        for label, key in (
            ("TTFT (ms)", "ttft_ms"),
            ("prefill (ms)", "server_prefill_ms"),
            ("decode (tok/s)", "decode_tokens_per_second"),
        ):
            stats = metrics["llm_by_role"][role][key]
            rows.append(
                [
                    role,
                    label,
                    stats["count"],
                    format_metric(stats["mean"]),
                    format_metric(stats["median"]),
                    format_metric(stats["p95_worst"]),
                ]
            )
    print_table(("role", "metric", "n", "mean", "median", "p95-worst"), rows)

    prefetch = metrics["prefetch"]
    print("\nPrefetch diagnostics")
    print_table(
        ("metric", "value"),
        [
            ["target queries", prefetch["target_query_count"]],
            ["draft queries", prefetch["draft_query_count"]],
            ["paired draft queries", prefetch["paired_draft_query_count"]],
            ["coverage (%)", _format_percent(prefetch["coverage_rate"])],
            ["warm-ready (%)", _format_percent(prefetch["warm_ready_rate"])],
            ["exact match (%)", _format_percent(prefetch["exact_match_rate"])],
            [
                "exact + warm-ready (%)",
                _format_percent(prefetch["exact_warm_ready_rate"]),
            ],
            ["cancelled draft generations", prefetch["cancelled_draft_generations"]],
            ["draft tool-call early stops", prefetch["draft_tool_call_early_stops"]],
            ["stale draft queries", prefetch["stale_draft_queries"]],
        ],
    )
    print_table(
        ("metric", "n", "mean", "median", "p95-worst"),
        [
            _summary_row("lead time (ms)", prefetch["lead_time_ms"]),
            _summary_row("result overlap", prefetch["result_source_overlap"]),
            _summary_row(
                "sync to draft request (ms)",
                prefetch["sync_to_draft_request_ms"],
            ),
            _summary_row(
                "target Qdrant exact+ready (ms)",
                prefetch["target_qdrant_ms_exact_warm_ready"],
            ),
            _summary_row(
                "target Qdrant other (ms)", prefetch["target_qdrant_ms_other"]
            ),
        ],
    )
    print("\nOutcome statuses")
    print_table(
        ("role", "status", "count"),
        [
            [role, status, count]
            for role, statuses in (
                ("target", summary["outcomes"]["target"]),
                ("draft", summary["outcomes"]["draft_attempts"]),
            )
            for status, count in (sorted(statuses.items()) or [("none", 0)])
        ],
    )


def _summary_row(label: str, stats: dict[str, Any]) -> list[object]:
    return [
        label,
        stats["count"],
        format_metric(stats["mean"]),
        format_metric(stats["median"]),
        format_metric(stats["p95_worst"]),
    ]


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}"


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
    results = _new_results()

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
                "execution": "parallel target with one draft query per synchronized transcript",
                "target_model_role": "main",
                "sync_policy": "strict latest-state after target retrieval",
                "draft_depth": 1,
                "agent_prompt": {
                    "text": SYSTEM_PROMPT,
                    "sha256": SYSTEM_PROMPT_SHA256,
                },
                "draft_stream_stop": "complete_valid_search_tool_call",
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
                            wiki_manifest,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                },
            },
        )
        with RunArtifacts("prefetched-toolcall", metadata) as artifacts:
            with ExitStack() as stack:
                servers = {
                    role: stack.enter_context(
                        ModelServer(
                            role,
                            binary,
                            model_paths[role],
                            port,
                            reasoning_budget=REASONING_BUDGET_TOKENS,
                        )
                    )
                    for role, port in (("main", MAIN_PORT), ("draft", DRAFT_PORT))
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
                progress = Progress("prefetched-toolcall", total)
                completed = 0
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
                            paired = run_benchmark(
                                question,
                                clients["main"],
                                clients["draft"],
                                retriever,
                                retriever,
                            )
                        record_result(results, paired)
                        artifacts.append(
                            {
                                "record_type": "prefetched_toolcall_run",
                                "repetition": repetition + 1,
                                "question": question.agent_value(),
                                **paired,
                            }
                        )
                        completed += 1
                        progress.update(
                            completed,
                            f"round={repetition + 1} question={question.id}",
                        )

            if not results["target_e2e"]:
                raise ExperimentError(
                    "prefetched-toolcall experiment produced no valid target runs"
                )
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
        "Repetitions per task", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(task_count, repetitions, model_specs=model_specs)
