import json
import threading
from copy import deepcopy
from typing import Any

from experiment import prefetched_toolcall
from fanoutqa.dataset import Question


def model_call(
    messages: list[dict[str, Any]],
    *,
    query: str | None = None,
    content: str = "",
    cancelled: bool = False,
    client_stop_reason: str | None = None,
    request_start_ns: int = 1_000_000,
) -> dict[str, Any]:
    tool_calls = []
    if query is not None:
        tool_calls = [
            {
                "id": f"call-{query}",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": query}),
                },
            }
        ]
    return {
        "request": {"messages": deepcopy(messages)},
        "request_start_ns": request_start_ns,
        "request_end_ns": request_start_ns + 4_000_000,
        "first_decode_ns": request_start_ns + 1_000_000,
        "chunks": [],
        "reasoning": "",
        "content": content,
        "tool_calls": tool_calls,
        "finish_reason": "tool_calls" if tool_calls else "stop",
        "cancelled": cancelled,
        "client_stop_reason": client_stop_reason,
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        "timing": {
            "request_wall_ms": 4.0,
            "ttft_ms": 1.0,
            "server_prompt_ms": 1.0,
            "server_decode_ms": 2.0,
            "decode_tokens_per_second": 1.5,
        },
    }


def retrieval(
    query: str,
    *,
    encode_start_ns: int = 10_000_000,
    qdrant_start_ns: int = 12_000_000,
    qdrant_end_ns: int = 14_000_000,
    source_ids: tuple[str, ...] = ("a",),
) -> dict[str, Any]:
    return {
        "query": query,
        "encode_start_ns": encode_start_ns,
        "encode_end_ns": qdrant_start_ns,
        "encode_duration_ms": (qdrant_start_ns - encode_start_ns) / 1_000_000,
        "qdrant_start_ns": qdrant_start_ns,
        "qdrant_end_ns": qdrant_end_ns,
        "qdrant_duration_ms": (qdrant_end_ns - qdrant_start_ns) / 1_000_000,
        "results": [
            {
                "rank": rank,
                "source_id": value,
                "title": value,
                "url": f"https://example.test/{value}",
                "score": 1.0,
                "snippet": value,
            }
            for rank, value in enumerate(source_ids, start=1)
        ],
    }


def test_coordinator_cancels_old_generation_and_runs_one_latest_query() -> None:
    first_started = threading.Event()
    retrieved = threading.Event()

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def stream_completion(
            self,
            messages,
            generation,
            *,
            cancel_event,
            stop_after_complete_tool_call,
        ):
            assert stop_after_complete_tool_call is True
            self.calls += 1
            if self.calls == 1:
                first_started.set()
                assert cancel_event.wait(timeout=1)
                return model_call(messages, cancelled=True)
            return model_call(messages, query="new query")

    class Retriever:
        def __init__(self) -> None:
            self.queries: list[str] = []

        def search(self, query):
            self.queries.append(query)
            retrieved.set()
            return retrieval(query)

    barrier = threading.Barrier(2)
    client = Client()
    retriever = Retriever()
    coordinator = prefetched_toolcall.DraftCoordinator(
        client,
        retriever,
        generation={},
        start_barrier=barrier,
    )
    coordinator.publish([{"role": "user", "content": "old"}], source="initial")
    coordinator.start()
    barrier.wait()
    assert first_started.wait(timeout=1)
    coordinator.publish(
        [{"role": "user", "content": "new"}],
        source="target_retrieval",
        target_retrieval_index=1,
    )
    assert retrieved.wait(timeout=1)
    with coordinator.condition:
        assert coordinator.condition.wait_for(
            lambda: len(coordinator.attempts) == 2, timeout=1
        )

    attempts, sync_events = coordinator.stop()

    assert [item["sync_id"] for item in sync_events] == [0, 1]
    assert [item["status"] for item in attempts] == ["superseded", "retrieved"]
    assert attempts[0]["superseded_by_sync_id"] == 1
    assert attempts[1]["model_call"]["request"]["messages"] == [
        {"role": "user", "content": "new"}
    ]
    assert retriever.queries == ["new query"]


def test_coordinator_stop_breaks_start_barrier_cleanly() -> None:
    coordinator = prefetched_toolcall.DraftCoordinator(
        object(),
        object(),
        generation={},
        start_barrier=threading.Barrier(2),
    )
    coordinator.publish([{"role": "user", "content": "q"}], source="initial")
    coordinator.start()

    attempts, sync_events = coordinator.stop()

    assert attempts == []
    assert len(sync_events) == 1


def test_pair_retrievals_measures_readiness_exact_match_and_overlap() -> None:
    target = retrieval(
        "same",
        qdrant_start_ns=300_000_000,
        qdrant_end_ns=340_000_000,
        source_ids=("a", "b"),
    )
    draft = retrieval(
        "same",
        qdrant_start_ns=100_000_000,
        qdrant_end_ns=200_000_000,
        source_ids=("b", "c"),
    )
    pairs = prefetched_toolcall.pair_retrievals(
        {"retrieval_calls": [target]},
        [{"sync_id": 0, "status": "retrieved", "retrieval_call": draft}],
    )

    assert pairs == [
        {
            "target_retrieval_index": 1,
            "draft_sync_id": 0,
            "draft_status": "retrieved",
            "target_query": "same",
            "draft_query": "same",
            "draft_retrieval_present": True,
            "draft_finished_before_target_qdrant": True,
            "query_exact_match": True,
            "exact_warm_ready": True,
            "lead_time_ms": 100.0,
            "result_source_overlap": 0.5,
            "target_qdrant_duration_ms": 40.0,
            "draft_qdrant_duration_ms": 100.0,
        }
    ]


def test_record_result_separates_role_latencies_and_coverage(
    capsys: Any,
) -> None:
    target_retrieval = retrieval(
        "same", qdrant_start_ns=30_000_000, qdrant_end_ns=40_000_000
    )
    draft_retrieval = retrieval("same", qdrant_end_ns=20_000_000)
    target_call = model_call([], content="answer")
    draft_call = model_call(
        [], query="same", client_stop_reason="complete_tool_call"
    )
    result = {
        "target_outcome": {
            "terminal_status": "final",
            "timing": {"end_to_end_ms": 50.0},
            "llm_calls": [target_call],
            "retrieval_calls": [target_retrieval],
        },
        "draft_attempts": [
            {
                "sync_id": 0,
                "sync_published_ns": 0,
                "status": "retrieved",
                "model_call": draft_call,
                "retrieval_call": draft_retrieval,
            }
        ],
    }
    result["query_pairs"] = prefetched_toolcall.pair_retrievals(
        result["target_outcome"], result["draft_attempts"]
    )
    results = prefetched_toolcall._new_results()

    prefetched_toolcall.record_result(results, result)
    summary = prefetched_toolcall.build_summary(results)

    metrics = summary["metrics"]
    assert metrics["target_end_to_end_ms"]["mean"] == 50
    assert metrics["retrieval_by_role"]["target"]["qdrant_duration_ms"]["count"] == 1
    assert metrics["retrieval_by_role"]["draft"]["qdrant_duration_ms"]["count"] == 1
    assert metrics["prefetch"]["coverage_rate"] == 1
    assert metrics["prefetch"]["exact_warm_ready_rate"] == 1
    assert metrics["prefetch"]["draft_tool_call_early_stops"] == 1
    assert metrics["prefetch"]["cancelled_draft_generations"] == 0

    prefetched_toolcall.render_report(results)
    output = capsys.readouterr().out
    assert "End-to-end latency histogram" in output
    assert "Qdrant RPC latency histogram" in output
    assert "#" in output


def test_early_stop_cancellation_and_stale_counts_are_independent() -> None:
    result = {
        "target_outcome": {
            "terminal_status": "final",
            "timing": {"end_to_end_ms": 50.0},
            "llm_calls": [],
            "retrieval_calls": [],
        },
        "draft_attempts": [
            {
                "sync_id": 0,
                "sync_published_ns": 0,
                "status": "retrieved",
                "model_call": model_call(
                    [], query="early", client_stop_reason="complete_tool_call"
                ),
                "retrieval_call": retrieval("early"),
            },
            {
                "sync_id": 1,
                "sync_published_ns": 0,
                "status": "superseded",
                "model_call": model_call([], cancelled=True),
                "retrieval_call": None,
            },
            {
                "sync_id": 2,
                "sync_published_ns": 0,
                "status": "retrieved_stale",
                "model_call": model_call([], query="stale"),
                "retrieval_call": retrieval("stale"),
            },
        ],
        "query_pairs": [],
    }
    results = prefetched_toolcall._new_results()

    prefetched_toolcall.record_result(results, result)

    assert results["draft_tool_call_early_stops"] == 1
    assert results["cancelled_draft_generations"] == 1
    assert results["stale_draft_queries"] == 1


def test_run_benchmark_syncs_after_target_retrieval() -> None:
    draft_retrieved = threading.Event()
    second_draft_started = threading.Event()

    class TargetClient:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def stream_completion(self, messages, generation):
            self.requests.append((deepcopy(messages), deepcopy(generation)))
            self.calls += 1
            if self.calls == 1:
                return model_call(messages, query="same")
            assert second_draft_started.wait(timeout=1)
            return model_call(messages, content="answer")

    class DraftClient:
        def __init__(self) -> None:
            self.calls = 0
            self.requests = []

        def stream_completion(
            self,
            messages,
            generation,
            *,
            cancel_event,
            stop_after_complete_tool_call,
        ):
            assert stop_after_complete_tool_call is True
            self.requests.append((deepcopy(messages), deepcopy(generation)))
            self.calls += 1
            if self.calls == 1:
                return model_call(
                    messages,
                    query="same",
                    client_stop_reason="complete_tool_call",
                )
            second_draft_started.set()
            cancel_event.wait(timeout=1)
            return model_call(messages, cancelled=True)

    class DraftRetriever:
        def search(self, query):
            draft_retrieved.set()
            return retrieval(query, qdrant_end_ns=20_000_000)

    class TargetRetriever:
        def search(self, query):
            assert draft_retrieved.wait(timeout=1)
            return retrieval(
                query, qdrant_start_ns=30_000_000, qdrant_end_ns=40_000_000
            )

    target_client = TargetClient()
    draft_client = DraftClient()
    result = prefetched_toolcall.run_benchmark(
        Question("q1", "question?", ()),
        target_client,
        draft_client,
        TargetRetriever(),
        DraftRetriever(),
    )

    assert result["target_outcome"]["terminal_status"] == "final"
    assert [event["source"] for event in result["sync_events"]] == [
        "initial",
        "target_retrieval",
    ]
    assert result["sync_events"][1]["message_count"] == 4
    assert result["query_pairs"][0]["exact_warm_ready"] is True
    assert target_client.requests[0] == draft_client.requests[0]
    assert target_client.requests[1] == draft_client.requests[1]
    assert all(
        attempt["sync_prompt_hash"]
        == prefetched_toolcall.prompt_hash(
            attempt["model_call"]["request"]["messages"]
        )
        for attempt in result["draft_attempts"]
        if attempt["model_call"] is not None
    )
    assert (
        result["draft_attempts"][0]["model_call"]["client_stop_reason"]
        == "complete_tool_call"
    )
    assert (
        sum(
            attempt["retrieval_call"] is not None
            for attempt in result["draft_attempts"]
        )
        == 1
    )
