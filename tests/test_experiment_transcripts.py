import json
import threading
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiment import (
    artifacts,
    independent_run,
    parallel,
    prefetched_toolcall,
    tooluse,
)
from fanoutqa.dataset import Question


class FakeServer:
    def __init__(self, role, *args, **kwargs) -> None:
        self.base_url = f"http://{role}"

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


class FakeClient:
    def __init__(self, base_url, **kwargs) -> None:
        self.role = base_url.rsplit("/", 1)[-1]

    def close(self) -> None:
        pass

    def stream_completion(self, messages, generation):
        return {
            "request": {
                "messages": deepcopy(messages),
                "stream": True,
                **deepcopy(generation),
            },
            "request_start_ns": 1_000_000_000,
            "first_decode_ns": 1_100_000_000,
            "request_end_ns": 2_000_000_000,
            "reasoning": f"thinking-{self.role}",
            "content": f"answer-{self.role}",
            "tool_calls": [],
            "chunks": [],
            "usage": {"completion_tokens": 10},
            "timing": {
                "ttft_ms": 100.0,
                "decode_tokens_per_second": 10.0,
                "server_decode_tokens": 10,
            },
        }


def _patch_inference_runtime(module, tmp_path, monkeypatch) -> Question:
    question = Question("q1", "question?", ("category",))
    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(module, "select_benchmark_questions", lambda count: [question])
    monkeypatch.setattr(
        module,
        "require_models",
        lambda: {"main": Path("/models/main"), "draft": Path("/models/draft")},
    )
    monkeypatch.setattr(
        module, "llama_server_binary", lambda: ("llama-server", {"build": 9000})
    )
    monkeypatch.setattr(module, "ModelServer", FakeServer)
    monkeypatch.setattr(module, "LlamaCppClient", FakeClient)
    monkeypatch.setattr(module, "render_report", lambda *args: None)
    return question


def _run_rows(root: Path, experiment: str) -> tuple[dict, list[dict], dict]:
    run_dir = next((root / experiment).iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text().splitlines()
    ]
    summary = json.loads((run_dir / "summary.json").read_text())
    return manifest, rows, summary


def test_parallel_persists_one_paired_record_per_benchmark_run(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inference_runtime(parallel, tmp_path, monkeypatch)

    parallel.run(1, 2)

    manifest, rows, summary = _run_rows(tmp_path, "parallel")
    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 2
    assert len(rows) == 2
    assert set(rows[0]["model_calls"]) == {"main", "draft"}
    assert rows[0]["question"]["question"] == "question?"
    assert (
        rows[0]["model_calls"]["main"]["request"]["messages"][1]["content"]
        == "question?"
    )
    assert summary["metrics"]["combined_decode_tokens_per_second"]["count"] == 2


def test_independent_run_persists_each_model_call(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_inference_runtime(independent_run, tmp_path, monkeypatch)

    independent_run.run(1, 2)

    manifest, rows, summary = _run_rows(tmp_path, "independent-run")
    assert manifest["record_count"] == 4
    assert [row["model_role"] for row in rows] == [
        "main",
        "main",
        "draft",
        "draft",
    ]
    assert all(row["record_type"] == "model_completion" for row in rows)
    assert summary["metrics"]["by_role"]["draft"]["ttft_ms"]["count"] == 2


def test_tooluse_persists_model_turns_queries_and_results(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = _patch_inference_runtime(tooluse, tmp_path, monkeypatch)
    environment = SimpleNamespace(
        paths=SimpleNamespace(model_dir=tmp_path / "model", bundle_dir=tmp_path),
        database=SimpleNamespace(
            config=SimpleNamespace(
                collection_name="wikipedia", endpoint="http://qdrant"
            )
        ),
        manifest={
            "dataset": {"resolved_revision": "dataset-revision"},
            "model": {"resolved_revision": "model-revision"},
        },
        points_count=123,
        incomplete=False,
    )

    @contextmanager
    def wikipedia_environment(*args, **kwargs):
        yield environment

    class FakeRetriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def search(self, query):
            return {"query": query, "results": []}

    class FakeAgentRunner:
        def __init__(self, client, *args, **kwargs) -> None:
            self.role = client.role

        def run(self, selected):
            query = f"query-{self.role}"
            retrieval = {
                "query": query,
                "encode_duration_ms": 1.0,
                "qdrant_duration_ms": 2.0,
                "results": [
                    {
                        "rank": 1,
                        "title": "Result",
                        "snippet": "evidence",
                        "truncated": False,
                    }
                ],
            }
            return {
                "terminal_status": "final",
                "final_response": f"answer-{self.role}",
                "error": None,
                "search_count": 1,
                "llm_calls": [
                    {
                        "request": {
                            "messages": [
                                {"role": "user", "content": selected.question}
                            ]
                        },
                        "reasoning": "find evidence",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "search",
                                    "arguments": json.dumps({"query": query}),
                                },
                            }
                        ],
                        "usage": {"completion_tokens": 7},
                    },
                    {
                        "request": {
                            "messages": [
                                {"role": "tool", "content": "evidence"}
                            ]
                        },
                        "reasoning": "done",
                        "content": f"answer-{self.role}",
                        "tool_calls": [],
                        "usage": {"completion_tokens": 3},
                    },
                ],
                "retrieval_calls": [retrieval],
                "timing": {"end_to_end_ms": 10.0},
            }

    monkeypatch.setattr(tooluse, "prepare_wikipedia", wikipedia_environment)
    monkeypatch.setattr(tooluse, "Encoder", lambda *args, **kwargs: object())
    monkeypatch.setattr(tooluse, "TimedRetriever", FakeRetriever)
    monkeypatch.setattr(tooluse, "AgentRunner", FakeAgentRunner)

    tooluse.run(1, 1)

    manifest, rows, summary = _run_rows(tmp_path, "tooluse")
    assert manifest["record_count"] == 2
    assert manifest["parameters"]["wikipedia"]["points_count"] == 123
    assert {row["model_role"] for row in rows} == {"main", "draft"}
    outcome = rows[0]["outcome"]
    assert outcome["llm_calls"][0]["reasoning"] == "find evidence"
    assert outcome["retrieval_calls"][0]["query"].startswith("query-")
    assert outcome["retrieval_calls"][0]["results"][0]["snippet"] == "evidence"
    assert summary["roles"]["main"]["valid_metrics"]["queries_per_run"]["count"] == 1


def test_prefetched_toolcall_persists_target_draft_sync_and_latency(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = Question("q1", "question?", ("category",))
    draft_retrieved = threading.Event()
    environment = SimpleNamespace(
        paths=SimpleNamespace(model_dir=tmp_path / "model", bundle_dir=tmp_path),
        database=SimpleNamespace(
            config=SimpleNamespace(
                collection_name="wikipedia", endpoint="http://qdrant"
            )
        ),
        manifest={"dataset": {}, "model": {}},
        points_count=123,
        incomplete=False,
    )

    @contextmanager
    def wikipedia_environment(*args, **kwargs):
        yield environment

    def call(messages, *, query=None, content="", cancelled=False):
        tool_calls = []
        if query:
            tool_calls = [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "search",
                        "arguments": json.dumps({"query": query}),
                    },
                }
            ]
        return {
            "request": {"messages": deepcopy(messages)},
            "request_start_ns": 1_000_000,
            "request_end_ns": 5_000_000,
            "first_decode_ns": 2_000_000,
            "chunks": [],
            "reasoning": "",
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": "tool_calls" if tool_calls else "stop",
            "cancelled": cancelled,
            "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            "timing": {
                "request_wall_ms": 4.0,
                "ttft_ms": 1.0,
                "server_prompt_ms": 1.0,
                "server_decode_ms": 2.0,
                "decode_tokens_per_second": 1.5,
                "server_decode_tokens": 3,
            },
        }

    class Client:
        def __init__(self, base_url, **kwargs) -> None:
            self.role = base_url.rsplit("/", 1)[-1]
            self.calls = 0

        def close(self):
            pass

        def stream_completion(self, messages, generation, *, cancel_event=None):
            if messages == [
                {"role": "user", "content": prefetched_toolcall.FIXED_LLM_WARMUP}
            ]:
                return call(messages, content="warm")
            self.calls += 1
            if self.role == "main":
                if self.calls == 1:
                    assert draft_retrieved.wait(timeout=1)
                    return call(messages, query="same")
                return call(messages, content="answer")
            if self.calls == 1:
                return call(messages, query="same")
            assert cancel_event is not None
            cancel_event.wait(timeout=1)
            return call(messages, cancelled=True)

    class Retriever:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def search(self, query):
            if threading.current_thread().name == "prefetched-toolcall-draft":
                draft_retrieved.set()
                start, end = 10_000_000, 20_000_000
            else:
                start, end = 30_000_000, 40_000_000
            return {
                "query": query,
                "encode_start_ns": start - 2_000_000,
                "encode_end_ns": start,
                "encode_duration_ms": 2.0,
                "qdrant_start_ns": start,
                "qdrant_end_ns": end,
                "qdrant_duration_ms": 10.0,
                "results": [
                    {
                        "rank": 1,
                        "source_id": "source",
                        "title": "Result",
                        "url": "https://example.test",
                        "score": 1.0,
                        "snippet": "evidence",
                    }
                ],
            }

    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        prefetched_toolcall, "select_benchmark_questions", lambda count: [question]
    )
    monkeypatch.setattr(
        prefetched_toolcall,
        "require_models",
        lambda: {"main": Path("/models/main"), "draft": Path("/models/draft")},
    )
    monkeypatch.setattr(
        prefetched_toolcall,
        "llama_server_binary",
        lambda: ("llama-server", {"build": 9000}),
    )
    monkeypatch.setattr(prefetched_toolcall, "ModelServer", FakeServer)
    monkeypatch.setattr(prefetched_toolcall, "LlamaCppClient", Client)
    monkeypatch.setattr(prefetched_toolcall, "prepare_wikipedia", wikipedia_environment)
    monkeypatch.setattr(
        prefetched_toolcall, "Encoder", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(prefetched_toolcall, "TimedRetriever", Retriever)
    monkeypatch.setattr(prefetched_toolcall, "render_report", lambda *args: None)

    prefetched_toolcall.run(1, 1)

    manifest, rows, summary = _run_rows(tmp_path, "prefetched-toolcall")
    assert manifest["status"] == "completed"
    assert manifest["parameters"]["target_model_role"] == "main"
    assert len(rows) == 1
    assert rows[0]["target_outcome"]["terminal_status"] == "final"
    assert rows[0]["sync_events"][1]["source"] == "target_retrieval"
    assert rows[0]["query_pairs"][0]["exact_warm_ready"] is True
    assert rows[0]["draft_attempts"][0]["retrieval_call"]["query"] == "same"
    assert summary["metrics"]["target_end_to_end_ms"]["count"] == 1
    assert (
        summary["metrics"]["retrieval_by_role"]["draft"]["qdrant_duration_ms"]["count"]
        == 1
    )
