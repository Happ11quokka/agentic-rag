import json
from typing import Any

import httpx

from agent.runner import (
    AgentRunner,
    LlamaCppClient,
    metric_deltas,
    summarize_llm_call,
)
from fanoutqa.dataset import Question


def call(
    content: str,
    reasoning: str = "thinking",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    chunks = []
    if reasoning:
        chunks.append(
            {
                "sequence": len(chunks),
                "channel": "reasoning",
                "text": reasoning,
                "received_ns": 2_000_000,
                "inter_arrival_ns": 1_000_000,
            }
        )
    if content:
        chunks.append(
            {
                "sequence": len(chunks),
                "channel": "content",
                "text": content,
                "received_ns": 4_000_000,
                "inter_arrival_ns": 2_000_000,
            }
        )
    if tool_calls:
        chunks.append(
            {
                "sequence": len(chunks),
                "channel": "tool_call",
                "text": tool_calls[0]["function"]["arguments"],
                "received_ns": 4_000_000,
                "inter_arrival_ns": 2_000_000,
            }
        )
    value = {
        "request_start_ns": 1_000_000,
        "request_end_ns": 5_000_000,
        "chunks": chunks,
        "reasoning": reasoning,
        "content": content,
        "tool_calls": tool_calls or [],
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": 2, "completion_tokens": 3},
        "metrics_delta": {
            "prompt_tokens": 2,
            "decode_tokens": 3,
            "prompt_seconds": 0.001,
            "decode_seconds": 0.002,
        },
    }
    value["timing"] = summarize_llm_call(value)
    return value


def search_call(query: str, identifier: str = "call_1") -> dict[str, Any]:
    return call(
        "",
        tool_calls=[
            {
                "id": identifier,
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps({"query": query}),
                },
            }
        ],
        finish_reason="tool_calls",
    )


class LLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = iter(responses)
        self.messages: list[list[dict[str, Any]]] = []
        self.generations: list[dict[str, Any]] = []

    def stream_completion(
        self, messages: list[dict[str, Any]], generation: dict[str, Any]
    ) -> dict[str, Any]:
        self.messages.append([dict(item) for item in messages])
        self.generations.append(generation)
        return next(self.responses)


class Retriever:
    def search(self, query: str) -> dict[str, Any]:
        return {
            "query": query,
            "encode_duration_ms": 1.0,
            "qdrant_duration_ms": 2.0,
            "results": [],
        }


class Clock:
    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        self.now += 10_000_000
        return self.now


def test_reasoning_not_added_to_history_and_search_limit_terminates() -> None:
    llm = LLM([search_call("one"), search_call("two", "call_2")])
    runner = AgentRunner(
        llm,
        Retriever(),
        generation={},
        max_searches=2,
        question_timeout_seconds=10,
        clock_ns=Clock(),
    )

    result = runner.run(Question("id", "question", ("cat",)))

    assert result["terminal_status"] == "search_limit"
    assert result["search_count"] == 2
    assert llm.generations[0]["parallel_tool_calls"] is False
    assert llm.generations[0]["tools"][0]["function"]["name"] == "search"
    assistant = next(item for item in llm.messages[1] if item["role"] == "assistant")
    tool = next(item for item in llm.messages[1] if item["role"] == "tool")
    assert "reasoning_content" not in assistant
    assert assistant["content"] == ""
    assert assistant["tool_calls"][0]["id"] == "call_1"
    assert tool["tool_call_id"] == "call_1"


def test_invalid_or_multiple_tool_calls_are_protocol_errors() -> None:
    invalid = search_call("one")
    invalid["tool_calls"].append(search_call("two", "call_2")["tool_calls"][0])
    runner = AgentRunner(
        LLM([invalid]),
        Retriever(),
        generation={},
        clock_ns=Clock(),
    )

    result = runner.run(Question("id", "question", ()))

    assert result["terminal_status"] == "protocol_error"
    assert result["search_count"] == 0
    assert "exactly one" in result["error"]


def test_plain_content_is_final_answer() -> None:
    runner = AgentRunner(
        LLM([call("answer")]),
        Retriever(),
        generation={},
        clock_ns=Clock(),
    )

    result = runner.run(Question("id", "question", ()))

    assert result["terminal_status"] == "final"
    assert result["final_response"] == "answer"


def test_call_timing_and_metrics_delta() -> None:
    timing = call("x")["timing"]
    assert timing["ttft_ms"] == 1
    assert timing["request_wall_ms"] == 4
    assert timing["server_prompt_ms"] == 1
    assert timing["server_decode_ms"] == 2
    assert timing["unaccounted_client_server_ms"] == 1
    assert metric_deltas(
        {"llamacpp:prompt_tokens_total": 10},
        {"llamacpp:prompt_tokens_total": 14},
    ) == {"prompt_tokens": 4}


def test_stream_reconstructs_native_tool_call_fragments() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        if request.url.path == "/metrics":
            requests += 1
            return httpx.Response(
                200,
                text=f"llamacpp:prompt_tokens_total {requests}\n",
            )
        body = "\n".join(
            [
                'data: {"choices":[{"delta":{"reasoning_content":"why"}}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{"}}]}}]}',
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\\"query\\\":\\\"x\\\"}"}}]}}]}',
                'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
                'data: {"choices":[],"usage":{"completion_tokens":7}}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlamaCppClient("http://test", client=client, clock_ns=Clock())
    messages = [{"role": "user", "content": "q"}]

    result = llm.stream_completion(messages, {"seed": 42})
    messages[0]["content"] = "changed later"

    assert result["request"]["messages"] == [{"role": "user", "content": "q"}]
    assert result["request"]["seed"] == 42
    assert result["request"]["stream"] is True
    assert result["reasoning"] == "why"
    assert result["content"] == ""
    assert result["tool_calls"] == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search", "arguments": '{"query":"x"}'},
        }
    ]
    assert result["finish_reason"] == "tool_calls"
    assert result["usage"]["completion_tokens"] == 7
    assert [item["channel"] for item in result["chunks"]] == [
        "reasoning",
        "tool_call",
        "tool_call",
    ]
    assert result["action_ready_ns"] is not None
    assert [item["received_ns"] for item in result["chunks"]] == sorted(
        item["received_ns"] for item in result["chunks"]
    )
