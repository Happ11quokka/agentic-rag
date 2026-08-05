import json
import threading
from typing import Any

import httpx
import pytest

from agent.runner import (
    SYSTEM_PROMPT,
    AgentRunner,
    LlamaCppClient,
    initial_messages,
    metric_deltas,
    summarize_llm_call,
    tool_generation,
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


def test_initial_messages_use_exact_shared_agent_prompt() -> None:
    expected = """Use thinking mode. /think

Answer FanOutQA questions using semantic search over English Wikipedia.

At every assistant turn, inspect the original question and all previous search
calls and results. Then choose exactly one action:

- If any requested item or attribute lacks explicit evidence in the search
  results, call search exactly once for the next missing atomic fact. Output no
  answer text.
- If every requested item and attribute has explicit evidence, output only the
  final answer, preserving the requested list or mapping.

Never use memory to fill missing evidence. Never give a partial answer. Each
search query must name exactly one entity and one missing attribute. For ranking
or list questions, first verify the entity set, then verify the requested
attribute for each entity. Do not repeat equivalent searches.

Reference date: 2023-11-20."""

    assert SYSTEM_PROMPT == expected
    assert initial_messages("question?") == [
        {"role": "system", "content": expected},
        {"role": "user", "content": "question?"},
    ]


def test_agent_runners_share_initial_messages_and_tool_generation() -> None:
    clients = [LLM([call("answer")]), LLM([call("answer")])]
    generation = {"seed": 42, "temperature": 0.6}

    for client in clients:
        AgentRunner(
            client,
            Retriever(),
            generation=generation,
            clock_ns=Clock(),
        ).run(Question("id", "question?", ()))

    assert clients[0].messages[0] == clients[1].messages[0]
    assert clients[0].generations[0] == clients[1].generations[0]
    assert clients[0].generations[0] == tool_generation(generation)


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


def test_retrieval_callback_receives_post_tool_transcript() -> None:
    snapshots: list[tuple[list[dict[str, Any]], dict[str, Any]]] = []
    runner = AgentRunner(
        LLM([search_call("one"), call("answer")]),
        Retriever(),
        generation={},
        clock_ns=Clock(),
        on_retrieval_complete=lambda messages, retrieval: snapshots.append(
            (messages, retrieval)
        ),
    )

    result = runner.run(Question("id", "question", ()))

    assert result["terminal_status"] == "final"
    assert len(snapshots) == 1
    messages, retrieval = snapshots[0]
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "tool"
    assert retrieval["query"] == "one"


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
    assert "client_stop_reason" not in result
    assert "stop_after_complete_tool_call" not in result["request"]
    assert [item["channel"] for item in result["chunks"]] == [
        "reasoning",
        "tool_call",
        "tool_call",
    ]
    assert result["action_ready_ns"] is not None
    assert [item["received_ns"] for item in result["chunks"]] == sorted(
        item["received_ns"] for item in result["chunks"]
    )


def test_stream_stops_only_when_fragmented_tool_arguments_become_valid() -> None:
    body = "\n".join(
        [
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{"}}]}}]}',
            'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"query\\":\\"x\\"}"}}]}}]}',
            'data: {"choices":[{"delta":{"content":"must not be consumed"}}]}',
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            'data: {"choices":[],"usage":{"completion_tokens":7}}',
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    llm = LlamaCppClient(
        "http://test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock_ns=Clock(),
    )
    result = llm.stream_completion(
        [{"role": "user", "content": "q"}],
        {"seed": 42},
        stop_after_complete_tool_call=True,
    )

    assert result["tool_calls"][0]["function"]["arguments"] == '{"query":"x"}'
    assert result["content"] == ""
    assert result["finish_reason"] is None
    assert result["usage"] == {}
    assert result["cancelled"] is False
    assert result["client_stop_reason"] == "complete_tool_call"
    assert result["action_ready_ns"] == result["chunks"][-1]["received_ns"]
    assert "stop_after_complete_tool_call" not in result["request"]


@pytest.mark.parametrize(
    "delta",
    [
        '{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{\\"query\\":"}}]}',
        '{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{\\"query\\":}"}}]}',
        '{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{\\"query\\":\\"x\\"}"}},{"index":1,"id":"call_2","function":{"name":"search","arguments":"{\\"query\\":\\"y\\"}"}}]}',
        '{"content":"answer","tool_calls":[{"index":0,"id":"call_1","function":{"name":"search","arguments":"{\\"query\\":\\"x\\"}"}}]}',
    ],
)
def test_stream_does_not_early_stop_invalid_tool_call_shapes(delta: str) -> None:
    body = "\n".join(
        [
            f'data: {{"choices":[{{"delta":{delta}}}]}}',
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            'data: {"choices":[],"usage":{"completion_tokens":7}}',
            "data: [DONE]",
            "",
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    llm = LlamaCppClient(
        "http://test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock_ns=Clock(),
    )
    result = llm.stream_completion(
        [{"role": "user", "content": "q"}],
        {},
        stop_after_complete_tool_call=True,
    )

    assert result["client_stop_reason"] is None
    assert result["finish_reason"] == "tool_calls"
    assert result["usage"]["completion_tokens"] == 7


def test_stream_can_be_cancelled_without_affecting_normal_call_shape() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/metrics":
            return httpx.Response(200, text="")
        return httpx.Response(
            200,
            text='data: {"choices":[{"delta":{"content":"late"}}]}\n\n',
            headers={"content-type": "text/event-stream"},
        )

    cancel = threading.Event()
    cancel.set()
    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlamaCppClient("http://test", client=client, clock_ns=Clock())

    result = llm.stream_completion(
        [{"role": "user", "content": "q"}],
        {},
        cancel_event=cancel,
        stop_after_complete_tool_call=True,
    )

    assert result["cancelled"] is True
    assert result["client_stop_reason"] is None
    assert result["chunks"] == []
    assert result["action_ready_ns"] is None
