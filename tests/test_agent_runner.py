from typing import Any

import httpx

from agent.runner import (
    AgentRunner,
    LlamaCppClient,
    metric_deltas,
    parse_action,
    summarize_llm_call,
)
from fanoutqa.dataset import Question


def call(content: str, reasoning: str = "thinking") -> dict[str, Any]:
    chunks = [
        {
            "sequence": 0,
            "channel": "reasoning",
            "text": reasoning,
            "received_ns": 2_000_000,
            "inter_arrival_ns": 1_000_000,
        },
        {
            "sequence": 1,
            "channel": "content",
            "text": content,
            "received_ns": 4_000_000,
            "inter_arrival_ns": 2_000_000,
        },
    ]
    value = {
        "request_start_ns": 1_000_000,
        "request_end_ns": 5_000_000,
        "chunks": chunks,
        "reasoning": reasoning,
        "content": content,
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


class LLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.messages: list[list[dict[str, str]]] = []

    def stream_completion(self, messages: list[dict[str, str]], generation: dict[str, Any]):
        self.messages.append([dict(item) for item in messages])
        return call(next(self.responses))


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


def test_parser_handles_search_final_split_content_and_malformed_fallback() -> None:
    assert parse_action("prefix <search>alpha</search> suffix").kind == "search"
    assert parse_action("<final>answer</final>").text == "answer"
    # Llama client concatenates content chunks before parsing, so split tags remain valid.
    assert parse_action("<sea" + "rch>alpha</sea" + "rch>").text == "alpha"
    malformed = parse_action("untagged answer")
    assert malformed.kind == "final"
    assert malformed.protocol_warning is not None


def test_reasoning_not_added_to_history_and_search_limit_terminates() -> None:
    llm = LLM(["<search>one</search>", "<search>two</search>"])
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
    assert all(item["content"] != "thinking" for item in llm.messages[1])
    assert {item["content"] for item in llm.messages[1] if item["role"] == "assistant"} == {
        "<search>one</search>"
    }


def test_call_timing_and_metrics_delta() -> None:
    timing = call("<final>x</final>")["timing"]
    assert timing["ttft_ms"] == 1
    assert timing["request_wall_ms"] == 4
    assert timing["server_prompt_ms"] == 1
    assert timing["server_decode_ms"] == 2
    assert timing["unaccounted_client_server_ms"] == 1
    assert metric_deltas(
        {"llamacpp:prompt_tokens_total": 10},
        {"llamacpp:prompt_tokens_total": 14},
    ) == {"prompt_tokens": 4}


def test_stream_keeps_reasoning_and_content_chunks_separate() -> None:
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
                'data: {"choices":[{"delta":{"content":"<sea"}}]}',
                'data: {"choices":[{"delta":{"content":"rch>x</search>"}}]}',
                "data: [DONE]",
                "",
            ]
        )
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    llm = LlamaCppClient("http://test", client=client, clock_ns=Clock())

    result = llm.stream_completion([{"role": "user", "content": "q"}], {})

    assert result["reasoning"] == "why"
    assert result["content"] == "<search>x</search>"
    assert [item["channel"] for item in result["chunks"]] == [
        "reasoning",
        "content",
        "content",
    ]
    assert result["action_ready_ns"] == result["chunks"][-1]["received_ns"]
    assert [item["received_ns"] for item in result["chunks"]] == sorted(
        item["received_ns"] for item in result["chunks"]
    )
