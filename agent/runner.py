from __future__ import annotations

import hashlib
import json
import math
import statistics
import threading
import time
from copy import deepcopy
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from fanoutqa.dataset import Question

from .retrieval import TimedRetriever, render_search_results

SYSTEM_PROMPT = """Use thinking mode. /think

Answer FanOutQA questions using semantic search over English Wikipedia.
These questions require evidence from multiple Wikipedia articles.

Maintain an internal coverage checklist. First identify the requested entity set
or ranking, then create one atomic fact to verify for every entity. Tool results
prove only facts they explicitly state; never fill missing facts from memory.

After each result, call search for the next unverified checklist item. Each search
must target exactly one entity and one missing attribute; never combine multiple
people or items in one query. Give a final answer only after every requested item
and attribute has explicit tool-result evidence. A top-five question normally
needs one list search plus five entity searches. Do not repeat equivalent searches.

When complete, answer only, preserving the requested list or mapping.
Reference date for the question is 2023-11-20."""

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search",
        "description": (
            "Search English Wikipedia passages for one atomic fact. "
            "Call once per missing checklist item."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "One standalone semantic query about exactly one "
                        "checklist item."
                    ),
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
METRIC_ALIASES = {
    "prompt_tokens": (
        "llamacpp:prompt_tokens_total",
        "llamacpp_prompt_tokens_total",
    ),
    "decode_tokens": (
        "llamacpp:tokens_predicted_total",
        "llamacpp_tokens_predicted_total",
    ),
    "prompt_seconds": (
        "llamacpp:prompt_seconds_total",
        "llamacpp_prompt_seconds_total",
    ),
    "decode_seconds": (
        "llamacpp:tokens_predicted_seconds_total",
        "llamacpp_tokens_predicted_seconds_total",
    ),
}


def initial_messages(question: Question | str) -> list[dict[str, Any]]:
    text = question.question if isinstance(question, Question) else question
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def prompt_hash(messages: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_prometheus(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{", 1)[0]
        try:
            values[name] = values.get(name, 0.0) + float(parts[-1])
        except ValueError:
            continue
    return values


def metric_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for output_name, aliases in METRIC_ALIASES.items():
        source = next((name for name in aliases if name in after), None)
        if source is not None:
            deltas[output_name] = max(0.0, after[source] - before.get(source, 0.0))
    return deltas


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_llm_call(call: dict[str, Any]) -> dict[str, Any]:
    chunks = call["chunks"]
    start = call["request_start_ns"]
    end = call["request_end_ns"]
    first = chunks[0]["received_ns"] if chunks else None
    intervals_ms = [item["inter_arrival_ns"] / 1_000_000 for item in chunks[1:]]
    reasoning = [item for item in chunks if item["channel"] == "reasoning"]
    content = [item for item in chunks if item["channel"] == "content"]
    tool_call = [item for item in chunks if item["channel"] == "tool_call"]
    action = [
        item for item in chunks if item["channel"] in {"content", "tool_call"}
    ]
    metrics = call.get("metrics_delta", {})
    prompt_ms = metrics.get("prompt_seconds", 0.0) * 1000
    decode_ms = metrics.get("decode_seconds", 0.0) * 1000
    prompt_tokens = metrics.get("prompt_tokens")
    decode_tokens = metrics.get("decode_tokens")
    wall_ms = (end - start) / 1_000_000
    return {
        "ttft_ms": None if first is None else (first - start) / 1_000_000,
        "request_wall_ms": wall_ms,
        "reasoning_duration_ms": _channel_duration_ms(reasoning),
        "content_action_duration_ms": _channel_duration_ms(action),
        "content_duration_ms": _channel_duration_ms(content),
        "tool_call_duration_ms": _channel_duration_ms(tool_call),
        "chunk_inter_arrival_p50_ms": (
            statistics.median(intervals_ms) if intervals_ms else None
        ),
        "chunk_inter_arrival_p95_ms": percentile(intervals_ms, 0.95),
        "chunk_inter_arrival_max_ms": max(intervals_ms) if intervals_ms else None,
        "server_prompt_tokens": prompt_tokens,
        "server_decode_tokens": decode_tokens,
        "server_prompt_ms": prompt_ms if "prompt_seconds" in metrics else None,
        "server_decode_ms": decode_ms if "decode_seconds" in metrics else None,
        "prompt_tokens_per_second": _rate(prompt_tokens, prompt_ms),
        "decode_tokens_per_second": _rate(decode_tokens, decode_ms),
        "unaccounted_client_server_ms": max(0.0, wall_ms - prompt_ms - decode_ms),
    }


def _channel_duration_ms(chunks: list[dict[str, Any]]) -> float | None:
    if not chunks:
        return None
    return (chunks[-1]["received_ns"] - chunks[0]["received_ns"]) / 1_000_000


def _rate(tokens: float | None, milliseconds: float) -> float | None:
    if tokens is None or milliseconds <= 0:
        return None
    return tokens / (milliseconds / 1000)


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 180,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.clock_ns = clock_ns
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(timeout_seconds, read=timeout_seconds)
        )
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def metrics(self) -> dict[str, float]:
        try:
            response = self.client.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return parse_prometheus(response.text)
        except httpx.HTTPError:
            return {}

    def stream_completion(
        self,
        messages: list[dict[str, Any]],
        generation: dict[str, Any],
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        before = self.metrics()
        request_start = self.clock_ns()
        chunks: list[dict[str, Any]] = []
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_call_parts: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        action_ready_ns: int | None = None
        extensions: dict[str, Any] = {}
        usage: dict[str, Any] = {}
        cancelled = False
        previous_ns = request_start
        payload = {
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload.update(generation)
        request = deepcopy(payload)

        with self.client.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json=payload
        ) as response:
            response.raise_for_status()
            for data in _sse_data(response.iter_lines()):
                if cancel_event is not None and cancel_event.is_set():
                    cancelled = True
                    break
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                extensions.update(
                    {
                        key: value
                        for key, value in event.items()
                        if key not in {"id", "object", "created", "model", "choices", "usage"}
                    }
                )
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                for channel, key in (
                    ("reasoning", "reasoning_content"),
                    ("content", "content"),
                ):
                    text = delta.get(key)
                    if not isinstance(text, str) or not text:
                        continue
                    received = self.clock_ns()
                    chunks.append(
                        {
                            "sequence": len(chunks),
                            "channel": channel,
                            "text": text,
                            "received_ns": received,
                            "inter_arrival_ns": received - previous_ns,
                        }
                    )
                    previous_ns = received
                    if channel == "reasoning":
                        reasoning_parts.append(text)
                    else:
                        content_parts.append(text)
                for fragment in delta.get("tool_calls") or []:
                    index = fragment.get("index")
                    if not isinstance(index, int) or index < 0:
                        continue
                    selected = tool_call_parts.setdefault(
                        index,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    identifier = fragment.get("id")
                    if isinstance(identifier, str):
                        selected["id"] += identifier
                    call_type = fragment.get("type")
                    if isinstance(call_type, str):
                        selected["type"] = call_type
                    function = fragment.get("function") or {}
                    streamed_text = ""
                    name = function.get("name")
                    if isinstance(name, str):
                        selected["function"]["name"] += name
                        streamed_text += name
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        selected["function"]["arguments"] += arguments
                        streamed_text += arguments
                    if streamed_text:
                        received = self.clock_ns()
                        chunks.append(
                            {
                                "sequence": len(chunks),
                                "channel": "tool_call",
                                "text": streamed_text,
                                "received_ns": received,
                                "inter_arrival_ns": received - previous_ns,
                            }
                        )
                        previous_ns = received
                value = choice.get("finish_reason")
                if isinstance(value, str):
                    finish_reason = value
                    if action_ready_ns is None:
                        action_ready_ns = self.clock_ns()
        request_end = self.clock_ns()
        if (
            action_ready_ns is None
            and not cancelled
            and (content_parts or tool_call_parts)
        ):
            action_ready_ns = request_end
        after = self.metrics()
        reasoning = "".join(reasoning_parts)
        content = "".join(content_parts)
        tool_calls = [tool_call_parts[index] for index in sorted(tool_call_parts)]
        call = {
            "request": request,
            "request_start_ns": request_start,
            "request_end_ns": request_end,
            "first_decode_ns": chunks[0]["received_ns"] if chunks else None,
            "reasoning_start_ns": reasoning and next(
                item["received_ns"] for item in chunks if item["channel"] == "reasoning"
            ) or None,
            "reasoning_end_ns": reasoning and next(
                item["received_ns"] for item in reversed(chunks) if item["channel"] == "reasoning"
            ) or None,
            "content_start_ns": content and next(
                item["received_ns"] for item in chunks if item["channel"] == "content"
            ) or None,
            "action_ready_ns": action_ready_ns,
            "chunks": chunks,
            "reasoning": reasoning,
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
            "cancelled": cancelled,
            "usage": usage,
            "response_extensions": extensions,
            "metrics_delta": metric_deltas(before, after),
        }
        call["timing"] = summarize_llm_call(call)
        return call


def _sse_data(lines: Iterable[str]) -> Iterable[str]:
    for line in lines:
        if line.startswith("data:"):
            yield line[5:].strip()


class AgentRunner:
    def __init__(
        self,
        llm: Any,
        retriever: TimedRetriever,
        *,
        generation: dict[str, Any],
        max_searches: int = 8,
        question_timeout_seconds: float | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        on_retrieval_complete: Callable[[list[dict[str, Any]], dict[str, Any]], None]
        | None = None,
    ) -> None:
        self.llm = llm
        self.retriever = retriever
        self.generation = {
            **generation,
            "tools": [SEARCH_TOOL],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
        }
        self.max_searches = max_searches
        self.question_timeout_seconds = question_timeout_seconds
        self.clock_ns = clock_ns
        self.on_retrieval_complete = on_retrieval_complete

    def run(self, question: Question) -> dict[str, Any]:
        messages = initial_messages(question)
        started = self.clock_ns()
        llm_calls: list[dict[str, Any]] = []
        retrieval_calls: list[dict[str, Any]] = []
        final_response = ""
        terminal_status = "server_error"
        error: str | None = None

        while True:
            if (
                self.question_timeout_seconds is not None
                and self.clock_ns() - started
                >= self.question_timeout_seconds * 1_000_000_000
            ):
                terminal_status = "timeout"
                break
            try:
                call = self.llm.stream_completion(messages, self.generation)
            except httpx.TimeoutException as exc:
                terminal_status, error = "timeout", str(exc)
                break
            except (httpx.HTTPError, OSError, ValueError, json.JSONDecodeError) as exc:
                detail = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    detail = f"{detail}: {exc.response.text}"
                terminal_status = "context_error" if "context" in detail.lower() else "server_error"
                error = detail
                break
            llm_calls.append(call)
            tool_calls = call.get("tool_calls") or []
            if not tool_calls:
                final_response = call["content"].strip()
                if not final_response:
                    terminal_status = "protocol_error"
                    error = "model returned neither a search tool call nor a final answer"
                    break
                terminal_status = "final"
                break
            try:
                tool_call_id, query = _parse_search_tool_call(call)
            except ValueError as exc:
                terminal_status, error = "protocol_error", str(exc)
                break
            if len(retrieval_calls) >= self.max_searches:
                terminal_status = "search_limit"
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": call["content"],
                    "tool_calls": tool_calls,
                }
            )
            try:
                retrieval = self.retriever.search(query)
            except Exception as exc:
                terminal_status, error = "retrieval_error", str(exc)
                break
            retrieval_calls.append(retrieval)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": render_search_results(retrieval),
                }
            )
            if self.on_retrieval_complete is not None:
                self.on_retrieval_complete(deepcopy(messages), deepcopy(retrieval))
            if len(retrieval_calls) >= self.max_searches:
                terminal_status = "search_limit"
                break

        ended = self.clock_ns()
        return {
            "terminal_status": terminal_status,
            "final_response": final_response,
            "error": error,
            "protocol_warning": None,
            "search_count": len(retrieval_calls),
            "llm_calls": llm_calls,
            "retrieval_calls": retrieval_calls,
            "timing": summarize_question(started, ended, llm_calls, retrieval_calls),
        }


def _parse_search_tool_call(call: dict[str, Any]) -> tuple[str, str]:
    tool_calls = call.get("tool_calls") or []
    if len(tool_calls) != 1:
        raise ValueError(f"expected exactly one search tool call; got {len(tool_calls)}")
    if call.get("content", "").strip():
        raise ValueError("model mixed final content with a search tool call")
    tool_call = tool_calls[0]
    identifier = tool_call.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("search tool call has no id")
    function = tool_call.get("function") or {}
    if function.get("name") != "search":
        raise ValueError(f"unsupported tool call: {function.get('name')!r}")
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        raise ValueError("search tool arguments are not JSON text")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"search tool arguments are malformed: {exc}") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"query"}:
        raise ValueError("search tool arguments must contain only query")
    query = parsed["query"]
    if not isinstance(query, str) or not query.strip():
        raise ValueError("search tool query must be a nonempty string")
    return identifier, query.strip()


def summarize_question(
    started_ns: int,
    ended_ns: int,
    llm_calls: list[dict[str, Any]],
    retrieval_calls: list[dict[str, Any]],
) -> dict[str, Any]:
    llm_wall = sum(call["timing"]["request_wall_ms"] for call in llm_calls)
    prefill = sum((call["timing"].get("server_prompt_ms") or 0) for call in llm_calls)
    decode = sum((call["timing"].get("server_decode_ms") or 0) for call in llm_calls)
    encoding = sum(call["encode_duration_ms"] for call in retrieval_calls)
    qdrant = sum(call["qdrant_duration_ms"] for call in retrieval_calls)
    end_to_end = (ended_ns - started_ns) / 1_000_000
    accounted = llm_wall + encoding + qdrant
    return {
        "start_ns": started_ns,
        "end_ns": ended_ns,
        "end_to_end_ms": end_to_end,
        "total_llm_wall_ms": llm_wall,
        "total_prefill_ms": prefill,
        "total_decode_ms": decode,
        "total_retrieval_encode_ms": encoding,
        "total_qdrant_search_ms": qdrant,
        "agent_overhead_ms": max(0.0, end_to_end - accounted),
        "prompt_tokens": sum(
            int(call.get("usage", {}).get("prompt_tokens", 0)) for call in llm_calls
        ),
        "decode_tokens": sum(
            int(call.get("usage", {}).get("completion_tokens", 0)) for call in llm_calls
        ),
        "chunk_count": sum(len(call["chunks"]) for call in llm_calls),
        "search_count": len(retrieval_calls),
        "llm_call_count": len(llm_calls),
    }
