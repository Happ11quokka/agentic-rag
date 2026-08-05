from __future__ import annotations

import codecs
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from agent.runner import summarize_llm_call

from .config import (
    DECODE_ORDER,
    PREFILL_CHUNK_TOKENS,
    PREFILL_ORDER,
    TIME_QUANTUM_NS,
    LlamaConfig,
)
from .engine import ScheduledNativeEngine
from .build import NativeEngine
from .protocol import CommandResult

OPEN_MARKER = "<think>"
CLOSE_MARKER = "</think>"


@dataclass(frozen=True, slots=True)
class ScheduledPairResult:
    calls: dict[str, dict[str, Any]]
    scheduler: dict[str, Any]


class ReasoningParser:
    """Incrementally split Qwen reasoning/content while hiding marker fragments."""

    def __init__(self) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.state = "opening"
        self.buffer = ""

    def feed(self, value: bytes, *, final: bool = False) -> list[tuple[str, str]]:
        text = self.decoder.decode(value, final=final)
        return self._consume(text, final=final)

    def _consume(self, text: str, *, final: bool) -> list[tuple[str, str]]:
        output: list[tuple[str, str]] = []
        self.buffer += text
        if self.state == "opening":
            if OPEN_MARKER.startswith(self.buffer) and self.buffer != OPEN_MARKER:
                if not final:
                    return output
            if self.buffer.startswith(OPEN_MARKER):
                self.buffer = self.buffer[len(OPEN_MARKER) :]
            self.state = "reasoning"

        if self.state == "reasoning":
            marker = self.buffer.find(CLOSE_MARKER)
            if marker >= 0:
                if marker:
                    output.append(("reasoning", self.buffer[:marker]))
                self.buffer = self.buffer[marker + len(CLOSE_MARKER) :]
                self.state = "content"
            elif final:
                if self.buffer:
                    output.append(("reasoning", self.buffer))
                self.buffer = ""
                return output
            else:
                held = _marker_prefix_suffix(self.buffer, CLOSE_MARKER)
                safe = self.buffer[: len(self.buffer) - held] if held else self.buffer
                if safe:
                    output.append(("reasoning", safe))
                self.buffer = self.buffer[len(safe) :]
                return output

        if self.state == "content" and self.buffer:
            output.append(("content", self.buffer))
            self.buffer = ""
        return output


def _marker_prefix_suffix(value: str, marker: str) -> int:
    maximum = min(len(value), len(marker) - 1)
    for length in range(maximum, 0, -1):
        if value.endswith(marker[:length]):
            return length
    return 0


class _RoleCall:
    def __init__(
        self,
        role: str,
        messages: list[Mapping[str, str]],
        generation: Mapping[str, object],
        request_start_ns: int,
        prompt_tokens: int,
    ) -> None:
        self.role = role
        self.messages = messages
        self.generation = generation
        self.request_start_ns = request_start_ns
        self.prompt_tokens = prompt_tokens
        self.parser = ReasoningParser()
        self.chunks: list[dict[str, Any]] = []
        self.reasoning: list[str] = []
        self.content: list[str] = []
        self.first_token_ns: int | None = None
        self.end_ns: int | None = None
        self.completion_tokens = 0
        self.finish_code = 0
        self.active_prefill_ns = 0
        self.active_decode_ns = 0
        self.slice_emitted_tokens = 0

    def accept(self, result: CommandResult, *, decode: bool) -> None:
        if decode:
            self.active_decode_ns += result.active_compute_ns
            self.slice_emitted_tokens += len(result.tokens)
        else:
            self.active_prefill_ns += result.active_compute_ns
        self.completion_tokens = result.output_tokens
        self.finish_code = result.finish_code
        for event in result.tokens:
            received_ns = self.request_start_ns + event.offset_ns
            if self.first_token_ns is None:
                self.first_token_ns = received_ns
            for channel, text in self.parser.feed(event.piece):
                if text:
                    self._append_chunk(channel, text, received_ns, event.token_id)
        if result.finished:
            self.end_ns = self.request_start_ns + result.end_offset_ns

    def finish_parser(self) -> None:
        received = self.end_ns or self.request_start_ns
        for channel, text in self.parser.feed(b"", final=True):
            if text:
                self._append_chunk(channel, text, received, None)

    def _append_chunk(
        self, channel: str, text: str, received_ns: int, token_id: int | None
    ) -> None:
        previous = (
            self.chunks[-1]["received_ns"] if self.chunks else self.request_start_ns
        )
        self.chunks.append(
            {
                "sequence": len(self.chunks),
                "channel": channel,
                "text": text,
                "received_ns": received_ns,
                "inter_arrival_ns": received_ns - previous,
                "token_id": token_id,
            }
        )
        (self.reasoning if channel == "reasoning" else self.content).append(text)

    def build(self) -> dict[str, Any]:
        self.finish_parser()
        if self.end_ns is None:
            raise RuntimeError(f"scheduled role did not finish: {self.role}")
        reasoning = "".join(self.reasoning)
        content = "".join(self.content)
        first_reasoning = next(
            (item["received_ns"] for item in self.chunks if item["channel"] == "reasoning"),
            None,
        )
        last_reasoning = next(
            (item["received_ns"] for item in reversed(self.chunks) if item["channel"] == "reasoning"),
            None,
        )
        content_start = next(
            (item["received_ns"] for item in self.chunks if item["channel"] == "content"),
            None,
        )
        prompt_seconds = (
            None
            if self.first_token_ns is None
            else (self.first_token_ns - self.request_start_ns) / 1_000_000_000
        )
        decode_seconds = (
            None
            if self.first_token_ns is None
            else (self.end_ns - self.first_token_ns) / 1_000_000_000
        )
        call = {
            "request": {
                "messages": [dict(message) for message in self.messages],
                "stream": True,
                "stream_options": {"include_usage": True},
                **dict(self.generation),
            },
            "request_start_ns": self.request_start_ns,
            "request_end_ns": self.end_ns,
            "first_decode_ns": self.first_token_ns,
            "reasoning_start_ns": first_reasoning,
            "reasoning_end_ns": last_reasoning,
            "content_start_ns": content_start,
            "action_ready_ns": self.end_ns if content else None,
            "chunks": self.chunks,
            "reasoning": reasoning,
            "content": content,
            "tool_calls": [],
            "finish_reason": "stop" if self.finish_code == 1 else "length",
            "cancelled": False,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.prompt_tokens + self.completion_tokens,
            },
            "response_extensions": {
                "scheduled": {
                    "active_prefill_ns": self.active_prefill_ns,
                    "active_decode_ns": self.active_decode_ns,
                }
            },
            "metrics_delta": {
                "prompt_tokens": self.prompt_tokens,
                "decode_tokens": self.completion_tokens,
                "prompt_seconds": prompt_seconds or 0.0,
                "decode_seconds": decode_seconds or 0.0,
            },
        }
        call["timing"] = summarize_llm_call(call)
        return call


class ScheduledPairRunner:
    def __init__(
        self,
        model_paths: Mapping[str, Path | str] | None = None,
        llama_config: LlamaConfig | None = None,
        *,
        engine: Any | None = None,
        native_engine: NativeEngine | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
    ) -> None:
        if engine is None and model_paths is None:
            raise ValueError("model_paths are required when engine is not supplied")
        self.llama_config = (llama_config or LlamaConfig()).resolved()
        self.engine = engine or ScheduledNativeEngine(
            model_paths or {}, self.llama_config, native_engine=native_engine
        )
        self.clock_ns = clock_ns

    def __enter__(self) -> ScheduledPairRunner:
        if hasattr(self.engine, "open"):
            self.engine.open()
        return self

    def __exit__(self, *_: object) -> None:
        if hasattr(self.engine, "close"):
            self.engine.close()

    def run_pair(
        self,
        messages: list[Mapping[str, str]],
        generation: Mapping[str, object],
    ) -> ScheduledPairResult:
        request_start = self.clock_ns()
        prompt_counts = self.engine.start_pair(messages, generation)
        roles = {
            role: _RoleCall(
                role,
                messages,
                generation,
                request_start,
                prompt_counts[role],
            )
            for role in ("main", "draft")
        }
        prefill_records: list[dict[str, Any]] = []
        for role in PREFILL_ORDER:
            remaining = prompt_counts[role]
            while remaining:
                result = self.engine.prefill(role, min(PREFILL_CHUNK_TOKENS, remaining))
                roles[role].accept(result, decode=False)
                prefill_records.append(
                    {
                        "role": role,
                        "processed_tokens": result.processed_tokens,
                        "remaining_tokens": result.remaining_tokens,
                        "active_compute_ns": result.active_compute_ns,
                        "start_offset_ns": result.start_offset_ns,
                        "end_offset_ns": result.end_offset_ns,
                        "emitted_tokens": len(result.tokens),
                        "finish_reason": _finish_reason(result.finish_code),
                    }
                )
                remaining = result.remaining_tokens

        active = {role for role in DECODE_ORDER if not roles[role].finish_code}
        turns: list[dict[str, Any]] = []
        role_switches = 0
        previous_role: str | None = None
        previous_both_active = False
        while active:
            progressed = False
            for role in DECODE_ORDER:
                if role not in active:
                    continue
                both_active = len(active) == 2
                result = self.engine.decode_slice(role, TIME_QUANTUM_NS)
                roles[role].accept(result, decode=True)
                actual = max(0, result.end_offset_ns - result.start_offset_ns)
                if (
                    both_active
                    and previous_both_active
                    and previous_role is not None
                    and previous_role != role
                ):
                    role_switches += 1
                turns.append(
                    {
                        "turn": len(turns) + 1,
                        "role": role,
                        "both_active": both_active,
                        "start_offset_ns": result.start_offset_ns,
                        "end_offset_ns": result.end_offset_ns,
                        "quota_ns": TIME_QUANTUM_NS,
                        "actual_ns": actual,
                        "overshoot_ns": max(0, actual - TIME_QUANTUM_NS),
                        "active_compute_ns": result.active_compute_ns,
                        "emitted_tokens": len(result.tokens),
                        "finish_reason": _finish_reason(result.finish_code),
                    }
                )
                previous_role = role
                previous_both_active = both_active
                if result.finished:
                    active.remove(role)
                progressed = True
            if not progressed:
                raise RuntimeError("scheduler made no decode progress")

        calls = {role: roles[role].build() for role in ("main", "draft")}
        scheduler = _scheduler_metrics(
            roles, turns, prefill_records, role_switches, request_start
        )
        return ScheduledPairResult(calls, scheduler)


def _finish_reason(code: int) -> str | None:
    return {0: None, 1: "stop", 2: "length"}.get(code, "error")


def _scheduler_metrics(
    roles: dict[str, _RoleCall],
    turns: list[dict[str, Any]],
    prefill: list[dict[str, Any]],
    role_switches: int,
    request_start_ns: int,
) -> dict[str, Any]:
    by_role: dict[str, Any] = {}
    both_active_ns = {"main": 0, "draft": 0}
    for turn in turns:
        if turn["both_active"]:
            both_active_ns[turn["role"]] += turn["active_compute_ns"]
    total_both_active = sum(both_active_ns.values())
    for role, state in roles.items():
        selected = [turn for turn in turns if turn["role"] == role]
        window_ns = (
            0
            if state.first_token_ns is None or state.end_ns is None
            else state.end_ns - state.first_token_ns
        )
        active_decode_ns = sum(turn["active_compute_ns"] for turn in selected)
        by_role[role] = {
            "slice_count": len(selected),
            "active_prefill_ms": state.active_prefill_ns / 1_000_000,
            "active_decode_ms": active_decode_ns / 1_000_000,
            "scheduled_decode_window_ms": window_ns / 1_000_000,
            "queued_decode_ms": max(0, window_ns - active_decode_ns) / 1_000_000,
            "tokens_per_slice": (
                sum(turn["emitted_tokens"] for turn in selected) / len(selected)
                if selected
                else 0.0
            ),
            "active_tokens_per_second": (
                state.slice_emitted_tokens / (active_decode_ns / 1_000_000_000)
                if active_decode_ns
                else None
            ),
            "both_active_time_share": (
                both_active_ns[role] / total_both_active if total_both_active else None
            ),
        }
    return {
        "policy": "equal-time-round-robin",
        "config": {
            "time_quantum_ns": TIME_QUANTUM_NS,
            "prefill_chunk_tokens": PREFILL_CHUNK_TOKENS,
            "prefill_order": list(PREFILL_ORDER),
            "decode_order": list(DECODE_ORDER),
        },
        "request_start_ns": request_start_ns,
        "prefill": prefill,
        "turns": turns,
        "role_metrics": by_role,
        "slice_count": len(turns),
        "role_switch_count": role_switches,
        "both_active_time_share": {
            role: (both_active_ns[role] / total_both_active if total_both_active else None)
            for role in ("main", "draft")
        },
    }
