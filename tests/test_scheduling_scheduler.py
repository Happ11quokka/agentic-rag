from __future__ import annotations

from agent.scheduling.config import TIME_QUANTUM_NS
from agent.scheduling.protocol import CommandResult, TokenEvent
from agent.scheduling.scheduler import ReasoningParser, ScheduledPairRunner


class FakeEngine:
    def __init__(self) -> None:
        self.remaining = {"main": 300, "draft": 300}
        self.output = {"main": 0, "draft": 0}
        self.slices_left = {"main": 3, "draft": 2}
        self.offset = 0
        self.prefill_calls: list[tuple[str, int]] = []
        self.slice_calls: list[tuple[str, int]] = []

    def start_pair(self, messages, generation):
        return dict(self.remaining)

    def prefill(self, role: str, maximum: int) -> CommandResult:
        self.prefill_calls.append((role, maximum))
        processed = min(maximum, self.remaining[role])
        start = self.offset
        self.offset += 5_000_000
        self.remaining[role] -= processed
        events = ()
        if self.remaining[role] == 0:
            self.output[role] += 1
            events = (TokenEvent(self.offset, self.output[role], b"<think>"),)
        return CommandResult(
            processed,
            self.remaining[role],
            self.output[role],
            0,
            5_000_000,
            start,
            self.offset,
            events,
        )

    def decode_slice(self, role: str, budget: int) -> CommandResult:
        self.slice_calls.append((role, budget))
        start = self.offset
        actual = 210_000_000 if len(self.slice_calls) == 1 else 190_000_000
        self.offset += actual
        self.slices_left[role] -= 1
        self.output[role] += 1
        finished = self.slices_left[role] == 0
        if finished:
            piece = f"</think>{role} done".encode()
        else:
            piece = f"{role} reasoning".encode()
        return CommandResult(
            0,
            0,
            self.output[role],
            1 if finished else 0,
            actual,
            start,
            self.offset,
            (TokenEvent(self.offset, self.output[role], piece),),
        )


def _generation() -> dict[str, object]:
    return {
        "max_tokens": 10,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
        "presence_penalty": 1.5,
        "seed": 42,
    }


def test_scheduler_uses_fixed_prefill_and_equal_time_order() -> None:
    engine = FakeEngine()
    result = ScheduledPairRunner(engine=engine, clock_ns=lambda: 1_000_000_000).run_pair(
        [{"role": "user", "content": "question"}], _generation()
    )

    assert engine.prefill_calls == [
        ("draft", 256),
        ("draft", 44),
        ("main", 256),
        ("main", 44),
    ]
    assert [role for role, _ in engine.slice_calls] == [
        "draft",
        "main",
        "draft",
        "main",
        "main",
    ]
    assert all(budget == TIME_QUANTUM_NS for _, budget in engine.slice_calls)
    assert result.scheduler["role_switch_count"] == 2
    assert result.scheduler["turns"][0]["overshoot_ns"] == 10_000_000
    assert result.scheduler["turns"][-2]["role"] == "main"
    assert result.scheduler["turns"][-1]["role"] == "main"
    assert result.calls["draft"]["content"] == "draft done"
    assert result.calls["main"]["finish_reason"] == "stop"


def test_reasoning_parser_handles_split_markers_and_utf8() -> None:
    parser = ReasoningParser()
    encoded = "<think>생각</think>정답🙂".encode("utf-8")
    parts = [encoded[:3], encoded[3:10], encoded[10:13], encoded[13:18]]
    parts.extend(encoded[index : index + 1] for index in range(18, len(encoded)))
    output = []
    for part in parts:
        output.extend(parser.feed(part))
    output.extend(parser.feed(b"", final=True))
    assert "".join(text for channel, text in output if channel == "reasoning") == "생각"
    assert "".join(text for channel, text in output if channel == "content") == "정답🙂"
    assert all("<think>" not in text and "</think>" not in text for _, text in output)


def test_reasoning_parser_accepts_output_without_open_marker() -> None:
    parser = ReasoningParser()
    output = parser.feed("reason</think>answer".encode())
    output.extend(parser.feed(b"", final=True))
    assert output == [("reasoning", "reason"), ("content", "answer")]
