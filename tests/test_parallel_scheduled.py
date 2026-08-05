from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent.scheduling.scheduler import ScheduledPairResult
from experiment import artifacts, parallel_scheduled
from experiment.common import ExperimentError
from fanoutqa.dataset import Question


def _call(role: str) -> dict[str, object]:
    return {
        "request": {"messages": []},
        "request_start_ns": 1_000_000_000,
        "first_decode_ns": 1_100_000_000,
        "request_end_ns": 2_100_000_000,
        "chunks": [],
        "reasoning": f"thinking-{role}",
        "content": f"answer-{role}",
        "tool_calls": [],
        "finish_reason": "stop",
        "cancelled": False,
        "usage": {"prompt_tokens": 2, "completion_tokens": 10},
        "metrics_delta": {
            "prompt_tokens": 2,
            "decode_tokens": 10,
            "prompt_seconds": 0.1,
            "decode_seconds": 1.0,
        },
        "timing": {"ttft_ms": 100.0, "decode_tokens_per_second": 10.0},
    }


def _scheduler() -> dict[str, object]:
    role_metrics = {
        role: {
            "slice_count": 2,
            "active_prefill_ms": 10.0,
            "active_decode_ms": 300.0,
            "scheduled_decode_window_ms": 600.0,
            "queued_decode_ms": 300.0,
            "tokens_per_slice": 5.0,
            "active_tokens_per_second": 30.0,
            "both_active_time_share": 0.5,
        }
        for role in ("main", "draft")
    }
    return {
        "policy": "equal-time-round-robin",
        "config": {},
        "prefill": [{"role": "draft"}, {"role": "main"}],
        "turns": [
            {
                "role": "draft",
                "quota_ns": 200_000_000,
                "overshoot_ns": 5_000_000,
            },
            {"role": "main", "quota_ns": 200_000_000, "overshoot_ns": 0},
        ],
        "role_metrics": role_metrics,
        "slice_count": 4,
        "role_switch_count": 3,
    }


class FakeRunner:
    calls = 0

    def __init__(self, **kwargs) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def run_pair(self, messages, generation):
        type(self).calls += 1
        return ScheduledPairResult(
            {role: _call(role) for role in ("main", "draft")}, _scheduler()
        )


def test_scheduled_experiment_persists_trace_manifest_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = Question("q1", "question?", ("category",))
    FakeRunner.calls = 0
    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        parallel_scheduled, "select_benchmark_questions", lambda count: [question]
    )
    monkeypatch.setattr(
        parallel_scheduled,
        "require_models",
        lambda model_specs=None: {
            "main": Path("/main.gguf"),
            "draft": Path("/draft.gguf"),
        },
    )
    monkeypatch.setattr(
        parallel_scheduled,
        "resolve_native_engine",
        lambda: SimpleNamespace(version={"build": 8360}),
    )
    monkeypatch.setattr(parallel_scheduled, "ScheduledPairRunner", FakeRunner)
    monkeypatch.setattr(parallel_scheduled, "render_report", lambda *args: None)

    parallel_scheduled.run(1, 2)

    run_dir = next((tmp_path / "parallel-scheduled").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text().splitlines()
    ]
    summary = json.loads((run_dir / "summary.json").read_text())
    assert FakeRunner.calls == 3  # one warmup plus two measured pairs
    assert manifest["status"] == "completed"
    assert manifest["record_count"] == 2
    assert manifest["parameters"]["scheduler"]["time_quantum_ns"] == 200_000_000
    assert len(rows) == 2
    assert rows[0]["record_type"] == "scheduled_paired_completion"
    assert rows[0]["scheduler"]["prefill"][0]["role"] == "draft"
    assert set(rows[0]["model_calls"]) == {"main", "draft"}
    assert summary["metrics"]["role_switch_count"]["count"] == 2
    assert summary["metrics"]["quota_overshoot_ms"]["p95_worst"] == 5.0


def test_scheduled_runtime_failure_is_recorded_as_experiment_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    question = Question("q1", "question?", ("category",))

    class FailingRunner(FakeRunner):
        def run_pair(self, messages, generation):
            raise RuntimeError("decode error; stderr log: /tmp/engine.log")

    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        parallel_scheduled, "select_benchmark_questions", lambda count: [question]
    )
    monkeypatch.setattr(
        parallel_scheduled,
        "require_models",
        lambda model_specs=None: {
            "main": Path("/main.gguf"),
            "draft": Path("/draft.gguf"),
        },
    )
    monkeypatch.setattr(
        parallel_scheduled,
        "resolve_native_engine",
        lambda: SimpleNamespace(version={"build": 8360}),
    )
    monkeypatch.setattr(parallel_scheduled, "ScheduledPairRunner", FailingRunner)

    with pytest.raises(ExperimentError, match="stderr log"):
        parallel_scheduled.run(1, 1)

    run_dir = next((tmp_path / "parallel-scheduled").iterdir())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["record_count"] == 0
    assert manifest["error"]["type"] == "ExperimentError"
