import json
from copy import deepcopy
from pathlib import Path

import pytest

from experiment import artifacts, cli, common, contention_measure as cm
from experiment.common import ExperimentError, GENERATION
from fanoutqa.dataset import Question


def call(start, first, last, end, *, ttft=10, throughput=20):
    return {
        "request_start_ns": start,
        "first_decode_ns": first,
        "request_end_ns": end,
        "chunks": [{"received_ns": first}, {"received_ns": last}],
        "timing": {"ttft_ms": ttft, "stream_chunks_per_second": throughput},
    }


def test_windows_separate_overlap_and_tail_from_request_drain():
    windows = cm.measurement_windows(
        "simultaneous",
        {
            "main": call(0, 20, 80, 100),
            "draft": call(0, 40, 60, 90),
        },
    )
    assert windows["simultaneous/decode-overlap"] == (40, 60)
    assert windows["simultaneous/decode-tail"] == (60, 80)
    assert windows["simultaneous/main/decode"] == (20, 80)
    assert windows["simultaneous/draft/before-first-chunk"] == (0, 40)
    assert windows["simultaneous/request"] == (0, 100)


def test_disjoint_or_missing_decode_has_no_overlap():
    windows = cm.measurement_windows(
        "simultaneous",
        {
            "main": call(0, 10, 20, 30),
            "draft": call(0, 40, 50, 60),
        },
    )
    assert "simultaneous/decode-overlap" not in windows
    windows = cm.measurement_windows("main-only", {"main": call(0, None, None, 20)})
    assert list(windows) == ["main-only/request"]


def test_matched_ratios_and_missing_throughput():
    conditions = {
        "main-only": {"main": call(0, 1, 2, 3, ttft=10, throughput=40)},
        "draft-only": {"draft": call(0, 1, 2, 3, ttft=5, throughput=0)},
        "simultaneous": {
            "main": call(0, 1, 2, 3, ttft=20, throughput=10),
            "draft": call(0, 1, 2, 3, ttft=10, throughput=None),
        },
    }
    result = cm.matched_slowdowns(conditions)
    assert result["main"] == {"ttft_ratio": 2, "stream_rate_ratio": 4}
    assert result["draft"]["stream_rate_ratio"] is None


def test_menu_appends_and_prompts_reuse_defaults(monkeypatch):
    assert cli.EXPERIMENTS[-1].name == "contention-measure"
    assert (
        cli.choose_experiment(input_fn=lambda _: "contention-measure").prompt_and_run
        == cm.prompt_and_run
    )
    assert [e.name for e in cli.EXPERIMENTS[:2]] == ["parallel", "parallel-scheduled"]
    calls = []
    monkeypatch.setattr(cm, "run", lambda *a, **kw: calls.append((a, kw)))
    cm.prompt_and_run(input_fn=lambda _: "")
    assert calls[0][0] == (cm.DEFAULT_TASKS, cm.DEFAULT_REPETITIONS)
    assert calls[0][0] == (1, 1)
    assert calls[0][1]["recording_seconds"] == 2


def test_fixed_deadline_keeps_partial_stream_and_kills_servers(monkeypatch):
    import threading
    import time
    import httpx

    killed = threading.Event()
    events = []

    class Client:
        def stream_completion(
            self, messages, generation, *, cancel_event, on_request_start, on_chunk
        ):
            now = time.perf_counter_ns()
            on_request_start(now)
            for offset in (0, 1_000_000):
                on_chunk(
                    {"received_ns": now + offset, "text": "word", "channel": "content"}
                )
            assert killed.wait(2)
            assert cancel_event.is_set()
            raise httpx.ReadError("server was killed at deadline")

    def kill(servers):
        events.append(set(servers))
        killed.set()

    monkeypatch.setattr(cm, "_kill_servers", kill)
    calls, interval = cm.fixed_duration_calls(
        {"main": Client(), "draft": Client()},
        {"main": object(), "draft": object()},
        [{"role": "user", "content": "test"}],
        GENERATION,
        0.05,
    )
    assert events == [{"main", "draft"}]
    assert interval[1] - interval[0] == 50_000_000
    for call in calls.values():
        assert call["cancelled"]
        assert call["stop_reason"] == "capture_deadline"
        assert call["timing"]["observed_chunks"] == 2
        assert call["timing"]["stream_chunks_per_second"] == 1000
        assert call["first_decode_ns"] is not None


def test_request_failure_is_not_misclassified_as_deadline(monkeypatch):
    import httpx

    class Client:
        def stream_completion(self, *args, **kwargs):
            raise httpx.ConnectError("server unavailable")

    killed = []
    monkeypatch.setattr(cm, "_kill_servers", lambda servers: killed.append(True))
    with pytest.raises(httpx.ConnectError, match="unavailable"):
        cm.fixed_duration_calls({"main": Client()}, {}, [], GENERATION, 2)
    assert killed == [True]


def test_oversized_capture_rejected_before_setup():
    with pytest.raises(ExperimentError, match="duration must be"):
        cm.run(1, 1, recording_seconds=6)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    active = set()
    requests = []
    recorded = []

    class Server:
        def __init__(self, role, *a):
            self.role = role
            self.base_url = role

        def __enter__(self):
            active.add(self.role)
            return self

        process = None

        def stop(self):
            active.discard(self.role)

        def __exit__(self, *a):
            self.stop()

    class Client:
        def __init__(self, role, **kw):
            self.role = role

        def close(self):
            pass

        def stream_completion(self, messages, generation):
            assert active == {"main", "draft"}
            requests.append((self.role, deepcopy(messages), deepcopy(generation)))
            return call(0, 10, 80, 100)

    class Recording:
        def __init__(self, directory, *, duration_seconds):
            self.directory = directory
            assert duration_seconds == 2

        def __enter__(self):
            recorded.append(self.directory.name)
            return self

        def __exit__(self, *a):
            assert not active, "servers must stop before trace finalization"

        def export(self, windows):
            assert not active, "servers must stop before counter export"
            return {
                "windows": {
                    key: {"counters": {name: {} for name in cm.REQUIRED_COUNTERS}}
                    for key in windows
                }
            }

    def fixed_calls(clients, servers, messages, generation, duration):
        assert active == {"main", "draft"}
        calls = {
            role: client.stream_completion(messages, generation)
            for role, client in clients.items()
        }
        cm._kill_servers(servers)
        return calls, (0, 100)

    monkeypatch.setattr(cm, "fixed_duration_calls", fixed_calls)
    monkeypatch.setattr(cm, "collector_metadata", lambda: {"backend": "fake"})
    monkeypatch.setattr(
        cm,
        "select_benchmark_questions",
        lambda count: [Question("q1", "Question?", ("cat",))],
    )
    monkeypatch.setattr(
        cm,
        "require_models",
        lambda selected: {role: Path(role) for role in active or ("main", "draft")},
    )
    monkeypatch.setattr(cm, "llama_server_binary", lambda: ("server", {"build": 10566}))
    monkeypatch.setattr(cm, "ModelServer", Server)
    monkeypatch.setattr(cm, "LlamaCppClient", Client)
    monkeypatch.setattr(cm, "MetalRecording", Recording)
    monkeypatch.setattr(cm, "render_report", lambda *a: None)
    monkeypatch.setattr(cm, "render_counter_report", lambda *a: None)
    monkeypatch.setattr(artifacts, "RESULTS_DIR", tmp_path)
    return active, requests, recorded, Recording


def test_resident_conditions_rotate_and_save_matched_results(tmp_path, runtime):
    active, requests, recorded, _ = runtime
    cm.run(1, 3)
    assert not active
    assert recorded == [
        condition for unit in range(3) for condition in cm.condition_order(unit)
    ]
    measured = [r for r in requests if r[2]["max_tokens"] != 32]
    assert len(measured) == 12
    assert all(r[2] == GENERATION for r in measured)
    assert all(r[1] == measured[0][1] for r in measured)
    directory = next((tmp_path / "contention-measure").iterdir())
    rows = [
        json.loads(line)
        for line in (directory / "traces.jsonl").read_text().splitlines()
    ]
    for unit in range(1, 4):
        selected = [
            r
            for r in rows
            if r["record_type"] == "contention_completion" and r["unit"] == unit
        ]
        assert [r["condition"] for r in selected] == list(cm.condition_order(unit - 1))
    assert (
        json.loads((directory / "manifest.json").read_text())["status"] == "completed"
    )
    summary = json.loads((directory / "summary.json").read_text())
    assert (
        summary["metrics"]["matched_slowdowns"]["main"]["stream_rate_ratio"]["count"]
        == 3
    )


def test_failed_export_preserves_calls_and_cleans_servers(
    tmp_path, runtime, monkeypatch
):
    active, _, _, Recording = runtime
    original = Recording.export

    def export(self, windows):
        if self.directory.name != "counter-preflight":
            raise ExperimentError("export failed")
        return original(self, windows)

    monkeypatch.setattr(Recording, "export", export)
    with pytest.raises(ExperimentError, match="export failed"):
        cm.run(1, 1)
    assert not active
    directory = next((tmp_path / "contention-measure").iterdir())
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["record_count"] == 1


@pytest.mark.parametrize(
    "version, expected",
    [
        ("version: 8360 (abc123)", 8360),
        ("version: 0.2.0 (build 10566, commit bb4caa754)", 10566),
    ],
)
def test_llama_build_formats(monkeypatch, version, expected):
    from types import SimpleNamespace

    monkeypatch.setenv("LLAMA_SERVER", "/test/server")
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(stdout=version, stderr="", returncode=0),
    )
    assert common.llama_server_binary()[1]["build"] == expected
