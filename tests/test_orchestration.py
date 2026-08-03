import hashlib
import io
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from experiment import common, independent_run, parallel, tooluse, vectordb
from experiment.common import (
    ExperimentError,
    ModelSpec,
    Progress,
    format_metric,
    summarize,
)
from wikipedia.bundle import BundlePaths


def test_statistics_and_formatting() -> None:
    stats = summarize([1, 2, 3, 4])
    assert stats.count == 4
    assert stats.mean == 2.5
    assert stats.median == 2
    assert stats.p95_worst == 4
    throughput = summarize([1, 2, 3, 4], higher_is_better=True)
    assert throughput.median == 2
    assert throughput.p95_worst == 1
    assert format_metric(None) == "n/a"


def test_progress_prints_first_and_final_for_non_tty() -> None:
    stream = io.StringIO()
    ticks = iter([0.0, 1.0, 2.0])
    progress = Progress("work", 100, stream=stream, clock=lambda: next(ticks))
    progress.update(1)
    progress.update(2)
    progress.update(100)
    output = stream.getvalue()
    assert "1/100" in output
    assert "2/100" not in output
    assert "100/100" in output


def test_validate_model_checks_size_and_hash(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"model"
    spec = ModelSpec(
        "test",
        "repo",
        "revision",
        "model.gguf",
        len(content),
        hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setattr(common, "MODELS_DIR", tmp_path)
    spec.path.parent.mkdir()
    spec.path.write_bytes(content)
    assert common.validate_model(spec) == spec.path.resolve()
    spec.path.write_bytes(b"bad")
    with pytest.raises(ExperimentError, match="size mismatch"):
        common.validate_model(spec)


def test_reasoning_budget_is_optional_per_model_server(tmp_path) -> None:
    default = common.server_arguments("llama-server", tmp_path / "model", 8080)
    bounded = common.server_arguments(
        "llama-server",
        tmp_path / "model",
        8080,
        reasoning_budget=512,
    )

    assert "--reasoning-budget" not in default
    assert bounded[bounded.index("--reasoning-budget") + 1] == "512"
    assert "--reasoning-budget-message" in bounded


def _wikipedia_environment_fakes(
    tmp_path, monkeypatch: pytest.MonkeyPatch, *, processes: list[str]
) -> object:
    paths = BundlePaths.from_dir(tmp_path)
    closed = SimpleNamespace(value=False)

    class Client:
        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return SimpleNamespace(points_count=123)

    class Database:
        def __init__(self, config):
            self.config = config
            self.client = Client()

        def close(self):
            closed.value = True

    monkeypatch.setattr(
        common.BundlePaths,
        "resolve",
        classmethod(lambda cls: paths),
    )
    monkeypatch.setattr(
        common,
        "load_manifest",
        lambda paths, require_complete: {
            "schema_version": 1,
            "status": "incomplete",
            "qdrant": {
                "url": "http://localhost:6333",
                "storage_dir": str(tmp_path / "qdrant"),
                "collection": "wikipedia",
            },
        },
    )
    monkeypatch.setattr(common, "model_is_downloaded", lambda paths: True)
    monkeypatch.setattr(common, "ensure_qdrant", lambda *args, **kwargs: "ready")
    monkeypatch.setattr(common, "QdrantVectorDB", Database)
    monkeypatch.setattr(common, "_local_ingestion_processes", lambda: processes)
    return closed


def test_prepare_wikipedia_warns_and_accepts_paused_partial_bundle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    closed = _wikipedia_environment_fakes(tmp_path, monkeypatch, processes=[])

    with common.prepare_wikipedia(stability_seconds=0) as environment:
        assert environment.incomplete is True
        assert environment.points_count == 123

    assert closed.value is True
    assert "bundle is incomplete" in capsys.readouterr().err


def test_prepare_wikipedia_rejects_active_ingestion_and_closes_database(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = _wikipedia_environment_fakes(
        tmp_path, monkeypatch, processes=["123 wikipedia-ingest qdrant"]
    )

    with pytest.raises(ExperimentError, match="pause it before measuring"):
        common.prepare_wikipedia(stability_seconds=0)

    assert closed.value is True


def test_paired_completion_really_runs_clients_concurrently() -> None:
    rendezvous = threading.Barrier(2)
    messages_seen: list[object] = []

    class Client:
        def stream_completion(self, messages, generation):
            messages_seen.append(messages)
            rendezvous.wait(timeout=1)
            return {"generation": generation}

    messages = [{"role": "user", "content": "same"}]
    result = parallel.paired_completion(
        {"main": Client(), "draft": Client()}, messages, {"seed": 42}
    )
    assert set(result) == {"main", "draft"}
    assert messages_seen == [messages, messages]


def test_combined_throughput_uses_pair_decode_window() -> None:
    calls = {
        "main": {
            "first_decode_ns": 1_000_000_000,
            "request_end_ns": 3_000_000_000,
            "timing": {"server_decode_tokens": 20},
        },
        "draft": {
            "first_decode_ns": 2_000_000_000,
            "request_end_ns": 4_000_000_000,
            "timing": {"server_decode_tokens": 10},
        },
    }
    assert parallel._combined_throughput(calls) == 10


def test_parallel_report_uses_worst_tail_in_metric_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[list[object]]] = []
    calls = [
        {"timing": {"ttft_ms": value, "decode_tokens_per_second": value * 10}}
        for value in (1.0, 2.0, 3.0)
    ]
    monkeypatch.setattr(
        parallel,
        "print_table",
        lambda headers, rows: captured.append(rows),
    )

    parallel.render_report({"main": calls, "draft": calls}, [40.0, 50.0, 60.0])

    rows = captured[0]
    assert rows[0][-1] == "3.00"  # TTFT: numeric p95, high latency is bad.
    assert rows[1][-1] == "10.00"  # Throughput: numeric p5, low rate is bad.
    assert rows[-1][-1] == "40.00"  # Combined throughput follows same direction.


def test_independent_run_model_server_phases_never_overlap() -> None:
    events: list[str] = []

    @contextmanager
    def server(role: str):
        events.append(f"start:{role}")
        try:
            yield role
        finally:
            events.append(f"stop:{role}")

    independent_run.run_phases(
        ("main", "draft"),
        server,
        lambda role, value: events.append(f"run:{role}:{value}"),
    )

    assert events == [
        "start:main",
        "run:main:main",
        "stop:main",
        "start:draft",
        "run:draft:draft",
        "stop:draft",
    ]


def test_independent_run_report_excludes_combined_throughput(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = [
        {"timing": {"ttft_ms": value, "decode_tokens_per_second": value * 10}}
        for value in (1.0, 2.0, 3.0)
    ]

    independent_run.render_report({"main": calls, "draft": calls})

    output = capsys.readouterr().out
    assert "Independent-run experiment metrics" in output
    assert output.count("TTFT (ms)") == 2
    assert output.count("decode (tok/s)") == 2
    assert "combined" not in output


def _call(completion_tokens: int, query: str | None = None) -> dict[str, object]:
    tool_calls = []
    if query is not None:
        tool_calls.append(
            {
                "id": f"call-{query}",
                "type": "function",
                "function": {"name": "search", "arguments": "{}"},
            }
        )
    return {
        "tool_calls": tool_calls,
        "usage": {"completion_tokens": completion_tokens},
    }


def test_tool_query_tokens_use_exact_completion_usage() -> None:
    outcome = {
        "llm_calls": [
            _call(12, "one"),
            _call(8, "two"),
            _call(3),
        ]
    }
    first, between = tooluse.query_token_metrics(outcome)
    assert first == 12
    assert between == [8]


def test_tool_query_tokens_handle_zero_queries() -> None:
    assert tooluse.query_token_metrics({"llm_calls": [_call(3)]}) == (None, [])


def test_tool_query_tokens_require_server_usage() -> None:
    search = _call(1, "one")
    search["usage"] = {}

    with pytest.raises(ExperimentError, match="completion_tokens"):
        tooluse.query_token_metrics({"llm_calls": [search]})


def test_tooluse_records_failed_run_metrics_and_status() -> None:
    selected = tooluse._new_role_result()
    outcome = {
        "terminal_status": "timeout",
        "search_count": 2,
        "llm_calls": [_call(12, "one"), _call(8, "two")],
    }

    tooluse._record_outcome(selected, outcome)

    assert selected["valid"] == 0
    assert selected["failed"] == 1
    assert selected["failure_statuses"] == {"timeout": 1}
    assert selected["incomplete_queries"] == [2.0]
    assert selected["incomplete_first"] == [12.0]
    assert selected["incomplete_between"] == [8.0]


def test_tooluse_failed_run_survives_missing_token_usage() -> None:
    selected = tooluse._new_role_result()
    search = _call(1, "one")
    search["usage"] = {}

    tooluse._record_outcome(
        selected,
        {
            "terminal_status": "server_error",
            "search_count": 0,
            "llm_calls": [search],
        },
    )

    assert selected["failed"] == 1
    assert selected["failure_statuses"] == {"server_error": 1}
    assert selected["incomplete_queries"] == [0.0]
    assert selected["incomplete_first"] == []


def test_tooluse_report_shows_incomplete_metrics_and_failure_status(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = {
        "main": tooluse._new_role_result(),
        "draft": tooluse._new_role_result(),
    }
    tooluse._record_outcome(
        results["main"],
        {
            "terminal_status": "timeout",
            "search_count": 1,
            "llm_calls": [_call(12, "one")],
        },
    )
    tooluse._record_outcome(
        results["draft"],
        {
            "terminal_status": "final",
            "search_count": 1,
            "llm_calls": [_call(8, "one"), _call(3)],
        },
    )

    tooluse.render_report(results)

    output = capsys.readouterr().out
    assert "Incomplete-run query metrics" in output
    assert "Failure status breakdown" in output
    rows = [line.split() for line in output.splitlines()]
    assert ["main", "timeout", "1"] in rows
    assert ["draft", "none", "0"] in rows


def test_vectordb_collection_modes_are_controlled() -> None:
    calls: list[dict[str, object]] = []

    class Client:
        def create_collection(self, **kwargs):
            calls.append(kwargs)
            return True

    names = vectordb.TemporaryCollections("memory", "disk")
    vectordb.create_collections(
        Client(), names, float16=True, dimension=1024, distance="DOT"
    )
    assert [call["collection_name"] for call in calls] == ["memory", "disk"]
    assert [call["vectors_config"].on_disk for call in calls] == [False, True]
    assert [call["on_disk_payload"] for call in calls] == [False, True]


def test_vectordb_sample_is_copied_identically() -> None:
    upserts: list[tuple[str, list[object]]] = []
    records = [
        SimpleNamespace(id=1, vector=[0.1, 0.2], payload={"text": "one"}),
        SimpleNamespace(id=2, vector=[0.3, 0.4], payload={"text": "two"}),
    ]

    class Client:
        def scroll(self, **kwargs):
            return records, None

        def upsert(self, *, collection_name, points, wait):
            assert wait is True
            upserts.append((collection_name, points))

    vectordb.copy_sample(
        Client(),
        "source",
        vectordb.TemporaryCollections("memory", "disk"),
        count=2,
    )

    assert [name for name, _ in upserts] == ["memory", "disk"]
    assert [point.id for point in upserts[0][1]] == [1, 2]
    assert upserts[0][1] == upserts[1][1]


def test_query_latency_checks_results() -> None:
    ticks = iter([1_000_000, 4_000_000])

    class Client:
        def query_points(self, **kwargs):
            return type("Response", (), {"points": [object()]})()

    assert (
        vectordb.query_latency_ms(
            Client(), "collection", [0.0] * 1024, timer_ns=lambda: next(ticks)
        )
        == 3
    )
