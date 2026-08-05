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


def test_histograms_share_bins_across_series(
    capsys: pytest.CaptureFixture[str],
) -> None:
    common.print_histograms(
        "Latency histogram (ms)",
        {"main": [0.0, 2.0], "draft": [1.0, 3.0]},
        max_bins=2,
        bar_width=4,
    )

    output = capsys.readouterr().out
    assert "Latency histogram (ms)" in output
    assert output.count("[0.00, 1.50)") == 2
    assert output.count("[1.50, 3.00]") == 2
    assert "main (n=2)" in output
    assert "draft (n=2)" in output


def test_histograms_handle_constant_and_empty_samples(
    capsys: pytest.CaptureFixture[str],
) -> None:
    common.print_histograms("Constant", {"one": [5.0], "empty": []})
    common.print_histograms("Empty", {"empty": []})

    output = capsys.readouterr().out
    assert "5.00" in output
    assert "one (n=1)" in output
    assert output.count("no samples") == 2


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
        key="test",
        label="Test model",
        directory="test",
        repo="repo",
        revision="revision",
        filename="model.gguf",
        bytes=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )
    monkeypatch.setattr(common, "MODELS_DIR", tmp_path)
    spec.path.parent.mkdir()
    spec.path.write_bytes(content)
    assert common.validate_model(spec) == spec.path.resolve()
    spec.path.write_bytes(b"bad")
    with pytest.raises(ExperimentError, match="size mismatch"):
        common.validate_model(spec)


def test_model_selection_defaults_and_requires_both_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected = common.default_model_specs()
    assert selected["main"].key == "qwen3-14b-q4-k-m"
    assert selected["draft"].key == "qwen3-1.7b-q8-0"

    validated: list[str] = []
    monkeypatch.setattr(
        common,
        "validate_model",
        lambda spec: validated.append(spec.key) or spec.path,
    )
    common.require_models(
        {
            "main": common.MODEL_SPECS["qwen3-0.6b-q8-0"],
            "draft": common.MODEL_SPECS["qwen3-14b-q4-k-m"],
        }
    )
    assert validated == ["qwen3-0.6b-q8-0", "qwen3-14b-q4-k-m"]

    with pytest.raises(ExperimentError, match="exactly main and draft"):
        common.resolve_model_specs({})


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


def test_restart_qdrant_restarts_configured_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    environment = SimpleNamespace(
        database=SimpleNamespace(
            config=SimpleNamespace(
                url=common.DEFAULT_QDRANT_URL,
                collection_name="wikipedia",
            )
        ),
        manifest={"qdrant": {"container": "test-qdrant"}},
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda arguments, **kwargs: calls.append(arguments)
        or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    monkeypatch.setattr(
        common.httpx,
        "get",
        lambda url, **kwargs: calls.append(url) or SimpleNamespace(status_code=200),
    )

    common.restart_qdrant(environment)

    assert calls == [
        ["docker", "restart", "test-qdrant"],
        f"{common.DEFAULT_QDRANT_URL}/collections/wikipedia",
    ]


def test_restart_qdrant_rejects_remote_endpoint() -> None:
    environment = SimpleNamespace(
        database=SimpleNamespace(
            config=SimpleNamespace(url="https://example.test", collection_name="wiki")
        ),
        manifest={},
    )

    with pytest.raises(ExperimentError, match="local Docker Qdrant"):
        common.restart_qdrant(environment)


def test_restart_qdrant_reports_docker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = SimpleNamespace(
        database=SimpleNamespace(
            config=SimpleNamespace(
                url=common.DEFAULT_QDRANT_URL,
                collection_name="wiki",
            )
        ),
        manifest={},
    )
    monkeypatch.setattr(
        common.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1, stdout="", stderr="restart failed"
        ),
    )

    with pytest.raises(ExperimentError, match="restart failed"):
        common.restart_qdrant(environment)


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
    selected = tooluse._new_result()
    outcome = {
        "terminal_status": "timeout",
        "search_count": 2,
        "llm_calls": [_call(12, "one"), _call(8, "two")],
        "timing": {"end_to_end_ms": 50.0},
        "retrieval_calls": [
            {"encode_duration_ms": 2.0, "qdrant_duration_ms": 3.0}
        ],
    }

    tooluse._record_outcome(selected, outcome)

    assert selected["valid"] == 0
    assert selected["failed"] == 1
    assert selected["failure_statuses"] == {"timeout": 1}
    assert selected["incomplete_queries"] == [2.0]
    assert selected["incomplete_first"] == [12.0]
    assert selected["incomplete_between"] == [8.0]
    assert selected["incomplete_end_to_end"] == [50.0]
    assert selected["retrieval"] == {
        "encode": [2.0],
        "qdrant": [3.0],
        "total": [5.0],
    }


def test_tooluse_failed_run_survives_missing_token_usage() -> None:
    selected = tooluse._new_result()
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
    results = {"model": tooluse._new_result()}
    tooluse._record_outcome(
        results["model"],
        {
            "terminal_status": "timeout",
            "search_count": 1,
            "llm_calls": [_call(12, "one")],
        },
    )
    tooluse.render_report(results)

    output = capsys.readouterr().out
    assert "Incomplete-run metrics" in output
    assert "Failure status breakdown" in output
    assert "End-to-end latency histogram" in output
    assert "Qdrant RPC latency histogram" in output
    rows = [line.split() for line in output.splitlines()]
    assert ["model", "timeout", "1"] in rows
    assert "main" not in output
    assert "draft" not in output


def test_vectordb_report_has_only_hit_and_non_hit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    vectordb.render_report({"hit": [1.0], "non-hit": [2.0]}, 123_456)

    output = capsys.readouterr().out
    rows = [line.split() for line in output.splitlines()]
    assert ["hit", "1", "1.00", "1.00", "1.00", "0.00"] in rows
    assert ["non-hit", "1", "2.00", "2.00", "2.00", "0.00"] in rows
    assert "memory" not in output


def test_vectordb_run_measures_source_non_hit_then_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point_count = 123_456
    measurements: list[tuple[str, str, bool]] = []
    warmed: list[tuple[object, str, list[float]]] = []
    reported: list[tuple[dict[str, list[float]], int]] = []
    config = SimpleNamespace(
        url=vectordb.DEFAULT_QDRANT_URL,
        collection_name="source",
    )
    environment = SimpleNamespace(
        database=SimpleNamespace(config=config, client=object()),
        points_count=point_count,
        paths=SimpleNamespace(model_dir="model", bundle_dir="bundle"),
    )

    @contextmanager
    def prepare_wikipedia(**kwargs):
        assert kwargs == {"require_idle": True}
        yield environment

    class Encoder:
        def __init__(self, bundle_dir, *, require_complete):
            assert bundle_dir == "bundle"
            assert require_complete is False

        def encode(self, question):
            assert question == "question"
            return [0.1, 0.2]

    class Database:
        def __init__(self, received_config):
            assert received_config is config
            self.client = object()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        vectordb,
        "select_benchmark_questions",
        lambda count: [SimpleNamespace(question="question")],
    )
    monkeypatch.setattr(vectordb, "prepare_wikipedia", prepare_wikipedia)
    monkeypatch.setattr(vectordb, "Encoder", Encoder)
    monkeypatch.setattr(vectordb, "QdrantVectorDB", Database)

    def measure_rounds(
        environment,
        collection,
        vectors,
        repetitions,
        *,
        label,
        restart_each_round=False,
    ):
        assert vectors == [[0.1, 0.2]]
        assert repetitions == 1
        measurements.append((collection, label, restart_each_round))
        return [1.0 if label == "non-hit" else 2.0]

    monkeypatch.setattr(vectordb, "measure_rounds", measure_rounds)
    monkeypatch.setattr(
        vectordb,
        "query_latency_ms",
        lambda client, collection, vector: warmed.append(
            (client, collection, vector)
        )
        or 1.0,
    )
    monkeypatch.setattr(
        vectordb,
        "render_report",
        lambda latencies, count: reported.append((latencies, count)),
    )

    vectordb.run(query_count=1, repetitions=1)

    assert measurements == [
        ("source", "non-hit", True),
        ("source", "hit", False),
    ]
    assert [(collection, vector) for _, collection, vector in warmed] == [
        ("source", [0.1, 0.2])
    ]
    assert reported == [({"non-hit": [1.0], "hit": [2.0]}, point_count)]


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
