from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from wikipedia.encoder import Encoder

from .common import (
    ExperimentError,
    Progress,
    WikipediaEnvironment,
    backend_name,
    format_metric,
    open_database,
    prepare_wikipedia,
    print_table,
    prompt_positive_int,
    reset_vector_cache,
    select_benchmark_questions,
    summarize,
)

DESCRIPTION = "Measure vector database cache hit and non-hit retrieval latency"
DEFAULT_QUERIES = 20
DEFAULT_REPETITIONS = 10
TOP_K = 5


def query_latency_ms(
    database: Any,
    collection: str,
    vector: Sequence[float],
    *,
    timer_ns: Callable[[], int] = time.perf_counter_ns,
) -> float:
    """Time one search through the backend's own API.

    Goes through VectorDB.search rather than a client-specific call so the same
    measurement works against Qdrant and Milvus, and so it times the same path
    the agent's retriever takes.
    """
    started = timer_ns()
    results = database.search(list(vector), limit=TOP_K)
    ended = timer_ns()
    if not results:
        raise ExperimentError(f"vector search returned nothing from {collection}")
    return (ended - started) / 1_000_000


def measure_rounds(
    environment: WikipediaEnvironment,
    collection: str,
    vectors: list[list[float]],
    repetitions: int,
    *,
    label: str,
    restart_each_round: bool = False,
) -> list[float]:
    total = len(vectors) * repetitions
    progress = Progress(f"vectordb {label}", total)
    latencies: list[float] = []
    completed = 0
    for repetition in range(repetitions):
        if restart_each_round:
            reset_vector_cache(environment, collection)
        with open_database(environment) as database:
            for vector in vectors:
                latencies.append(query_latency_ms(database, collection, vector))
                completed += 1
                progress.update(completed, f"round={repetition + 1}")
    return latencies


def render_report(latencies: dict[str, list[float]], point_count: int) -> None:
    print(
        f"\nVector DB metrics (query_points only; points={point_count:,}, top_k={TOP_K})"
    )
    rows: list[list[object]] = []
    for target in ("hit", "non-hit"):
        stats = summarize(latencies[target])
        rows.append(
            [
                target,
                stats.count,
                format_metric(stats.mean),
                format_metric(stats.median),
                format_metric(stats.p95_worst),
                format_metric(stats.standard_deviation),
            ]
        )
    print_table(
        ("target", "n", "mean ms", "median ms", "p95-worst ms", "stddev ms"),
        rows,
    )
    print("p95-worst is numeric latency p95; lower is better.")
    print(
        "non-hit resets the backend before each round and remains best-effort. On "
        "Qdrant the container is restarted, which clears its in-process caches but "
        "not the host page cache. On Milvus the host page cache is dropped, which "
        "is where a DiskANN index warms because Knowhere reads it with pread, but "
        "Milvus's own node cache survives. Neither is a process-fresh measurement."
    )


def run(query_count: int, repetitions: int) -> None:
    questions = select_benchmark_questions(query_count)
    with prepare_wikipedia(require_idle=True) as environment:
        backend = backend_name(environment)
        point_count = environment.points_count
        if point_count < TOP_K:
            raise ExperimentError(
                f"collection has {point_count} points; at least {TOP_K} are required"
            )
        print(f"setup: backend {backend}", flush=True)
        print(f"setup: loading BGE-M3 from {environment.paths.model_dir}", flush=True)
        try:
            encoder = Encoder(environment.paths.bundle_dir, require_complete=False)
            query_vectors = [
                [float(value) for value in encoder.encode(question.question)]
                for question in questions
            ]
        except Exception as exc:
            raise ExperimentError(f"could not prepare query embeddings: {exc}") from exc

        collection = environment.database.config.collection_name
        latencies = {
            "non-hit": measure_rounds(
                environment,
                collection,
                query_vectors,
                repetitions,
                label="non-hit",
                restart_each_round=True,
            )
        }
        print("setup: warming collection for cache-hit target", flush=True)
        with open_database(environment) as database:
            for vector in query_vectors:
                query_latency_ms(database, collection, vector)
        latencies["hit"] = measure_rounds(
            environment,
            collection,
            query_vectors,
            repetitions,
            label="hit",
        )
        render_report(latencies, point_count)


def prompt_and_run(*, input_fn: Callable[[str], str] = input) -> None:
    query_count = prompt_positive_int(
        "FanOutQA query count", DEFAULT_QUERIES, input_fn=input_fn
    )
    repetitions = prompt_positive_int(
        "Repetitions per measurement target", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(query_count, repetitions)
