from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any

from wikipedia.encoder import Encoder
from wikipedia.qdrant import QdrantVectorDB
from wikipedia.qdrant_runtime import DEFAULT_QDRANT_URL

from .common import (
    ExperimentError,
    Progress,
    WikipediaEnvironment,
    format_metric,
    prepare_wikipedia,
    print_table,
    prompt_positive_int,
    restart_qdrant,
    select_benchmark_questions,
    summarize,
)

DESCRIPTION = "Measure Qdrant cache hit and non-hit retrieval latency"
DEFAULT_QUERIES = 20
DEFAULT_REPETITIONS = 10
TOP_K = 5


def query_latency_ms(
    client: Any,
    collection: str,
    vector: Sequence[float],
    *,
    timer_ns: Callable[[], int] = time.perf_counter_ns,
) -> float:
    started = timer_ns()
    response = client.query_points(
        collection_name=collection,
        query=list(vector),
        limit=TOP_K,
        with_payload=True,
        with_vectors=False,
    )
    ended = timer_ns()
    points = getattr(response, "points", [])
    if not points:
        raise ExperimentError(f"Qdrant returned no points from {collection}")
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
            restart_qdrant(environment, collection)
        with QdrantVectorDB(environment.database.config) as database:
            for vector in vectors:
                latencies.append(query_latency_ms(database.client, collection, vector))
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
        "non-hit resets Qdrant before each round, but remains best-effort: host OS "
        "page-cache eviction is not performed and later queries may benefit from warming."
    )


def run(query_count: int, repetitions: int) -> None:
    questions = select_benchmark_questions(query_count)
    with prepare_wikipedia(require_idle=True) as environment:
        url = environment.database.config.url or ""
        if url.rstrip("/") != DEFAULT_QDRANT_URL:
            raise ExperimentError(
                "vectordb experiment requires local Docker Qdrant so it can control "
                f"cold starts; configured endpoint is {url}"
            )
        point_count = environment.points_count
        if point_count < TOP_K:
            raise ExperimentError(
                f"Qdrant has {point_count} points; at least {TOP_K} are required"
            )
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
        with QdrantVectorDB(environment.database.config) as database:
            for vector in query_vectors:
                query_latency_ms(database.client, collection, vector)
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
