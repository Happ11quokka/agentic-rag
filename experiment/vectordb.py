from __future__ import annotations

import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import httpx
from qdrant_client import models

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
    select_benchmark_questions,
    summarize,
)

DESCRIPTION = "Measure controlled Qdrant cache, memory, and SSD retrieval latency"
DEFAULT_QUERIES = 20
DEFAULT_REPETITIONS = 10
COPY_BATCH_SIZE = 256
TOP_K = 5
INDEXING_THRESHOLD = 10_000
OPTIMIZATION_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True, slots=True)
class TemporaryCollections:
    memory: str
    disk: str


def _collection_names() -> TemporaryCollections:
    suffix = uuid.uuid4().hex[:12]
    return TemporaryCollections(
        memory=f"experiment_memory_{suffix}",
        disk=f"experiment_disk_{suffix}",
    )


def create_collections(
    client: Any,
    names: TemporaryCollections,
    *,
    float16: bool,
    dimension: int,
    distance: str,
) -> None:
    datatype = models.Datatype.FLOAT16 if float16 else models.Datatype.FLOAT32
    metric = getattr(models.Distance, distance.upper())
    for name, on_disk in ((names.memory, False), (names.disk, True)):
        created = client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=dimension,
                distance=metric,
                on_disk=on_disk,
                datatype=datatype,
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=on_disk),
            on_disk_payload=on_disk,
            optimizers_config=models.OptimizersConfigDiff(
                indexing_threshold=INDEXING_THRESHOLD
            ),
        )
        if not created:
            raise ExperimentError(f"Qdrant did not create temporary collection {name}")


def copy_points(
    client: Any,
    source: str,
    names: TemporaryCollections,
    count: int,
) -> None:
    progress = Progress("vectordb prepare", count)
    offset: Any = None
    copied = 0
    while copied < count:
        records, offset = client.scroll(
            collection_name=source,
            limit=min(COPY_BATCH_SIZE, count - copied),
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )
        if not records:
            raise ExperimentError(
                f"source collection ended after {copied:,} of {count:,} requested points"
            )
        points = [
            models.PointStruct(
                id=record.id, vector=record.vector, payload=record.payload
            )
            for record in records
        ]
        for target in (names.memory, names.disk):
            client.upsert(collection_name=target, points=points, wait=True)
        copied += len(points)
        progress.update(copied, "copied into RAM and disk collections")
        if offset is None and copied < count:
            raise ExperimentError(
                f"source collection has only {copied:,} scrollable points; expected {count:,}"
            )


def wait_until_optimized(
    client: Any,
    collection: str,
    expected_points: int,
    *,
    timeout_seconds: float = OPTIMIZATION_TIMEOUT_SECONDS,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    minimum_indexed = max(0, expected_points - INDEXING_THRESHOLD)
    last_message = 0.0
    while time.monotonic() < deadline:
        info = client.get_collection(collection)
        raw_status = getattr(info, "status", "")
        status = str(getattr(raw_status, "value", raw_status)).lower()
        points = int(getattr(info, "points_count", 0) or 0)
        indexed = int(getattr(info, "indexed_vectors_count", 0) or 0)
        optimizer = getattr(info, "optimizer_status", "")
        optimizer_value = str(getattr(optimizer, "value", optimizer)).lower()
        optimizer_ok = "error" not in optimizer_value
        if (
            status == "green"
            and optimizer_ok
            and points == expected_points
            and indexed >= minimum_indexed
        ):
            return
        now = time.monotonic()
        if now - last_message >= 5:
            print(
                f"[vectordb optimize] {collection} status={status} "
                f"points={points:,}/{expected_points:,} indexed={indexed:,}",
                flush=True,
            )
            last_message = now
        time.sleep(1)
    raise ExperimentError(
        f"Qdrant collection {collection} did not finish indexing within "
        f"{timeout_seconds:.0f} seconds"
    )


def delete_collections(
    environment: WikipediaEnvironment, names: TemporaryCollections
) -> None:
    try:
        with QdrantVectorDB(environment.database.config) as database:
            for name in (names.memory, names.disk):
                if database.client.collection_exists(name):
                    database.client.delete_collection(name)
    except Exception as exc:
        print(
            f"warning: could not delete temporary Qdrant collections: {exc}",
            file=sys.stderr,
            flush=True,
        )


def restart_qdrant(environment: WikipediaEnvironment, collection: str) -> None:
    saved = environment.manifest.get("qdrant", {})
    url = environment.database.config.url or ""
    if url.rstrip("/") != DEFAULT_QDRANT_URL:
        raise ExperimentError(
            "vectordb storage tiers require the local Docker Qdrant endpoint "
            f"{DEFAULT_QDRANT_URL}; configured endpoint is {url}"
        )
    container = str(saved.get("container", "wikipedia-qdrant"))
    result = subprocess.run(
        ["docker", "restart", container], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise ExperimentError(
            f"could not restart Qdrant container {container}: {detail}"
        )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{url}/collections/{collection}", timeout=1)
            if response.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise ExperimentError(
        "Qdrant did not become ready within 120 seconds after restart"
    )


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
    for tier in ("cache", "memory", "ssd-best-effort"):
        stats = summarize(latencies[tier])
        rows.append(
            [
                tier,
                stats.count,
                format_metric(stats.mean),
                format_metric(stats.median),
                format_metric(stats.p95_worst),
                format_metric(stats.standard_deviation),
            ]
        )
    print_table(
        ("tier", "n", "mean ms", "median ms", "p95-worst ms", "stddev ms"),
        rows,
    )
    print("p95-worst is numeric latency p95; lower is better.")
    print(
        "ssd-best-effort resets the Qdrant process before each round; host OS page-cache "
        "eviction is not performed."
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

        names = _collection_names()
        try:
            create_collections(
                environment.database.client,
                names,
                float16=environment.database.config.float16,
                dimension=environment.database.config.dimension,
                distance=environment.database.config.distance,
            )
            copy_points(
                environment.database.client,
                environment.database.config.collection_name,
                names,
                point_count,
            )
            wait_until_optimized(environment.database.client, names.memory, point_count)
            wait_until_optimized(environment.database.client, names.disk, point_count)

            latencies = {
                "ssd-best-effort": measure_rounds(
                    environment,
                    names.disk,
                    query_vectors,
                    repetitions,
                    label="ssd",
                    restart_each_round=True,
                )
            }
            print("setup: warming on-disk collection for cache tier", flush=True)
            with QdrantVectorDB(environment.database.config) as database:
                for vector in query_vectors:
                    query_latency_ms(database.client, names.disk, vector)
            latencies["cache"] = measure_rounds(
                environment,
                names.disk,
                query_vectors,
                repetitions,
                label="cache",
            )
            latencies["memory"] = measure_rounds(
                environment,
                names.memory,
                query_vectors,
                repetitions,
                label="memory",
            )
            render_report(latencies, point_count)
        finally:
            delete_collections(environment, names)


def prompt_and_run(*, input_fn: Callable[[str], str] = input) -> None:
    query_count = prompt_positive_int(
        "FanOutQA query count", DEFAULT_QUERIES, input_fn=input_fn
    )
    repetitions = prompt_positive_int(
        "Repetitions per storage tier", DEFAULT_REPETITIONS, input_fn=input_fn
    )
    run(query_count, repetitions)
