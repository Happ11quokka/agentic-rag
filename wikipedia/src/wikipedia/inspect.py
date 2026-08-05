from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bundle import BundlePaths, load_manifest
from .encoder import Encoder, search_text
from .ingest import default_checkpoint_path
from .milvus import (
    DEFAULT_LOAD_TIMEOUT,
    DEFAULT_SEARCH_LIST,
    MilvusConfig,
    MilvusVectorDB,
)
from .milvus_runtime import (
    DEFAULT_MILVUS_IMAGE,
    DEFAULT_MILVUS_PROJECT,
    DEFAULT_MILVUS_URI,
    ensure_milvus,
)
from .qdrant import QdrantConfig, QdrantVectorDB
from .qdrant_runtime import DEFAULT_QDRANT_URL, ensure_qdrant
from .types import SearchResult

DEFAULT_COLLECTION = "wikipedia_2024_06_bge_m3_en_v1"

# Timing the same query repeatedly measures the page cache, not the index: the
# second search walks the graph path the first one already faulted in. That is
# the wrong measurement for an on-disk index, where the disk reads are the cost
# under study. These are semantically spread so consecutive searches land in
# different regions of the embedding space and touch different pages.
DEFAULT_QUERIES: tuple[str, ...] = (
    "What causes auroras?",
    "How does photosynthesis convert light into chemical energy?",
    "Who composed the Brandenburg Concertos?",
    "How do mRNA vaccines train the immune system?",
    "Why did the Roman Republic become an empire?",
    "How are black holes detected?",
    "What role do mitochondria play in a cell?",
    "How does a semiconductor transistor switch current?",
    "What caused the 1929 stock market crash?",
    "How do migratory birds navigate?",
    "What is plate tectonics?",
    "Who wrote the Tale of Genji?",
    "How does anesthesia produce unconsciousness?",
    "What is the Riemann hypothesis?",
    "How did the printing press change Europe?",
    "What are the stages of stellar evolution?",
    "How does the human eye perceive colour?",
    "What were the origins of the Silk Road?",
    "How do glaciers shape valleys?",
    "What is the function of the blood-brain barrier?",
)


@dataclass(frozen=True, slots=True)
class BundleSummary:
    paths: BundlePaths
    complete: bool
    shard_count: int
    completed_shards: int
    dataset_bytes: int
    model_bytes: int
    qdrant: dict[str, Any] | None
    milvus: dict[str, Any] | None = None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _tree_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(root).parts
        and not path.name.startswith("._")
    )


def inspect_bundle(
    bundle_dir: str | Path | None = None, *, backend: str = "qdrant"
) -> BundleSummary:
    paths = BundlePaths.resolve(bundle_dir)
    manifest = load_manifest(paths, require_complete=False)
    declared = manifest.get("dataset", {}).get("shards", [])
    if not isinstance(declared, list) or not declared:
        raise ValueError("Manifest contains no dataset shards")
    shards = [(paths.bundle_dir / str(item)).resolve() for item in declared]
    qdrant = manifest.get("qdrant") if isinstance(manifest.get("qdrant"), dict) else None
    milvus = manifest.get("milvus") if isinstance(manifest.get("milvus"), dict) else None
    saved = milvus if backend == "milvus" else qdrant
    completed_shards = 0
    if saved is not None:
        checkpoint_value = saved.get("checkpoint")
        if checkpoint_value is None:
            endpoint = (
                f"{str(saved.get('uri', DEFAULT_MILVUS_URI)).rstrip('/')}"
                f"|db={saved.get('database', 'default')}"
                if backend == "milvus"
                else str(saved.get("url", DEFAULT_QDRANT_URL))
            )
            collection = str(saved.get("collection", DEFAULT_COLLECTION))
            checkpoint = default_checkpoint_path(
                paths, backend, collection, endpoint
            )
        else:
            checkpoint = Path(str(checkpoint_value)).expanduser().resolve()
        if checkpoint.is_file():
            try:
                checkpoint_data = json.loads(checkpoint.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Malformed ingestion checkpoint {checkpoint}: {exc}") from exc
            completed = checkpoint_data.get("completed_shards", [])
            if not isinstance(completed, list):
                raise ValueError(f"Malformed completed_shards in {checkpoint}")
            completed_shards = len({str(item) for item in completed} & set(declared))
    return BundleSummary(
        paths=paths,
        complete=manifest.get("status") == "complete",
        shard_count=len(shards),
        completed_shards=completed_shards,
        dataset_bytes=sum(path.stat().st_size for path in shards if path.is_file()),
        model_bytes=_tree_bytes(paths.model_dir),
        qdrant=qdrant,
        milvus=milvus,
    )


def benchmark_query(
    database: Any,
    encoder: Any,
    queries: Sequence[str],
    *,
    limit: int,
    timer: Callable[[], float] = time.perf_counter,
    on_result: Callable[[int, str, float], None] | None = None,
) -> tuple[list[float], list[SearchResult]]:
    """Time one search per query, in order.

    Queries must be distinct. Repeating one would report the page cache rather
    than the index, which is the measurement this benchmark exists to avoid.
    """
    if not queries:
        raise ValueError("benchmark needs at least one query")
    if len(set(queries)) != len(queries):
        raise ValueError(
            "benchmark queries must be distinct; a repeated query re-walks the "
            "graph path the previous search already cached and reports the page "
            "cache instead of the index"
        )
    latencies: list[float] = []
    results: list[SearchResult] = []
    for query in queries:
        started = timer()
        results = search_text(database, encoder, query, limit=limit)
        latency_ms = (timer() - started) * 1000
        latencies.append(latency_ms)
        if on_result is not None:
            on_result(len(latencies), query, latency_ms)
        if not results:
            raise RuntimeError("vector database returned no chunks")
    return latencies, results


def _preview(text: str, limit: int = 160) -> str:
    compact = " ".join(text.split())
    return compact if len(compact) <= limit else compact[: limit - 1] + "…"


def _print_results(results: Sequence[SearchResult]) -> None:
    for index, result in enumerate(results, 1):
        print(f"[{index}] score={result.score:.4f} title={result.title}")
        print(f"    {_preview(result.text)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect a Wikipedia bundle and benchmark text-to-chunks latency"
    )
    parser.add_argument("--backend", choices=("qdrant", "milvus"), default="qdrant")
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--runs", type=_positive_int, default=10)
    parser.add_argument("--limit", type=_positive_int, default=5)
    parser.add_argument("--url", help=f"Qdrant URL (default: {DEFAULT_QDRANT_URL})")
    parser.add_argument("--uri", help=f"Milvus URI (default: {DEFAULT_MILVUS_URI})")
    parser.add_argument(
        "--search-list",
        type=_positive_int,
        help=f"DiskANN candidate pool size (default: {DEFAULT_SEARCH_LIST})",
    )
    parser.add_argument("--collection", help="collection name")
    parser.add_argument(
        "--load-timeout",
        type=_positive_float,
        help=(
            "seconds to wait for the Milvus collection load (default: "
            f"{DEFAULT_LOAD_TIMEOUT:.0f}). Load time scales with the "
            "collection and the storage medium — 1M took 1,529 s off this HDD — "
            "so a larger collection needs a larger budget than the default"
        ),
    )
    parser.add_argument(
        "--query",
        action="append",
        dest="queries",
        metavar="TEXT",
        help=(
            "query to time; repeat to supply your own set. Each run uses a "
            "different query, so at least --runs of them are needed. Defaults to "
            f"the first --runs of {len(DEFAULT_QUERIES)} built-in queries"
        ),
    )
    return parser


def select_queries(runs: int, queries: Sequence[str] | None) -> list[str]:
    """Pick one distinct query per run, from the built-in pool unless overridden.

    Refuses to recycle a query rather than silently reporting the page cache on
    every run past the first pass.
    """
    pool = list(queries) if queries else list(DEFAULT_QUERIES)
    if len(set(pool)) != len(pool):
        raise ValueError("queries must be distinct; a repeated query measures the page cache")
    if runs > len(pool):
        raise ValueError(
            f"--runs {runs} needs {runs} distinct queries but only {len(pool)} "
            "are available; pass more --query values or lower --runs. Reusing a "
            "query would measure the page cache instead of the index"
        )
    return pool[:runs]


def _open_qdrant(args: argparse.Namespace, summary: BundleSummary) -> Any:
    if summary.qdrant is None:
        raise RuntimeError(
            "Qdrant is not configured; run `uv run wikipedia-ingest qdrant "
            "--bundle-dir /absolute/path` first"
        )
    saved = summary.qdrant
    url = args.url or os.environ.get("QDRANT_URL") or str(
        saved.get("url", DEFAULT_QDRANT_URL)
    )
    collection = args.collection or os.environ.get("QDRANT_COLLECTION") or str(
        saved.get("collection", DEFAULT_COLLECTION)
    )
    runtime = ensure_qdrant(
        url,
        storage_dir=saved.get("storage_dir"),
        container=str(saved.get("container", "wikipedia-qdrant")),
        image=str(saved.get("image", "qdrant/qdrant:latest")),
    )
    print(f"qdrant: {runtime} at {url}")

    config = QdrantConfig(
        url=url,
        api_key=os.environ.get("QDRANT_API_KEY"),
        collection_name=collection,
        float16=bool(saved.get("float16", False)),
        prefer_grpc=bool(saved.get("prefer_grpc", True)),
        grpc_port=int(saved.get("grpc_port", 6334)),
        timeout=float(saved.get("timeout", 60.0)),
    )
    database = QdrantVectorDB(config)
    if not database.client.collection_exists(collection):
        raise RuntimeError(f"Qdrant collection does not exist: {collection}")
    info = database.client.get_collection(collection)
    status = getattr(info, "status", "unknown")
    optimizer = getattr(info, "optimizer_status", "unknown")
    print(
        f"collection: name={collection}, "
        f"status={getattr(status, 'value', status)}, "
        f"optimizer={getattr(optimizer, 'value', optimizer)}, "
        f"points={getattr(info, 'points_count', None)}, "
        f"indexed={getattr(info, 'indexed_vectors_count', None)}, "
        f"segments={getattr(info, 'segments_count', None)}"
    )
    return database


def _open_milvus(args: argparse.Namespace, summary: BundleSummary) -> Any:
    return open_milvus(
        summary,
        uri=args.uri,
        collection=args.collection,
        search_list=args.search_list,
        load_timeout=args.load_timeout,
    )


def open_milvus(
    summary: BundleSummary,
    *,
    uri: str | None = None,
    collection: str | None = None,
    search_list: int | None = None,
    load_timeout: float | None = None,
) -> Any:
    """Bring the stack up, validate the collection, and load it.

    Shared by every command that measures against Milvus, so they all report the
    same collection facts and pay the load cost in the same place.
    """
    if summary.milvus is None:
        raise RuntimeError(
            "Milvus is not configured; run `uv run wikipedia-ingest milvus "
            "--bundle-dir /absolute/path` first"
        )
    saved = summary.milvus
    uri = uri or os.environ.get("MILVUS_URI") or str(
        saved.get("uri", DEFAULT_MILVUS_URI)
    )
    collection = collection or os.environ.get("MILVUS_COLLECTION") or str(
        saved.get("collection", DEFAULT_COLLECTION)
    )
    search_list = (
        search_list
        if search_list is not None
        else int(saved.get("search_list", DEFAULT_SEARCH_LIST))
    )
    runtime = ensure_milvus(
        uri,
        storage_dir=saved.get("storage_dir"),
        project=str(saved.get("project", DEFAULT_MILVUS_PROJECT)),
        image=str(saved.get("image", DEFAULT_MILVUS_IMAGE)),
        # Recorded in the bundle rather than passed per-run: ensure_milvus
        # rewrites the compose file when it differs, so a run that forgot this
        # would move etcd back and restart the stack mid-experiment.
        etcd_dir=saved.get("etcd_dir"),
    )
    print(f"milvus: {runtime} at {uri}")

    load_timeout = (
        load_timeout
        if load_timeout is not None
        else float(saved.get("load_timeout", DEFAULT_LOAD_TIMEOUT))
    )
    config = MilvusConfig(
        uri=uri,
        token=os.environ.get("MILVUS_TOKEN"),
        database=str(saved.get("database", "default")),
        collection_name=collection,
        index_type=str(saved.get("index_type", "DISKANN")),
        search_params={"search_list": search_list},
        timeout=float(saved.get("timeout", 60.0)),
        load_timeout=load_timeout,
    )
    database = MilvusVectorDB(config)
    if not database.client.has_collection(collection, timeout=config.timeout):
        raise RuntimeError(f"Milvus collection does not exist: {collection}")
    state = database.index_state()
    print(
        f"collection: name={collection}, index={state['index_type']}, "
        f"state={state['state']}, rows={state['total_rows']:,}, "
        f"indexed={state['indexed_rows']:,}, pending={state['pending_rows']:,}"
    )
    # pending_rows is not a usable signal here — Milvus 2.5.27 reports it equal to
    # total_rows once the build is done. Uncovered rows show up as indexed < total.
    uncovered = state["total_rows"] - state["indexed_rows"]
    if uncovered > 0:
        print(
            f"warning: {uncovered:,} rows are not covered by the index; those "
            "segments are answered by brute-force scan, so latency below is not a "
            "clean on-disk index measurement",
            file=sys.stderr,
            flush=True,
        )
    print(f"search_list: {search_list}")

    # Load before benchmarking so the first query does not report the cost of
    # reading the on-disk index off the storage medium as its search latency.
    print(f"milvus: loading collection (timeout {load_timeout:.0f} s)", flush=True)
    started = time.perf_counter()
    database.load()
    print(f"milvus: collection loaded in {time.perf_counter() - started:.1f} s")
    return database


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = BundlePaths.resolve(args.bundle_dir)
        print(f"bundle: inspecting {paths.bundle_dir}", flush=True)
        summary = inspect_bundle(paths.bundle_dir, backend=args.backend)
        if not summary.complete:
            print(
                "warning: bundle is incomplete; inspection reflects only currently "
                "ingested data",
                file=sys.stderr,
                flush=True,
            )
        bundle_status = "complete" if summary.complete else "incomplete"
        print(
            f"bundle: {bundle_status}, {summary.completed_shards}/{summary.shard_count} "
            f"shards ingested, {_format_bytes(summary.dataset_bytes)} raw data retained"
        )
        print(f"model: {_format_bytes(summary.model_bytes)}")
        database = (
            _open_milvus(args, summary)
            if args.backend == "milvus"
            else _open_qdrant(args, summary)
        )
        with database:
            if not summary.complete and summary.model_bytes == 0:
                print(
                    "warning: BGE-M3 model is not available; skipping query benchmark",
                    file=sys.stderr,
                    flush=True,
                )
                return

            queries = select_queries(args.runs, args.queries)
            print("encoder: loading BGE-M3", flush=True)
            encoder_started = time.perf_counter()
            encoder = Encoder(summary.paths.bundle_dir, require_complete=False)
            print(
                f"encoder: ready in {time.perf_counter() - encoder_started:.1f} s",
                flush=True,
            )
            print(
                f"queries: {len(queries)} distinct, runs={args.runs}, limit={args.limit}",
                flush=True,
            )
            latencies, results = benchmark_query(
                database,
                encoder,
                queries,
                limit=args.limit,
                # A cold search on this collection runs for minutes. Without a
                # line per query the log is silent from here to the summary,
                # which is indistinguishable from a hang.
                on_result=lambda index, query, ms: print(
                    f"  [{index}/{len(queries)}] {ms:9.1f} ms  {query}", flush=True
                ),
            )

        values = ", ".join(f"{value:.1f}" for value in latencies)
        print(f"latency_ms: [{values}]")
        print(
            f"summary: runs={args.runs}, cold={latencies[0]:.1f} ms, "
            f"mean={statistics.fmean(latencies):.1f} ms, "
            f"median={statistics.median(latencies):.1f} ms, "
            f"min={min(latencies):.1f} ms, max={max(latencies):.1f} ms"
        )
        print(f"last query: {queries[-1]!r}")
        _print_results(results)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
