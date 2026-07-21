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
from .qdrant import QdrantConfig, QdrantVectorDB
from .qdrant_runtime import DEFAULT_QDRANT_URL, ensure_qdrant
from .types import SearchResult

DEFAULT_COLLECTION = "wikipedia_2024_06_bge_m3_en_v1"
DEFAULT_QUERY = "What causes auroras?"


@dataclass(frozen=True, slots=True)
class BundleSummary:
    paths: BundlePaths
    complete: bool
    shard_count: int
    completed_shards: int
    dataset_bytes: int
    model_bytes: int
    qdrant: dict[str, Any] | None


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
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


def inspect_bundle(bundle_dir: str | Path | None = None) -> BundleSummary:
    paths = BundlePaths.resolve(bundle_dir)
    manifest = load_manifest(paths, require_complete=False)
    declared = manifest.get("dataset", {}).get("shards", [])
    if not isinstance(declared, list) or not declared:
        raise ValueError("Manifest contains no dataset shards")
    shards = [(paths.bundle_dir / str(item)).resolve() for item in declared]
    qdrant = manifest.get("qdrant") if isinstance(manifest.get("qdrant"), dict) else None
    completed_shards = 0
    if qdrant is not None:
        checkpoint_value = qdrant.get("checkpoint")
        if checkpoint_value is None:
            endpoint = str(qdrant.get("url", DEFAULT_QDRANT_URL))
            collection = str(qdrant.get("collection", DEFAULT_COLLECTION))
            checkpoint = default_checkpoint_path(
                paths, "qdrant", collection, endpoint
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
    )


def benchmark_query(
    database: Any,
    encoder: Any,
    query: str,
    *,
    runs: int,
    limit: int,
    timer: Callable[[], float] = time.perf_counter,
) -> tuple[list[float], list[SearchResult]]:
    latencies: list[float] = []
    results: list[SearchResult] = []
    for _ in range(runs):
        started = timer()
        results = search_text(database, encoder, query, limit=limit)
        latencies.append((timer() - started) * 1000)
        if not results:
            raise RuntimeError("Qdrant returned no chunks")
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
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--runs", type=_positive_int, default=10)
    parser.add_argument("--limit", type=_positive_int, default=5)
    parser.add_argument("--url", help=f"Qdrant URL (default: {DEFAULT_QDRANT_URL})")
    parser.add_argument("--collection", help="Qdrant collection (or QDRANT_COLLECTION)")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        paths = BundlePaths.resolve(args.bundle_dir)
        print(f"bundle: inspecting {paths.bundle_dir}", flush=True)
        summary = inspect_bundle(paths.bundle_dir)
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
        with QdrantVectorDB(config) as database:
            if not database.client.collection_exists(collection):
                raise RuntimeError(f"Qdrant collection does not exist: {collection}")
            info = database.client.get_collection(collection)
            points = getattr(info, "points_count", None)
            status = getattr(info, "status", "unknown")
            status = getattr(status, "value", status)
            indexed = getattr(info, "indexed_vectors_count", None)
            segments = getattr(info, "segments_count", None)
            optimizer = getattr(info, "optimizer_status", "unknown")
            optimizer = getattr(optimizer, "value", optimizer)
            print(
                f"collection: name={collection}, status={status}, optimizer={optimizer}, "
                f"points={points}, indexed={indexed}, segments={segments}"
            )

            if not summary.complete and summary.model_bytes == 0:
                print(
                    "warning: BGE-M3 model is not available; skipping query benchmark",
                    file=sys.stderr,
                    flush=True,
                )
                return

            print("encoder: loading BGE-M3", flush=True)
            encoder_started = time.perf_counter()
            encoder = Encoder(summary.paths.bundle_dir, require_complete=False)
            print(f"encoder: ready in {time.perf_counter() - encoder_started:.1f} s")
            print(f"query: {DEFAULT_QUERY!r}, runs={args.runs}, limit={args.limit}")
            latencies, results = benchmark_query(
                database,
                encoder,
                DEFAULT_QUERY,
                runs=args.runs,
                limit=args.limit,
            )

        values = ", ".join(f"{value:.1f}" for value in latencies)
        print(f"latency_ms: [{values}]")
        print(
            f"summary: runs={args.runs}, mean={statistics.fmean(latencies):.1f} ms, "
            f"min={min(latencies):.1f} ms, max={max(latencies):.1f} ms"
        )
        _print_results(results)
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
