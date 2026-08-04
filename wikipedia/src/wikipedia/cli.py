from __future__ import annotations

import argparse
import os
from pathlib import Path

from .bundle import BundlePaths, atomic_json, load_manifest, write_bundle_marker
from .download import (
    DATASET_REVISION,
    DEFAULT_MAX_WORKERS,
    MODEL_REVISION,
    download_model,
    model_is_downloaded,
    prepare_bundle,
    status,
)
from .ingest import default_checkpoint_path, endpoint_fingerprint, ingest_wikipedia
from .milvus import (
    DEFAULT_INDEX_TYPE,
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
from .qdrant_runtime import (
    DEFAULT_QDRANT_CONTAINER,
    DEFAULT_QDRANT_GRPC_PORT,
    DEFAULT_QDRANT_IMAGE,
    DEFAULT_QDRANT_URL,
    ensure_qdrant,
)

DEFAULT_BATCH_SIZE = 256
# Milvus round-trips cost more than Qdrant's, so batches are larger by default.
DEFAULT_MILVUS_BATCH_SIZE = 1000
DEFAULT_TIMEOUT = 60.0
DEFAULT_COLLECTION = "wikipedia_2024_06_bge_m3_en_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and ingest Wikipedia vectors with bounded local storage"
    )
    parser.add_argument("backend", choices=("qdrant", "milvus"))
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        help="absolute working directory; remembered for later commands",
    )
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "parallel shard pipelines for remote Qdrant and model download workers "
            f"(default: {DEFAULT_MAX_WORKERS})"
        ),
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30,
        metavar="SECONDS",
        help="print download and ingest heartbeats (0 disables them)",
    )
    parser.add_argument(
        "--download-timeout",
        type=int,
        metavar="SECONDS",
        help="fail and retry a stalled Hugging Face request after this many seconds",
    )
    transfer = parser.add_mutually_exclusive_group()
    transfer.add_argument(
        "--high-performance",
        action="store_true",
        help="let hf-xet use more CPU, network, and disk bandwidth",
    )
    transfer.add_argument(
        "--disable-xet",
        action="store_true",
        help="use resumable HTTP instead of hf-xet when Xet repeatedly stalls",
    )
    parser.add_argument("--collection")
    parser.add_argument("--url", help="Qdrant URL")
    parser.add_argument("--path", type=Path, help="Local Qdrant storage path")
    parser.add_argument(
        "--float16",
        action="store_true",
        help="store Qdrant vectors as float16 instead of float32",
    )
    parser.add_argument(
        "--prefer-grpc",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--grpc-port", type=int)
    parser.add_argument("--uri", help="Milvus URI")
    parser.add_argument("--database", help="Milvus database")
    parser.add_argument(
        "--index-type",
        help=f"Milvus vector index (default: {DEFAULT_INDEX_TYPE})",
    )
    parser.add_argument(
        "--search-list",
        type=int,
        help=(
            "DiskANN candidate pool size; larger raises recall and latency "
            f"(default: {DEFAULT_SEARCH_LIST})"
        ),
    )
    parser.add_argument(
        "--milvus-storage-dir",
        type=Path,
        help="Local Milvus stack storage (default: <bundle-dir>/milvus)",
    )
    parser.add_argument(
        "--milvus-truncate-text",
        action="store_true",
        help=(
            "store chunks longer than Milvus' 65535-byte VARCHAR limit with the text "
            "cut to fit instead of failing; the embedding is unaffected and the number "
            "of truncated records is reported"
        ),
    )
    parser.add_argument(
        "--milvus-upsert",
        action="store_true",
        help=(
            "write with upsert instead of insert; idempotent but costs one delete "
            "tombstone per record. Needed only when resuming into rows that may "
            "already exist"
        ),
    )
    parser.add_argument("--timeout", type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        _main(argv)
    except KeyboardInterrupt:
        status("interrupted; incomplete downloads and shards retained for resume")
        raise SystemExit(130) from None


def _recorded_model_revision(paths: BundlePaths) -> str | None:
    try:
        manifest = load_manifest(paths, require_complete=False)
    except Exception:
        return None
    model = manifest.get("model")
    if not isinstance(model, dict):
        return None
    revision = model.get("resolved_revision")
    return revision if isinstance(revision, str) else None


def _finalize_milvus_index(database: MilvusVectorDB) -> None:
    """Seal segments and wait out the index build.

    Without this the newest rows stay in growing segments, where Milvus answers
    by brute-force scan instead of the on-disk index — which would silently
    invalidate any latency measured afterwards.
    """
    truncated = getattr(database, "truncated_records", 0)
    if truncated:
        status(
            f"truncated {truncated:,} chunks to Milvus' {database.config.text_max_bytes}-byte "
            "text limit; embeddings are unaffected but the stored payload is shorter "
            "than in the Qdrant collection"
        )
    status("flushing Milvus segments so the index can cover every row")
    database.flush()

    def report(state: dict[str, object]) -> None:
        status(
            f"index {state['index_type']}: indexed={state['indexed_rows']:,} "
            f"pending={state['pending_rows']:,} of {state['total_rows']:,}"
        )

    final = database.wait_for_index(poll=30.0, on_progress=report)
    status(
        f"index ready: type={final['index_type']}, "
        f"indexed={final['indexed_rows']:,} rows"
    )


def _main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bundle_dir is not None:
        bundle_dir = args.bundle_dir.expanduser()
        if not bundle_dir.is_absolute():
            parser.error("--bundle-dir must be an absolute path")
        bundle_dir = bundle_dir.resolve()
        bundle_dir.mkdir(parents=True, exist_ok=True)
        write_bundle_marker(bundle_dir)
        paths = BundlePaths.from_dir(bundle_dir)
    else:
        paths = BundlePaths.resolve()

    recorded_model_revision = _recorded_model_revision(paths)
    manifest = prepare_bundle(
        paths,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        max_workers=args.max_workers,
        max_shards=args.max_shards,
        progress_interval=args.progress_interval,
        download_timeout=args.download_timeout,
        high_performance=args.high_performance,
        disable_xet=args.disable_xet,
        token=os.environ.get("HF_TOKEN"),
    )
    saved: dict[str, object] = {}
    if args.backend == "qdrant":
        manifest_qdrant = manifest.get("qdrant")
        saved = manifest_qdrant if isinstance(manifest_qdrant, dict) else {}
        saved_float16 = bool(saved.get("float16", False))
        if saved and saved_float16 != args.float16:
            parser.error(
                "--float16 does not match saved Qdrant configuration; clean up "
                "existing Qdrant storage and remove the qdrant manifest section first"
            )

    resolved_model_revision = str(manifest["model"]["resolved_revision"])
    if (
        recorded_model_revision != resolved_model_revision
        or not model_is_downloaded(paths)
    ):
        download_model(
            paths,
            revision=resolved_model_revision,
            max_workers=args.max_workers,
            progress_interval=args.progress_interval,
            download_timeout=args.download_timeout,
            token=os.environ.get("HF_TOKEN"),
        )
    else:
        status("BGE-M3 model already downloaded; skipping model transfer")

    if args.backend == "qdrant":
        saved_url = (
            args.url
            or os.environ.get("QDRANT_URL")
            or saved.get("url")
            or DEFAULT_QDRANT_URL
        )
        url = None if args.path is not None else str(saved_url)
        collection = str(
            args.collection
            or os.environ.get("QDRANT_COLLECTION")
            or saved.get("collection")
            or "wikipedia_2024_06_bge_m3_en_v1"
        )
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else int(saved.get("batch_size", DEFAULT_BATCH_SIZE))
        )
        timeout = (
            args.timeout
            if args.timeout is not None
            else float(saved.get("timeout", DEFAULT_TIMEOUT))
        )
        prefer_grpc = (
            args.prefer_grpc
            if args.prefer_grpc is not None
            else bool(saved.get("prefer_grpc", True))
        )
        grpc_port = (
            args.grpc_port
            if args.grpc_port is not None
            else int(saved.get("grpc_port", DEFAULT_QDRANT_GRPC_PORT))
        )

        if args.path is None:
            assert url is not None
            stored_path = saved.get("storage_dir")
            if url.rstrip("/") == DEFAULT_QDRANT_URL:
                storage_dir = (
                    Path(stored_path).expanduser().resolve()
                    if stored_path is not None
                    else (paths.bundle_dir / "qdrant").resolve()
                )
            else:
                storage_dir = None
            container = str(saved.get("container", DEFAULT_QDRANT_CONTAINER))
            image = os.environ.get("QDRANT_IMAGE") or str(
                saved.get("image", DEFAULT_QDRANT_IMAGE)
            )
            runtime = ensure_qdrant(
                url,
                storage_dir=storage_dir,
                container=container,
                image=image,
            )
            print(f"Qdrant runtime: {runtime}")
            manifest["qdrant"] = {
                "schema_version": 1,
                "url": url,
                "storage_dir": str(storage_dir) if storage_dir is not None else None,
                "container": container,
                "image": image,
                "collection": collection,
                "float16": args.float16,
                "prefer_grpc": prefer_grpc,
                "grpc_port": grpc_port,
                "batch_size": batch_size,
                "timeout": timeout,
            }

        config = QdrantConfig(
            url=url,
            path=args.path,
            api_key=os.environ.get("QDRANT_API_KEY"),
            collection_name=collection,
            float16=args.float16,
            prefer_grpc=prefer_grpc,
            grpc_port=grpc_port,
            timeout=timeout,
        )
        database = QdrantVectorDB(config)
        metric = config.distance
    else:
        manifest_milvus = manifest.get("milvus")
        saved = manifest_milvus if isinstance(manifest_milvus, dict) else {}
        uri = str(
            args.uri
            or os.environ.get("MILVUS_URI")
            or saved.get("uri")
            or DEFAULT_MILVUS_URI
        )
        collection = str(
            args.collection
            or os.environ.get("MILVUS_COLLECTION")
            or saved.get("collection")
            or DEFAULT_COLLECTION
        )
        index_type = str(
            args.index_type or saved.get("index_type") or DEFAULT_INDEX_TYPE
        ).upper()
        search_list = (
            args.search_list
            if args.search_list is not None
            else int(saved.get("search_list", DEFAULT_SEARCH_LIST))
        )
        batch_size = (
            args.batch_size
            if args.batch_size is not None
            else int(saved.get("batch_size", DEFAULT_MILVUS_BATCH_SIZE))
        )
        timeout = (
            args.timeout
            if args.timeout is not None
            else float(saved.get("timeout", DEFAULT_TIMEOUT))
        )
        if args.milvus_storage_dir is not None:
            storage_dir = args.milvus_storage_dir.expanduser().resolve()
        else:
            stored_path = saved.get("storage_dir")
            storage_dir = (
                Path(str(stored_path)).expanduser().resolve()
                if stored_path is not None
                else (paths.bundle_dir / "milvus").resolve()
            )
        image = os.environ.get("MILVUS_IMAGE") or str(
            saved.get("image", DEFAULT_MILVUS_IMAGE)
        )
        project = str(saved.get("project", DEFAULT_MILVUS_PROJECT))
        runtime = ensure_milvus(
            uri, storage_dir=storage_dir, project=project, image=image
        )
        print(f"Milvus runtime: {runtime}")
        manifest["milvus"] = {
            "schema_version": 1,
            "uri": uri,
            "storage_dir": str(storage_dir),
            "project": project,
            "image": image,
            "database": args.database or os.environ.get("MILVUS_DB_NAME", "default"),
            "collection": collection,
            "index_type": index_type,
            "search_list": search_list,
            "batch_size": batch_size,
            "timeout": timeout,
        }
        config = MilvusConfig(
            uri=uri,
            token=os.environ.get("MILVUS_TOKEN"),
            database=str(manifest["milvus"]["database"]),
            collection_name=collection,
            index_type=index_type,
            search_params={"search_list": search_list},
            upsert_existing=args.milvus_upsert,
            truncate_text=args.milvus_truncate_text,
            timeout=timeout,
        )
        database = MilvusVectorDB(config)
        metric = config.metric_type

    checkpoint = args.checkpoint or default_checkpoint_path(
        paths, args.backend, config.collection_name, config.endpoint
    )
    checkpoint = Path(checkpoint).expanduser().resolve()
    if (
        isinstance(manifest.get("qdrant"), dict)
        and args.backend == "qdrant"
        and args.path is None
    ):
        manifest["qdrant"]["checkpoint"] = str(checkpoint)
    if args.backend == "milvus" and isinstance(manifest.get("milvus"), dict):
        manifest["milvus"]["checkpoint"] = str(checkpoint)
    atomic_json(paths.manifest_path, manifest)

    print(
        f"Ingesting into {args.backend} collection={config.collection_name} "
        f"endpoint_fingerprint={endpoint_fingerprint(config.endpoint)}"
    )
    parallel_qdrant = args.backend == "qdrant" and args.path is None
    ingest_workers = args.max_workers if parallel_qdrant else 1
    worker_database_factory = (
        (lambda: QdrantVectorDB(config))
        if parallel_qdrant and ingest_workers > 1
        else None
    )
    if parallel_qdrant:
        status(f"ingest workers: {ingest_workers}")
    else:
        status(
            "ingest workers: 1; parallel shard ingestion is only enabled for "
            "remote Qdrant (--max-workers still controls model download)"
        )
    with database:
        if args.backend == "milvus":
            # Load state survives restarts, so a collection left loaded by an
            # earlier benchmark would hold every inserted row in query-node
            # memory as well and eventually get the process OOM-killed.
            status("releasing Milvus collection so ingest does not fill query-node memory")
            database.release()
        count = ingest_wikipedia(
            database,
            paths.bundle_dir,
            backend=args.backend,
            endpoint=config.endpoint,
            collection=config.collection_name,
            metric=metric,
            batch_size=batch_size,
            checkpoint_path=checkpoint,
            max_shards=args.max_shards,
            max_records=args.max_records,
            progress_interval=args.progress_interval,
            download_timeout=args.download_timeout,
            token=os.environ.get("HF_TOKEN"),
            max_workers=ingest_workers,
            worker_database_factory=worker_database_factory,
        )
        if args.backend == "milvus":
            _finalize_milvus_index(database)
    manifest["status"] = "complete"
    atomic_json(paths.manifest_path, manifest)
    print(f"Ingested {count} records")


if __name__ == "__main__":
    main()
