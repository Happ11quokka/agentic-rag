from __future__ import annotations

import argparse
import os
from pathlib import Path

from .ingest import endpoint_fingerprint, ingest_wikipedia
from .milvus import MilvusConfig, MilvusVectorDB
from .qdrant import QdrantConfig, QdrantVectorDB


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest local Wikipedia vectors")
    parser.add_argument("backend", choices=("qdrant", "milvus"))
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--collection")
    parser.add_argument("--url", help="Qdrant URL")
    parser.add_argument("--path", type=Path, help="Local Qdrant storage path")
    parser.add_argument("--prefer-grpc", action="store_true")
    parser.add_argument("--grpc-port", type=int)
    parser.add_argument("--uri", help="Milvus URI")
    parser.add_argument("--database", help="Milvus database")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == "qdrant":
        url = args.url or os.environ.get("QDRANT_URL")
        if not url and args.path is None:
            parser.error("Qdrant requires QDRANT_URL, --url, or --path")
        config = QdrantConfig(
            url=url,
            path=args.path,
            api_key=os.environ.get("QDRANT_API_KEY"),
            collection_name=args.collection
            or os.environ.get("QDRANT_COLLECTION", "wikipedia_2024_06_bge_m3_en_v1"),
            prefer_grpc=args.prefer_grpc,
            grpc_port=args.grpc_port,
            timeout=args.timeout,
        )
        database = QdrantVectorDB(config)
        metric = config.distance
    else:
        config = MilvusConfig(
            uri=args.uri or os.environ.get("MILVUS_URI", "http://localhost:19530"),
            token=os.environ.get("MILVUS_TOKEN"),
            database=args.database or os.environ.get("MILVUS_DB_NAME", "default"),
            collection_name=args.collection
            or os.environ.get("MILVUS_COLLECTION", "wikipedia_2024_06_bge_m3_en_v1"),
            timeout=args.timeout,
        )
        database = MilvusVectorDB(config)
        metric = config.metric_type

    print(
        f"Ingesting into {args.backend} collection={config.collection_name} "
        f"endpoint_fingerprint={endpoint_fingerprint(config.endpoint)}"
    )
    with database:
        count = ingest_wikipedia(
            database,
            args.bundle_dir,
            backend=args.backend,
            endpoint=config.endpoint,
            collection=config.collection_name,
            metric=metric,
            batch_size=args.batch_size,
            checkpoint_path=args.checkpoint,
            max_shards=args.max_shards,
            max_records=args.max_records,
        )
    print(f"Ingested {count} records")


if __name__ == "__main__":
    main()
