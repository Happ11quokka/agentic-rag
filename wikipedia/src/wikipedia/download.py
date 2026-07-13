from __future__ import annotations

import argparse
import fnmatch
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bundle import BundlePaths, atomic_json, write_bundle_marker

DATASET_REPOSITORY = "Upstash/wikipedia-2024-06-bge-m3"
DATASET_REVISION = "ce4e0ea49276d99975816b8bc85a85bf416b8b62"
MODEL_REPOSITORY = "BAAI/bge-m3"
MODEL_REVISION = "babcf60cae0a1f438d7ade582983d4ba462303c2"
DATASET_PATTERN = "data/en/*.parquet"
MODEL_ALLOW_PATTERNS = [
    "*.json",
    "*.model",
    "*.txt",
    "*.safetensors",
    "pytorch_model*.bin",
    "tokenizer*",
    "1_Pooling/*.json",
]
MODEL_IGNORE_PATTERNS = [
    "onnx/**",
    "*.onnx",
    "*.png",
    "*.jpg",
    "*.jpeg",
    "colbert_linear.pt",
    "sparse_linear.pt",
]


def download_bundle(
    output_dir: str | Path | None = None,
    *,
    dataset_revision: str = DATASET_REVISION,
    model_revision: str = MODEL_REVISION,
    max_workers: int = 8,
    max_shards: int | None = None,
    token: str | None = None,
    api: Any = None,
    snapshot: Any = None,
) -> dict[str, Any]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be at least 1")

    if output_dir is not None:
        absolute = Path(output_dir).expanduser()
        if not absolute.is_absolute():
            raise ValueError("--output-dir must be an absolute path")
        absolute = absolute.resolve()
        absolute.mkdir(parents=True, exist_ok=True)
        write_bundle_marker(absolute)
        paths = BundlePaths.from_dir(absolute)
    else:
        paths = BundlePaths.resolve()

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.unlink(missing_ok=True)

    if api is None or snapshot is None:
        from huggingface_hub import HfApi, snapshot_download

        api = api or HfApi(token=token)
        snapshot = snapshot or snapshot_download

    dataset_info = api.dataset_info(DATASET_REPOSITORY, revision=dataset_revision)
    model_info = api.model_info(MODEL_REPOSITORY, revision=model_revision)
    resolved_dataset_revision = dataset_info.sha
    resolved_model_revision = model_info.sha
    files = api.list_repo_files(
        DATASET_REPOSITORY, revision=resolved_dataset_revision, repo_type="dataset"
    )
    shards = sorted(name for name in files if fnmatch.fnmatch(name, DATASET_PATTERN))
    if not shards:
        raise RuntimeError(f"No files matched {DATASET_PATTERN} in {DATASET_REPOSITORY}")
    selected_shards = shards[:max_shards]

    common = {"token": token, "max_workers": max_workers}
    snapshot(
        repo_id=DATASET_REPOSITORY,
        repo_type="dataset",
        revision=resolved_dataset_revision,
        local_dir=paths.bundle_dir / "dataset",
        allow_patterns=selected_shards,
        **common,
    )
    snapshot(
        repo_id=MODEL_REPOSITORY,
        repo_type="model",
        revision=resolved_model_revision,
        local_dir=paths.model_dir,
        allow_patterns=MODEL_ALLOW_PATTERNS,
        ignore_patterns=MODEL_IGNORE_PATTERNS,
        **common,
    )

    missing = [name for name in selected_shards if not (paths.bundle_dir / "dataset" / name).is_file()]
    if missing:
        raise RuntimeError(f"Downloaded dataset shard is missing: {missing[0]}")
    if not any(path.is_file() for path in paths.model_dir.rglob("*")):
        raise RuntimeError(f"Downloaded model directory is empty: {paths.model_dir}")

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now(UTC).isoformat(),
        "language": "en",
        "partial": len(selected_shards) < len(shards),
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "requested_revision": dataset_revision,
            "resolved_revision": resolved_dataset_revision,
            "shards": [f"dataset/{name}" for name in selected_shards],
            "selected_shard_count": len(selected_shards),
            "available_shard_count": len(shards),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "requested_revision": model_revision,
            "resolved_revision": resolved_model_revision,
            "path": str(paths.model_dir.relative_to(paths.bundle_dir)),
        },
    }
    atomic_json(paths.manifest_path, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download English Wikipedia vectors and BGE-M3")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-shards", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = download_bundle(
        args.output_dir,
        dataset_revision=args.dataset_revision,
        model_revision=args.model_revision,
        max_workers=args.max_workers,
        max_shards=args.max_shards,
        token=os.environ.get("HF_TOKEN"),
    )
    kind = "partial" if manifest["partial"] else "full"
    print(f"Wikipedia bundle ready ({kind}, {manifest['dataset']['selected_shard_count']} shards)")


if __name__ == "__main__":
    main()
