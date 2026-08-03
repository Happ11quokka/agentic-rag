from __future__ import annotations

import fnmatch
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .bundle import BundlePaths, atomic_json, load_manifest

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
MODEL_REQUIRED_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
MODEL_WEIGHT_PATTERNS = ("*.safetensors", "pytorch_model*.bin")
DEFAULT_MAX_WORKERS = 4


def status(message: str) -> None:
    print(f"[wikipedia-ingest] {message}", file=sys.stderr, flush=True)


def _format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and _size(path) > 0


def model_is_downloaded(paths: BundlePaths) -> bool:
    if not all(
        _is_nonempty_file(paths.model_dir / filename)
        for filename in MODEL_REQUIRED_FILES
    ):
        return False
    return any(
        _is_nonempty_file(path)
        for pattern in MODEL_WEIGHT_PATTERNS
        for path in paths.model_dir.glob(pattern)
    )


def _local_progress(root: Path, files: Sequence[str] | None) -> tuple[int, int]:
    if files is None:
        completed = [
            path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts
        ]
    else:
        completed = [root / name for name in files if (root / name).is_file()]
    partial = root / ".cache" / "huggingface" / "download"
    partial_bytes = (
        sum(_size(path) for path in partial.rglob("*.incomplete") if path.is_file())
        if partial.is_dir()
        else 0
    )
    return len(completed), sum(_size(path) for path in completed) + partial_bytes


@contextmanager
def progress_heartbeat(
    label: str,
    interval: float,
    detail: Callable[[], str] | None = None,
) -> Iterator[None]:
    if interval == 0:
        yield
        return

    stopped = threading.Event()
    started = time.monotonic()

    def report() -> None:
        while not stopped.wait(interval):
            elapsed = int(time.monotonic() - started)
            suffix = f", {detail()}" if detail is not None else ""
            status(
                f"{label}: still running ({elapsed // 60:02d}:{elapsed % 60:02d})"
                f"{suffix}"
            )

    thread = threading.Thread(target=report, name=f"{label}-progress", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def _configure_transfer(
    *,
    max_workers: int,
    progress_interval: float,
    download_timeout: int | None,
    high_performance: bool,
    disable_xet: bool,
) -> None:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1")
    if progress_interval < 0:
        raise ValueError("progress_interval must not be negative")
    if download_timeout is not None and download_timeout < 1:
        raise ValueError("download_timeout must be at least 1")
    if high_performance and disable_xet:
        raise ValueError("high_performance and disable_xet cannot be used together")

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except ImportError:
        pass
    if download_timeout is not None:
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(download_timeout)
    if high_performance:
        os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    if disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"


def cancel_transfers() -> None:
    try:
        from huggingface_hub.utils import close_session

        close_session()
    except Exception:
        pass
    try:
        from huggingface_hub.utils._xet import abort_xet_session

        abort_xet_session()
    except Exception:
        pass


def prepare_bundle(
    paths: BundlePaths,
    *,
    dataset_revision: str = DATASET_REVISION,
    model_revision: str = MODEL_REVISION,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_shards: int | None = None,
    progress_interval: float = 30,
    download_timeout: int | None = None,
    high_performance: bool = False,
    disable_xet: bool = False,
    token: str | None = None,
    api: Any = None,
) -> dict[str, Any]:
    if max_shards is not None and max_shards < 1:
        raise ValueError("max_shards must be at least 1")
    _configure_transfer(
        max_workers=max_workers,
        progress_interval=progress_interval,
        download_timeout=download_timeout,
        high_performance=high_performance,
        disable_xet=disable_xet,
    )

    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    paths.model_dir.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)

    previous: dict[str, Any] = {}
    if paths.manifest_path.is_file():
        try:
            previous = load_manifest(paths, require_complete=False)
        except Exception:
            pass

    if api is None:
        from huggingface_hub import HfApi, constants

        if download_timeout is not None:
            constants.HF_HUB_DOWNLOAD_TIMEOUT = download_timeout
        if high_performance:
            constants.HF_XET_HIGH_PERFORMANCE = True
        if disable_xet:
            constants.HF_HUB_DISABLE_XET = True
        api = HfApi(token=token)

    status(f"bundle directory: {paths.bundle_dir}")
    status("resolving pinned dataset and model revisions")
    info_options = {"timeout": download_timeout} if download_timeout is not None else {}
    dataset_info = api.dataset_info(
        DATASET_REPOSITORY, revision=dataset_revision, **info_options
    )
    model_info = api.model_info(MODEL_REPOSITORY, revision=model_revision, **info_options)
    files = api.list_repo_files(
        DATASET_REPOSITORY, revision=dataset_info.sha, repo_type="dataset"
    )
    shards = sorted(name for name in files if fnmatch.fnmatch(name, DATASET_PATTERN))
    if not shards:
        raise RuntimeError(f"No files matched {DATASET_PATTERN} in {DATASET_REPOSITORY}")
    selected = shards[:max_shards]

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "incomplete",
        "created_at": previous.get("created_at", datetime.now(UTC).isoformat()),
        "language": "en",
        "partial": len(selected) < len(shards),
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "requested_revision": dataset_revision,
            "resolved_revision": dataset_info.sha,
            "retention": "ephemeral",
            "shards": [f"dataset/{name}" for name in selected],
            "selected_shard_count": len(selected),
            "available_shard_count": len(shards),
        },
        "model": {
            "repository": MODEL_REPOSITORY,
            "requested_revision": model_revision,
            "resolved_revision": model_info.sha,
            "path": str(paths.model_dir.relative_to(paths.bundle_dir)),
        },
    }
    for backend in ("qdrant", "milvus"):
        if isinstance(previous.get(backend), dict):
            manifest[backend] = previous[backend]
    atomic_json(paths.manifest_path, manifest)
    status(f"dataset: {len(selected)}/{len(shards)} shards selected")
    return manifest


def download_dataset_shard(
    paths: BundlePaths,
    filename: str,
    *,
    revision: str,
    progress_interval: float,
    download_timeout: int | None,
    token: str | None,
    downloader: Any = None,
) -> Path:
    if downloader is None:
        from huggingface_hub import hf_hub_download

        downloader = hf_hub_download

    root = paths.bundle_dir / "dataset"
    destination = root / filename
    completed, size = _local_progress(root, [filename])
    status(
        f"downloading {filename}: {completed}/1 files complete, "
        f"{_format_bytes(size)} on disk"
    )
    kwargs: dict[str, Any] = {
        "repo_id": DATASET_REPOSITORY,
        "repo_type": "dataset",
        "revision": revision,
        "filename": filename,
        "local_dir": root,
        "token": token,
    }
    if download_timeout is not None:
        kwargs["etag_timeout"] = download_timeout

    def detail() -> str:
        count, current = _local_progress(root, [filename])
        return f"{count}/1 files complete, {_format_bytes(current)} on disk"

    with progress_heartbeat(f"download {filename}", progress_interval, detail):
        downloader(**kwargs)
    if not destination.is_file():
        raise RuntimeError(f"Downloaded dataset shard is missing: {destination}")
    return destination


def download_model(
    paths: BundlePaths,
    *,
    revision: str,
    max_workers: int,
    progress_interval: float,
    download_timeout: int | None,
    token: str | None,
    snapshot: Any = None,
) -> None:
    if snapshot is None:
        from huggingface_hub import snapshot_download

        snapshot = snapshot_download

    status("ensuring BGE-M3 model before dataset ingestion")
    kwargs: dict[str, Any] = {
        "repo_id": MODEL_REPOSITORY,
        "repo_type": "model",
        "revision": revision,
        "local_dir": paths.model_dir,
        "allow_patterns": MODEL_ALLOW_PATTERNS,
        "ignore_patterns": MODEL_IGNORE_PATTERNS,
        "token": token,
        "max_workers": max_workers,
    }
    if download_timeout is not None:
        kwargs["etag_timeout"] = download_timeout

    def detail() -> str:
        count, size = _local_progress(paths.model_dir, None)
        return f"{count} files complete, {_format_bytes(size)} on disk"

    with progress_heartbeat("model download", progress_interval, detail):
        snapshot(**kwargs)
    if not model_is_downloaded(paths):
        raise RuntimeError(f"Downloaded model is incomplete: {paths.model_dir}")
