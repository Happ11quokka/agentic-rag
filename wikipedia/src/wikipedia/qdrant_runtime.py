from __future__ import annotations

import shutil  # noqa: F401  (tests patch qdrant_runtime.shutil.which)
from pathlib import Path

from .docker_runtime import UNSUPPORTED_FILESYSTEMS as UNSUPPORTED_QDRANT_FILESYSTEMS
from .docker_runtime import (
    docker as _docker,
)
from .docker_runtime import (
    ensure_compatible_storage,
    ensure_docker,
)
from .docker_runtime import (
    url_ready as _url_ready,
)
from .docker_runtime import (
    wait_until_ready as _wait_until_ready,
)

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_GRPC_PORT = 6334
DEFAULT_QDRANT_CONTAINER = "wikipedia-qdrant"
DEFAULT_QDRANT_IMAGE = "qdrant/qdrant:latest"


def _ensure_compatible_storage(storage: Path) -> None:
    ensure_compatible_storage(storage, engine="Qdrant")


def _ensure_docker() -> None:
    ensure_docker("Qdrant")


def ensure_qdrant(
    url: str,
    *,
    storage_dir: str | Path | None = None,
    container: str = DEFAULT_QDRANT_CONTAINER,
    image: str = DEFAULT_QDRANT_IMAGE,
) -> str:
    local = url.rstrip("/") == DEFAULT_QDRANT_URL
    if not local:
        if _url_ready(url):
            return "already running"
        raise RuntimeError(f"Qdrant is not reachable: {url}")
    if storage_dir is None:
        raise RuntimeError(
            "Local Qdrant storage is not configured; run "
            "`uv run wikipedia-ingest qdrant --bundle-dir /absolute/path` first"
        )

    storage = Path(storage_dir).expanduser()
    if not storage.is_absolute():
        raise ValueError("Qdrant storage directory must be an absolute path")
    storage = storage.resolve()
    _ensure_compatible_storage(storage)
    storage.mkdir(parents=True, exist_ok=True)
    _ensure_docker()

    existing = _docker("container", "inspect", container)
    container_exists = existing.returncode == 0
    if container_exists:
        mount = _docker(
            "container",
            "inspect",
            "--format",
            '{{range .Mounts}}{{if eq .Destination "/qdrant/storage"}}{{.Source}}{{end}}{{end}}',
            container,
        )
        source = mount.stdout.strip() if mount.returncode == 0 else ""
        if not source or Path(source).resolve() != storage:
            print(f"qdrant: reconfiguring {container} storage -> {storage}", flush=True)
            removed = _docker("rm", "-f", container)
            if removed.returncode != 0:
                detail = removed.stderr.strip() or removed.stdout.strip()
                raise RuntimeError(f"Could not replace Qdrant container: {detail}")
            container_exists = False
        elif _url_ready(url):
            return "already running"

    if container_exists:
        print(f"qdrant: starting existing container {container}", flush=True)
        started = _docker("start", container)
    else:
        print(f"qdrant: creating container {container} with storage {storage}", flush=True)
        started = _docker(
            "run",
            "-d",
            "--name",
            container,
            "-p",
            "6333:6333",
            "-p",
            "6334:6334",
            "-v",
            f"{storage}:/qdrant/storage",
            image,
        )
    if started.returncode != 0:
        detail = started.stderr.strip() or started.stdout.strip()
        raise RuntimeError(f"Could not start Qdrant container: {detail}")
    if not _wait_until_ready(url):
        raise RuntimeError("Qdrant did not become ready within 120 seconds")
    return "started"
