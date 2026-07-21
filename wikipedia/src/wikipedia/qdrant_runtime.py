from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_GRPC_PORT = 6334
DEFAULT_QDRANT_CONTAINER = "wikipedia-qdrant"
DEFAULT_QDRANT_IMAGE = "qdrant/qdrant:latest"

UNSUPPORTED_QDRANT_FILESYSTEMS = {
    "cifs",
    "exfat",
    "msdos",
    "nfs",
    "ntfs",
    "smbfs",
    "vfat",
}


def _filesystem_type(path: Path) -> str | None:
    target = path
    while not target.exists() and target != target.parent:
        target = target.parent
    result = subprocess.run(
        ["mount"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None

    resolved = target.resolve()
    matches: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        match = re.match(r".+ on (.+?) \(([^, )]+)", line)
        if match is None:
            match = re.match(r".+ on (.+?) type ([^ ]+)", line)
        if match is None:
            continue
        mount_point = Path(match.group(1))
        if resolved == mount_point or mount_point in resolved.parents:
            matches.append((len(mount_point.parts), match.group(2).lower()))
    return max(matches)[1] if matches else None


def _ensure_compatible_storage(storage: Path) -> None:
    filesystem = _filesystem_type(storage)
    if filesystem not in UNSUPPORTED_QDRANT_FILESYSTEMS:
        return
    raise RuntimeError(
        f"Qdrant storage {storage} is on {filesystem}, but Qdrant requires "
        "block-level access on a POSIX-compatible filesystem. Use APFS or ext4 "
        "storage instead of ExFAT, NTFS, or a network filesystem."
    )


def _url_ready(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=1):
            return True
    except OSError:
        return False


def _wait_until_ready(url: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _url_ready(url):
            return True
        time.sleep(1)
    return False


def _docker(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _ensure_docker() -> None:
    if shutil.which("docker") is None:
        raise RuntimeError("Docker is required to start local Qdrant")
    if _docker("info").returncode == 0:
        return
    if sys.platform != "darwin":
        raise RuntimeError("Docker daemon is not running")
    print("qdrant: starting Docker Desktop", flush=True)
    subprocess.run(
        ["open", "-a", "Docker"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if _docker("info").returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError("Docker Desktop did not become ready within 120 seconds")


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
